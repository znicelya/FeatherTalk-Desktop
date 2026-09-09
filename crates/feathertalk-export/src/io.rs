use std::{
    fs::{self, File, OpenOptions},
    io::{Read, Write},
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};

use sha2::{Digest, Sha256};

use crate::{
    FileManifest, LICENSE_FILE_NAME, MANIFEST_FILE_NAME, MAX_LICENSE_BYTES, MAX_MANIFEST_BYTES,
    MAX_MODEL_BYTES, MODEL_FILE_NAME, PackageError,
};

static NEXT_STAGING_ID: AtomicU64 = AtomicU64::new(1);

pub(crate) struct StagingDirectory {
    path: PathBuf,
    armed: bool,
}

impl StagingDirectory {
    pub(crate) fn path(&self) -> &Path {
        &self.path
    }

    pub(crate) fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for StagingDirectory {
    fn drop(&mut self) {
        if self.armed {
            let _ = fs::remove_dir_all(&self.path);
        }
    }
}

pub(crate) fn validate_parent(path: &Path) -> Result<&Path, PackageError> {
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    reject_symlink_components(parent)?;
    let metadata = fs::symlink_metadata(parent).map_err(|error| {
        if error.kind() == std::io::ErrorKind::NotFound {
            PackageError::InvalidRequest(format!(
                "destination parent does not exist: {}",
                parent.display()
            ))
        } else {
            PackageError::Io(error)
        }
    })?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        return Err(PackageError::InvalidRequest(format!(
            "destination parent must be a real directory: {}",
            parent.display()
        )));
    }
    Ok(parent)
}

pub(crate) fn reject_symlink_components(path: &Path) -> Result<(), PackageError> {
    let mut current = PathBuf::new();
    for component in path.components() {
        current.push(component.as_os_str());
        match fs::symlink_metadata(&current) {
            Ok(metadata) if metadata.file_type().is_symlink() => {
                return Err(PackageError::InvalidRequest(format!(
                    "path component must not be a symbolic link: {}",
                    current.display()
                )));
            }
            Ok(_) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => break,
            Err(error) => return Err(PackageError::Io(error)),
        }
    }
    Ok(())
}

pub(crate) fn ensure_destination_absent(path: &Path) -> Result<(), PackageError> {
    match fs::symlink_metadata(path) {
        Ok(_) => Err(PackageError::InvalidRequest(format!(
            "destination already exists: {}",
            path.display()
        ))),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(PackageError::Io(error)),
    }
}

pub(crate) fn create_staging_directory(parent: &Path) -> Result<StagingDirectory, PackageError> {
    let process_id = std::process::id();
    for _ in 0..1024 {
        let id = NEXT_STAGING_ID.fetch_add(1, Ordering::Relaxed);
        let path = parent.join(format!(".feathertalk-model-{process_id}-{id}.staging"));
        match fs::create_dir(&path) {
            Ok(()) => return Ok(StagingDirectory { path, armed: true }),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(PackageError::Io(error)),
        }
    }
    Err(PackageError::Publication(
        "unable to allocate a unique staging directory".to_owned(),
    ))
}

pub(crate) fn read_bounded_regular(
    path: &Path,
    max_bytes: u64,
    label: &str,
) -> Result<Vec<u8>, PackageError> {
    reject_symlink_components(path)?;
    let metadata = fs::symlink_metadata(path)?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(PackageError::InvalidRequest(format!(
            "{label} must be a regular non-symlink file: {}",
            path.display()
        )));
    }
    if metadata.len() > max_bytes {
        return Err(PackageError::InvalidRequest(format!(
            "{label} exceeds {max_bytes} bytes: {}",
            path.display()
        )));
    }
    let mut file = File::open(path)?;
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)?;
    if u64::try_from(bytes.len()).unwrap_or(u64::MAX) > max_bytes {
        return Err(PackageError::InvalidRequest(format!(
            "{label} exceeds {max_bytes} bytes while reading"
        )));
    }
    Ok(bytes)
}

pub(crate) fn write_synced_create_new(path: &Path, bytes: &[u8]) -> Result<(), PackageError> {
    let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
    file.write_all(bytes)?;
    file.flush()?;
    file.sync_all()?;
    Ok(())
}

pub(crate) fn sync_regular_file(path: &Path) -> Result<(), PackageError> {
    let metadata = fs::symlink_metadata(path)?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(PackageError::Publication(format!(
            "file to sync must be a regular non-symlink file: {}",
            path.display()
        )));
    }
    OpenOptions::new()
        .read(true)
        .write(true)
        .open(path)?
        .sync_all()?;
    Ok(())
}

pub(crate) fn sha256_file(path: &Path) -> Result<(u64, String), PackageError> {
    let mut file = File::open(path)?;
    let mut digest = Sha256::new();
    let mut bytes = 0_u64;
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        bytes = bytes
            .checked_add(u64::try_from(count).expect("buffer count fits u64"))
            .ok_or_else(|| PackageError::Publication("file size overflowed u64".to_owned()))?;
        digest.update(&buffer[..count]);
    }
    Ok((bytes, hex::encode(digest.finalize())))
}

pub(crate) fn file_manifest(
    path: &Path,
    expected_name: &str,
) -> Result<FileManifest, PackageError> {
    let metadata = fs::symlink_metadata(path)?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(PackageError::Publication(format!(
            "staged file must be a regular file: {}",
            path.display()
        )));
    }
    let (bytes, sha256) = sha256_file(path)?;
    let max_bytes = max_bytes_for(expected_name);
    if bytes > max_bytes {
        return Err(PackageError::Publication(format!(
            "{expected_name} exceeds {max_bytes} bytes"
        )));
    }
    let manifest = FileManifest {
        file_name: expected_name.to_owned(),
        bytes,
        sha256,
    };
    manifest.validate(expected_name)?;
    Ok(manifest)
}

pub(crate) fn exact_directory_entries(directory: &Path) -> Result<Vec<String>, PackageError> {
    let mut entries = fs::read_dir(directory)?
        .map(|entry| {
            entry.and_then(|entry| {
                entry.file_name().into_string().map_err(|_| {
                    std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        "package entry name is not valid UTF-8",
                    )
                })
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    entries.sort();
    Ok(entries)
}

pub(crate) fn validate_package_directory(
    directory: &Path,
    training: bool,
) -> Result<(), PackageError> {
    reject_symlink_components(directory)?;
    let metadata = fs::symlink_metadata(directory)?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        return Err(PackageError::InvalidRequest(format!(
            "package directory must be a real directory: {}",
            directory.display()
        )));
    }
    let mut expected = vec![
        LICENSE_FILE_NAME.to_owned(),
        MANIFEST_FILE_NAME.to_owned(),
        MODEL_FILE_NAME.to_owned(),
    ];
    if training {
        expected.push(crate::OPTIMIZER_FILE_NAME.to_owned());
        expected.push(crate::TRAINING_STATE_FILE_NAME.to_owned());
    }
    expected.sort();
    let actual = exact_directory_entries(directory)?;
    if actual != expected {
        return Err(PackageError::InvalidRequest(format!(
            "package directory entries must be exactly {expected:?}, got {actual:?}"
        )));
    }
    for entry in &actual {
        let path = directory.join(entry);
        let metadata = fs::symlink_metadata(&path)?;
        if metadata.file_type().is_symlink() || !metadata.is_file() {
            return Err(PackageError::InvalidRequest(format!(
                "package entry must be a regular non-symlink file: {}",
                path.display()
            )));
        }
    }
    Ok(())
}

pub(crate) fn validate_declared_file(
    path: &Path,
    declared: &FileManifest,
) -> Result<(), PackageError> {
    let max_bytes = max_bytes_for(&declared.file_name);
    let metadata = fs::symlink_metadata(path)?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(PackageError::InvalidRequest(format!(
            "declared package file is not regular: {}",
            path.display()
        )));
    }
    if metadata.len() > max_bytes {
        return Err(PackageError::InvalidRequest(format!(
            "package file exceeds {max_bytes} bytes: {}",
            path.display()
        )));
    }
    let (bytes, actual) = sha256_file(path)?;
    if bytes != declared.bytes {
        return Err(PackageError::HashMismatch {
            file: declared.file_name.clone(),
            expected: format!("{} bytes", declared.bytes),
            actual: format!("{bytes} bytes"),
        });
    }
    if actual != declared.sha256 {
        return Err(PackageError::HashMismatch {
            file: declared.file_name.clone(),
            expected: declared.sha256.clone(),
            actual,
        });
    }
    Ok(())
}

pub(crate) fn sync_directory(path: &Path) -> Result<(), PackageError> {
    #[cfg(unix)]
    {
        File::open(path)?.sync_all()?;
    }
    #[cfg(not(unix))]
    {
        let _ = path;
    }
    Ok(())
}

pub(crate) fn publish_no_clobber(
    mut staging: StagingDirectory,
    destination: &Path,
) -> Result<(), PackageError> {
    ensure_destination_absent(destination)?;
    rename_noreplace(staging.path(), destination).map_err(|error| {
        PackageError::Publication(format!(
            "rename staging {} to {}: {error}",
            staging.path().display(),
            destination.display()
        ))
    })?;
    staging.disarm();
    sync_directory(
        destination
            .parent()
            .filter(|path| !path.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new(".")),
    )?;
    Ok(())
}

fn max_bytes_for(file_name: &str) -> u64 {
    match file_name {
        MANIFEST_FILE_NAME => MAX_MANIFEST_BYTES,
        LICENSE_FILE_NAME => MAX_LICENSE_BYTES,
        MODEL_FILE_NAME => MAX_MODEL_BYTES,
        _ => MAX_MODEL_BYTES,
    }
}

fn rename_noreplace(source: &Path, destination: &Path) -> std::io::Result<()> {
    platform::rename_noreplace(source, destination)
}

#[cfg(unix)]
mod platform {
    use std::path::Path;

    use rustix::fs::{CWD, RenameFlags, renameat_with};

    pub(super) fn rename_noreplace(source: &Path, destination: &Path) -> std::io::Result<()> {
        renameat_with(CWD, source, CWD, destination, RenameFlags::NOREPLACE)
            .map_err(std::io::Error::from)
    }
}

#[cfg(windows)]
mod platform {
    use std::{iter, os::windows::ffi::OsStrExt, path::Path};

    use windows_sys::Win32::Storage::FileSystem::{MOVEFILE_WRITE_THROUGH, MoveFileExW};

    pub(super) fn rename_noreplace(source: &Path, destination: &Path) -> std::io::Result<()> {
        let wide = |path: &Path| {
            path.as_os_str()
                .encode_wide()
                .chain(iter::once(0))
                .collect::<Vec<_>>()
        };
        let source = wide(source);
        let destination = wide(destination);
        if unsafe {
            MoveFileExW(
                source.as_ptr(),
                destination.as_ptr(),
                MOVEFILE_WRITE_THROUGH,
            )
        } == 0
        {
            return Err(std::io::Error::last_os_error());
        }
        Ok(())
    }
}

#[cfg(not(any(unix, windows)))]
mod platform {
    use std::path::Path;

    pub(super) fn rename_noreplace(_source: &Path, _destination: &Path) -> std::io::Result<()> {
        Err(std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            "atomic no-replace rename is unsupported on this platform",
        ))
    }
}
