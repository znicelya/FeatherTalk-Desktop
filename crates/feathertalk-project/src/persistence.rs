use crate::platform::{replace_file_atomic, sync_parent_directory};
use crate::{AssetManifest, ProjectError, ProjectManifest};
use std::{
    fs::{self, File, OpenOptions},
    io::{Read, Write},
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};

const MAX_MANIFEST_BYTES: usize = 1_048_576;
const MAX_TEMP_ATTEMPTS: u64 = 32;
static COUNTER: AtomicU64 = AtomicU64::new(0);

pub fn read_project_manifest(path: &Path) -> Result<ProjectManifest, ProjectError> {
    let manifest: ProjectManifest = read_json(path)?;
    manifest.validate()?;
    Ok(manifest)
}
pub fn read_asset_manifest(path: &Path) -> Result<AssetManifest, ProjectError> {
    let manifest: AssetManifest = read_json(path)?;
    match manifest.state {
        crate::AssetPackageState::Preparing => manifest.validate_preparing()?,
        crate::AssetPackageState::Locked => manifest.validate_locked()?,
    }
    Ok(manifest)
}

fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T, ProjectError> {
    let mut bytes = Vec::new();
    File::open(path)
        .map_err(|e| io_err("open", path, e))?
        .take((MAX_MANIFEST_BYTES + 1) as u64)
        .read_to_end(&mut bytes)
        .map_err(|e| io_err("read", path, e))?;
    if bytes.len() > MAX_MANIFEST_BYTES {
        return Err(ProjectError::ManifestTooLarge {
            path: path.to_path_buf(),
            limit: MAX_MANIFEST_BYTES,
        });
    }
    let json = std::str::from_utf8(&bytes).map_err(|_| ProjectError::InvalidUtf8 {
        path: path.to_path_buf(),
    })?;
    serde_json::from_str(json).map_err(|source| ProjectError::InvalidJson {
        path: path.to_path_buf(),
        source,
    })
}

pub fn write_project_manifest_atomic(
    path: &Path,
    manifest: &ProjectManifest,
) -> Result<(), ProjectError> {
    manifest.validate()?;
    write_atomic(path, manifest)
}
pub fn write_asset_manifest_atomic(
    path: &Path,
    manifest: &AssetManifest,
) -> Result<(), ProjectError> {
    if path
        .symlink_metadata()
        .is_ok_and(|m| m.file_type().is_symlink())
    {
        return Err(ProjectError::Symlink {
            path: path.to_path_buf(),
        });
    }
    if path.exists()
        && let Ok(old) = read_asset_manifest(path)
        && old.validate_locked().is_ok()
    {
        return Err(ProjectError::LockedAssetMutation {
            path: path.to_path_buf(),
        });
    }
    match manifest.state {
        crate::AssetPackageState::Preparing => manifest.validate_preparing()?,
        crate::AssetPackageState::Locked => manifest.validate_locked()?,
    }
    write_atomic(path, manifest)
}

fn write_atomic<T: serde::Serialize>(path: &Path, value: &T) -> Result<(), ProjectError> {
    let parent = path.parent().ok_or_else(|| ProjectError::InvalidField {
        field: "path".into(),
        message: "missing parent".into(),
    })?;
    reject_symlink_components(parent)?;
    fs::create_dir_all(parent).map_err(|e| io_err("create_dir", parent, e))?;
    if path
        .symlink_metadata()
        .is_ok_and(|metadata| metadata.file_type().is_symlink())
    {
        return Err(ProjectError::Symlink {
            path: path.to_path_buf(),
        });
    }
    let mut bytes = serde_json::to_vec_pretty(value).map_err(|e| ProjectError::InvalidField {
        field: "manifest".into(),
        message: e.to_string(),
    })?;
    bytes.push(b'\n');
    if bytes.len() > MAX_MANIFEST_BYTES {
        return Err(ProjectError::ManifestTooLarge {
            path: path.to_path_buf(),
            limit: MAX_MANIFEST_BYTES,
        });
    }
    let (temp, mut file) = create_temp_file(parent, path)?;
    file.write_all(&bytes)
        .and_then(|_| file.flush())
        .and_then(|_| file.sync_all())
        .map_err(|e| io_err("write_temp", &temp, e))?;
    drop(file);
    if let Err(error) = replace_file_atomic(&temp, path) {
        let _ = fs::remove_file(&temp);
        return Err(error);
    }
    sync_parent_directory(parent)?;
    Ok(())
}

fn create_temp_file(parent: &Path, destination: &Path) -> Result<(PathBuf, File), ProjectError> {
    for _ in 0..MAX_TEMP_ATTEMPTS {
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let temp = parent.join(format!(
            ".{}.{}.{}.tmp",
            destination
                .file_name()
                .and_then(|x| x.to_str())
                .unwrap_or("manifest"),
            std::process::id(),
            n
        ));
        match OpenOptions::new().write(true).create_new(true).open(&temp) {
            Ok(file) => return Ok((temp, file)),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(io_err("create_temp", &temp, error)),
        }
    }
    Err(ProjectError::AtomicReplacementUnsupported {
        path: destination.to_path_buf(),
    })
}

fn reject_symlink_components(parent: &Path) -> Result<(), ProjectError> {
    let mut current = PathBuf::new();
    for component in parent.components() {
        current.push(component.as_os_str());
        if let Ok(metadata) = fs::symlink_metadata(&current)
            && metadata.file_type().is_symlink()
        {
            return Err(ProjectError::Symlink { path: current });
        }
    }
    Ok(())
}
fn io_err(operation: &'static str, path: &Path, source: std::io::Error) -> ProjectError {
    ProjectError::Io {
        operation,
        path: PathBuf::from(path),
        source,
    }
}
