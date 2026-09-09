use std::path::Path;

pub(crate) fn rename_noreplace(source: &Path, destination: &Path) -> std::io::Result<()> {
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

#[cfg(test)]
mod tests {
    use super::rename_noreplace;

    #[test]
    fn rename_noreplace_preserves_existing_destination() {
        let directory = tempfile::tempdir().unwrap();
        let source = directory.path().join("source.tmp");
        let destination = directory.path().join("destination.mp4");
        std::fs::write(&source, b"staging").unwrap();
        std::fs::write(&destination, b"sentinel").unwrap();

        assert!(rename_noreplace(&source, &destination).is_err());
        assert_eq!(std::fs::read(&destination).unwrap(), b"sentinel");
        assert_eq!(std::fs::read(&source).unwrap(), b"staging");
    }

    #[test]
    fn rename_noreplace_moves_into_an_absent_destination() {
        let directory = tempfile::tempdir().unwrap();
        let source = directory.path().join("source.tmp");
        let destination = directory.path().join("destination.mp4");
        std::fs::write(&source, b"staging").unwrap();

        rename_noreplace(&source, &destination).unwrap();
        assert!(!source.exists());
        assert_eq!(std::fs::read(destination).unwrap(), b"staging");
    }
}
