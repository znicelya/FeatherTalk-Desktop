use std::path::Path;

pub(crate) fn rename_noreplace(source: &Path, destination: &Path) -> std::io::Result<()> {
    platform::rename_noreplace(source, destination)
}

#[cfg(unix)]
mod platform {
    use std::{
        fs,
        io::{self, ErrorKind},
        path::Path};

    use rustix::{
        fs::{CWD, RenameFlags, renameat_with},
        io::Errno};

    pub(super) fn rename_noreplace(source: &Path, destination: &Path) -> std::io::Result<()> {
        match renameat_with(CWD, source, CWD, destination, RenameFlags::NOREPLACE) {
            Ok(()) => Ok(()),
            Err(error) => {
                let error = io::Error::from(error);
                if retryable_rename_error(&error) {
                    rename_fallback(source, destination)
                } else {
                    Err(error)
                }
            }
        }
    }

    /// Some overlay/NFS filesystems reject `RENAME_NOREPLACE` even when a
    /// normal rename would work. The fallback rechecks for the destination so
    /// the no-clobber contract is preserved where the filesystem allows.
    fn rename_fallback(source: &Path, destination: &Path) -> io::Result<()> {
        ensure_absent(destination)?;
        match fs::rename(source, destination) {
            Ok(()) => Ok(()),
            Err(error) if cross_device_error(&error) => copy_noreplace(source, destination),
            Err(error) => Err(error)}
    }

    fn copy_noreplace(source: &Path, destination: &Path) -> io::Result<()> {
        ensure_absent(destination)?;
        let mut from = fs::File::open(source)?;
        let mut to = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(destination)?;
        let copy_result = io::copy(&mut from, &mut to).and_then(|_| to.sync_all());
        if let Err(error) = copy_result {
            drop(to);
            let _ = fs::remove_file(destination);
            return Err(error);
        }
        drop(to);
        fs::remove_file(source)
    }

    fn ensure_absent(path: &Path) -> io::Result<()> {
        match fs::symlink_metadata(path) {
            Ok(_) => Err(io::Error::from(ErrorKind::AlreadyExists)),
            Err(error) if error.kind() == ErrorKind::NotFound => Ok(()),
            Err(error) => Err(error)}
    }

    fn retryable_rename_error(error: &io::Error) -> bool {
        unsupported_flag_error(error) || cross_device_error(error)
    }

    fn unsupported_flag_error(error: &io::Error) -> bool {
        matches_os_error(
            error,
            &[Errno::INVAL, Errno::NOSYS, Errno::OPNOTSUPP, Errno::PERM],
        )
    }

    fn cross_device_error(error: &io::Error) -> bool {
        matches_os_error(error, &[Errno::XDEV])
    }

    fn matches_os_error(error: &io::Error, codes: &[Errno]) -> bool {
        let Some(code) = error.raw_os_error() else {
            return false;
        };
        codes.iter().any(|errno| errno.raw_os_error() == code)
    }

    #[cfg(test)]
    mod tests {
        use super::{copy_noreplace, rename_fallback, retryable_rename_error};
        use rustix::io::Errno;
        use std::io;

        #[test]
        fn fallback_preserves_an_existing_destination() {
            let directory = tempfile::tempdir().unwrap();
            let source = directory.path().join("source.tmp");
            let destination = directory.path().join("destination.mp4");
            std::fs::write(&source, b"staging").unwrap();
            std::fs::write(&destination, b"sentinel").unwrap();

            assert!(rename_fallback(&source, &destination).is_err());
            assert_eq!(std::fs::read(&destination).unwrap(), b"sentinel");
            assert_eq!(std::fs::read(&source).unwrap(), b"staging");
        }

        #[test]
        fn fallback_moves_into_an_absent_destination() {
            let directory = tempfile::tempdir().unwrap();
            let source = directory.path().join("source.tmp");
            let destination = directory.path().join("destination.mp4");
            std::fs::write(&source, b"staging").unwrap();

            rename_fallback(&source, &destination).unwrap();
            assert!(!source.exists());
            assert_eq!(std::fs::read(destination).unwrap(), b"staging");
        }

        #[test]
        fn copy_fallback_preserves_an_existing_destination() {
            let directory = tempfile::tempdir().unwrap();
            let source = directory.path().join("source.tmp");
            let destination = directory.path().join("destination.mp4");
            std::fs::write(&source, b"staging").unwrap();
            std::fs::write(&destination, b"sentinel").unwrap();

            assert!(copy_noreplace(&source, &destination).is_err());
            assert_eq!(std::fs::read(&destination).unwrap(), b"sentinel");
            assert_eq!(std::fs::read(&source).unwrap(), b"staging");
        }

        #[test]
        fn copy_fallback_moves_into_an_absent_destination() {
            let directory = tempfile::tempdir().unwrap();
            let source = directory.path().join("source.tmp");
            let destination = directory.path().join("destination.mp4");
            std::fs::write(&source, b"staging").unwrap();

            copy_noreplace(&source, &destination).unwrap();
            assert!(!source.exists());
            assert_eq!(std::fs::read(destination).unwrap(), b"staging");
        }

        #[test]
        fn overlay_rename_failures_are_retried() {
            for errno in [
                Errno::INVAL,
                Errno::NOSYS,
                Errno::OPNOTSUPP,
                Errno::PERM,
                Errno::XDEV,
            ] {
                let error = io::Error::from_raw_os_error(errno.raw_os_error());
                assert!(
                    retryable_rename_error(&error),
                    "{errno:?} should fall back instead of failing the publish"
                );
            }
            let unrelated = io::Error::from_raw_os_error(Errno::BUSY.raw_os_error());
            assert!(!retryable_rename_error(&unrelated));
        }
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
