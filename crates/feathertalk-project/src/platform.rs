use crate::ProjectError;
use std::path::Path;

pub fn replace_file_atomic(temp: &Path, destination: &Path) -> Result<(), ProjectError> {
    #[cfg(unix)]
    {
        std::fs::rename(temp, destination).map_err(|source| ProjectError::Io {
            operation: "rename",
            path: destination.to_path_buf(),
            source,
        })
    }
    #[cfg(windows)]
    {
        replace_windows(temp, destination)
    }
}

pub fn sync_parent_directory(_parent: &Path) -> Result<(), ProjectError> {
    #[cfg(unix)]
    {
        let file = std::fs::File::open(_parent).map_err(|source| ProjectError::Io {
            operation: "open_parent",
            path: _parent.to_path_buf(),
            source,
        })?;
        file.sync_all().map_err(|source| ProjectError::Io {
            operation: "sync_parent",
            path: _parent.to_path_buf(),
            source,
        })
    }
    #[cfg(windows)]
    {
        Ok(())
    }
}

#[cfg(windows)]
fn replace_windows(temp: &Path, destination: &Path) -> Result<(), ProjectError> {
    use std::os::windows::ffi::OsStrExt;
    use windows_sys::Win32::Storage::FileSystem::{
        MOVEFILE_WRITE_THROUGH, MoveFileExW, ReplaceFileW,
    };
    let wide = |p: &Path| {
        p.as_os_str()
            .encode_wide()
            .chain(std::iter::once(0))
            .collect::<Vec<_>>()
    };
    let src = wide(temp);
    let dst = wide(destination);
    let ok = if destination.exists() {
        unsafe {
            ReplaceFileW(
                dst.as_ptr(),
                src.as_ptr(),
                std::ptr::null(),
                0,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            )
        }
    } else {
        unsafe { MoveFileExW(src.as_ptr(), dst.as_ptr(), MOVEFILE_WRITE_THROUGH) }
    };
    if ok == 0 {
        return Err(ProjectError::Io {
            operation: "replace",
            path: destination.to_path_buf(),
            source: std::io::Error::last_os_error(),
        });
    }
    Ok(())
}
