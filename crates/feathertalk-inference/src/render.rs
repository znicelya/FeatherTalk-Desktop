use std::{
    ffi::OsString,
    fs,
    path::{Component, Path, PathBuf},
};

use crate::InferenceError;

const OUTPUT_FPS: u32 = 25;
const MAX_TASK_ID_BYTES: usize = 64;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RenderGeometry {
    crop_size: u32,
    inner_size: u32,
    border: u32,
}

impl RenderGeometry {
    pub fn standard() -> Self {
        let crop = feathertalk_preprocess::default_crop_spec();
        let geometry = Self {
            crop_size: crop.crop_size,
            inner_size: crop.inner_size,
            border: crop.border,
        };
        debug_assert_eq!(
            geometry.crop_size,
            geometry.inner_size + 2 * geometry.border
        );
        geometry
    }

    pub fn crop_size(&self) -> u32 {
        self.crop_size
    }

    pub fn inner_size(&self) -> u32 {
        self.inner_size
    }

    pub fn border(&self) -> u32 {
        self.border
    }

    pub fn replacement_offset(&self) -> (u32, u32) {
        (self.border, self.border)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RawFrameRenderSpec {
    width: u32,
    height: u32,
    audio_path: PathBuf,
    output_path: PathBuf,
}

impl RawFrameRenderSpec {
    pub fn new(
        width: u32,
        height: u32,
        audio_path: &Path,
        output_path: &Path,
    ) -> Result<Self, InferenceError> {
        if width == 0 {
            return Err(invalid("width", "must be greater than zero"));
        }
        if height == 0 {
            return Err(invalid("height", "must be greater than zero"));
        }
        if audio_path.as_os_str().is_empty() {
            return Err(invalid("audio_path", "must not be empty"));
        }
        if output_path.as_os_str().is_empty() {
            return Err(invalid("output_path", "must not be empty"));
        }
        Ok(Self {
            width,
            height,
            audio_path: audio_path.to_owned(),
            output_path: output_path.to_owned(),
        })
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    pub fn fps(&self) -> u32 {
        OUTPUT_FPS
    }

    pub fn audio_path(&self) -> &Path {
        &self.audio_path
    }

    pub fn output_path(&self) -> &Path {
        &self.output_path
    }
}

pub fn validate_output_destination(path: &Path) -> Result<(), InferenceError> {
    if path.as_os_str().is_empty() {
        return Err(invalid("output_path", "must not be empty"));
    }
    reject_symlink_components(path)?;
    let parent = path
        .parent()
        .ok_or_else(|| InferenceError::OutputParentInvalid {
            path: path.to_owned(),
        })?;
    let parent_metadata =
        fs::symlink_metadata(parent).map_err(|_| InferenceError::OutputParentInvalid {
            path: parent.to_owned(),
        })?;
    if parent_metadata.file_type().is_symlink() || !parent_metadata.is_dir() {
        return Err(InferenceError::OutputParentInvalid {
            path: parent.to_owned(),
        });
    }
    match fs::symlink_metadata(path) {
        Ok(metadata) if metadata.file_type().is_symlink() => Err(InferenceError::OutputSymlink {
            path: path.to_owned(),
        }),
        Ok(metadata) if metadata.is_file() => Err(InferenceError::OutputExists {
            path: path.to_owned(),
        }),
        Ok(_) => Err(InferenceError::OutputNotRegular {
            path: path.to_owned(),
        }),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(_) => Err(InferenceError::OutputParentInvalid {
            path: path.to_owned(),
        }),
    }
}

pub fn staging_output_path(path: &Path, task_id: &str) -> Result<PathBuf, InferenceError> {
    validate_output_destination(path)?;
    validate_task_id(task_id)?;
    let parent = path
        .parent()
        .ok_or_else(|| InferenceError::OutputParentInvalid {
            path: path.to_owned(),
        })?;
    let stem = path
        .file_stem()
        .ok_or_else(|| invalid("output_path", "must have a file name"))?;
    let extension = path.extension().map(|value| {
        let mut ext = OsString::from(".");
        ext.push(value);
        ext
    });
    let mut name = OsString::from(".");
    name.push(stem);
    name.push(".");
    name.push(task_id);
    name.push(".staging");
    if let Some(extension) = extension {
        name.push(extension);
    }
    let staging = parent.join(name);
    if fs::symlink_metadata(&staging).is_ok() {
        return Err(InferenceError::OutputExists { path: staging });
    }
    Ok(staging)
}

fn reject_symlink_components(path: &Path) -> Result<(), InferenceError> {
    let mut current = PathBuf::new();
    for component in path.components() {
        match component {
            Component::Prefix(prefix) => current.push(prefix.as_os_str()),
            Component::RootDir => current.push(component.as_os_str()),
            _ => current.push(component.as_os_str()),
        }
        match fs::symlink_metadata(&current) {
            Ok(metadata) if metadata.file_type().is_symlink() => {
                return Err(InferenceError::OutputSymlink { path: current });
            }
            Ok(_) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => break,
            Err(_) => break,
        }
    }
    Ok(())
}

fn validate_task_id(task_id: &str) -> Result<(), InferenceError> {
    if task_id.is_empty()
        || task_id.len() > MAX_TASK_ID_BYTES
        || task_id == "."
        || task_id == ".."
        || !task_id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-'))
    {
        return Err(InferenceError::InvalidTaskId {
            task_id: task_id.to_owned(),
        });
    }
    Ok(())
}

fn invalid(field: &'static str, message: &str) -> InferenceError {
    InferenceError::InvalidField {
        field,
        message: message.to_owned(),
    }
}
