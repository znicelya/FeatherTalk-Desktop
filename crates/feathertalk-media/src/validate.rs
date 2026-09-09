use std::{
    fs,
    path::{Component, Path, PathBuf},
};

use crate::{MediaError, MediaInput, NormalizationSpec, NormalizedMediaLayout, ValidatedInput};

pub fn validate_input(input: &MediaInput) -> Result<ValidatedInput, MediaError> {
    reject_existing_symlink_components(&input.source)?;
    let metadata = match fs::symlink_metadata(&input.source) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            return Err(MediaError::InputMissing {
                path: input.source.clone(),
            });
        }
        Err(source) => return Err(io_error("stat_input", &input.source, source)),
    };
    if metadata.file_type().is_symlink() {
        return Err(MediaError::SymlinkNotAllowed {
            path: input.source.clone(),
        });
    }
    if !metadata.is_file() {
        return Err(MediaError::InputNotRegularFile {
            path: input.source.clone(),
        });
    }
    let canonical = fs::canonicalize(&input.source)
        .map_err(|source| io_error("canonicalize_input", &input.source, source))?;
    Ok(ValidatedInput::new(canonical))
}

pub fn validate_normalization(
    input: &ValidatedInput,
    spec: &NormalizationSpec,
) -> Result<NormalizedMediaLayout, MediaError> {
    validate_target(
        "target_video_fps",
        "25",
        spec.target_video_fps.to_string(),
        spec.target_video_fps == 25,
    )?;
    validate_target(
        "target_audio_sample_rate",
        "16000",
        spec.target_audio_sample_rate.to_string(),
        spec.target_audio_sample_rate == 16_000,
    )?;
    validate_target(
        "target_audio_channels",
        "1",
        spec.target_audio_channels.to_string(),
        spec.target_audio_channels == 1,
    )?;

    reject_existing_symlink_components(&spec.output_dir)?;
    if fs::symlink_metadata(&spec.output_dir).is_ok_and(|metadata| !metadata.is_dir()) {
        return Err(MediaError::OutputDirectoryInvalid {
            path: spec.output_dir.clone(),
        });
    }
    fs::create_dir_all(&spec.output_dir)
        .map_err(|source| io_error("create_output_dir", &spec.output_dir, source))?;
    reject_existing_symlink_components(&spec.output_dir)?;
    let output_metadata = fs::symlink_metadata(&spec.output_dir)
        .map_err(|source| io_error("stat_output_dir", &spec.output_dir, source))?;
    if output_metadata.file_type().is_symlink() || !output_metadata.is_dir() {
        return Err(MediaError::OutputDirectoryInvalid {
            path: spec.output_dir.clone(),
        });
    }
    let output_dir = fs::canonicalize(&spec.output_dir)
        .map_err(|source| io_error("canonicalize_output_dir", &spec.output_dir, source))?;
    let video_path = output_dir.join("video_25fps.mp4");
    let audio_path = output_dir.join("audio_16k_mono.wav");
    validate_destination(&video_path, input.source())?;
    validate_destination(&audio_path, input.source())?;
    if input.source().starts_with(&output_dir) {
        return Err(MediaError::OutputInsideInput {
            input: input.source().to_path_buf(),
            output: output_dir,
        });
    }
    Ok(NormalizedMediaLayout::new(
        output_dir, video_path, audio_path,
    ))
}

fn validate_destination(path: &Path, source: &Path) -> Result<(), MediaError> {
    if path == source {
        return Err(MediaError::OutputConflictsWithInput {
            path: path.to_path_buf(),
        });
    }
    match fs::symlink_metadata(path) {
        Ok(metadata) if metadata.file_type().is_symlink() || !metadata.is_file() => {
            Err(MediaError::OutputDestinationInvalid {
                path: path.to_path_buf(),
            })
        }
        Ok(_) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(source_error) => Err(io_error("stat_output", path, source_error)),
    }
}

fn validate_target(
    field: &'static str,
    expected: &'static str,
    actual: String,
    valid: bool,
) -> Result<(), MediaError> {
    if valid {
        Ok(())
    } else {
        Err(MediaError::UnsupportedTarget {
            field,
            expected,
            actual,
        })
    }
}

fn reject_existing_symlink_components(path: &Path) -> Result<(), MediaError> {
    let mut current = PathBuf::new();
    for component in path.components() {
        match component {
            Component::Prefix(_) | Component::RootDir => current.push(component.as_os_str()),
            _ => current.push(component.as_os_str()),
        }
        match fs::symlink_metadata(&current) {
            Ok(metadata) if metadata.file_type().is_symlink() => {
                return Err(MediaError::SymlinkNotAllowed { path: current });
            }
            Ok(_) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => break,
            Err(source) => return Err(io_error("stat_path_component", &current, source)),
        }
    }
    Ok(())
}

fn io_error(operation: &'static str, path: &Path, source: std::io::Error) -> MediaError {
    MediaError::Io {
        operation,
        path: path.to_path_buf(),
        source,
    }
}
