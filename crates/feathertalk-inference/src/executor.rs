use std::{
    collections::BTreeSet,
    fs::{self, File, OpenOptions},
    path::{Path, PathBuf},
};

use burn::tensor::backend::Backend;
use feathertalk_audio::{FeatureMatrix, read_feature_file};
use feathertalk_models::unet::TalkingHeadModel;
use feathertalk_preprocess::{compute_face_bbox, read_landmarks};

use crate::{
    FrameReader, InferenceError, RawFrameRenderSpec, RawVideoSinkFactory, RenderGeometry,
    RenderPlan, publish::rename_noreplace, raw_video_command, render_planned_frame,
    staging_output_path, validate_output_destination,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OfflineRenderRequest {
    frame_dir: PathBuf,
    landmark_dir: PathBuf,
    feature_path: PathBuf,
    audio_path: PathBuf,
    ffmpeg_path: PathBuf,
    output_path: PathBuf,
    task_id: String,
    source_frame_count: usize,
    max_output_frames: Option<usize>,
}

impl OfflineRenderRequest {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        frame_dir: PathBuf,
        landmark_dir: PathBuf,
        feature_path: PathBuf,
        audio_path: PathBuf,
        ffmpeg_path: PathBuf,
        output_path: PathBuf,
        task_id: impl Into<String>,
        source_frame_count: usize,
        max_output_frames: Option<usize>,
    ) -> Result<Self, InferenceError> {
        let task_id = task_id.into();
        for (field, path) in [
            ("frame_dir", &frame_dir),
            ("landmark_dir", &landmark_dir),
            ("feature_path", &feature_path),
            ("audio_path", &audio_path),
            ("ffmpeg_path", &ffmpeg_path),
            ("output_path", &output_path),
        ] {
            validate_absolute_non_empty(field, path)?;
        }
        if source_frame_count < 2 {
            return Err(InferenceError::FrameCountTooSmall {
                actual: source_frame_count,
                minimum: 2,
            });
        }
        if max_output_frames == Some(0) {
            return Err(InferenceError::InvalidField {
                field: "max_output_frames",
                message: "must be greater than zero when provided".into(),
            });
        }
        // Reuse the established destination and task-id contract without creating a file.
        staging_output_path(&output_path, &task_id)?;
        Ok(Self {
            frame_dir,
            landmark_dir,
            feature_path,
            audio_path,
            ffmpeg_path,
            output_path,
            task_id,
            source_frame_count,
            max_output_frames,
        })
    }

    pub fn frame_dir(&self) -> &Path {
        &self.frame_dir
    }

    pub fn landmark_dir(&self) -> &Path {
        &self.landmark_dir
    }

    pub fn feature_path(&self) -> &Path {
        &self.feature_path
    }

    pub fn audio_path(&self) -> &Path {
        &self.audio_path
    }

    pub fn ffmpeg_path(&self) -> &Path {
        &self.ffmpeg_path
    }

    pub fn output_path(&self) -> &Path {
        &self.output_path
    }

    pub fn task_id(&self) -> &str {
        &self.task_id
    }

    pub fn source_frame_count(&self) -> usize {
        self.source_frame_count
    }

    pub fn max_output_frames(&self) -> Option<usize> {
        self.max_output_frames
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OfflineRenderResult {
    output_path: PathBuf,
    frame_count: usize,
    width: u32,
    height: u32,
}

impl OfflineRenderResult {
    pub fn output_path(&self) -> &Path {
        &self.output_path
    }

    pub fn frame_count(&self) -> usize {
        self.frame_count
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }
}

pub fn execute_offline_render<B, M, R, F>(
    model: &M,
    device: &B::Device,
    request: &OfflineRenderRequest,
    frame_reader: &R,
    sink_factory: &F,
) -> Result<OfflineRenderResult, InferenceError>
where
    B: Backend,
    M: TalkingHeadModel<B>,
    R: FrameReader + ?Sized,
    F: RawVideoSinkFactory + ?Sized,
{
    validate_request_inputs(request)?;

    let features = read_feature_file(request.feature_path()).map_err(|error| {
        invalid_artifact("feature_path", request.feature_path(), error.to_string())
    })?;
    validate_features(&features)?;
    let feature_frame_count = features.tokens() / 2;
    let plan = RenderPlan::new(
        request.source_frame_count(),
        feature_frame_count,
        request.max_output_frames(),
    )?;
    validate_planned_artifacts(request, &plan)?;

    let first_frame_path = frame_path(request, 0);
    let first_frame = frame_reader.read(0, &first_frame_path)?;
    let width = first_frame.width();
    let height = first_frame.height();

    let staging_path = staging_output_path(request.output_path(), request.task_id())
        .map_err(|error| map_staging_error(error, request.output_path()))?;
    let mut staging = reserve_staging(&staging_path)?;
    let render_spec = RawFrameRenderSpec::new(width, height, request.audio_path(), staging.path())?;
    let command = raw_video_command(request.ffmpeg_path(), &render_spec)?;
    let mut sink = sink_factory.start(&command)?;
    let geometry = RenderGeometry::standard();

    for output_index in 0..plan.output_frame_count() {
        let frame_plan = plan.frame(output_index)?;
        let source_index = frame_plan.source_frame_index;
        let source_path = frame_path(request, source_index);
        let frame = if source_index == 0 && output_index == 0 {
            first_frame.clone()
        } else {
            frame_reader.read(source_index, &source_path)?
        };
        if frame.width() != width || frame.height() != height {
            return Err(InferenceError::FrameDimensionsMismatch {
                index: source_index,
                expected_width: width,
                expected_height: height,
                actual_width: frame.width(),
                actual_height: frame.height(),
            });
        }
        let landmark_path = landmark_path(request, source_index);
        validate_input_file(&landmark_path, "landmark_path")?;
        let landmarks = read_landmarks(&landmark_path).map_err(|error| {
            invalid_artifact("landmark_path", &landmark_path, error.to_string())
        })?;
        let bbox = compute_face_bbox(&landmarks).map_err(|error| {
            invalid_artifact("landmark_path", &landmark_path, error.to_string())
        })?;
        let rendered = render_planned_frame::<B, M>(
            model,
            &frame,
            &bbox,
            &features,
            &frame_plan,
            &geometry,
            device,
        )?;
        sink.write_frame(&rendered)?;
    }
    sink.finish()?;
    verify_staging_output(staging.path())?;
    publish_staging(staging.path(), request.output_path())?;
    staging.disarm();
    Ok(OfflineRenderResult {
        output_path: request.output_path().to_owned(),
        frame_count: plan.output_frame_count(),
        width,
        height,
    })
}

fn validate_request_inputs(request: &OfflineRenderRequest) -> Result<(), InferenceError> {
    validate_output_destination(request.output_path())?;
    validate_directory(request.frame_dir(), "frame_dir")?;
    validate_directory(request.landmark_dir(), "landmark_dir")?;
    validate_input_file(request.feature_path(), "feature_path")?;
    validate_input_file(request.audio_path(), "audio_path")?;
    validate_input_file(request.ffmpeg_path(), "ffmpeg_path")?;
    Ok(())
}

fn validate_planned_artifacts(
    request: &OfflineRenderRequest,
    plan: &RenderPlan,
) -> Result<(), InferenceError> {
    let mut source_indexes = BTreeSet::new();
    for output_index in 0..plan.output_frame_count() {
        source_indexes.insert(plan.frame(output_index)?.source_frame_index);
    }
    for source_index in source_indexes {
        validate_input_file(&frame_path(request, source_index), "frame_path")?;
        validate_input_file(&landmark_path(request, source_index), "landmark_path")?;
    }
    Ok(())
}

fn validate_directory(path: &Path, field: &'static str) -> Result<(), InferenceError> {
    reject_symlink_components(path, field)?;
    let metadata =
        fs::symlink_metadata(path).map_err(|_| InferenceError::InvalidInputDirectory {
            field,
            path: path.to_owned(),
        })?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        return Err(InferenceError::InvalidInputDirectory {
            field,
            path: path.to_owned(),
        });
    }
    Ok(())
}

fn validate_input_file(path: &Path, field: &'static str) -> Result<(), InferenceError> {
    reject_symlink_components(path, field)?;
    let metadata = fs::symlink_metadata(path)
        .map_err(|_| invalid_artifact(field, path, "missing or invalid filesystem entry"))?;
    if metadata.file_type().is_symlink() || !metadata.is_file() || metadata.len() == 0 {
        return Err(invalid_artifact(
            field,
            path,
            "must be a regular non-symlink file",
        ));
    }
    Ok(())
}

fn reject_symlink_components(path: &Path, field: &'static str) -> Result<(), InferenceError> {
    let mut current = PathBuf::new();
    for component in path.components() {
        current.push(component.as_os_str());
        match fs::symlink_metadata(&current) {
            Ok(metadata) if metadata.file_type().is_symlink() => {
                if field == "frame_dir" || field == "landmark_dir" {
                    return Err(InferenceError::InvalidInputDirectory {
                        field,
                        path: current,
                    });
                }
                return Err(invalid_artifact(
                    field,
                    &current,
                    "path contains a symbolic link component",
                ));
            }
            Ok(_) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => break,
            Err(error) => {
                if field == "frame_dir" || field == "landmark_dir" {
                    return Err(InferenceError::InvalidInputDirectory {
                        field,
                        path: current,
                    });
                }
                return Err(invalid_artifact(field, &current, error.to_string()));
            }
        }
    }
    Ok(())
}

fn validate_features(features: &FeatureMatrix) -> Result<(), InferenceError> {
    if features.tokens() == 0 || features.dims() != 1024 || !features.tokens().is_multiple_of(2) {
        return Err(InferenceError::InvalidFeatureShape {
            tokens: features.tokens(),
            dims: features.dims(),
        });
    }
    Ok(())
}

fn frame_path(request: &OfflineRenderRequest, index: usize) -> PathBuf {
    request.frame_dir().join(format!("{index:06}.jpg"))
}

fn landmark_path(request: &OfflineRenderRequest, index: usize) -> PathBuf {
    request.landmark_dir().join(format!("{index:06}.lms"))
}

struct StagingGuard {
    path: PathBuf,
    armed: bool,
}

impl StagingGuard {
    fn path(&self) -> &Path {
        &self.path
    }

    fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for StagingGuard {
    fn drop(&mut self) {
        if self.armed {
            let _ = fs::remove_file(&self.path);
        }
    }
}

fn reserve_staging(path: &Path) -> Result<StagingGuard, InferenceError> {
    match OpenOptions::new().write(true).create_new(true).open(path) {
        Ok(_) => Ok(StagingGuard {
            path: path.to_owned(),
            armed: true,
        }),
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
            Err(InferenceError::StagingCollision {
                path: path.to_owned(),
            })
        }
        Err(error) => Err(InferenceError::StagingOutputInvalid {
            path: path.to_owned(),
            message: error.to_string(),
        }),
    }
}

fn verify_staging_output(path: &Path) -> Result<(), InferenceError> {
    let metadata =
        fs::symlink_metadata(path).map_err(|error| InferenceError::StagingOutputInvalid {
            path: path.to_owned(),
            message: error.to_string(),
        })?;
    if metadata.file_type().is_symlink() || !metadata.is_file() || metadata.len() == 0 {
        return Err(InferenceError::StagingOutputInvalid {
            path: path.to_owned(),
            message: "must be a regular non-empty non-symlink file".into(),
        });
    }
    OpenOptions::new()
        .read(true)
        .write(true)
        .open(path)
        .and_then(|file| file.sync_all())
        .map_err(|error| InferenceError::StagingOutputInvalid {
            path: path.to_owned(),
            message: error.to_string(),
        })
}

fn publish_staging(staging: &Path, destination: &Path) -> Result<(), InferenceError> {
    if let Err(error) = validate_output_destination(destination) {
        return Err(InferenceError::AtomicPublishFailed {
            path: destination.to_owned(),
            message: error.to_string(),
        });
    }
    rename_noreplace(staging, destination).map_err(|error| {
        InferenceError::AtomicPublishFailed {
            path: destination.to_owned(),
            message: error.to_string(),
        }
    })?;
    if let Some(parent) = destination.parent()
        && let Ok(directory) = File::open(parent)
    {
        let _ = directory.sync_all();
    }
    Ok(())
}

fn map_staging_error(error: InferenceError, output: &Path) -> InferenceError {
    match error {
        InferenceError::OutputExists { path } if path == output => {
            InferenceError::OutputExists { path }
        }
        InferenceError::OutputExists { path } => InferenceError::StagingCollision { path },
        other => other,
    }
}

fn invalid_artifact(
    field: &'static str,
    path: &Path,
    message: impl Into<String>,
) -> InferenceError {
    InferenceError::InvalidInputArtifact {
        field,
        path: path.to_owned(),
        message: bounded_message(message.into()),
    }
}

const MAX_ERROR_MESSAGE_BYTES: usize = 512;

fn bounded_message(mut message: String) -> String {
    if message.len() <= MAX_ERROR_MESSAGE_BYTES {
        return message;
    }
    let mut end = MAX_ERROR_MESSAGE_BYTES;
    while !message.is_char_boundary(end) {
        end -= 1;
    }
    message.truncate(end);
    message
}

fn validate_absolute_non_empty(field: &'static str, path: &Path) -> Result<(), InferenceError> {
    if path.as_os_str().is_empty() || !path.is_absolute() {
        return Err(InferenceError::InvalidField {
            field,
            message: "must be a non-empty absolute path".into(),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use crate::InferenceError;

    use super::invalid_artifact;

    #[test]
    fn invalid_artifact_messages_are_bounded_on_utf8_boundaries() {
        let error = invalid_artifact("feature_path", Path::new("feature.f32"), "界".repeat(300));
        let InferenceError::InvalidInputArtifact { message, .. } = error else {
            panic!("unexpected error variant");
        };

        assert!(message.len() <= 512);
        assert!(message.is_char_boundary(message.len()));
    }
}
