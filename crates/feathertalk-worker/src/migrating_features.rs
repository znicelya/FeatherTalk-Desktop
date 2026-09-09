//! Convert the legacy NumPy feature matrix into the versioned feature artifact.

use std::{fmt, fs};

use feathertalk_audio::{FeatureMatrix, MAX_FEATURE_FILE_BYTES, write_feature_file_no_clobber};
use feathertalk_domain::{MigrateLegacyFeaturesParams, Progress, TaskStage};
use feathertalk_media::CancellationToken;
use ndarray::{ArrayD, Ix3};
use ndarray_npy::ReadNpyExt;

use crate::TaskReporter;

const FEATURE_DIMS: usize = 1024;
const FEATURE_PAIR_WIDTH: usize = 2;

#[derive(Debug)]
pub enum MigrateLegacyFeaturesError {
    Cancelled { stage: TaskStage },
    Failed { detail: String, stage: TaskStage },
}

impl fmt::Display for MigrateLegacyFeaturesError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cancelled { .. } => formatter.write_str("legacy feature migration cancelled"),
            Self::Failed { detail, .. } => formatter.write_str(detail),
        }
    }
}

impl MigrateLegacyFeaturesError {
    pub fn stage(&self) -> TaskStage {
        match self {
            Self::Cancelled { stage } | Self::Failed { stage, .. } => stage.clone(),
        }
    }

    pub fn is_cancelled(&self) -> bool {
        matches!(self, Self::Cancelled { .. })
    }
}

pub fn execute_migrate_legacy_features(
    params: &MigrateLegacyFeaturesParams,
    token: &CancellationToken,
    reporter: &dyn TaskReporter,
) -> Result<serde_json::Value, MigrateLegacyFeaturesError> {
    validate_request(params)?;
    reporter.report(TaskStage::Preparing, None);
    if token.is_cancelled() {
        return Err(MigrateLegacyFeaturesError::Cancelled {
            stage: TaskStage::Preparing,
        });
    }
    reporter.report(
        TaskStage::Importing,
        Some(Progress {
            completed: 0,
            total: Some(1),
        }),
    );

    let file = fs::File::open(&params.source)
        .map_err(|error| failure(TaskStage::Importing, error.to_string()))?;
    let array = ArrayD::<f32>::read_npy(std::io::BufReader::new(file)).map_err(|error| {
        failure(
            TaskStage::Importing,
            format!(
                "invalid NPY {}; expected an f32 array: {error}",
                params.source.display()
            ),
        )
    })?;
    if token.is_cancelled() {
        return Err(MigrateLegacyFeaturesError::Cancelled {
            stage: TaskStage::Importing,
        });
    }
    let rank = array.ndim();
    if rank != 3 {
        return Err(failure(
            TaskStage::Importing,
            format!("invalid NPY rank {rank}; expected rank 3 [video_frames, 2, 1024]"),
        ));
    }
    let array = array.into_dimensionality::<Ix3>().map_err(|error| {
        failure(
            TaskStage::Importing,
            format!("invalid NPY dimensions: {error}"),
        )
    })?;
    let shape = array.shape();
    if shape[0] == 0 || shape[1] != FEATURE_PAIR_WIDTH || shape[2] != FEATURE_DIMS {
        return Err(failure(
            TaskStage::Importing,
            format!(
                "invalid NPY shape {:?}; expected [video_frames, 2, 1024] with video_frames > 0",
                shape
            ),
        ));
    }
    let source_shape = [shape[0], shape[1], shape[2]];
    let tokens = shape[0]
        .checked_mul(shape[1])
        .ok_or_else(|| failure(TaskStage::Importing, "feature token count overflowed usize"))?;
    let values = array.iter().copied().collect();
    let matrix = FeatureMatrix::new(tokens, shape[2], values)
        .map_err(|error| failure(TaskStage::Importing, error.to_string()))?;
    if token.is_cancelled() {
        return Err(MigrateLegacyFeaturesError::Cancelled {
            stage: TaskStage::Importing,
        });
    }
    let artifact = write_feature_file_no_clobber(&params.destination, &matrix)
        .map_err(|error| failure(TaskStage::Importing, error.to_string()))?;
    reporter.report(
        TaskStage::Importing,
        Some(Progress {
            completed: 1,
            total: Some(1),
        }),
    );
    Ok(serde_json::json!({
        "kind": "migrate_legacy_features",
        "source": params.source,
        "destination": params.destination,
        "source_shape": source_shape,
        "tokens": artifact.tokens(),
        "dims": artifact.dims(),
        "bytes": artifact.bytes(),
        "sha256": artifact.sha256(),
    }))
}

fn validate_request(
    params: &MigrateLegacyFeaturesParams,
) -> Result<(), MigrateLegacyFeaturesError> {
    if !params.source.is_absolute() {
        return Err(failure(
            TaskStage::Preparing,
            "source path must be absolute",
        ));
    }
    let metadata = fs::symlink_metadata(&params.source)
        .map_err(|error| failure(TaskStage::Preparing, error.to_string()))?;
    if metadata.file_type().is_symlink() || !metadata.file_type().is_file() {
        return Err(failure(
            TaskStage::Preparing,
            "source must be a regular non-symlink file",
        ));
    }
    let file_name = params
        .source
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| failure(TaskStage::Preparing, "source file name must be valid UTF-8"))?;
    if !file_name.ends_with(".npy") {
        return Err(failure(
            TaskStage::Preparing,
            "legacy feature source must end in .npy",
        ));
    }
    if metadata.len() > MAX_FEATURE_FILE_BYTES {
        return Err(failure(
            TaskStage::Preparing,
            format!(
                "legacy feature source exceeds {MAX_FEATURE_FILE_BYTES} bytes: {}",
                metadata.len()
            ),
        ));
    }
    if !params.destination.is_absolute() {
        return Err(failure(
            TaskStage::Preparing,
            "destination path must be absolute",
        ));
    }
    match fs::symlink_metadata(&params.destination) {
        Ok(_) => {
            return Err(failure(
                TaskStage::Preparing,
                "destination must not already exist",
            ));
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(failure(TaskStage::Preparing, error.to_string())),
    }
    let parent = params
        .destination
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .ok_or_else(|| failure(TaskStage::Preparing, "destination parent is unavailable"))?;
    let parent_metadata = fs::symlink_metadata(parent)
        .map_err(|error| failure(TaskStage::Preparing, error.to_string()))?;
    if parent_metadata.file_type().is_symlink() || !parent_metadata.is_dir() {
        return Err(failure(
            TaskStage::Preparing,
            "destination parent must be an existing non-symlink directory",
        ));
    }
    Ok(())
}

fn failure(stage: TaskStage, detail: impl Into<String>) -> MigrateLegacyFeaturesError {
    MigrateLegacyFeaturesError::Failed {
        detail: detail.into(),
        stage,
    }
}
