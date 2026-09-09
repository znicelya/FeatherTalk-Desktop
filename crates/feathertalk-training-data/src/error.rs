use std::path::PathBuf;

use feathertalk_training::TrainingError;

#[derive(Debug, thiserror::Error)]
pub enum TrainingDataError {
    #[error("invalid training project at {path}: {message}")]
    Project { path: PathBuf, message: String },
    #[error("unable to read audio features from {path}: {message}")]
    Features { path: PathBuf, message: String },
    #[error(
        "feature file {path} holds {actual_tokens} tokens of {dims} dims but the asset package declares {expected_tokens} tokens"
    )]
    FeatureShape {
        path: PathBuf,
        expected_tokens: usize,
        actual_tokens: usize,
        dims: usize,
    },
    #[error("frame index {index} is out of range for {frame_count} frames")]
    FrameIndexOutOfRange { index: u64, frame_count: u64 },
    #[error("unable to read frame {index} from {path}: {message}")]
    Frame {
        index: usize,
        path: PathBuf,
        message: String,
    },
    #[error("unable to read landmarks for frame {index} from {path}: {message}")]
    Landmarks {
        index: usize,
        path: PathBuf,
        message: String,
    },
    #[error("unable to build the training sample for frame {index}: {message}")]
    Sample { index: usize, message: String },
    #[error("unable to stack a training batch: {message}")]
    Batch { message: String },
}

impl From<TrainingDataError> for TrainingError {
    fn from(error: TrainingDataError) -> Self {
        TrainingError::InvalidInput(error.to_string())
    }
}
