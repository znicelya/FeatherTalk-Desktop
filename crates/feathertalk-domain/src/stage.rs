use serde::{Deserialize, Serialize};

use crate::{ErrorCode, TaskStatus};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(
    tag = "stage",
    content = "data",
    rename_all = "snake_case",
    deny_unknown_fields
)]
pub enum TaskStage {
    Queued,
    Preparing,
    ExtractingAudio,
    ExtractingFrames,
    DetectingFaces,
    ExtractingFeatures,
    Training { epoch: u32, step: u64, loss: f64 },
    Importing,
    Exporting,
    Rendering { frame: u64, total: u64 },
    Completed,
    Failed { code: ErrorCode, message: String },
    Cancelled,
}

impl TaskStage {
    /// One sample per variant, for exhaustive tests. Data-carrying variants use
    /// arbitrary but fixed payloads.
    pub const ALL_UNIT_SAMPLES: [Self; 13] = [
        Self::Queued,
        Self::Preparing,
        Self::ExtractingAudio,
        Self::ExtractingFrames,
        Self::DetectingFaces,
        Self::ExtractingFeatures,
        Self::Training {
            epoch: 0,
            step: 0,
            loss: 0.0,
        },
        Self::Importing,
        Self::Exporting,
        Self::Rendering { frame: 0, total: 1 },
        Self::Completed,
        Self::Failed {
            code: ErrorCode::WorkerCrashed,
            message: String::new(),
        },
        Self::Cancelled,
    ];

    pub fn status(&self) -> TaskStatus {
        match self {
            Self::Queued => TaskStatus::Queued,
            Self::Completed => TaskStatus::Completed,
            Self::Failed { .. } => TaskStatus::Failed,
            Self::Cancelled => TaskStatus::Cancelled,
            Self::Preparing
            | Self::ExtractingAudio
            | Self::ExtractingFrames
            | Self::DetectingFaces
            | Self::ExtractingFeatures
            | Self::Training { .. }
            | Self::Importing
            | Self::Exporting
            | Self::Rendering { .. } => TaskStatus::Running,
        }
    }

    pub fn is_terminal(&self) -> bool {
        matches!(
            self,
            Self::Completed | Self::Failed { .. } | Self::Cancelled
        )
    }

    pub fn as_slug(&self) -> &'static str {
        match self {
            Self::Queued => "queued",
            Self::Preparing => "preparing",
            Self::ExtractingAudio => "extracting_audio",
            Self::ExtractingFrames => "extracting_frames",
            Self::DetectingFaces => "detecting_faces",
            Self::ExtractingFeatures => "extracting_features",
            Self::Training { .. } => "training",
            Self::Importing => "importing",
            Self::Exporting => "exporting",
            Self::Rendering { .. } => "rendering",
            Self::Completed => "completed",
            Self::Failed { .. } => "failed",
            Self::Cancelled => "cancelled",
        }
    }
}
