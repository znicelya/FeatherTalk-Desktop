use serde::{Deserialize, Serialize};

use crate::{DomainError, TaskStage};

pub const MAX_SUMMARY_CHARS: usize = 200;
pub const MAX_DETAIL_CHARS: usize = 4_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ErrorCode {
    #[serde(rename = "MEDIA_INVALID")]
    MediaInvalid,
    #[serde(rename = "FACE_NOT_FOUND")]
    FaceNotFound,
    #[serde(rename = "LANDMARK_INVALID")]
    LandmarkInvalid,
    #[serde(rename = "FEATURE_SHAPE_MISMATCH")]
    FeatureShapeMismatch,
    #[serde(rename = "MODEL_INCOMPATIBLE")]
    ModelIncompatible,
    #[serde(rename = "GPU_OUT_OF_MEMORY")]
    GpuOutOfMemory,
    #[serde(rename = "GPU_DEVICE_LOST")]
    GpuDeviceLost,
    #[serde(rename = "DISK_SPACE_LOW")]
    DiskSpaceLow,
    #[serde(rename = "WORKER_CRASHED")]
    WorkerCrashed,
    #[serde(rename = "TASK_CANCELLED")]
    TaskCancelled,
}

impl ErrorCode {
    pub const ALL: [Self; 10] = [
        Self::MediaInvalid,
        Self::FaceNotFound,
        Self::LandmarkInvalid,
        Self::FeatureShapeMismatch,
        Self::ModelIncompatible,
        Self::GpuOutOfMemory,
        Self::GpuDeviceLost,
        Self::DiskSpaceLow,
        Self::WorkerCrashed,
        Self::TaskCancelled,
    ];

    pub fn as_wire(self) -> &'static str {
        match self {
            Self::MediaInvalid => "MEDIA_INVALID",
            Self::FaceNotFound => "FACE_NOT_FOUND",
            Self::LandmarkInvalid => "LANDMARK_INVALID",
            Self::FeatureShapeMismatch => "FEATURE_SHAPE_MISMATCH",
            Self::ModelIncompatible => "MODEL_INCOMPATIBLE",
            Self::GpuOutOfMemory => "GPU_OUT_OF_MEMORY",
            Self::GpuDeviceLost => "GPU_DEVICE_LOST",
            Self::DiskSpaceLow => "DISK_SPACE_LOW",
            Self::WorkerCrashed => "WORKER_CRASHED",
            Self::TaskCancelled => "TASK_CANCELLED",
        }
    }

    pub fn default_recovery(self) -> Recovery {
        match self {
            Self::MediaInvalid => Recovery::Retry,
            Self::FaceNotFound | Self::LandmarkInvalid => Recovery::ExcludeBadFrames,
            Self::FeatureShapeMismatch => Recovery::Retry,
            Self::ModelIncompatible => Recovery::ReimportModel,
            Self::GpuOutOfMemory => Recovery::SelectDifferentAdapter,
            Self::GpuDeviceLost | Self::WorkerCrashed => Recovery::ResumeFromCheckpoint,
            Self::DiskSpaceLow => Recovery::FreeDiskSpace,
            Self::TaskCancelled => Recovery::NotRecoverable,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Recovery {
    Retry,
    ResumeFromCheckpoint,
    FreeDiskSpace,
    SelectDifferentAdapter,
    ExcludeBadFrames,
    ReimportModel,
    NotRecoverable,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TaskError {
    pub code: ErrorCode,
    pub summary: String,
    pub detail: String,
    pub stage: TaskStage,
    pub recovery: Recovery,
}

impl TaskError {
    pub fn new(code: ErrorCode, summary: &str, detail: &str, stage: TaskStage) -> Self {
        Self {
            code,
            summary: summary.to_owned(),
            detail: detail.to_owned(),
            stage,
            recovery: code.default_recovery(),
        }
    }

    pub fn validate(&self) -> Result<(), DomainError> {
        let summary_chars = self.summary.trim().chars().count();
        if summary_chars == 0 || self.summary.chars().count() > MAX_SUMMARY_CHARS {
            return Err(DomainError::InvalidField {
                field: "summary",
                reason: format!("must be 1-{MAX_SUMMARY_CHARS} characters after trimming"),
            });
        }
        if self.detail.chars().count() > MAX_DETAIL_CHARS {
            return Err(DomainError::InvalidField {
                field: "detail",
                reason: format!("must be at most {MAX_DETAIL_CHARS} characters"),
            });
        }
        if self.stage.is_terminal() {
            return Err(DomainError::InvalidField {
                field: "stage",
                reason: "must be the stage the failure occurred in, not a terminal stage".into(),
            });
        }
        Ok(())
    }
}
