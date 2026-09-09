use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use time::{OffsetDateTime, format_description::well_known::Rfc3339};

use crate::ProjectError;

const CURRENT_SCHEMA_VERSION: u32 = 1;
const MAX_IDENTIFIER_BYTES: usize = 128;
const MAX_DISPLAY_NAME_CHARS: usize = 256;
const MAX_TASK_HISTORY: usize = 10_000;
const MAX_FRAME_COUNT: u64 = 100_000_000;
const MAX_FRAME_DIMENSION: u32 = 32_768;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProjectManifest {
    pub schema_version: u32,
    pub project_id: String,
    pub display_name: String,
    pub asset_package: String,
    pub default_model: ModelSelection,
    pub task_history: Vec<TaskHistoryEntry>,
}

impl ProjectManifest {
    pub fn validate(&self) -> Result<(), ProjectError> {
        if self.schema_version != CURRENT_SCHEMA_VERSION {
            return Err(ProjectError::UnsupportedSchemaVersion {
                path: "project.json".into(),
                version: self.schema_version,
            });
        }
        validate_identifier("project_id", &self.project_id)?;
        if self.display_name.trim() != self.display_name
            || self.display_name.is_empty()
            || self.display_name.chars().count() > MAX_DISPLAY_NAME_CHARS
        {
            return Err(invalid(
                "display_name",
                "must be trimmed and 1-256 characters",
            ));
        }
        validate_relative_manifest_path(&self.asset_package)?;
        if self.asset_package != "assets/assets.json" {
            return Err(invalid("asset_package", "must be assets/assets.json"));
        }
        if self.task_history.len() > MAX_TASK_HISTORY {
            return Err(invalid("task_history", "too many entries"));
        }
        let mut task_ids = HashSet::new();
        for entry in &self.task_history {
            validate_identifier("task_id", &entry.task_id)?;
            validate_kind(&entry.kind)?;
            OffsetDateTime::parse(&entry.updated_at, &Rfc3339)
                .map_err(|_| invalid("updated_at", "must be RFC 3339"))?;
            if !task_ids.insert(&entry.task_id) {
                return Err(invalid("task_id", "duplicate task id"));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelSelection {
    OriginalUnet,
    MobileOneUnet,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TaskHistoryEntry {
    pub task_id: String,
    pub kind: String,
    pub status: TaskHistoryStatus,
    pub updated_at: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskHistoryStatus {
    Queued,
    Running,
    Completed,
    Failed,
    Cancelled,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AssetManifest {
    pub schema_version: u32,
    pub state: AssetPackageState,
    pub video_fps: u32,
    pub audio_sample_rate: u32,
    pub audio_channels: u16,
    pub frame_count: u64,
    pub frame_width: u32,
    pub frame_height: u32,
    pub feature_type: FeatureType,
    pub feature_shape: [u64; 3],
    pub landmark_model_sha256: String,
    pub feature_model_sha256: String,
}

impl AssetManifest {
    pub fn validate_preparing(&self) -> Result<(), ProjectError> {
        self.validate_common()?;
        if !matches!(self.state, AssetPackageState::Preparing) {
            return Err(invalid("state", "expected preparing"));
        }
        if !valid_progress_shape(self.feature_shape) {
            return Err(invalid(
                "feature_shape",
                "must be [0,0,0] or [tokens,2,1024]",
            ));
        }
        if !matches!(self.video_fps, 0 | 25) {
            return Err(invalid("video_fps", "must be 0 or 25 while preparing"));
        }
        if !matches!(self.audio_sample_rate, 0 | 16_000) {
            return Err(invalid(
                "audio_sample_rate",
                "must be 0 or 16000 while preparing",
            ));
        }
        if !matches!(self.audio_channels, 0 | 1) {
            return Err(invalid("audio_channels", "must be 0 or 1 while preparing"));
        }
        if self.feature_shape[0] != 0
            && self.frame_count != 0
            && self.feature_shape[0] != self.frame_count
        {
            return Err(invalid(
                "feature_shape",
                "token count must match frame_count",
            ));
        }
        validate_optional_sha256("landmark_model_sha256", &self.landmark_model_sha256)?;
        validate_optional_sha256("feature_model_sha256", &self.feature_model_sha256)?;
        Ok(())
    }

    pub fn validate_locked(&self) -> Result<(), ProjectError> {
        self.validate_common()?;
        if !matches!(self.state, AssetPackageState::Locked) {
            return Err(invalid("state", "expected locked"));
        }
        if self.video_fps != 25 {
            return Err(invalid("video_fps", "must be 25"));
        }
        if self.audio_sample_rate != 16_000 {
            return Err(invalid("audio_sample_rate", "must be 16000"));
        }
        if self.audio_channels != 1 {
            return Err(invalid("audio_channels", "must be 1"));
        }
        if self.frame_count == 0 || self.frame_width == 0 || self.frame_height == 0 {
            return Err(invalid("frame_count", "locked dimensions must be non-zero"));
        }
        if self.feature_shape != [self.frame_count, 2, 1024] {
            return Err(invalid("feature_shape", "must be [frame_count,2,1024]"));
        }
        validate_sha256("landmark_model_sha256", &self.landmark_model_sha256)?;
        validate_sha256("feature_model_sha256", &self.feature_model_sha256)?;
        Ok(())
    }

    fn validate_common(&self) -> Result<(), ProjectError> {
        if self.schema_version != CURRENT_SCHEMA_VERSION {
            return Err(ProjectError::UnsupportedSchemaVersion {
                path: "assets/assets.json".into(),
                version: self.schema_version,
            });
        }
        if self.frame_count > MAX_FRAME_COUNT {
            return Err(invalid("frame_count", "exceeds maximum"));
        }
        if self.frame_width > MAX_FRAME_DIMENSION || self.frame_height > MAX_FRAME_DIMENSION {
            return Err(invalid("frame_width", "exceeds maximum dimension"));
        }
        if !matches!(self.feature_type, FeatureType::FeatherHubert) {
            return Err(invalid("feature_type", "unsupported feature type"));
        }
        Ok(())
    }
}

pub fn validate_relative_manifest_path(value: &str) -> Result<(), ProjectError> {
    if value.is_empty()
        || value.contains('\\')
        || value.starts_with('/')
        || value.contains(':')
        || value
            .split('/')
            .any(|part| part.is_empty() || part == "." || part == "..")
    {
        return Err(ProjectError::UnsafeRelativePath {
            path: value.to_owned(),
        });
    }
    Ok(())
}

fn validate_identifier(field: &str, value: &str) -> Result<(), ProjectError> {
    if value.is_empty()
        || value.len() > MAX_IDENTIFIER_BYTES
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-'))
    {
        return Err(invalid(field, "must be 1-128 ASCII identifier characters"));
    }
    Ok(())
}

fn validate_kind(value: &str) -> Result<(), ProjectError> {
    if value.is_empty()
        || value.len() > 64
        || !value.bytes().all(|byte| {
            byte.is_ascii_lowercase() || byte.is_ascii_digit() || matches!(byte, b'_' | b'-')
        })
    {
        return Err(invalid("kind", "must be 1-64 lowercase ASCII characters"));
    }
    Ok(())
}

fn validate_sha256(field: &str, value: &str) -> Result<(), ProjectError> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    {
        return Err(invalid(
            field,
            "must be 64 lowercase hexadecimal characters",
        ));
    }
    Ok(())
}

fn validate_optional_sha256(field: &str, value: &str) -> Result<(), ProjectError> {
    if value.is_empty() {
        Ok(())
    } else {
        validate_sha256(field, value)
    }
}

fn valid_progress_shape(shape: [u64; 3]) -> bool {
    shape == [0, 0, 0] || (shape[0] != 0 && shape[1] == 2 && shape[2] == 1024)
}

fn invalid(field: &str, message: &str) -> ProjectError {
    ProjectError::InvalidField {
        field: field.to_owned(),
        message: message.to_owned(),
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AssetPackageState {
    Preparing,
    Locked,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FeatureType {
    FeatherHubert,
}
