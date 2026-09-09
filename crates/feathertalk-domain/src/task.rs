use serde::{Deserialize, Deserializer, Serialize, de::Error as DeError};

use crate::DomainError;

pub const TASK_ID_MILLIS_DIGITS: usize = 13;
pub const TASK_ID_SUFFIX_DIGITS: usize = 8;
pub const TASK_ID_LEN: usize = TASK_ID_MILLIS_DIGITS + 1 + TASK_ID_SUFFIX_DIGITS;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[serde(transparent)]
pub struct TaskId(String);

impl TaskId {
    pub fn parse(value: &str) -> Result<Self, DomainError> {
        let invalid = |reason: &str| DomainError::InvalidTaskId {
            reason: reason.to_owned(),
        };
        if value.len() != TASK_ID_LEN {
            return Err(invalid("must be exactly 22 characters"));
        }
        let bytes = value.as_bytes();
        if !bytes[..TASK_ID_MILLIS_DIGITS]
            .iter()
            .all(u8::is_ascii_digit)
        {
            return Err(invalid("millis must be 13 decimal digits"));
        }
        if bytes[TASK_ID_MILLIS_DIGITS] != b'-' {
            return Err(invalid("must separate millis and suffix with '-'"));
        }
        if !bytes[TASK_ID_MILLIS_DIGITS + 1..]
            .iter()
            .all(|byte| byte.is_ascii_digit() || byte.is_ascii_lowercase() && *byte <= b'f')
        {
            return Err(invalid("suffix must be 8 lowercase hex digits"));
        }
        Ok(Self(value.to_owned()))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl<'de> Deserialize<'de> for TaskId {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Self::parse(&value).map_err(D::Error::custom)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskKind {
    ProbeMedia,
    NormalizeMedia,
    ValidateProject,
    LockAssetPackage,
    ExtractFrames,
    ExtractFeatures,
    Train,
    Render,
    InspectModel,
    ImportLegacyModel,
    ExportModelPackage,
    ExportOnnx,
    MigrateLegacyFeatures,
}

impl TaskKind {
    pub const ALL: [Self; 13] = [
        Self::ProbeMedia,
        Self::NormalizeMedia,
        Self::ValidateProject,
        Self::LockAssetPackage,
        Self::ExtractFrames,
        Self::ExtractFeatures,
        Self::Train,
        Self::Render,
        Self::InspectModel,
        Self::ImportLegacyModel,
        Self::ExportModelPackage,
        Self::ExportOnnx,
        Self::MigrateLegacyFeatures,
    ];

    pub fn as_slug(self) -> &'static str {
        match self {
            Self::ProbeMedia => "probe_media",
            Self::NormalizeMedia => "normalize_media",
            Self::ValidateProject => "validate_project",
            Self::LockAssetPackage => "lock_asset_package",
            Self::ExtractFrames => "extract_frames",
            Self::ExtractFeatures => "extract_features",
            Self::Train => "train",
            Self::Render => "render",
            Self::InspectModel => "inspect_model",
            Self::ImportLegacyModel => "import_legacy_model",
            Self::ExportModelPackage => "export_model_package",
            Self::ExportOnnx => "export_onnx",
            Self::MigrateLegacyFeatures => "migrate_legacy_features",
        }
    }

    pub fn from_slug(slug: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|kind| kind.as_slug() == slug)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskStatus {
    Queued,
    Running,
    Completed,
    Failed,
    Cancelled,
}

impl TaskStatus {
    pub const ALL: [Self; 5] = [
        Self::Queued,
        Self::Running,
        Self::Completed,
        Self::Failed,
        Self::Cancelled,
    ];

    pub fn is_incomplete(self) -> bool {
        matches!(self, Self::Queued | Self::Running)
    }
}
