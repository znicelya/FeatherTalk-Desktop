use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::TaskKind;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TrainingMode {
    Baseline,
    MouthRoi,
    Temporal,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UnetVariant {
    OriginalUnet,
    MobileOneUnet,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LegacyModelKind {
    FeatherHubert,
    Pfld,
    OriginalUnet,
    MobileOneUnet,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OnnxExportKind {
    FeatherHubert,
    OriginalUnet,
    MobileOneUnet,
}

macro_rules! params {
    ($name:ident { $($field:ident : $ty:ty),* $(,)? }) => {
        #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
        #[serde(deny_unknown_fields)]
        pub struct $name {
            $(pub $field: $ty),*
        }
    };
}

params!(ProbeMediaParams { input: PathBuf });
params!(NormalizeMediaParams {
    input: PathBuf,
    output_dir: PathBuf
});
params!(ProjectDirParams {
    project_dir: PathBuf
});
params!(ExtractFramesParams {
    project_dir: PathBuf,
    video: PathBuf
});
params!(ExtractFeaturesParams {
    project_dir: PathBuf,
    audio: PathBuf
});
params!(InspectModelParams { source: PathBuf });
params!(ExportModelPackageParams {
    source: PathBuf,
    destination: PathBuf
});
params!(MigrateLegacyFeaturesParams {
    source: PathBuf,
    destination: PathBuf
});
params!(ImportLegacyModelParams {
    source: PathBuf,
    kind: LegacyModelKind,
    destination: PathBuf
});
params!(ExportOnnxParams {
    source: PathBuf,
    kind: OnnxExportKind,
    destination: PathBuf
});

/// Keep existing clients and newly opened training forms on one sample per step.
pub const DEFAULT_BATCH_SIZE: u32 = 1;

fn default_batch_size() -> u32 {
    DEFAULT_BATCH_SIZE
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrainParams {
    pub project_dir: PathBuf,
    pub mode: TrainingMode,
    pub variant: UnetVariant,
    pub epochs: u32,
    /// Samples per optimizer step. Older requests omit this field.
    #[serde(default = "default_batch_size")]
    pub batch_size: u32,
    pub resume: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RenderParams {
    pub project_dir: PathBuf,
    pub checkpoint: PathBuf,
    pub audio: PathBuf,
    pub output: PathBuf,
    /// Fixed-width `u64` frame cap used by the JSON wire contract.
    ///
    /// `None` renders the full sequence; `Some(n)` caps output frames and is how
    /// a short preview is requested. Preview and full render share this one path.
    /// A worker mapping this value to inference's local `Option<usize>` must use
    /// checked conversion and reject values that do not fit.
    pub max_output_frames: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(
    tag = "command",
    content = "params",
    rename_all = "snake_case",
    deny_unknown_fields
)]
pub enum Request {
    ProbeMedia(ProbeMediaParams),
    NormalizeMedia(NormalizeMediaParams),
    ValidateProject(ProjectDirParams),
    LockAssetPackage(ProjectDirParams),
    ExtractFrames(ExtractFramesParams),
    ExtractFeatures(ExtractFeaturesParams),
    Train(TrainParams),
    Render(RenderParams),
    InspectModel(InspectModelParams),
    ImportLegacyModel(ImportLegacyModelParams),
    ExportModelPackage(ExportModelPackageParams),
    ExportOnnx(ExportOnnxParams),
    MigrateLegacyFeatures(MigrateLegacyFeaturesParams),
}

impl Request {
    pub fn kind(&self) -> TaskKind {
        match self {
            Self::ProbeMedia(_) => TaskKind::ProbeMedia,
            Self::NormalizeMedia(_) => TaskKind::NormalizeMedia,
            Self::ValidateProject(_) => TaskKind::ValidateProject,
            Self::LockAssetPackage(_) => TaskKind::LockAssetPackage,
            Self::ExtractFrames(_) => TaskKind::ExtractFrames,
            Self::ExtractFeatures(_) => TaskKind::ExtractFeatures,
            Self::Train(_) => TaskKind::Train,
            Self::Render(_) => TaskKind::Render,
            Self::InspectModel(_) => TaskKind::InspectModel,
            Self::ImportLegacyModel(_) => TaskKind::ImportLegacyModel,
            Self::ExportModelPackage(_) => TaskKind::ExportModelPackage,
            Self::ExportOnnx(_) => TaskKind::ExportOnnx,
            Self::MigrateLegacyFeatures(_) => TaskKind::MigrateLegacyFeatures,
        }
    }
}
