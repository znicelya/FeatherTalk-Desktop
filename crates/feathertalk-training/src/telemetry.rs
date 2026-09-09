use serde::{Deserialize, Serialize};

use crate::{TrainingError, TrainingMode};

pub const TRAINING_METRICS_SCHEMA_VERSION: u32 = 1;
pub const PREVIEW_ARTIFACT_SCHEMA_VERSION: u32 = 1;
pub const PREVIEW_ARTIFACT_FORMAT: &str = "feathertalk-preview-f32-v1";
pub const PREVIEW_TENSOR_SHAPE: [u32; 3] = [3, 160, 160];
pub const PREVIEW_TENSOR_ELEMENTS: usize = 3 * 160 * 160;
pub const PREVIEW_MANIFEST_FILE_NAME: &str = "manifest.json";
pub const PREVIEW_PREDICTION_FILE_NAME: &str = "prediction.f32";
pub const PREVIEW_TARGET_FILE_NAME: &str = "target.f32";
pub const PREVIEW_MOUTH_ROI_FILE_NAME: &str = "mouth-roi.f32";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrainingMetrics {
    pub schema_version: u32,
    pub mode: TrainingMode,
    pub epoch: u64,
    pub global_step: u64,
    pub total_loss: f64,
    pub full_loss: f64,
    pub perceptual_loss: f64,
    pub mouth_loss: Option<f64>,
    pub temporal_loss: Option<f64>,
    pub temporal_mouth_loss: Option<f64>,
    pub samples_seen: u64,
    pub samples_per_second: f64,
    pub estimated_remaining_seconds: f64,
    pub gpu_memory_bytes: Option<u64>,
    pub worker_state: String,
}

impl TrainingMetrics {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        mode: TrainingMode,
        epoch: u64,
        global_step: u64,
        total_loss: f64,
        full_loss: f64,
        perceptual_loss: f64,
        mouth_loss: Option<f64>,
        temporal_loss: Option<f64>,
        temporal_mouth_loss: Option<f64>,
        samples_seen: u64,
        samples_per_second: f64,
        estimated_remaining_seconds: f64,
        gpu_memory_bytes: Option<u64>,
        worker_state: impl Into<String>,
    ) -> Result<Self, TrainingError> {
        let value = Self {
            schema_version: TRAINING_METRICS_SCHEMA_VERSION,
            mode,
            epoch,
            global_step,
            total_loss,
            full_loss,
            perceptual_loss,
            mouth_loss,
            temporal_loss,
            temporal_mouth_loss,
            samples_seen,
            samples_per_second,
            estimated_remaining_seconds,
            gpu_memory_bytes,
            worker_state: worker_state.into(),
        };
        value.validate()?;
        Ok(value)
    }

    pub fn validate(&self) -> Result<(), TrainingError> {
        if self.schema_version != TRAINING_METRICS_SCHEMA_VERSION {
            return invalid_checkpoint(format!(
                "training_metrics.schema_version must be {}, got {}",
                TRAINING_METRICS_SCHEMA_VERSION, self.schema_version
            ));
        }

        validate_metric("total_loss", self.total_loss)?;
        validate_metric("full_loss", self.full_loss)?;
        validate_metric("perceptual_loss", self.perceptual_loss)?;
        validate_optional_metric("mouth_loss", self.mouth_loss)?;
        validate_optional_metric("temporal_loss", self.temporal_loss)?;
        validate_optional_metric("temporal_mouth_loss", self.temporal_mouth_loss)?;
        validate_metric("samples_per_second", self.samples_per_second)?;
        validate_metric(
            "estimated_remaining_seconds",
            self.estimated_remaining_seconds,
        )?;
        validate_worker_state(&self.worker_state)?;

        match self.mode {
            TrainingMode::Baseline => {
                if self.mouth_loss.is_some()
                    || self.temporal_loss.is_some()
                    || self.temporal_mouth_loss.is_some()
                {
                    return invalid_checkpoint(
                        "baseline metrics must not contain mouth or temporal loss components",
                    );
                }
            }
            TrainingMode::MouthRoi => {
                if self.mouth_loss.is_none() {
                    return invalid_checkpoint(
                        "mouth_roi metrics must contain mouth_loss component",
                    );
                }
                if self.temporal_loss.is_some() || self.temporal_mouth_loss.is_some() {
                    return invalid_checkpoint(
                        "mouth_roi metrics must not contain temporal loss components",
                    );
                }
            }
            TrainingMode::MouthRoiTemporal => {
                if self.mouth_loss.is_none()
                    || self.temporal_loss.is_none()
                    || self.temporal_mouth_loss.is_none()
                {
                    return invalid_checkpoint(
                        "mouth_roi_temporal metrics must contain all optional loss components",
                    );
                }
            }
        }

        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct PreviewArtifact {
    sample_index: u64,
    reference_index: u64,
    epoch: u64,
    global_step: u64,
    model_kind: String,
    model_config_sha256: String,
    worker_state: String,
    prediction: Vec<f32>,
    target: Vec<f32>,
    mouth_roi: Vec<f32>,
}

impl PreviewArtifact {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        sample_index: u64,
        reference_index: u64,
        epoch: u64,
        global_step: u64,
        model_kind: impl Into<String>,
        model_config_sha256: impl Into<String>,
        worker_state: impl Into<String>,
        prediction: Vec<f32>,
        target: Vec<f32>,
        mouth_roi: Vec<f32>,
    ) -> Result<Self, TrainingError> {
        let value = Self {
            sample_index,
            reference_index,
            epoch,
            global_step,
            model_kind: model_kind.into(),
            model_config_sha256: model_config_sha256.into(),
            worker_state: worker_state.into(),
            prediction,
            target,
            mouth_roi,
        };
        value.validate()?;
        Ok(value)
    }

    pub fn sample_index(&self) -> u64 {
        self.sample_index
    }

    pub fn reference_index(&self) -> u64 {
        self.reference_index
    }

    pub fn epoch(&self) -> u64 {
        self.epoch
    }

    pub fn global_step(&self) -> u64 {
        self.global_step
    }

    pub fn model_kind(&self) -> &str {
        &self.model_kind
    }

    pub fn model_config_sha256(&self) -> &str {
        &self.model_config_sha256
    }

    pub fn worker_state(&self) -> &str {
        &self.worker_state
    }

    pub const fn shape(&self) -> [u32; 3] {
        PREVIEW_TENSOR_SHAPE
    }

    pub fn prediction(&self) -> &[f32] {
        &self.prediction
    }

    pub fn target(&self) -> &[f32] {
        &self.target
    }

    pub fn mouth_roi(&self) -> &[f32] {
        &self.mouth_roi
    }

    pub fn validate(&self) -> Result<(), TrainingError> {
        validate_identifier("preview.model_kind", &self.model_kind, 128)?;
        validate_sha256("preview.model_config_sha256", &self.model_config_sha256)?;
        validate_worker_state(&self.worker_state)?;
        validate_tensor("preview.prediction", &self.prediction)?;
        validate_tensor("preview.target", &self.target)?;
        validate_tensor("preview.mouth_roi", &self.mouth_roi)?;
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PreviewFileManifest {
    pub file_name: String,
    pub bytes: u64,
    pub sha256: String,
}

impl PreviewFileManifest {
    pub fn validate(&self, expected_file_name: &str) -> Result<(), TrainingError> {
        if self.file_name != expected_file_name {
            return invalid_checkpoint(format!(
                "preview file_name must be {expected_file_name}, got {}",
                self.file_name
            ));
        }
        if self.bytes == 0 {
            return invalid_checkpoint(format!(
                "preview file {} must contain at least one byte",
                self.file_name
            ));
        }
        validate_sha256(&format!("preview.{}.sha256", self.file_name), &self.sha256)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PreviewArtifactManifest {
    pub schema_version: u32,
    pub format: String,
    pub sample_index: u64,
    pub reference_index: u64,
    pub epoch: u64,
    pub global_step: u64,
    pub model_kind: String,
    pub model_config_sha256: String,
    pub worker_state: String,
    pub shape: [u32; 3],
    pub prediction: PreviewFileManifest,
    pub target: PreviewFileManifest,
    pub mouth_roi: PreviewFileManifest,
}

impl PreviewArtifactManifest {
    pub fn validate(&self) -> Result<(), TrainingError> {
        if self.schema_version != PREVIEW_ARTIFACT_SCHEMA_VERSION {
            return invalid_checkpoint(format!(
                "preview manifest schema_version must be {}, got {}",
                PREVIEW_ARTIFACT_SCHEMA_VERSION, self.schema_version
            ));
        }
        if self.format != PREVIEW_ARTIFACT_FORMAT {
            return invalid_checkpoint(format!(
                "preview manifest format must be {PREVIEW_ARTIFACT_FORMAT}, got {}",
                self.format
            ));
        }
        if self.shape != PREVIEW_TENSOR_SHAPE {
            return invalid_checkpoint(format!(
                "preview manifest shape must be {PREVIEW_TENSOR_SHAPE:?}, got {:?}",
                self.shape
            ));
        }
        validate_identifier("preview.model_kind", &self.model_kind, 128)?;
        validate_sha256("preview.model_config_sha256", &self.model_config_sha256)?;
        validate_worker_state(&self.worker_state)?;
        self.prediction.validate(PREVIEW_PREDICTION_FILE_NAME)?;
        self.target.validate(PREVIEW_TARGET_FILE_NAME)?;
        self.mouth_roi.validate(PREVIEW_MOUTH_ROI_FILE_NAME)?;
        Ok(())
    }

    pub fn validate_against(&self, artifact: &PreviewArtifact) -> Result<(), TrainingError> {
        self.validate()?;
        artifact.validate()?;
        if self.sample_index != artifact.sample_index
            || self.reference_index != artifact.reference_index
            || self.epoch != artifact.epoch
            || self.global_step != artifact.global_step
            || self.model_kind != artifact.model_kind
            || self.model_config_sha256 != artifact.model_config_sha256
            || self.worker_state != artifact.worker_state
        {
            return invalid_checkpoint(
                "preview manifest metadata does not match decoded preview artifact",
            );
        }
        Ok(())
    }
}

fn validate_metric(name: &str, value: f64) -> Result<(), TrainingError> {
    if !value.is_finite() || value < 0.0 {
        return invalid_checkpoint(format!(
            "training_metrics.{name} must be finite and non-negative, got {value}"
        ));
    }
    Ok(())
}

fn validate_optional_metric(name: &str, value: Option<f64>) -> Result<(), TrainingError> {
    if let Some(value) = value {
        validate_metric(name, value)?;
    }
    Ok(())
}

fn validate_worker_state(value: &str) -> Result<(), TrainingError> {
    if value.is_empty()
        || value.len() > 128
        || value.trim() != value
        || !value.bytes().all(|byte| {
            byte.is_ascii_lowercase() || byte.is_ascii_digit() || matches!(byte, b'_' | b'-')
        })
    {
        return invalid_checkpoint(
            "worker_state must be 1-128 trimmed lower-case ASCII [a-z0-9_-] characters",
        );
    }
    Ok(())
}

fn validate_identifier(name: &str, value: &str, max_bytes: usize) -> Result<(), TrainingError> {
    if value.is_empty()
        || value.len() > max_bytes
        || value.trim() != value
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-'))
    {
        return invalid_checkpoint(format!(
            "{name} must be 1-{max_bytes} trimmed ASCII identifier characters"
        ));
    }
    Ok(())
}

fn validate_sha256(name: &str, value: &str) -> Result<(), TrainingError> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return invalid_checkpoint(format!(
            "{name} must be 64 lowercase hexadecimal characters"
        ));
    }
    Ok(())
}

fn validate_tensor(name: &str, values: &[f32]) -> Result<(), TrainingError> {
    if values.len() != PREVIEW_TENSOR_ELEMENTS {
        return invalid_checkpoint(format!(
            "{name} must contain {PREVIEW_TENSOR_ELEMENTS} values, got {}",
            values.len()
        ));
    }
    if let Some(index) = values.iter().position(|value| !value.is_finite()) {
        return invalid_checkpoint(format!(
            "{name}[{index}] must be finite, got {}",
            values[index]
        ));
    }
    Ok(())
}

fn invalid_checkpoint<T>(message: impl Into<String>) -> Result<T, TrainingError> {
    Err(TrainingError::InvalidCheckpoint(message.into()))
}
