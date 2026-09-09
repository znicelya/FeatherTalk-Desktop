use feathertalk_models::{
    feather_hubert::FeatherHubertConfig,
    unet::{MobileOneUnetConfig, OriginalUnetConfig},
};
use serde::{Deserialize, Serialize};
use time::{OffsetDateTime, format_description::well_known::Rfc3339};

use crate::PackageError;

pub const MODEL_PACKAGE_SCHEMA_VERSION: u32 = 1;
pub const MODEL_LICENSE_SCHEMA_VERSION: u32 = 1;
pub const FEATHER_HUBERT_ARCHITECTURE_VERSION: &str = "feather-hubert-burn-v1";
pub const ORIGINAL_UNET_ARCHITECTURE_VERSION: &str = "original-unet-burn-v1";
pub const MOBILEONE_UNET_ARCHITECTURE_VERSION: &str = "mobileone-unet-burn-v1";
pub const MODEL_FILE_NAME: &str = "model.safetensors";
pub const LICENSE_FILE_NAME: &str = "LICENSES.json";
pub const MANIFEST_FILE_NAME: &str = "manifest.json";
pub const OPTIMIZER_FILE_NAME: &str = "optimizer.safetensors";
pub const TRAINING_STATE_FILE_NAME: &str = "training-state.json";
pub const MAX_SOURCE_BYTES: u64 = 512 * 1024 * 1024;
pub const MAX_MANIFEST_BYTES: u64 = 64 * 1024;
pub const MAX_LICENSE_BYTES: u64 = 1024 * 1024;
pub const MAX_MODEL_BYTES: u64 = 2 * 1024 * 1024 * 1024;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TensorSpec {
    pub name: String,
    pub shape: Vec<i64>,
    pub dtype: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TensorContract {
    pub tensor_count: usize,
    pub total_elements: u64,
    pub entries: Vec<TensorSpec>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FileManifest {
    pub file_name: String,
    pub bytes: u64,
    pub sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SourceManifest {
    pub format: String,
    pub identifier: String,
    pub version: String,
    pub file_name: String,
    pub sha256: String,
    pub url: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TrainingMode {
    Inference,
    Baseline,
    MouthRoi,
    MouthRoiTemporal,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrainingManifest {
    pub mode: TrainingMode,
    pub mouth_weight: f64,
    pub temporal_weight: f64,
    pub temporal_mouth_weight: f64,
    pub perceptual_weight: f64,
}

impl Default for TrainingManifest {
    fn default() -> Self {
        Self {
            mode: TrainingMode::Inference,
            mouth_weight: 0.0,
            temporal_weight: 0.0,
            temporal_mouth_weight: 0.0,
            perceptual_weight: 0.0,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ModelConfiguration {
    FeatherHubert {
        channels: usize,
        expansion: usize,
        num_blocks: usize,
        output_dim: usize,
        dropout: f64,
    },
    OriginalUnet {
        channels: [usize; 5],
    },
    MobileOneUnet {
        channels: [usize; 5],
        num_conv_branches: usize,
        reparameterized: bool,
    },
}

impl ModelConfiguration {
    pub fn feather_hubert(config: &FeatherHubertConfig) -> Self {
        Self::FeatherHubert {
            channels: config.channels,
            expansion: config.expansion,
            num_blocks: config.num_blocks,
            output_dim: config.output_dim,
            dropout: config.dropout,
        }
    }

    pub fn original_unet(config: &OriginalUnetConfig) -> Self {
        Self::OriginalUnet {
            channels: config.channels,
        }
    }

    pub fn mobileone_unet(config: &MobileOneUnetConfig, reparameterized: bool) -> Self {
        Self::MobileOneUnet {
            channels: config.channels,
            num_conv_branches: config.num_conv_branches,
            reparameterized,
        }
    }

    pub fn model_type(&self) -> &'static str {
        match self {
            Self::FeatherHubert { .. } => "feather_hubert",
            Self::OriginalUnet { .. } => "original_unet",
            Self::MobileOneUnet { .. } => "mobileone_unet",
        }
    }

    pub fn architecture_version(&self) -> &'static str {
        match self {
            Self::FeatherHubert { .. } => FEATHER_HUBERT_ARCHITECTURE_VERSION,
            Self::OriginalUnet { .. } => ORIGINAL_UNET_ARCHITECTURE_VERSION,
            Self::MobileOneUnet { .. } => MOBILEONE_UNET_ARCHITECTURE_VERSION,
        }
    }

    fn expected_io(&self) -> (Vec<TensorSpec>, Vec<TensorSpec>) {
        match self {
            Self::FeatherHubert { output_dim, .. } => (
                vec![TensorSpec::new("waveform", vec![1, -1])],
                vec![TensorSpec::new("hidden", vec![1, -1, *output_dim as i64])],
            ),
            Self::OriginalUnet { .. } | Self::MobileOneUnet { .. } => (
                vec![
                    TensorSpec::new("input", vec![1, 6, 160, 160]),
                    TensorSpec::new("audio", vec![1, 16, 32, 32]),
                ],
                vec![TensorSpec::new("output", vec![1, 3, 160, 160])],
            ),
        }
    }

    pub fn validate(&self) -> Result<(), PackageError> {
        match self {
            Self::FeatherHubert {
                channels,
                expansion,
                num_blocks,
                output_dim,
                dropout,
            } => {
                require_positive("configuration.channels", *channels)?;
                require_positive("configuration.expansion", *expansion)?;
                require_positive("configuration.num_blocks", *num_blocks)?;
                require_positive("configuration.output_dim", *output_dim)?;
                if !dropout.is_finite() || !(0.0..1.0).contains(dropout) {
                    return invalid("configuration.dropout", "must be finite and in [0,1)");
                }
            }
            Self::OriginalUnet { channels } => {
                for (index, channel) in channels.iter().enumerate() {
                    require_positive(&format!("configuration.channels[{index}]"), *channel)?;
                }
            }
            Self::MobileOneUnet {
                channels,
                num_conv_branches,
                ..
            } => {
                for (index, channel) in channels.iter().enumerate() {
                    require_positive(&format!("configuration.channels[{index}]"), *channel)?;
                }
                require_positive("configuration.num_conv_branches", *num_conv_branches)?;
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelDescription {
    pub model_type: String,
    pub architecture_version: String,
    pub configuration: ModelConfiguration,
    pub inputs: Vec<TensorSpec>,
    pub outputs: Vec<TensorSpec>,
}

impl ModelDescription {
    pub fn feather_hubert(config: FeatherHubertConfig) -> Self {
        Self::from_configuration(ModelConfiguration::feather_hubert(&config))
    }

    pub fn original_unet(config: OriginalUnetConfig) -> Self {
        Self::from_configuration(ModelConfiguration::original_unet(&config))
    }

    pub fn mobileone_unet(config: MobileOneUnetConfig, reparameterized: bool) -> Self {
        Self::from_configuration(ModelConfiguration::mobileone_unet(&config, reparameterized))
    }

    pub fn from_configuration(configuration: ModelConfiguration) -> Self {
        let (inputs, outputs) = configuration.expected_io();
        Self {
            model_type: configuration.model_type().to_owned(),
            architecture_version: configuration.architecture_version().to_owned(),
            configuration,
            inputs,
            outputs,
        }
    }

    pub fn validate(&self) -> Result<(), PackageError> {
        self.configuration.validate()?;
        if self.model_type != self.configuration.model_type() {
            return invalid(
                "model_type",
                format!("expected {}", self.configuration.model_type()),
            );
        }
        if self.architecture_version != self.configuration.architecture_version() {
            return invalid(
                "architecture_version",
                format!("expected {}", self.configuration.architecture_version()),
            );
        }
        validate_io_tensor_specs("inputs", &self.inputs)?;
        validate_io_tensor_specs("outputs", &self.outputs)?;
        let expected = Self::from_configuration(self.configuration.clone());
        if self.inputs != expected.inputs || self.outputs != expected.outputs {
            return invalid(
                "io",
                "input/output tensor contract does not match configuration",
            );
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LicenseBundle {
    pub schema_version: u32,
    pub entries: Vec<LicenseEntry>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LicenseEntry {
    pub component: String,
    pub license_id: String,
    pub source_url: String,
    pub notice: String,
}

impl LicenseBundle {
    pub fn validate(&self) -> Result<(), PackageError> {
        if self.schema_version != MODEL_LICENSE_SCHEMA_VERSION {
            return invalid_license(
                "schema_version",
                format!("expected {MODEL_LICENSE_SCHEMA_VERSION}"),
            );
        }
        if self.entries.is_empty() {
            return invalid_license("entries", "must contain at least one entry");
        }
        for (index, entry) in self.entries.iter().enumerate() {
            require_non_empty_license(index, "component", &entry.component)?;
            require_non_empty_license(index, "license_id", &entry.license_id)?;
            require_non_empty_license(index, "source_url", &entry.source_url)?;
            require_non_empty_license(index, "notice", &entry.notice)?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelPackageManifest {
    pub schema_version: u32,
    pub model_type: String,
    pub architecture_version: String,
    pub configuration: ModelConfiguration,
    pub inputs: Vec<TensorSpec>,
    pub outputs: Vec<TensorSpec>,
    pub training: TrainingManifest,
    pub source: SourceManifest,
    pub created_at: String,
    pub minimum_app_version: String,
    pub tensors: TensorContract,
    pub model: FileManifest,
    pub licenses: FileManifest,
    pub optimizer: Option<FileManifest>,
    pub training_state: Option<FileManifest>,
}

impl ModelPackageManifest {
    pub fn validate(&self) -> Result<(), PackageError> {
        if self.schema_version != MODEL_PACKAGE_SCHEMA_VERSION {
            return invalid(
                "schema_version",
                format!("expected {MODEL_PACKAGE_SCHEMA_VERSION}"),
            );
        }
        let description = ModelDescription {
            model_type: self.model_type.clone(),
            architecture_version: self.architecture_version.clone(),
            configuration: self.configuration.clone(),
            inputs: self.inputs.clone(),
            outputs: self.outputs.clone(),
        };
        description.validate()?;
        validate_training(&self.training)?;
        validate_source(&self.source)?;
        validate_rfc3339("created_at", &self.created_at)?;
        validate_version("minimum_app_version", &self.minimum_app_version)?;
        self.tensors.validate()?;
        self.model.validate(MODEL_FILE_NAME)?;
        self.licenses.validate(LICENSE_FILE_NAME)?;
        match (&self.optimizer, &self.training_state) {
            (Some(optimizer), Some(state)) => {
                optimizer.validate(OPTIMIZER_FILE_NAME)?;
                state.validate(TRAINING_STATE_FILE_NAME)?;
            }
            (None, None) => {}
            _ => {
                return invalid(
                    "training_files",
                    "optimizer and training_state must be paired",
                );
            }
        }
        Ok(())
    }

    pub fn description(&self) -> ModelDescription {
        ModelDescription {
            model_type: self.model_type.clone(),
            architecture_version: self.architecture_version.clone(),
            configuration: self.configuration.clone(),
            inputs: self.inputs.clone(),
            outputs: self.outputs.clone(),
        }
    }
}

impl TensorSpec {
    pub fn new(name: impl Into<String>, shape: Vec<i64>) -> Self {
        Self {
            name: name.into(),
            shape,
            dtype: "f32".to_owned(),
        }
    }

    pub fn validate(&self, field: &str, allow_dynamic: bool) -> Result<(), PackageError> {
        if self.name.trim().is_empty() {
            return invalid(field, "name must be non-empty");
        }
        if self.dtype != "f32" {
            return invalid(field, "dtype must be f32");
        }
        if self.shape.is_empty() {
            return invalid(field, "shape must not be empty");
        }
        for dimension in &self.shape {
            if *dimension == -1 && allow_dynamic {
                continue;
            }
            if *dimension <= 0 {
                return invalid(
                    field,
                    "dimensions must be positive or -1 for dynamic dimensions",
                );
            }
        }
        Ok(())
    }
}

impl TensorContract {
    pub fn validate(&self) -> Result<(), PackageError> {
        if self.tensor_count != self.entries.len() {
            return invalid("tensors.tensor_count", "must equal entries length");
        }
        validate_tensor_specs("tensors.entries", &self.entries, false)?;
        let mut elements = 0_u64;
        for entry in &self.entries {
            let count = entry.shape.iter().try_fold(1_u64, |total, dimension| {
                let dimension = u64::try_from(*dimension).map_err(|_| ())?;
                total.checked_mul(dimension).ok_or(())
            });
            let count = count.map_err(|_| {
                PackageError::InvalidManifest("tensor element count overflowed u64".to_owned())
            })?;
            elements = elements.checked_add(count).ok_or_else(|| {
                PackageError::InvalidManifest("tensor element count overflowed u64".to_owned())
            })?;
        }
        if elements != self.total_elements {
            return invalid("tensors.total_elements", "does not match tensor shapes");
        }
        Ok(())
    }
}

impl FileManifest {
    pub fn validate(&self, expected_name: &str) -> Result<(), PackageError> {
        if self.file_name != expected_name {
            return invalid(
                "file_name",
                format!("expected {expected_name}, got {}", self.file_name),
            );
        }
        if self.bytes == 0 {
            return invalid("bytes", "must be greater than zero");
        }
        validate_hash("sha256", &self.sha256)
    }
}

fn validate_tensor_specs(
    field: &str,
    entries: &[TensorSpec],
    allow_dynamic: bool,
) -> Result<(), PackageError> {
    let mut previous: Option<&str> = None;
    for (index, entry) in entries.iter().enumerate() {
        entry.validate(&format!("{field}[{index}]"), allow_dynamic)?;
        if let Some(previous) = previous
            && previous >= entry.name.as_str()
        {
            return invalid(field, "tensor names must be sorted and unique");
        }
        previous = Some(&entry.name);
    }
    Ok(())
}

fn validate_io_tensor_specs(field: &str, entries: &[TensorSpec]) -> Result<(), PackageError> {
    if entries.is_empty() {
        return invalid(field, "must contain at least one tensor");
    }
    let mut names = std::collections::BTreeSet::new();
    for (index, entry) in entries.iter().enumerate() {
        entry.validate(&format!("{field}[{index}]"), true)?;
        if !names.insert(entry.name.as_str()) {
            return invalid(field, "tensor names must be unique");
        }
    }
    Ok(())
}

fn validate_training(training: &TrainingManifest) -> Result<(), PackageError> {
    for (name, value) in [
        ("mouth_weight", training.mouth_weight),
        ("temporal_weight", training.temporal_weight),
        ("temporal_mouth_weight", training.temporal_mouth_weight),
        ("perceptual_weight", training.perceptual_weight),
    ] {
        if !value.is_finite() || value < 0.0 {
            return invalid(
                &format!("training.{name}"),
                "must be finite and non-negative",
            );
        }
    }
    Ok(())
}

fn validate_source(source: &SourceManifest) -> Result<(), PackageError> {
    for (field, value) in [
        ("format", source.format.as_str()),
        ("identifier", source.identifier.as_str()),
        ("version", source.version.as_str()),
        ("file_name", source.file_name.as_str()),
    ] {
        if value.trim().is_empty() {
            return invalid(&format!("source.{field}"), "must be non-empty");
        }
    }
    if let Some(url) = &source.url
        && url.trim().is_empty()
    {
        return invalid("source.url", "must be non-empty when present");
    }
    validate_hash("source.sha256", &source.sha256)
}

fn validate_rfc3339(field: &str, value: &str) -> Result<(), PackageError> {
    OffsetDateTime::parse(value, &Rfc3339)
        .map(|_| ())
        .map_err(|_| PackageError::InvalidManifest(format!("{field} must be RFC 3339")))
}

fn validate_version(field: &str, value: &str) -> Result<(), PackageError> {
    let parts = value.split('.').collect::<Vec<_>>();
    if parts.len() != 3
        || parts
            .iter()
            .any(|part| part.is_empty() || part.parse::<u64>().is_err())
    {
        return invalid(field, "must contain three numeric dot-separated components");
    }
    Ok(())
}

fn validate_hash(field: &str, value: &str) -> Result<(), PackageError> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return invalid(field, "must be 64 lowercase hexadecimal characters");
    }
    Ok(())
}

fn require_positive(field: &str, value: usize) -> Result<(), PackageError> {
    if value == 0 {
        return invalid(field, "must be positive");
    }
    Ok(())
}

fn require_non_empty_license(index: usize, field: &str, value: &str) -> Result<(), PackageError> {
    if value.trim().is_empty() {
        return invalid_license(&format!("entries[{index}].{field}"), "must be non-empty");
    }
    Ok(())
}

fn invalid<T>(field: &str, message: impl Into<String>) -> Result<T, PackageError> {
    Err(PackageError::InvalidManifest(format!(
        "{field}: {}",
        message.into()
    )))
}

fn invalid_license<T>(field: &str, message: impl Into<String>) -> Result<T, PackageError> {
    Err(PackageError::InvalidLicense(format!(
        "{field}: {}",
        message.into()
    )))
}
