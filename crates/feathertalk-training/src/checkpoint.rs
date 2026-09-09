use std::{collections::BTreeMap, path::Path};

use burn::{module::AutodiffModule, optim::Optimizer, tensor::backend::AutodiffBackend};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

use crate::{DataLoaderState, TrainingError};

pub const TRAINING_CHECKPOINT_MANIFEST_SCHEMA_VERSION: u32 = 1;
pub const TRAINING_STATE_SCHEMA_VERSION: u32 = 1;
pub const TRAINING_CHECKPOINT_RECORD_FORMAT: &str = "burn-bin-full-precision-v1";
pub const TRAINING_CHECKPOINT_OPTIMIZER_KIND: &str = "adam";
pub const TRAINING_CHECKPOINT_OPTIMIZER_SCHEMA_VERSION: u32 = 1;

pub const CHECKPOINT_MANIFEST_FILE_NAME: &str = "manifest.json";
pub const CHECKPOINT_MODEL_FILE_NAME: &str = "model.bin";
pub const CHECKPOINT_OPTIMIZER_FILE_NAME: &str = "optimizer.bin";
pub const CHECKPOINT_STATE_FILE_NAME: &str = "training-state.json";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Provenance {
    pub entries: BTreeMap<String, String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TrainingMode {
    Baseline,
    MouthRoi,
    MouthRoiTemporal,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrainingConfig {
    pub mode: TrainingMode,
    pub batch_size: u64,
    pub learning_rate: f64,
    pub total_epochs: u64,
    pub temporal_stride: u64,
    pub mouth_weight: f64,
    pub temporal_weight: f64,
    pub temporal_mouth_weight: f64,
    pub perceptual_weight: f64,
}

impl TrainingConfig {
    pub fn validate(&self) -> Result<(), TrainingError> {
        if self.batch_size == 0 {
            return invalid_checkpoint("training_config.batch_size must be greater than zero");
        }
        if self.total_epochs == 0 {
            return invalid_checkpoint("training_config.total_epochs must be greater than zero");
        }
        validate_finite_non_negative("training_config.learning_rate", self.learning_rate)?;
        validate_finite_non_negative("training_config.mouth_weight", self.mouth_weight)?;
        validate_finite_non_negative("training_config.temporal_weight", self.temporal_weight)?;
        validate_finite_non_negative(
            "training_config.temporal_mouth_weight",
            self.temporal_mouth_weight,
        )?;
        validate_finite_non_negative("training_config.perceptual_weight", self.perceptual_weight)?;

        match self.mode {
            TrainingMode::Baseline | TrainingMode::MouthRoi if self.temporal_stride != 0 => {
                invalid_checkpoint(
                    "training_config.temporal_stride must be zero for non-temporal modes",
                )
            }
            TrainingMode::MouthRoiTemporal if self.temporal_stride == 0 => invalid_checkpoint(
                "training_config.temporal_stride must be greater than zero for temporal mode",
            ),
            _ => Ok(()),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrainingCheckpointState {
    pub schema_version: u32,
    pub epoch: u64,
    pub global_step: u64,
    pub random_seed: u64,
    pub data_loader: DataLoaderState,
    pub training_config: TrainingConfig,
    pub asset_provenance: Provenance,
    pub model_provenance: Provenance,
}

impl TrainingCheckpointState {
    pub fn validate(&self) -> Result<(), TrainingError> {
        if self.schema_version != TRAINING_STATE_SCHEMA_VERSION {
            return invalid_checkpoint(format!(
                "unsupported training state schema_version {}, expected {}",
                self.schema_version, TRAINING_STATE_SCHEMA_VERSION
            ));
        }
        self.training_config.validate()?;
        self.data_loader.validate(self.data_loader.frame_count)?;
        if self.epoch != self.data_loader.epoch {
            return invalid_checkpoint("training state epoch must equal data_loader.epoch");
        }
        if self.random_seed != self.data_loader.config.seed {
            return invalid_checkpoint("random_seed must equal data_loader.config.seed");
        }
        if self.training_config.batch_size != self.data_loader.config.batch_size {
            return invalid_checkpoint(
                "training_config.batch_size must equal data_loader.config.batch_size",
            );
        }
        if self.training_config.temporal_stride != self.data_loader.config.sampling.temporal_stride
        {
            return invalid_checkpoint(
                "training_config.temporal_stride must equal data_loader sampling stride",
            );
        }
        validate_provenance("asset_provenance", &self.asset_provenance)?;
        validate_provenance("model_provenance", &self.model_provenance)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CheckpointFileManifest {
    pub file_name: String,
    pub bytes: u64,
    pub sha256: String,
}

impl CheckpointFileManifest {
    pub fn validate(&self, expected_file_name: &str) -> Result<(), TrainingError> {
        if self.file_name != expected_file_name {
            return invalid_checkpoint(format!(
                "file_name must be {expected_file_name}, got {}",
                self.file_name
            ));
        }
        if self.bytes == 0 {
            return invalid_checkpoint(format!(
                "{} must contain at least one byte",
                self.file_name
            ));
        }
        validate_sha256(&format!("{}.sha256", self.file_name), &self.sha256)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CheckpointDescriptor {
    pub model_kind: String,
    pub architecture_version: String,
    pub model_config_sha256: String,
    pub optimizer_kind: String,
    pub optimizer_schema_version: u32,
}

impl CheckpointDescriptor {
    pub fn new(
        model_kind: impl Into<String>,
        architecture_version: impl Into<String>,
        model_config_sha256: impl Into<String>,
    ) -> Self {
        Self {
            model_kind: model_kind.into(),
            architecture_version: architecture_version.into(),
            model_config_sha256: model_config_sha256.into(),
            optimizer_kind: TRAINING_CHECKPOINT_OPTIMIZER_KIND.to_owned(),
            optimizer_schema_version: TRAINING_CHECKPOINT_OPTIMIZER_SCHEMA_VERSION,
        }
    }

    pub fn validate(&self) -> Result<(), TrainingError> {
        validate_non_empty("model_kind", &self.model_kind)?;
        validate_non_empty("architecture_version", &self.architecture_version)?;
        validate_sha256("model_config_sha256", &self.model_config_sha256)?;
        if self.optimizer_kind != TRAINING_CHECKPOINT_OPTIMIZER_KIND {
            return invalid_checkpoint(format!(
                "optimizer_kind must be {}, got {}",
                TRAINING_CHECKPOINT_OPTIMIZER_KIND, self.optimizer_kind
            ));
        }
        if self.optimizer_schema_version != TRAINING_CHECKPOINT_OPTIMIZER_SCHEMA_VERSION {
            return invalid_checkpoint(format!(
                "unsupported optimizer_schema_version {}, expected {}",
                self.optimizer_schema_version, TRAINING_CHECKPOINT_OPTIMIZER_SCHEMA_VERSION
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrainingCheckpointManifest {
    pub schema_version: u32,
    pub record_format: String,
    pub model_kind: String,
    pub architecture_version: String,
    pub model_config_sha256: String,
    pub optimizer_kind: String,
    pub optimizer_schema_version: u32,
    pub model: CheckpointFileManifest,
    pub optimizer: CheckpointFileManifest,
    pub training_state: CheckpointFileManifest,
    pub training_state_sha256: String,
    pub burn_version: String,
    pub rust_version: String,
}

impl TrainingCheckpointManifest {
    pub fn validate(&self) -> Result<(), TrainingError> {
        if self.schema_version != TRAINING_CHECKPOINT_MANIFEST_SCHEMA_VERSION {
            return invalid_checkpoint(format!(
                "unsupported manifest schema_version {}, expected {}",
                self.schema_version, TRAINING_CHECKPOINT_MANIFEST_SCHEMA_VERSION
            ));
        }
        if self.record_format != TRAINING_CHECKPOINT_RECORD_FORMAT {
            return invalid_checkpoint(format!(
                "unsupported record_format {}, expected {}",
                self.record_format, TRAINING_CHECKPOINT_RECORD_FORMAT
            ));
        }
        CheckpointDescriptor {
            model_kind: self.model_kind.clone(),
            architecture_version: self.architecture_version.clone(),
            model_config_sha256: self.model_config_sha256.clone(),
            optimizer_kind: self.optimizer_kind.clone(),
            optimizer_schema_version: self.optimizer_schema_version,
        }
        .validate()?;
        self.model.validate(CHECKPOINT_MODEL_FILE_NAME)?;
        self.optimizer.validate(CHECKPOINT_OPTIMIZER_FILE_NAME)?;
        self.training_state.validate(CHECKPOINT_STATE_FILE_NAME)?;
        validate_sha256("training_state_sha256", &self.training_state_sha256)?;
        if self.training_state_sha256 != self.training_state.sha256 {
            return invalid_checkpoint("training_state_sha256 must equal training_state.sha256");
        }
        validate_non_empty("burn_version", &self.burn_version)?;
        validate_non_empty("rust_version", &self.rust_version)
    }

    pub fn descriptor(&self) -> CheckpointDescriptor {
        CheckpointDescriptor {
            model_kind: self.model_kind.clone(),
            architecture_version: self.architecture_version.clone(),
            model_config_sha256: self.model_config_sha256.clone(),
            optimizer_kind: self.optimizer_kind.clone(),
            optimizer_schema_version: self.optimizer_schema_version,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct CheckpointCompatibility {
    pub descriptor: CheckpointDescriptor,
    pub training_config: TrainingConfig,
    pub frame_count: u64,
    pub asset_provenance: Provenance,
    pub model_provenance: Provenance,
}

impl CheckpointCompatibility {
    pub fn new(
        descriptor: CheckpointDescriptor,
        training_config: TrainingConfig,
        frame_count: u64,
    ) -> Self {
        Self {
            descriptor,
            training_config,
            frame_count,
            asset_provenance: Provenance {
                entries: BTreeMap::new(),
            },
            model_provenance: Provenance {
                entries: BTreeMap::new(),
            },
        }
    }

    pub fn validate(&self) -> Result<(), TrainingError> {
        self.descriptor.validate()?;
        self.training_config.validate()?;
        if self.frame_count == 0 {
            return Err(TrainingError::CheckpointCompatibility(
                "frame_count must be greater than zero".to_owned(),
            ));
        }
        validate_provenance("asset_provenance", &self.asset_provenance)?;
        validate_provenance("model_provenance", &self.model_provenance)
    }

    pub fn validate_manifest_state(
        &self,
        manifest: &TrainingCheckpointManifest,
        state: &TrainingCheckpointState,
    ) -> Result<(), TrainingError> {
        self.validate()?;
        manifest.validate()?;
        state.validate()?;
        if manifest.descriptor() != self.descriptor {
            return Err(TrainingError::CheckpointCompatibility(
                "model or optimizer descriptor does not match expected compatibility".to_owned(),
            ));
        }
        if state.data_loader.frame_count != self.frame_count {
            return Err(TrainingError::CheckpointCompatibility(
                "data loader frame_count does not match expected compatibility".to_owned(),
            ));
        }
        if state.training_config != self.training_config {
            return Err(TrainingError::CheckpointCompatibility(
                "training configuration does not match expected compatibility".to_owned(),
            ));
        }
        if state.asset_provenance != self.asset_provenance {
            return Err(TrainingError::CheckpointCompatibility(
                "asset provenance does not match expected compatibility".to_owned(),
            ));
        }
        if state.model_provenance != self.model_provenance {
            return Err(TrainingError::CheckpointCompatibility(
                "model provenance does not match expected compatibility".to_owned(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct RestoredTrainingState<M, O> {
    pub model: M,
    pub optimizer: O,
    pub state: TrainingCheckpointState,
    pub manifest: TrainingCheckpointManifest,
}

pub fn save_training_checkpoint<B, M, O>(
    destination: impl AsRef<Path>,
    model: &M,
    optimizer: &O,
    descriptor: CheckpointDescriptor,
    state: TrainingCheckpointState,
) -> Result<TrainingCheckpointManifest, TrainingError>
where
    B: AutodiffBackend,
    M: AutodiffModule<B> + Clone,
    O: Optimizer<M, B> + Clone,
{
    descriptor.validate()?;
    state.validate()?;

    let destination = destination.as_ref();
    let parent = destination
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    crate::checkpoint_io::reject_symlink_components(parent)?;
    std::fs::create_dir_all(parent)?;

    match std::fs::symlink_metadata(destination) {
        Ok(_) => {
            return Err(TrainingError::CheckpointDirectory(format!(
                "checkpoint destination already exists: {}",
                destination.display()
            )));
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(error.into()),
    }

    let mut staging = crate::checkpoint_io::create_staging_directory(parent)?;
    let staging_path = staging.path().to_path_buf();

    let model_path =
        crate::checkpoint_io::write_model_record::<B, M>(model, &staging_path.join("model"))?;
    crate::checkpoint_io::sync_file(&model_path)?;
    let optimizer_path = crate::checkpoint_io::write_optimizer_record::<B, M, O>(
        optimizer,
        &staging_path.join("optimizer"),
    )?;
    crate::checkpoint_io::sync_file(&optimizer_path)?;

    let state_bytes = serde_json::to_vec(&state)
        .map_err(|error| TrainingError::Store(format!("serialize training state: {error}")))?;
    if u64::try_from(state_bytes.len()).unwrap_or(u64::MAX) > crate::checkpoint_io::STATE_MAX_BYTES
    {
        return Err(TrainingError::InvalidCheckpoint(format!(
            "training state exceeds maximum size of {} bytes",
            crate::checkpoint_io::STATE_MAX_BYTES
        )));
    }
    let state_path = staging_path.join(CHECKPOINT_STATE_FILE_NAME);
    crate::checkpoint_io::write_synced_bytes(&state_path, &state_bytes)?;

    let model_manifest = file_manifest(&model_path, CHECKPOINT_MODEL_FILE_NAME)?;
    let optimizer_manifest = file_manifest(&optimizer_path, CHECKPOINT_OPTIMIZER_FILE_NAME)?;
    let state_manifest = file_manifest(&state_path, CHECKPOINT_STATE_FILE_NAME)?;
    let manifest = TrainingCheckpointManifest {
        schema_version: TRAINING_CHECKPOINT_MANIFEST_SCHEMA_VERSION,
        record_format: TRAINING_CHECKPOINT_RECORD_FORMAT.to_owned(),
        model_kind: descriptor.model_kind.clone(),
        architecture_version: descriptor.architecture_version.clone(),
        model_config_sha256: descriptor.model_config_sha256.clone(),
        optimizer_kind: descriptor.optimizer_kind.clone(),
        optimizer_schema_version: descriptor.optimizer_schema_version,
        model: model_manifest,
        optimizer: optimizer_manifest,
        training_state_sha256: state_manifest.sha256.clone(),
        training_state: state_manifest,
        // Burn 0.21 is pinned by the workspace dependency.  Keep this
        // identifier explicit so a checkpoint cannot silently change format
        // when the training crate's own package version changes.
        burn_version: "0.21.0".to_owned(),
        rust_version: "1.94.0".to_owned(),
    };
    manifest.validate()?;

    let manifest_bytes = serde_json::to_vec(&manifest)
        .map_err(|error| TrainingError::Store(format!("serialize checkpoint manifest: {error}")))?;
    if u64::try_from(manifest_bytes.len()).unwrap_or(u64::MAX)
        > crate::checkpoint_io::MANIFEST_MAX_BYTES
    {
        return Err(TrainingError::InvalidCheckpoint(format!(
            "checkpoint manifest exceeds maximum size of {} bytes",
            crate::checkpoint_io::MANIFEST_MAX_BYTES
        )));
    }
    let manifest_path = staging_path.join(CHECKPOINT_MANIFEST_FILE_NAME);
    crate::checkpoint_io::write_synced_bytes(&manifest_path, &manifest_bytes)?;

    crate::checkpoint_io::sync_directory(&staging_path)?;
    std::fs::rename(&staging_path, destination)?;
    staging.disarm();
    crate::checkpoint_io::sync_directory(parent)?;

    Ok(manifest)
}

pub fn load_training_checkpoint<B, M, O>(
    directory: impl AsRef<Path>,
    model_template: &M,
    optimizer_template: &O,
    device: &B::Device,
    expected: &CheckpointCompatibility,
) -> Result<RestoredTrainingState<M, O>, TrainingError>
where
    B: AutodiffBackend,
    M: AutodiffModule<B> + Clone,
    O: Optimizer<M, B> + Clone,
{
    let directory = directory.as_ref();

    // No Burn record is touched before this complete filesystem and JSON
    // preflight has succeeded.
    crate::checkpoint_io::reject_symlink_components(directory)?;
    crate::checkpoint_io::validate_checkpoint_directory(directory)?;
    let manifest: TrainingCheckpointManifest = read_checkpoint_json(
        &directory.join(CHECKPOINT_MANIFEST_FILE_NAME),
        crate::checkpoint_io::MANIFEST_MAX_BYTES,
        "checkpoint manifest",
    )?;
    manifest.validate()?;
    let state: TrainingCheckpointState = read_checkpoint_json(
        &directory.join(CHECKPOINT_STATE_FILE_NAME),
        crate::checkpoint_io::STATE_MAX_BYTES,
        "training checkpoint state",
    )?;
    state.validate()?;
    expected.validate_manifest_state(&manifest, &state)?;

    crate::checkpoint_io::validate_declared_file(
        &directory.join(CHECKPOINT_MODEL_FILE_NAME),
        &manifest.model,
    )?;
    crate::checkpoint_io::validate_declared_file(
        &directory.join(CHECKPOINT_OPTIMIZER_FILE_NAME),
        &manifest.optimizer,
    )?;
    crate::checkpoint_io::validate_declared_file(
        &directory.join(CHECKPOINT_STATE_FILE_NAME),
        &manifest.training_state,
    )?;

    // Restore into clones only.  If either record fails, the caller's
    // templates remain untouched and no partially restored value is exposed.
    let model = crate::checkpoint_io::load_model_record::<B, M>(
        model_template.clone(),
        &directory.join(CHECKPOINT_MODEL_FILE_NAME),
        device,
    )?;
    let optimizer = crate::checkpoint_io::load_optimizer_record::<B, M, O>(
        optimizer_template.clone(),
        &directory.join(CHECKPOINT_OPTIMIZER_FILE_NAME),
        device,
    )?;

    Ok(RestoredTrainingState {
        model,
        optimizer,
        state,
        manifest,
    })
}

/// Everything a checkpoint says about itself, with no Burn record read.
#[derive(Debug, Clone, PartialEq)]
pub struct TrainingCheckpointMetadata {
    pub manifest: TrainingCheckpointManifest,
    pub state: TrainingCheckpointState,
}

/// A model restored from a checkpoint, next to the metadata that described it.
#[derive(Debug, Clone)]
pub struct RestoredCheckpointModel<M> {
    pub model: M,
    pub metadata: TrainingCheckpointMetadata,
}

/// Reads a checkpoint's manifest and training state, and nothing else.
///
/// The preflight is `load_training_checkpoint`'s, in the same order and with the
/// same bounds: no symbolic link on the path, a validated checkpoint directory,
/// then the two JSON documents read under their size caps and validated.
///
/// This exists for a caller that cannot build a model template yet, because the
/// template depends on the model variant and the variant is only written in the
/// manifest. Rendering reads this first, picks the configuration it names, and
/// then calls [`load_training_checkpoint_model`].
pub fn read_training_checkpoint(
    directory: impl AsRef<Path>,
) -> Result<TrainingCheckpointMetadata, TrainingError> {
    let directory = directory.as_ref();
    crate::checkpoint_io::reject_symlink_components(directory)?;
    crate::checkpoint_io::validate_checkpoint_directory(directory)?;
    let manifest: TrainingCheckpointManifest = read_checkpoint_json(
        &directory.join(CHECKPOINT_MANIFEST_FILE_NAME),
        crate::checkpoint_io::MANIFEST_MAX_BYTES,
        "checkpoint manifest",
    )?;
    manifest.validate()?;
    let state: TrainingCheckpointState = read_checkpoint_json(
        &directory.join(CHECKPOINT_STATE_FILE_NAME),
        crate::checkpoint_io::STATE_MAX_BYTES,
        "training checkpoint state",
    )?;
    state.validate()?;
    Ok(TrainingCheckpointMetadata { manifest, state })
}

/// Restores only the model record of a checkpoint.
///
/// Inference has no optimizer to continue and no training configuration to
/// match, so the compatibility gate is the descriptor alone: the model kind, the
/// architecture version and the digest of the model configuration all have to be
/// the ones the caller expects, or the weights would be poured into the wrong
/// shapes and the failure would be a bad video rather than an error.
///
/// The `AutodiffBackend` bound is not decoration: the record was written by a
/// module on `Autodiff<_>`, so reading it back with the same types is what makes
/// it certainly compatible instead of probably compatible. The caller drops the
/// autodiff shell afterwards with `AutodiffModule::valid`.
///
/// The template is only ever cloned. A failed load leaves the caller's template
/// untouched, the same rule `load_training_checkpoint` follows.
pub fn load_training_checkpoint_model<B, M>(
    directory: impl AsRef<Path>,
    model_template: &M,
    device: &B::Device,
    expected: &CheckpointDescriptor,
) -> Result<RestoredCheckpointModel<M>, TrainingError>
where
    B: AutodiffBackend,
    M: AutodiffModule<B> + Clone,
{
    let directory = directory.as_ref();
    let metadata = read_training_checkpoint(directory)?;
    if metadata.manifest.descriptor() != *expected {
        return Err(TrainingError::CheckpointCompatibility(
            "checkpoint descriptor does not match the expected model".to_owned(),
        ));
    }
    let model_path = directory.join(CHECKPOINT_MODEL_FILE_NAME);
    crate::checkpoint_io::validate_declared_file(&model_path, &metadata.manifest.model)?;
    let model = crate::checkpoint_io::load_model_record::<B, M>(
        model_template.clone(),
        &model_path,
        device,
    )?;
    Ok(RestoredCheckpointModel { model, metadata })
}

fn read_checkpoint_json<T>(path: &Path, max_bytes: u64, label: &str) -> Result<T, TrainingError>
where
    T: DeserializeOwned,
{
    let bytes = crate::checkpoint_io::read_bounded(path, max_bytes)?;
    serde_json::from_slice(&bytes).map_err(|error| {
        TrainingError::InvalidCheckpoint(format!("{label} JSON is invalid: {error}"))
    })
}

fn validate_provenance(name: &str, values: &Provenance) -> Result<(), TrainingError> {
    for (key, value) in &values.entries {
        validate_non_empty(&format!("{name} key"), key)?;
        validate_sha256(&format!("{name}.{key}"), value)?;
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

fn validate_non_empty(name: &str, value: &str) -> Result<(), TrainingError> {
    if value.trim().is_empty() {
        return invalid_checkpoint(format!("{name} must be non-empty"));
    }
    Ok(())
}

fn validate_finite_non_negative(name: &str, value: f64) -> Result<(), TrainingError> {
    if !value.is_finite() || value < 0.0 {
        return invalid_checkpoint(format!(
            "{name} must be finite and non-negative, got {value}"
        ));
    }
    Ok(())
}

fn file_manifest(
    path: &Path,
    expected_name: &str,
) -> Result<CheckpointFileManifest, TrainingError> {
    let (bytes, sha256) = crate::checkpoint_io::sha256_file(path)?;
    let manifest = CheckpointFileManifest {
        file_name: expected_name.to_owned(),
        bytes,
        sha256,
    };
    manifest.validate(expected_name)?;
    Ok(manifest)
}

fn invalid_checkpoint<T>(message: impl Into<String>) -> Result<T, TrainingError> {
    Err(TrainingError::InvalidCheckpoint(message.into()))
}
