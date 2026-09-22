use std::{
    fs, io,
    path::{Path, PathBuf}};

use burn::tensor::Device;
use feathertalk_domain::{TrainParams, TrainingMode as DomainTrainingMode, UnetVariant};
use feathertalk_export::ModelConfiguration;
use feathertalk_models::backend::CpuAutodiffBackend;
use feathertalk_training::{
    CheckpointDescriptor, PreviewArtifact, TrainingConfig, TrainingError, TrainingMetrics,
    TrainingMode, TrainingSample, write_preview_artifact, write_training_metrics};
use sha2::{Digest, Sha256};

/// The default backend used by CPU convenience functions. Production dispatch
/// can instead pass the selected WGPU autodiff device to the generic runner.
pub type TrainBackend = CpuAutodiffBackend;

/// The default CPU training device.
pub type TrainDevice = Device;

/// The backend name in results from the CPU convenience functions.
pub const TRAIN_BACKEND_NAME: &str = "flex-cpu";

/// The protocol's single-sample default, expressed in training-config units.
pub const DEFAULT_BATCH_SIZE: u64 = feathertalk_domain::DEFAULT_BATCH_SIZE as u64;

/// The migration design's learning rate (section 8.2).
pub const DEFAULT_LEARNING_RATE: f64 = 1e-4;

/// The sampling seed. Fixed, so two runs of the same project see the same order.
pub const TRAINING_SEED: u64 = 1;

/// The largest accepted `epochs`. `TrainingConfig::validate` would reject zero
/// anyway; the ceiling exists so a typo cannot ask for a run no one can finish.
pub const MAX_EPOCHS: u32 = 10_000;

/// The `worker_state` every metrics file and preview manifest records.
/// `validate_worker_state` accepts 1 to 128 lowercase letters, digits, hyphens
/// and underscores.
pub const WORKER_STATE: &str = "training";

/// Loss weights, all from the migration design section 8.2.
const MOUTH_WEIGHT: f64 = 4.0;
const TEMPORAL_WEIGHT: f64 = 0.5;
const TEMPORAL_MOUTH_WEIGHT: f64 = 4.0;
const PERCEPTUAL_WEIGHT: f64 = 0.01;

const MODELS_DIR: &str = "models";
const UNET_DIR: &str = "unet";
const OUTPUTS_DIR: &str = "outputs";
const METRICS_DIR: &str = "metrics";
const PREVIEW_DIR: &str = "preview";
const CHECKPOINT_PREFIX: &str = "checkpoint-";
const STEP_PREFIX: &str = "step-";

/// How many digits a step number is padded to in an artefact name.
const STEP_DIGITS: usize = 8;

/// Maps the request's mode onto the training crate's mode. The two enums have
/// three variants each and only two names in common.
pub fn training_mode(mode: DomainTrainingMode) -> TrainingMode {
    match mode {
        DomainTrainingMode::Baseline => TrainingMode::Baseline,
        DomainTrainingMode::MouthRoi => TrainingMode::MouthRoi,
        DomainTrainingMode::Temporal => TrainingMode::MouthRoiTemporal}
}

/// `TrainingConfig::validate` demands a zero stride outside the temporal mode
/// and a positive one inside it, and `DataLoaderConfig::sample_count` subtracts
/// the stride from the frame count.
fn temporal_stride(mode: DomainTrainingMode) -> u64 {
    match mode {
        DomainTrainingMode::Baseline | DomainTrainingMode::MouthRoi => 0,
        DomainTrainingMode::Temporal => 1}
}

/// How many samples one epoch holds. A temporal pair needs a successor, so the
/// last frame starts no sample.
pub fn sample_count(mode: DomainTrainingMode, frame_count: u64) -> u64 {
    match mode {
        DomainTrainingMode::Baseline | DomainTrainingMode::MouthRoi => frame_count,
        DomainTrainingMode::Temporal => frame_count.saturating_sub(1)}
}

/// Combine the requested mode, batch size and epoch target with the worker's
/// loss defaults and the mode's temporal stride.
pub fn training_config(params: &TrainParams) -> TrainingConfig {
    TrainingConfig {
        mode: training_mode(params.mode),
        batch_size: u64::from(params.batch_size),
        learning_rate: DEFAULT_LEARNING_RATE,
        total_epochs: u64::from(params.epochs),
        temporal_stride: temporal_stride(params.mode),
        mouth_weight: MOUTH_WEIGHT,
        temporal_weight: TEMPORAL_WEIGHT,
        temporal_mouth_weight: TEMPORAL_MOUTH_WEIGHT,
        perceptual_weight: PERCEPTUAL_WEIGHT}
}

/// Derives the checkpoint descriptor from the model configuration instead of
/// hand-writing its three fields.
///
/// `ModelConfiguration` is a fixed-field, map-free structure, so its serialised
/// bytes are stable and their digest is the natural canonical form of "this
/// model configuration". `CheckpointDescriptor::validate` requires 64 lowercase
/// hex characters, which is exactly what `hex::encode` produces. This is the
/// first place in the workspace that computes the value; every existing test
/// uses a repeated-digit placeholder.
pub fn checkpoint_descriptor(
    configuration: &ModelConfiguration,
) -> Result<CheckpointDescriptor, TrainingError> {
    let bytes = serde_json::to_vec(configuration).map_err(|error| {
        TrainingError::InvalidConfig(format!("serialize model configuration: {error}"))
    })?;
    let descriptor = CheckpointDescriptor::new(
        configuration.model_type(),
        configuration.architecture_version(),
        hex::encode(Sha256::digest(&bytes)),
    );
    descriptor.validate()?;
    Ok(descriptor)
}

/// Where a training run writes. Every directory is created by its writer, so
/// nothing here touches the filesystem.
#[derive(Debug, Clone)]
pub struct TrainingPaths {
    checkpoints: PathBuf,
    metrics: PathBuf,
    previews: PathBuf}

impl TrainingPaths {
    pub fn new(project_dir: &Path) -> Self {
        Self {
            checkpoints: project_dir.join(MODELS_DIR).join(UNET_DIR),
            metrics: project_dir.join(OUTPUTS_DIR).join(METRICS_DIR),
            previews: project_dir.join(OUTPUTS_DIR).join(PREVIEW_DIR)}
    }

    /// The directory every checkpoint of this project lives in.
    pub fn checkpoints(&self) -> &Path {
        &self.checkpoints
    }

    /// `models/unet/checkpoint-00000188`.
    ///
    /// Named by step, never by epoch: a cancellation lands mid-epoch, and
    /// `DataLoaderState.next_position` already carries the position inside the
    /// epoch, so one naming scheme covers both save points and resume only has
    /// to take the largest number.
    pub fn checkpoint(&self, global_step: u64) -> PathBuf {
        self.checkpoints
            .join(format!("{CHECKPOINT_PREFIX}{global_step:08}"))
    }

    /// `outputs/metrics/step-00000188.json`.
    pub fn metrics(&self, global_step: u64) -> PathBuf {
        self.metrics
            .join(format!("{STEP_PREFIX}{global_step:08}.json"))
    }

    /// `outputs/metrics/step-profile.jsonl`.
    pub fn step_profile(&self) -> PathBuf {
        self.metrics.join("step-profile.jsonl")
    }

    /// `outputs/preview/step-00000188`.
    pub fn preview(&self, global_step: u64) -> PathBuf {
        self.previews.join(format!("{STEP_PREFIX}{global_step:08}"))
    }

    /// The step a checkpoint directory name encodes, or `None` if the name is
    /// not one of ours.
    ///
    /// At least eight ASCII digits are required, so a hand-made
    /// `checkpoint-188` or a stray `.publish-*` never becomes a resume
    /// candidate. A step past eight digits still round-trips because `{:08}`
    /// only pads.
    pub fn checkpoint_step(name: &str) -> Option<u64> {
        let digits = name.strip_prefix(CHECKPOINT_PREFIX)?;
        if digits.len() < STEP_DIGITS || !digits.bytes().all(|byte| byte.is_ascii_digit()) {
            return None;
        }
        digits.parse::<u64>().ok()
    }
}

/// Everything the command settled before the first optimizer step.
#[derive(Debug, Clone)]
pub struct TrainingPlan {
    pub mode: DomainTrainingMode,
    pub variant: UnetVariant,
    pub epochs_requested: u32,
    pub frame_count: u64,
    pub config: TrainingConfig,
    pub descriptor: CheckpointDescriptor,
    pub paths: TrainingPaths,
    /// The checkpoint this run resumes from, `None` for a fresh run. It is also
    /// what the result payload reports as `resumed_from`.
    pub resume_from: Option<PathBuf>}

/// Prefix of the directory a checkpoint is written into before it takes its
/// step name.
const PUBLISH_PREFIX: &str = ".publish-";
/// Prefix of the directory an overwritten checkpoint is moved aside into.
const RETIRED_PREFIX: &str = ".retired-";
/// How many names to try before giving up on finding a free one.
const MAX_PUBLISH_ATTEMPTS: u32 = 1024;

/// A tensor readback may unwind as well as return an error. The reserved
/// staging name belongs to this save, so either failure must remove it.
struct CheckpointStagingGuard<'a>(&'a Path);

impl Drop for CheckpointStagingGuard<'_> {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(self.0);
    }
}

/// Writes the checkpoint for `global_step` through a staging directory so the
/// step's own name only ever holds a complete checkpoint.
///
/// `save` is handed a directory that does not exist yet -- the contract
/// `save_training_checkpoint` already expects -- and only has to write. The
/// rename into place happens here.
pub fn publish_checkpoint<F>(
    paths: &TrainingPaths,
    global_step: u64,
    save: F,
) -> Result<PathBuf, TrainingError>
where
    F: FnOnce(&Path) -> Result<(), TrainingError>,
{
    let root = paths.checkpoints();
    fs::create_dir_all(root)?;
    let staged = reserve_name(root, PUBLISH_PREFIX)?;
    let _staging_guard = CheckpointStagingGuard(&staged);
    save(&staged)?;

    let destination = paths.checkpoint(global_step);
    let retired = match fs::symlink_metadata(&destination) {
        Ok(_) => {
            let retired = reserve_name(root, RETIRED_PREFIX)?;
            fs::rename(&destination, &retired)?;
            Some(retired)
        }
        Err(error) if error.kind() == io::ErrorKind::NotFound => None,
        Err(error) => return Err(TrainingError::from(error))};

    fs::rename(&staged, &destination)?;
    if let Some(retired) = retired {
        // The destination is already correct, so a leftover copy is untidy
        // rather than broken and is not worth failing the step over.
        let _ = fs::remove_dir_all(&retired);
    }
    Ok(destination)
}

/// Reserves a free name under `root` by creating the directory and removing it
/// again, which proves the name is both free and creatable.
///
/// The name comes back unoccupied because `save_training_checkpoint` refuses a
/// destination that exists. Two workers on one project would race here, which
/// the pid in the name makes harmless in practice and which the
/// one-worker-per-project rule rules out anyway.
fn reserve_name(root: &Path, prefix: &str) -> Result<PathBuf, TrainingError> {
    let pid = std::process::id();
    for attempt in 0..MAX_PUBLISH_ATTEMPTS {
        let candidate = root.join(format!("{prefix}{pid}-{attempt}"));
        match fs::create_dir(&candidate) {
            Ok(()) => {
                fs::remove_dir(&candidate)?;
                return Ok(candidate);
            }
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(TrainingError::from(error))}
    }
    Err(TrainingError::CheckpointDirectory(format!(
        "no free staging name under {} after {MAX_PUBLISH_ATTEMPTS} attempts",
        root.display()
    )))
}

/// The checkpoint with the largest step, or `None` when the project has never
/// trained.
pub fn latest_checkpoint(paths: &TrainingPaths) -> Result<Option<PathBuf>, TrainingError> {
    let entries = match fs::read_dir(paths.checkpoints()) {
        Ok(entries) => entries,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(TrainingError::from(error))};

    let mut best: Option<(u64, PathBuf)> = None;
    for entry in entries {
        let entry = entry?;
        // Bind the name before borrowing from it: `file_name` returns an owned
        // `OsString`, and `entry.file_name().to_str()` would not live long
        // enough (E0716).
        let name = entry.file_name();
        let Some(step) = name.to_str().and_then(TrainingPaths::checkpoint_step) else {
            continue;
        };
        if !entry.file_type()?.is_dir() {
            continue;
        }
        if best.as_ref().is_none_or(|(seen, _)| step > *seen) {
            best = Some((step, entry.path()));
        }
    }
    Ok(best.map(|(_, path)| path))
}

/// Writes the metrics file for `global_step`, or reports that one was already
/// there.
pub fn write_metrics_unless_present(
    paths: &TrainingPaths,
    global_step: u64,
    metrics: &TrainingMetrics,
) -> Result<bool, TrainingError> {
    let path = paths.metrics(global_step);
    if exists(&path)? {
        return Ok(false);
    }
    write_training_metrics(&path, metrics)?;
    Ok(true)
}

/// Writes the preview directory for `global_step`, or reports that one was
/// already there.
pub fn write_preview_unless_present(
    paths: &TrainingPaths,
    global_step: u64,
    artifact: &PreviewArtifact,
) -> Result<bool, TrainingError> {
    let destination = paths.preview(global_step);
    if exists(&destination)? {
        return Ok(false);
    }
    write_preview_artifact(&destination, artifact)?;
    Ok(true)
}

/// The pair every preview renders: frame zero, driven by the audio and judged
/// against the ground truth of frame zero, wearing the middle frame of the clip
/// as its appearance reference.
///
/// Fixed rather than sampled, because the whole point of the artifact is to
/// watch one frame improve from epoch to epoch. A one-frame project references
/// itself, which is in range and therefore not an error.
pub fn preview_sample(frame_count: u64) -> TrainingSample {
    TrainingSample::SingleFrame {
        target_index: 0,
        reference_index: frame_count / 2}
}

/// Whether `path` is already taken, without following a symlink.
fn exists(path: &Path) -> Result<bool, TrainingError> {
    match fs::symlink_metadata(path) {
        Ok(_) => Ok(true),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(false),
        Err(error) => Err(TrainingError::from(error))}
}
