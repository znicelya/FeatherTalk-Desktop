//! What a project's training has left on disk.
//!
//! `TrainingMetrics` and `TrainingCheckpointState` live in `feathertalk-training`,
//! which depends on burn: pulling it into the window process would pull the whole
//! training stack in with it. So these are narrow structures over the same JSON,
//! declaring only the fields a panel paints, and the schema version is checked
//! before a single number is shown -- an unrecognised version is reported, not
//! rendered.

use std::fs::{self, File};
use std::io::Read;
use std::path::{Path, PathBuf};

use feathertalk_domain::{DEFAULT_BATCH_SIZE, Request, TrainParams, TrainingMode, UnetVariant};
use serde::Deserialize;
use thiserror::Error;

use crate::assets::AssetSurvey;
use crate::facts::{Fact, FactValue};

/// The metrics schema this page knows how to read.
const METRICS_SCHEMA_VERSION: u32 = 1;

/// The training state schema this page knows how to read.
const STATE_SCHEMA_VERSION: u32 = 1;

/// The most a metrics file may hold, which is the bound the worker reads it under.
const METRICS_MAX_BYTES: u64 = 64 * 1024;

/// The most a `training-state.json` may hold, likewise.
const STATE_MAX_BYTES: u64 = 256 * 1024;

/// The file a checkpoint's training state is written to.
///
/// The generate page joins it onto a directory a dialog handed over, so the name
/// is declared once instead of spelled twice.
pub const CHECKPOINT_STATE_FILE: &str = "training-state.json";

/// One `outputs/metrics/step-*.json`, as far as the page is concerned.
///
/// `mode` is read as text because the two crates spell the third mode
/// differently -- `feathertalk-training` writes `mouth_roi_temporal` where the
/// protocol says `temporal` -- so there is no one enum both sides mean. The page
/// maps the string to a catalog key and leaves an unknown spelling as it is.
///
/// `worker_state` is in the file and not here: it is the constant `"training"`,
/// and a constant is not news.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Metrics {
    /// Which shape of this file the worker wrote.
    pub schema_version: u32,
    /// The training mode, in the worker's spelling.
    pub mode: String,
    /// Zero-based index of the epoch that produced this metric's batch.
    pub epoch: u64,
    /// How many optimiser steps have run.
    pub global_step: u64,
    /// The loss that was minimised, components included.
    pub total_loss: f64,
    /// The reconstruction loss over the whole frame.
    pub full_loss: f64,
    /// The perceptual loss over the whole frame.
    pub perceptual_loss: f64,
    /// The mouth region loss, in the modes that weigh one.
    pub mouth_loss: Option<f64>,
    /// The frame to frame loss, in temporal mode.
    pub temporal_loss: Option<f64>,
    /// The frame to frame loss inside the mouth region, in temporal mode.
    pub temporal_mouth_loss: Option<f64>,
    /// How many samples the run has consumed.
    pub samples_seen: u64,
    /// The rate over the last window the worker measured.
    pub samples_per_second: f64,
    /// What the worker's own arithmetic says is left.
    pub estimated_remaining_seconds: f64,
    /// Device memory, when the backend reports it. The CPU backend does not.
    pub gpu_memory_bytes: Option<u64>,
}

/// The `training-state.json` beside a checkpoint's weights.
///
/// The file also carries the data loader position and the provenance of the
/// assets and the model. Those decide whether a resume is sound, which is the
/// worker's judgement to make, so they are not declared here.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct CheckpointState {
    /// Which shape of this file the worker wrote.
    pub schema_version: u32,
    /// How many passes over the dataset were finished when it was written.
    pub epoch: u64,
    /// How many optimiser steps had run.
    pub global_step: u64,
    /// The seed the run started from.
    pub random_seed: u64,
    /// What the run was configured to do.
    pub training_config: TrainingConfigView,
}

/// The nine settings a run was started with.
///
/// The desktop form chooses the mode, batch size and epoch target; the worker
/// derives the rest. Showing all nine describes how a saved checkpoint trained.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct TrainingConfigView {
    /// The training mode, in the worker's spelling.
    pub mode: String,
    /// Samples per optimiser step.
    pub batch_size: u64,
    /// The optimiser's learning rate.
    pub learning_rate: f64,
    /// The epoch the run stops at, not a count of epochs still to do.
    pub total_epochs: u64,
    /// The gap between the frames a temporal pair is drawn from.
    pub temporal_stride: u64,
    /// How much the mouth region loss counts.
    pub mouth_weight: f64,
    /// How much the frame to frame loss counts.
    pub temporal_weight: f64,
    /// How much the frame to frame mouth loss counts.
    pub temporal_mouth_weight: f64,
    /// How much the perceptual loss counts.
    pub perceptual_weight: f64,
}

/// Why one of these files could not be turned into numbers.
///
/// The messages are English technical detail, like `ClientError`'s: they name a
/// path's problem for whoever has to fix it, and the page shows them beside a
/// Chinese sentence saying which file is at fault.
#[derive(Debug, Error)]
pub enum ReadError {
    /// The file could not be opened or read.
    #[error("read failed: {0}")]
    Io(#[from] std::io::Error),
    /// The file is larger than a file of this kind is allowed to be.
    #[error("file exceeds maximum size of {limit} bytes")]
    TooLarge {
        /// The bound that was exceeded.
        limit: u64,
    },
    /// The bytes are not the JSON this expects.
    #[error("invalid JSON: {0}")]
    Json(#[from] serde_json::Error),
    /// The file was written by a version this does not know how to read.
    #[error("unsupported schema_version {found}, expected {expected}")]
    Schema {
        /// The version this build reads.
        expected: u32,
        /// The version the file declares.
        found: u32,
    },
}

/// Read one `outputs/metrics/step-*.json`.
pub fn read_metrics(path: &Path) -> Result<Metrics, ReadError> {
    let bytes = read_bounded(path, METRICS_MAX_BYTES)?;
    let metrics: Metrics = serde_json::from_slice(&bytes)?;
    if metrics.schema_version == METRICS_SCHEMA_VERSION {
        Ok(metrics)
    } else {
        Err(ReadError::Schema {
            expected: METRICS_SCHEMA_VERSION,
            found: metrics.schema_version,
        })
    }
}

/// Read one checkpoint's `training-state.json`.
pub fn read_checkpoint_state(path: &Path) -> Result<CheckpointState, ReadError> {
    let bytes = read_bounded(path, STATE_MAX_BYTES)?;
    let state: CheckpointState = serde_json::from_slice(&bytes)?;
    if state.schema_version == STATE_SCHEMA_VERSION {
        Ok(state)
    } else {
        Err(ReadError::Schema {
            expected: STATE_SCHEMA_VERSION,
            found: state.schema_version,
        })
    }
}

/// Where the worker publishes checkpoints: `<project>/models/unet`.
///
/// `TrainingPaths` in `feathertalk-worker` owns this layout, and that crate
/// depends on burn, so the names are spelled out here the way `assets.rs` spells
/// out `assets/video_25fps.mp4`. Both sides reading the same four strings is the
/// cost of keeping the training stack out of the window process.
pub fn checkpoints_dir(project_dir: &Path) -> PathBuf {
    project_dir.join("models").join("unet")
}

/// Where the worker writes one metrics file per epoch boundary.
pub fn metrics_dir(project_dir: &Path) -> PathBuf {
    project_dir.join("outputs").join("metrics")
}

/// Where the worker writes one preview directory per epoch boundary.
///
/// A preview holds three `[3, 160, 160]` f32 tensors and a manifest rather than an
/// image, so these directories are counted and never opened.
pub fn previews_dir(project_dir: &Path) -> PathBuf {
    project_dir.join("outputs").join("preview")
}

/// The step in a checkpoint directory's name, when the name is one of ours.
pub fn checkpoint_step(name: &str) -> Option<u64> {
    step_after("checkpoint-", name)
}

/// The step in a metrics file's name.
pub fn metrics_step(name: &str) -> Option<u64> {
    step_after("step-", name.strip_suffix(".json")?)
}

/// The step in a preview directory's name.
pub fn preview_step(name: &str) -> Option<u64> {
    step_after("step-", name)
}

/// The digits after `prefix`, when there are at least eight of them and nothing
/// else.
///
/// This is the worker's `{:08}` read backwards: the format pads to eight digits
/// and never truncates, so more than eight digits is a real step and fewer is not
/// a name it wrote. That is what keeps a hand made `checkpoint-188` out of the
/// count, along with the `.publish-*` and `.retired-*` directories the worker
/// stages a publish through.
fn step_after(prefix: &str, name: &str) -> Option<u64> {
    let digits = name.strip_prefix(prefix)?;
    if digits.len() < 8 || !digits.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    digits.parse::<u64>().ok()
}

/// One published checkpoint.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Checkpoint {
    /// The step its name declares.
    pub step: u64,
    /// The directory itself.
    pub path: PathBuf,
}

/// One photograph of what a project's training has produced.
///
/// The asset page's rule again: rendering never reads the filesystem, so the disk
/// is photographed after a directory is chosen, after a task ends and when the
/// user asks, and the page paints the photograph. What it answers is what the four
/// cards need -- is there anything to resume from, what were the last numbers, and
/// which files refused to be read.
#[derive(Debug, Default, Clone)]
pub struct TrainingSurvey {
    /// The checkpoint with the largest step, which is the one a resume continues.
    pub checkpoint: Option<Checkpoint>,
    /// How many checkpoint directories were recognised.
    pub checkpoint_count: usize,
    /// The training state of that checkpoint, when it has one that reads.
    pub state: Option<CheckpointState>,
    /// Why it did not read, in English, for the notes card.
    pub state_error: Option<String>,
    /// The most recent metrics file, when it reads.
    pub metrics: Option<Metrics>,
    /// Why it did not read, likewise.
    pub metrics_error: Option<String>,
    /// The largest step a preview was written for.
    pub preview_step: Option<u64>,
    /// How many preview directories were recognised.
    pub preview_count: usize,
}

impl TrainingSurvey {
    /// Photograph the training output of `project_dir`.
    pub fn inspect(project_dir: &Path) -> Self {
        let (latest, checkpoint_count) = scan(&checkpoints_dir(project_dir), checkpoint_step);
        let checkpoint = latest.map(|(step, path)| Checkpoint { step, path });
        let (state, state_error) = match &checkpoint {
            Some(checkpoint) => read_state(&checkpoint.path),
            None => (None, None),
        };
        let (metrics, metrics_error) = read_latest_metrics(&metrics_dir(project_dir));
        let (preview, preview_count) = scan(&previews_dir(project_dir), preview_step);
        Self {
            checkpoint,
            checkpoint_count,
            state,
            state_error,
            metrics,
            metrics_error,
            preview_step: preview.map(|(step, _path)| step),
            preview_count,
        }
    }

    /// Whether anything a run leaves behind was found.
    ///
    /// A file that could not be read counts. It is still the trace of a run, and
    /// the page has an error line for it; hiding the panels behind "nothing has
    /// been trained yet" would be a different claim than the one that is true.
    pub fn has_history(&self) -> bool {
        self.checkpoint_count > 0
            || self.metrics.is_some()
            || self.metrics_error.is_some()
            || self.preview_count > 0
    }
}

/// The entry with the largest step under `dir`, and how many entries were ours.
///
/// A directory that cannot be read counts as empty, the way `assets::has_entries`
/// treats one: a project that never trained has no `outputs/preview`, and an
/// unreadable one changes nothing the page could do about it. Names decide what
/// counts, not file types -- the worker writes directories under two of these
/// three prefixes and files under the third, and a stray file wearing a
/// checkpoint's name simply has no state file to read.
fn scan(dir: &Path, step_of: fn(&str) -> Option<u64>) -> (Option<(u64, PathBuf)>, usize) {
    let Ok(entries) = fs::read_dir(dir) else {
        return (None, 0);
    };
    let mut latest: Option<(u64, PathBuf)> = None;
    let mut count: usize = 0;
    for entry in entries.flatten() {
        let name = entry.file_name();
        let Some(name) = name.to_str() else {
            continue;
        };
        let Some(step) = step_of(name) else {
            continue;
        };
        count = count.saturating_add(1);
        let further = match &latest {
            Some((best, _path)) => step > *best,
            None => true,
        };
        if further {
            latest = Some((step, entry.path()));
        }
    }
    (latest, count)
}

/// Read the training state inside a checkpoint directory.
///
/// An absent file is not an error: the worker writes the weights before the state,
/// and a directory caught mid publish has none yet. "Not written" and "written
/// wrong" are different things to tell a user, so only the second gets a line.
fn read_state(checkpoint: &Path) -> (Option<CheckpointState>, Option<String>) {
    let path = checkpoint.join(CHECKPOINT_STATE_FILE);
    if !path.is_file() {
        return (None, None);
    }
    match read_checkpoint_state(&path) {
        Ok(state) => (Some(state), None),
        Err(error) => (None, Some(error.to_string())),
    }
}

/// Read the metrics file with the largest step under `dir`.
fn read_latest_metrics(dir: &Path) -> (Option<Metrics>, Option<String>) {
    let (latest, _count) = scan(dir, metrics_step);
    match latest {
        Some((_step, path)) => match read_metrics(&path) {
            Ok(metrics) => (Some(metrics), None),
            Err(error) => (None, Some(error.to_string())),
        },
        None => (None, None),
    }
}

/// What the metrics panel paints.
///
/// The components a mode does not compute are left out rather than drawn as `-`:
/// a baseline run has no mouth loss at all, and an empty line beside a label
/// suggests the number was measured and came out zero.
pub fn metrics_facts(metrics: &Metrics) -> Vec<Fact> {
    let mut facts = vec![
        Fact {
            label: "workflow.training.metrics_step",
            value: FactValue::Text(metrics.global_step.to_string()),
        },
        Fact {
            label: "training.metrics.mode",
            value: mode_value(&metrics.mode),
        },
        Fact {
            label: "training.metrics.total_loss",
            value: FactValue::Text(loss(metrics.total_loss)),
        },
        Fact {
            label: "training.metrics.full_loss",
            value: FactValue::Text(loss(metrics.full_loss)),
        },
        Fact {
            label: "training.metrics.perceptual_loss",
            value: FactValue::Text(loss(metrics.perceptual_loss)),
        },
    ];
    if let Some(value) = metrics.mouth_loss {
        facts.push(Fact {
            label: "training.metrics.mouth_loss",
            value: FactValue::Text(loss(value)),
        });
    }
    if let Some(value) = metrics.temporal_loss {
        facts.push(Fact {
            label: "training.metrics.temporal_loss",
            value: FactValue::Text(loss(value)),
        });
    }
    if let Some(value) = metrics.temporal_mouth_loss {
        facts.push(Fact {
            label: "training.metrics.temporal_mouth_loss",
            value: FactValue::Text(loss(value)),
        });
    }
    facts.push(Fact {
        label: "training.metrics.samples_seen",
        value: FactValue::Text(metrics.samples_seen.to_string()),
    });
    facts.push(Fact {
        label: "training.metrics.samples_per_second",
        value: FactValue::Text(measured(metrics.samples_per_second)),
    });
    facts.push(Fact {
        label: "training.metrics.remaining",
        value: FactValue::Text(measured(metrics.estimated_remaining_seconds)),
    });
    facts.push(Fact {
        label: "training.metrics.gpu_memory",
        value: match metrics.gpu_memory_bytes {
            // The CPU backend measures no device memory, and a `0` would read as
            // "the run used none" instead of "nobody looked".
            Some(bytes) => FactValue::Text(bytes.to_string()),
            None => FactValue::Key("training.metrics.gpu_memory_none"),
        },
    });
    facts
}

/// The primary summary uses epoch numbers. Cumulative step/sample counts remain
/// available in the detailed metrics so they cannot be mistaken for dataset size.
pub fn metric_summary(metrics: &Metrics) -> Vec<Fact> {
    let mut facts = vec![Fact {
        label: "workflow.training.metrics_epoch",
        value: FactValue::Text(
            metrics
                .epoch
                .checked_add(1)
                .map(|epoch| epoch.to_string())
                .unwrap_or_else(|| "—".into()),
        ),
    }];
    facts.extend(metrics_facts(metrics).into_iter().filter(|fact| {
        matches!(
            fact.label,
            "training.metrics.mode"
                | "training.metrics.total_loss"
                | "training.metrics.samples_per_second"
        )
    }));
    facts
}

/// What the checkpoint card paints about the settings a run actually used.
///
/// Seven of these nine are the worker's own constants. They are here because a
/// checkpoint outlives the build that wrote it: reading them off the state file is
/// how a user tells what an old checkpoint was trained under.
pub fn config_facts(state: &CheckpointState) -> Vec<Fact> {
    let config = &state.training_config;
    vec![
        Fact {
            label: "training.config.batch_size",
            value: FactValue::Text(config.batch_size.to_string()),
        },
        Fact {
            label: "training.config.learning_rate",
            value: FactValue::Text(configured(config.learning_rate)),
        },
        Fact {
            label: "training.config.total_epochs",
            value: FactValue::Text(config.total_epochs.to_string()),
        },
        Fact {
            label: "training.config.temporal_stride",
            value: FactValue::Text(config.temporal_stride.to_string()),
        },
        Fact {
            label: "training.config.mouth_weight",
            value: FactValue::Text(configured(config.mouth_weight)),
        },
        Fact {
            label: "training.config.temporal_weight",
            value: FactValue::Text(configured(config.temporal_weight)),
        },
        Fact {
            label: "training.config.temporal_mouth_weight",
            value: FactValue::Text(configured(config.temporal_mouth_weight)),
        },
        Fact {
            label: "training.config.perceptual_weight",
            value: FactValue::Text(configured(config.perceptual_weight)),
        },
        Fact {
            label: "training.config.random_seed",
            value: FactValue::Text(state.random_seed.to_string()),
        },
    ]
}

/// What the checkpoint card paints about the checkpoints themselves.
///
/// The two counts are always there, including as zeroes: a project with metrics
/// but no checkpoint is a real state -- a run that was stopped before its first
/// epoch ended -- and "0" says so. The step and the epoch only appear once
/// something has been published to name them.
pub fn checkpoint_facts(survey: &TrainingSurvey) -> Vec<Fact> {
    let mut facts = Vec::new();
    if let Some(checkpoint) = &survey.checkpoint {
        facts.push(Fact {
            label: "training.checkpoint.latest",
            value: FactValue::Text(checkpoint.step.to_string()),
        });
    }
    if let Some(state) = &survey.state {
        facts.push(Fact {
            label: "training.checkpoint.epoch",
            value: FactValue::Text(state.epoch.to_string()),
        });
    }
    facts.push(Fact {
        label: "training.checkpoint.count",
        value: FactValue::Text(survey.checkpoint_count.to_string()),
    });
    facts.push(Fact {
        label: "training.checkpoint.preview_count",
        value: FactValue::Text(survey.preview_count.to_string()),
    });
    facts
}

/// The catalog key of a training mode, or the spelling itself when it is new.
///
/// The three arms are `feathertalk-training`'s spellings, which is what the file
/// carries; the protocol's `temporal` never reaches this text. A mode this build
/// does not know is shown as written rather than dropped: the wrong word beats a
/// silently missing line.
///
/// The generate page reads the same field out of a checkpoint a user chose, so the
/// three keys are shared rather than spelled twice: two copies would drift, and
/// one mode would end up named two ways on two pages.
pub fn mode_value(mode: &str) -> FactValue {
    match mode {
        "baseline" => FactValue::Key("training.metrics.mode_baseline"),
        "mouth_roi" => FactValue::Key("training.metrics.mode_mouth"),
        "mouth_roi_temporal" => FactValue::Key("training.metrics.mode_temporal"),
        unknown => FactValue::Text(unknown.to_owned()),
    }
}

/// A loss, at the precision a loss curve is read at.
fn loss(value: f64) -> String {
    format!("{value:.6}")
}

/// A measured rate or a count of seconds, where a second decimal is noise.
fn measured(value: f64) -> String {
    format!("{value:.1}")
}

/// A configured constant, printed the way it was configured.
///
/// Six decimals would turn a learning rate of `0.0001` into `0.000100` and a
/// weight of `4` into `4.000000`. The default for `f64` is the shortest text that
/// reads back as the same number, which is how the worker's defaults are written.
fn configured(value: f64) -> String {
    format!("{value}")
}

/// The smallest epoch target the worker accepts.
pub const MIN_EPOCHS: u32 = 1;

/// The largest, which is `feathertalk-training`'s own `MAX_EPOCHS`.
pub const MAX_EPOCHS: u32 = 10_000;

/// What the form starts at, which is `train.py --epochs` in the Python
/// implementation.
///
/// The two enhanced scripts there default to 60. Switching preset does not move
/// this number: `number_input` holds its own text and only seeds it from `value`
/// while that text is empty, so writing the field behind the control's back would
/// leave the page showing one number and the state holding another. The 60 goes in
/// the preset's hint instead, for whoever wants it.
pub const DEFAULT_EPOCHS: u32 = 200;

/// The three modes, in the order the page paints them.
pub const ALL_MODES: [TrainingMode; 3] = [
    TrainingMode::Baseline,
    TrainingMode::MouthRoi,
    TrainingMode::Temporal,
];

/// Both model variants, likewise.
pub const ALL_VARIANTS: [UnetVariant; 2] = [UnetVariant::OriginalUnet, UnetVariant::MobileOneUnet];

/// The editable parameters of a training request.
///
/// Learning rate, seed, loss weights and temporal stride are derived by the
/// worker and shown from the latest saved checkpoint instead of edited here.
#[derive(Debug, Clone)]
pub struct TrainingForm {
    mode: TrainingMode,
    variant: UnetVariant,
    epochs: u32,
    batch_size: u32,
    resume: bool,
}

impl Default for TrainingForm {
    fn default() -> Self {
        Self {
            mode: TrainingMode::Baseline,
            variant: UnetVariant::OriginalUnet,
            epochs: DEFAULT_EPOCHS,
            batch_size: DEFAULT_BATCH_SIZE,
            resume: false,
        }
    }
}

impl TrainingForm {
    /// Which loss the run minimises.
    pub fn mode(&self) -> TrainingMode {
        self.mode
    }

    /// Choose the loss.
    pub fn set_mode(&mut self, mode: TrainingMode) {
        self.mode = mode;
    }

    /// Which network is trained.
    pub fn variant(&self) -> UnetVariant {
        self.variant
    }

    /// Choose the network.
    pub fn set_variant(&mut self, variant: UnetVariant) {
        self.variant = variant;
    }

    /// The epoch the run stops at.
    pub fn epochs(&self) -> u32 {
        self.epochs
    }

    /// Take an epoch target from the number input, which hands over an `f64`.
    ///
    /// A non-finite value keeps the previous one: there is no number to show and
    /// no reason to invent one. Anything else is clamped into the worker's range
    /// and rounded first, so the conversion is exact -- a whole number between 1
    /// and 10000 is a `u32` -- and a float to integer cast saturates rather than
    /// wrapping, which is the answer the clamp already gave.
    pub fn set_epochs(&mut self, value: f64) {
        if !value.is_finite() {
            return;
        }
        let clamped = value
            .clamp(f64::from(MIN_EPOCHS), f64::from(MAX_EPOCHS))
            .round();
        self.epochs = clamped as u32;
    }

    /// Samples per optimizer step.
    pub fn batch_size(&self) -> u32 {
        self.batch_size
    }

    /// Accept only positive whole counts from the numeric input. As with the
    /// epoch target, a non-finite value leaves the previous selection intact.
    pub fn set_batch_size(&mut self, value: f64) {
        if value.is_finite() {
            self.batch_size = value.clamp(1.0, f64::from(u32::MAX)).round() as u32;
        }
    }

    /// Whether the run continues from the latest checkpoint.
    pub fn resume(&self) -> bool {
        self.resume
    }

    /// Choose whether to continue or start from new weights.
    pub fn set_resume(&mut self, resume: bool) {
        self.resume = resume;
    }

    /// The request this form submits for the project at `project_dir`.
    pub fn request(&self, project_dir: &Path) -> Request {
        Request::Train(TrainParams {
            project_dir: project_dir.to_path_buf(),
            mode: self.mode,
            variant: self.variant,
            epochs: self.epochs,
            batch_size: self.batch_size,
            resume: self.resume,
        })
    }
}

/// The catalog key of a preset's name.
pub fn mode_label_key(mode: TrainingMode) -> &'static str {
    match mode {
        TrainingMode::Baseline => "training.preset.fast",
        TrainingMode::MouthRoi => "training.preset.mouth",
        TrainingMode::Temporal => "training.preset.temporal",
    }
}

/// The catalog key of the sentence saying what a preset optimises.
pub fn mode_hint_key(mode: TrainingMode) -> &'static str {
    match mode {
        TrainingMode::Baseline => "training.preset.fast_hint",
        TrainingMode::MouthRoi => "training.preset.mouth_hint",
        TrainingMode::Temporal => "training.preset.temporal_hint",
    }
}

/// A stable element id fragment, so a re-render keeps each radio's state.
pub fn mode_element_id(mode: TrainingMode) -> &'static str {
    match mode {
        TrainingMode::Baseline => "baseline",
        TrainingMode::MouthRoi => "mouth-roi",
        TrainingMode::Temporal => "temporal",
    }
}

/// The catalog key of a variant's name.
pub fn variant_label_key(variant: UnetVariant) -> &'static str {
    match variant {
        UnetVariant::OriginalUnet => "training.variant.original",
        UnetVariant::MobileOneUnet => "training.variant.mobileone",
    }
}

/// A stable element id fragment for a variant's radio.
pub fn variant_element_id(variant: UnetVariant) -> &'static str {
    match variant {
        UnetVariant::OriginalUnet => "original-unet",
        UnetVariant::MobileOneUnet => "mobileone-unet",
    }
}

/// What the form may do right now.
///
/// The shell-wide gate -- no project, no worker, a task already running -- is
/// `tasks::blocked_key`, the same as on the asset page. This is only about
/// training's own prerequisites.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FormState {
    /// A prerequisite is missing; the key says which one.
    Blocked(&'static str),
    /// The worker would accept this.
    Ready,
}

impl FormState {
    /// Whether training's own prerequisites allow a submission.
    pub fn is_submittable(self) -> bool {
        match self {
            Self::Ready => true,
            Self::Blocked(_) => false,
        }
    }
}

/// Decide whether this form could be submitted.
///
/// The order is the worker's. An unlocked package is refused before anything else
/// is looked at -- `validate_project_dir` is the first thing `Train` does -- and
/// the resume conditions only matter once resuming is what was asked for.
///
/// The epoch condition is only judged when the state file reads. `epochs` is the
/// epoch to stop at, not a number of epochs to add, so resuming a checkpoint that
/// already finished epoch 200 with a target of 200 would run nothing at all. When
/// the state cannot be read there is no way to know which epoch it stopped at, and
/// the interface does not guess: the worker gets to say.
pub fn form_state(
    form: &TrainingForm,
    assets: &AssetSurvey,
    training: &TrainingSurvey,
) -> FormState {
    if !assets.is_locked() {
        return FormState::Blocked("training.blocked.not_locked");
    }
    if !form.resume {
        return FormState::Ready;
    }
    if training.checkpoint.is_none() {
        return FormState::Blocked("training.blocked.no_checkpoint");
    }
    match &training.state {
        Some(state) => {
            if u64::from(form.batch_size) != state.training_config.batch_size {
                FormState::Blocked("training.blocked.batch_size_mismatch")
            } else if u64::from(form.epochs) > state.epoch {
                FormState::Ready
            } else {
                FormState::Blocked("training.blocked.epoch_reached")
            }
        }
        None => FormState::Ready,
    }
}

/// Whether a fresh run would start on top of checkpoints that already exist.
///
/// Not a block -- starting over is a legitimate thing to want -- but worth a
/// sentence first. A fresh run numbers its steps from zero again, and publishing a
/// checkpoint replaces a directory of the same name, so the last run's early
/// checkpoints are overwritten while its later ones stay where they are. "The
/// latest checkpoint" can therefore still belong to the previous run afterwards.
pub fn fresh_over_checkpoint(form: &TrainingForm, training: &TrainingSurvey) -> bool {
    !form.resume && training.checkpoint.is_some()
}

/// Read at most `limit` bytes of `path`, refusing the file if there are more.
///
/// These paths are written by a worker, but the directory is a user's: reading a
/// truncated prefix of a file someone replaced with a gigabyte of something else
/// would only push the failure into the parser. Asking for one byte over the
/// bound is how the difference between "exactly the limit" and "more than that"
/// is told, and it is what `feathertalk-training` does on the writing side.
fn read_bounded(path: &Path, limit: u64) -> Result<Vec<u8>, ReadError> {
    let file = File::open(path)?;
    let mut reader = file.take(limit.saturating_add(1));
    let mut bytes = Vec::new();
    reader.read_to_end(&mut bytes)?;
    if u64::try_from(bytes.len()).unwrap_or(u64::MAX) > limit {
        return Err(ReadError::TooLarge { limit });
    }
    Ok(bytes)
}
