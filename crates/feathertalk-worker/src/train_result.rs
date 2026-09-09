//! The JSON payload a finished training run returns.

use std::path::Path;

use feathertalk_domain::{TrainingMode, UnetVariant};
use feathertalk_training::CheckpointDescriptor;
use serde_json::{Value, json};

use crate::TRAIN_BACKEND_NAME;

/// What a finished run has to say for itself.
///
/// A struct rather than seventeen positional arguments: eight of them are `u64`
/// counters, so a wrong order would type-check and report the wrong numbers
/// without a word.
#[derive(Debug)]
pub struct TrainSummary<'a> {
    /// The mode the request asked for, echoed in the request's own words.
    pub mode: TrainingMode,
    pub variant: UnetVariant,
    /// Supplies `model_kind`, `architecture_version` and `model_config_sha256`
    /// -- the three values a later resume has to match.
    pub descriptor: &'a CheckpointDescriptor,
    pub frame_count: u64,
    pub epochs_requested: u32,
    /// Epochs this run finished, which is below `epochs_requested` when the
    /// task was cancelled.
    pub epochs_completed: u64,
    pub global_step: u64,
    /// Samples this run saw; a resume starts the count again (design section 8).
    pub samples_seen: u64,
    /// The total loss of the last step, `None` when this run never stepped.
    pub total_loss: Option<f64>,
    pub resumed_from: Option<&'a Path>,
    /// The newest checkpoint this run published, `None` when it published none.
    pub checkpoint_dir: Option<&'a Path>,
    pub checkpoints_written: u64,
    pub metrics_written: u64,
    pub previews_written: u64,
}

/// Shapes the payload the `completed` event of a training task carries.
///
/// The three fields beyond the minimum are deliberate (design section 12):
/// `backend` puts the backend that actually ran into the artifact,
/// `model_config_sha256` is both the audit trail and the value the next resume
/// must match, and `resumed_from` says which checkpoint this run continued.
/// Loss curves stay out -- they would blow up a single-line JSON event, and
/// `outputs/metrics/` holds every step.
pub fn train_to_json(summary: &TrainSummary<'_>) -> Value {
    train_to_json_on(summary, TRAIN_BACKEND_NAME)
}

pub(crate) fn train_to_json_on(summary: &TrainSummary<'_>, backend: &str) -> Value {
    json!({
        "mode": mode_slug(summary.mode),
        "variant": variant_slug(summary.variant),
        "backend": backend,
        "model_kind": summary.descriptor.model_kind.as_str(),
        "architecture_version": summary.descriptor.architecture_version.as_str(),
        "model_config_sha256": summary.descriptor.model_config_sha256.as_str(),
        "frame_count": summary.frame_count,
        "epochs_requested": summary.epochs_requested,
        "epochs_completed": summary.epochs_completed,
        "global_step": summary.global_step,
        "samples_seen": summary.samples_seen,
        "total_loss": summary.total_loss,
        "resumed_from": summary.resumed_from.map(path_text),
        "checkpoint_dir": summary.checkpoint_dir.map(path_text),
        "checkpoints_written": summary.checkpoints_written,
        "metrics_written": summary.metrics_written,
        "previews_written": summary.previews_written,
    })
}

/// The request's spelling of the mode.
///
/// Matched exhaustively rather than serialised, so a fourth mode is a compile
/// error here instead of a surprise string in the payload.
fn mode_slug(mode: TrainingMode) -> &'static str {
    match mode {
        TrainingMode::Baseline => "baseline",
        TrainingMode::MouthRoi => "mouth_roi",
        TrainingMode::Temporal => "temporal",
    }
}

/// The request's spelling of the variant.
fn variant_slug(variant: UnetVariant) -> &'static str {
    match variant {
        UnetVariant::OriginalUnet => "original_unet",
        UnetVariant::MobileOneUnet => "mobileone_unet",
    }
}

fn path_text(path: &Path) -> String {
    path.display().to_string()
}
