//! The JSON payload a finished render returns.

use std::path::Path;

use feathertalk_inference::OfflineRenderResult;
use feathertalk_training::CheckpointDescriptor;
use serde_json::{Value, json};

use crate::{RENDER_BACKEND_NAME, RENDER_FPS};

/// Everything the completed event of a render task reports.
///
/// Borrowed rather than cloned: the caller owns the result and the job while the
/// payload is built. A struct rather than eight positional arguments, four of
/// which are `u64` counters that would type-check in the wrong order.
#[derive(Debug)]
pub struct RenderSummary<'a> {
    pub result: &'a OfflineRenderResult,
    /// Supplies `model_kind`, `architecture_version` and `model_config_sha256`
    /// -- the identity of the weights this video came from.
    pub descriptor: &'a CheckpointDescriptor,
    pub checkpoint_dir: &'a Path,
    pub checkpoint_epoch: u64,
    pub checkpoint_global_step: u64,
    /// The locked manifest's frame count, which is the render's upper bound.
    pub source_frame_count: u64,
    /// The request's cap, echoed as `null` when it did not set one.
    pub max_output_frames: Option<u64>}

/// Shapes the payload the `completed` event of a render task carries.
///
/// Inference fixes the container's frame rate. This convenience entry point
/// describes a CPU render; execution passes the actual backend below. The four
/// checkpoint fields are the audit trail -- given this payload, the exact
/// weights that produced the video can be found again (design section 12).
pub fn render_to_json(summary: &RenderSummary<'_>) -> Value {
    render_to_json_on(summary, RENDER_BACKEND_NAME)
}

pub(crate) fn render_to_json_on(summary: &RenderSummary<'_>, backend: &str) -> Value {
    json!({
        "output_path": path_text(summary.result.output_path()),
        "frame_count": summary.result.frame_count(),
        "width": summary.result.width(),
        "height": summary.result.height(),
        "fps": RENDER_FPS,
        "backend": backend,
        "checkpoint_dir": path_text(summary.checkpoint_dir),
        "model_kind": summary.descriptor.model_kind.as_str(),
        "architecture_version": summary.descriptor.architecture_version.as_str(),
        "model_config_sha256": summary.descriptor.model_config_sha256.as_str(),
        "checkpoint_epoch": summary.checkpoint_epoch,
        "checkpoint_global_step": summary.checkpoint_global_step,
        "source_frame_count": summary.source_frame_count,
        "max_output_frames": summary.max_output_frames})
}

fn path_text(path: &Path) -> String {
    path.display().to_string()
}
