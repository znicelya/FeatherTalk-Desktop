use std::path::Path;

use feathertalk_frame_pipeline::{FramePipelineSpec, QualityReport};
use serde_json::{Value, json};

/// Shapes a published frame set as the JSON object a `completed` event carries.
///
/// Like a normalization and unlike a probe, the payload names the locations:
/// the caller asked for a project directory and the worker chose the layout
/// inside it, so a later task would otherwise have to guess. The per-frame
/// array is deliberately absent -- one JSON line per task must stay small, and
/// `quality.json` at the reported path holds every record.
pub fn quality_to_json(spec: &FramePipelineSpec, report: &QualityReport) -> Value {
    json!({
        "output_dir": path_text(spec.output_root()),
        "frames_dir": path_text(&spec.frames_dir()),
        "landmarks_dir": path_text(&spec.landmarks_dir()),
        "quality_report": path_text(&spec.quality_path()),
        "frame_count": report.frame_count(),
        "frame_width": spec.image_width(),
        "frame_height": spec.image_height(),
    })
}

fn path_text(path: &Path) -> String {
    path.display().to_string()
}
