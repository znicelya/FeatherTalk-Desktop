use std::path::Path;

use feathertalk_audio::FeatureArtifact;
use serde_json::{Value, json};

/// Shapes a published feature file as the JSON object a `completed` event
/// carries.
///
/// Like a published frame set and unlike a probe, the payload names the
/// locations: the caller asked for a project directory and the worker chose the
/// layout inside it. The tokens themselves are deliberately absent -- one JSON
/// line per task must stay small, the file at the reported path holds every
/// number, and the digest is here to say which file was meant. `model_sha256`
/// comes from the package manifest, not from the artifact: it is what lets a
/// later run decide whether these features still match the encoder.
pub fn feature_to_json(output_dir: &Path, artifact: &FeatureArtifact, model_sha256: &str) -> Value {
    json!({
        "output_dir": path_text(output_dir),
        "feature_file": path_text(artifact.path()),
        "tokens": artifact.tokens(),
        "dims": artifact.dims(),
        "frame_count": artifact.tokens() / 2,
        "bytes": artifact.bytes(),
        "sha256": artifact.sha256(),
        "model_sha256": model_sha256,
    })
}

fn path_text(path: &Path) -> String {
    path.display().to_string()
}
