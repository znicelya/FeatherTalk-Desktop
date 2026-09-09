//! The JSON payload the asset lock returns on success.

use std::path::Path;

use feathertalk_audio::{FeatureArtifact, FeatureCommitSpec};
use serde_json::{Value, json};

/// Shape the result of a successful lock.
///
/// Reports what the caller cannot see for itself: where the manifest landed,
/// the geometry that was verified, the feature file that was committed, and
/// how far the feature stream had to move to match the frame count.
/// `token_adjustment` is signed on purpose -- a negative value means tokens
/// were dropped, which is the case an operator may want to look at.
pub fn lock_to_json(
    project_dir: &Path,
    spec: &FeatureCommitSpec,
    artifact: &FeatureArtifact,
    token_adjustment: i64,
) -> Value {
    let manifest_file = project_dir.join("assets").join("assets.json");
    json!({
        "project_dir": path_text(project_dir),
        "manifest_file": path_text(&manifest_file),
        "frame_count": spec.frame_count,
        "frame_width": spec.frame_width,
        "frame_height": spec.frame_height,
        "feature_file": path_text(artifact.path()),
        "tokens": artifact.tokens(),
        "dims": artifact.dims(),
        "bytes": artifact.bytes(),
        "sha256": artifact.sha256(),
        "token_adjustment": token_adjustment,
        "landmark_model_sha256": spec.landmark_model_sha256.as_str(),
        "feature_model_sha256": spec.feature_model_sha256.as_str(),
    })
}

fn path_text(path: &Path) -> String {
    path.display().to_string()
}
