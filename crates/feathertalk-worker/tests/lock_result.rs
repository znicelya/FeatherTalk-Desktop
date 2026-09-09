//! The shape of the asset lock's result payload.

use feathertalk_audio::{FeatureCommitSpec, FeatureMatrix, write_feature_file_no_clobber};
use feathertalk_worker::lock_to_json;
use tempfile::TempDir;

const LANDMARK_SHA256: &str = "e131dd764236fde54a27b2f7084906119f06c28b140bf127b459ec967e92915b";
const FEATURE_SHA256: &str = "1111111111111111111111111111111111111111111111111111111111111111";

#[test]
fn the_payload_carries_every_field_the_desktop_needs() {
    let dir = TempDir::new().unwrap();
    let project_dir = dir.path().join("project");
    let feature_path = dir.path().join("feather_hubert.f32");
    let matrix = FeatureMatrix::new(2, 4, vec![0.5; 8]).unwrap();
    let artifact = write_feature_file_no_clobber(&feature_path, &matrix).unwrap();
    let spec = FeatureCommitSpec {
        project_root: project_dir.clone(),
        frame_count: 1,
        frame_width: 1280,
        frame_height: 720,
        landmark_model_sha256: LANDMARK_SHA256.to_owned(),
        feature_model_sha256: FEATURE_SHA256.to_owned(),
    };

    let value = lock_to_json(&project_dir, &spec, &artifact, -3);
    let object = value.as_object().expect("the payload must be an object");

    assert_eq!(object["project_dir"], project_dir.display().to_string());
    let manifest = project_dir.join("assets").join("assets.json");
    assert_eq!(object["manifest_file"], manifest.display().to_string());
    assert_eq!(object["frame_count"], 1);
    assert_eq!(object["frame_width"], 1280);
    assert_eq!(object["frame_height"], 720);
    assert_eq!(object["feature_file"], feature_path.display().to_string());
    assert_eq!(object["tokens"], 2);
    assert_eq!(object["dims"], 4);
    assert_eq!(object["bytes"], 76);
    assert_eq!(object["sha256"], artifact.sha256());
    assert_eq!(object["token_adjustment"], -3);
    assert_eq!(object["landmark_model_sha256"], LANDMARK_SHA256);
    assert_eq!(object["feature_model_sha256"], FEATURE_SHA256);
    // Every key is asserted above, so the count keeps a future field from
    // slipping into the protocol untested.
    assert_eq!(object.len(), 13);
}
