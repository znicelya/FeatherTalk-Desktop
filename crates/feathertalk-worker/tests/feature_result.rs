use std::path::Path;

use feathertalk_audio::{FeatureArtifact, FeatureMatrix, write_feature_file_no_clobber};
use feathertalk_worker::feature_to_json;

fn artifact(directory: &Path) -> FeatureArtifact {
    let path = directory.join("features").join("feather_hubert.f32");
    let matrix = FeatureMatrix::new(2, 4, vec![0.5; 8]).unwrap();
    write_feature_file_no_clobber(&path, &matrix).unwrap()
}

#[test]
fn the_payload_names_every_published_location() {
    let directory = tempfile::tempdir().unwrap();
    let output_dir = directory.path().join("features");
    let artifact = artifact(directory.path());

    let value = feature_to_json(&output_dir, &artifact, &"c".repeat(64));

    assert_eq!(value["output_dir"], output_dir.display().to_string());
    assert_eq!(
        value["feature_file"],
        output_dir.join("feather_hubert.f32").display().to_string()
    );
    assert_eq!(value["tokens"], 2);
    assert_eq!(value["dims"], 4);
    // Two tokens per video frame, and the odd one was already dropped.
    assert_eq!(value["frame_count"], 1);
    // The 44-byte header plus 2 * 4 f32 values.
    assert_eq!(value["bytes"], 76);
    assert_eq!(artifact.sha256().len(), 64);
    assert_eq!(value["sha256"], artifact.sha256());
    assert_eq!(value["model_sha256"], "c".repeat(64));
}

#[test]
fn the_payload_omits_the_per_token_detail() {
    let directory = tempfile::tempdir().unwrap();
    let artifact = artifact(directory.path());

    let value = feature_to_json(&directory.path().join("features"), &artifact, "d");

    let object = value.as_object().expect("the payload is an object");
    assert_eq!(object.len(), 8);
    assert!(object.get("values").is_none());
    assert!(object.get("waveform").is_none());
}
