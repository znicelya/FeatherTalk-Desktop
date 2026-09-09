use std::{
    fs,
    path::{Path, PathBuf},
};

use feathertalk_training::{
    PREVIEW_MOUTH_ROI_FILE_NAME, PREVIEW_PREDICTION_FILE_NAME, PREVIEW_TARGET_FILE_NAME,
    PreviewArtifact, TrainingError, read_preview_artifact, write_preview_artifact,
};

fn preview() -> PreviewArtifact {
    let prediction = (0..76_800).map(|index| index as f32 / 10_000.0).collect();
    let target = (0..76_800)
        .map(|index| (index % 97) as f32 / 100.0)
        .collect();
    let mouth_roi = (0..76_800)
        .map(|index| (index % 31) as f32 / 31.0)
        .collect();
    PreviewArtifact::new(
        4,
        9,
        2,
        17,
        "original-unet",
        "a".repeat(64),
        "training",
        prediction,
        target,
        mouth_roi,
    )
    .unwrap()
}

fn sorted_entries(path: &Path) -> Vec<String> {
    let mut entries = fs::read_dir(path)
        .unwrap()
        .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
        .collect::<Vec<_>>();
    entries.sort();
    entries
}

fn destination() -> (tempfile::TempDir, PathBuf) {
    let root = tempfile::tempdir().unwrap();
    let destination = root.path().join("preview-000001");
    (root, destination)
}

fn save_valid() -> (tempfile::TempDir, PathBuf) {
    let (root, destination) = destination();
    write_preview_artifact(&destination, &preview()).unwrap();
    (root, destination)
}

fn load_error(path: &Path, model_kind: &str, hash: &str) -> TrainingError {
    read_preview_artifact(path, model_kind, hash).unwrap_err()
}

#[test]
fn preview_round_trip_uses_exact_directory_and_manifest() {
    let (_root, destination) = destination();
    let original = preview();
    let written = write_preview_artifact(&destination, &original).unwrap();
    assert_eq!(
        sorted_entries(&destination),
        vec![
            "manifest.json",
            PREVIEW_MOUTH_ROI_FILE_NAME,
            PREVIEW_PREDICTION_FILE_NAME,
            PREVIEW_TARGET_FILE_NAME,
        ]
    );
    let (restored, manifest) =
        read_preview_artifact(&destination, "original-unet", &"a".repeat(64)).unwrap();
    assert_eq!(restored, original);
    assert_eq!(manifest, written);
}

#[test]
fn payload_hash_mismatch_is_rejected_before_decode() {
    let (_root, destination) = save_valid();
    let path = destination.join(PREVIEW_PREDICTION_FILE_NAME);
    let mut bytes = fs::read(&path).unwrap();
    bytes[40] ^= 1;
    fs::write(path, bytes).unwrap();
    assert!(matches!(
        load_error(&destination, "original-unet", &"a".repeat(64)),
        TrainingError::HashMismatch { ref file, .. } if file == PREVIEW_PREDICTION_FILE_NAME
    ));
}

#[test]
fn malformed_header_is_rejected() {
    let (_root, destination) = save_valid();
    fs::write(
        destination.join(PREVIEW_TARGET_FILE_NAME),
        b"malformed preview payload",
    )
    .unwrap();
    assert!(matches!(
        load_error(&destination, "original-unet", &"a".repeat(64)),
        TrainingError::HashMismatch { .. }
    ));
}

#[test]
fn metadata_compatibility_is_checked_before_payload_decode() {
    let (_root, destination) = save_valid();
    fs::write(
        destination.join(PREVIEW_PREDICTION_FILE_NAME),
        b"not a record",
    )
    .unwrap();
    assert!(matches!(
        load_error(&destination, "different-model", &"a".repeat(64)),
        TrainingError::CheckpointCompatibility(_)
    ));
}

#[test]
fn missing_and_extra_entries_are_rejected() {
    let (_root, destination) = save_valid();
    fs::remove_file(destination.join(PREVIEW_TARGET_FILE_NAME)).unwrap();
    assert!(matches!(
        load_error(&destination, "original-unet", &"a".repeat(64)),
        TrainingError::CheckpointDirectory(_)
    ));

    let (_root, destination) = save_valid();
    fs::write(destination.join("notes.txt"), b"extra").unwrap();
    assert!(matches!(
        load_error(&destination, "original-unet", &"a".repeat(64)),
        TrainingError::CheckpointDirectory(_)
    ));
}
