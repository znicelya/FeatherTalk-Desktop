use std::{
    fs,
    path::{Path, PathBuf},
};

use feathertalk_training::{
    PREVIEW_MOUTH_ROI_FILE_NAME, PREVIEW_PREDICTION_FILE_NAME, PREVIEW_TARGET_FILE_NAME,
    PREVIEW_TENSOR_ELEMENTS, PreviewArtifact, TRAINING_METRICS_SCHEMA_VERSION, TrainingError,
    TrainingMetrics, TrainingMode, read_preview_artifact, read_training_metrics,
    write_preview_artifact, write_training_metrics,
};
use sha2::{Digest, Sha256};

fn preview() -> PreviewArtifact {
    let prediction = vec![0.125_f32; PREVIEW_TENSOR_ELEMENTS];
    let target = vec![0.25_f32; PREVIEW_TENSOR_ELEMENTS];
    let mouth_roi = vec![0.5_f32; PREVIEW_TENSOR_ELEMENTS];
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

fn metrics() -> TrainingMetrics {
    TrainingMetrics::new(
        TrainingMode::MouthRoiTemporal,
        2,
        17,
        1.5,
        1.0,
        0.5,
        Some(0.25),
        Some(0.1),
        Some(0.2),
        34,
        12.5,
        8.0,
        Some(4_000_000),
        "training",
    )
    .unwrap()
}

fn staging_entries(parent: &Path) -> Vec<PathBuf> {
    let mut entries = fs::read_dir(parent)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with(".preview-") && name.ends_with(".staging"))
        })
        .collect::<Vec<_>>();
    entries.sort();
    entries
}

fn create_file_symlink(target: &Path, link: &Path) -> std::io::Result<()> {
    #[cfg(windows)]
    {
        std::os::windows::fs::symlink_file(target, link)
    }
    #[cfg(unix)]
    {
        std::os::unix::fs::symlink(target, link)
    }
    #[cfg(not(any(windows, unix)))]
    {
        let _ = (target, link);
        Err(std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            "file symlinks are unsupported on this platform",
        ))
    }
}

fn create_directory_symlink(target: &Path, link: &Path) -> std::io::Result<()> {
    #[cfg(windows)]
    {
        std::os::windows::fs::symlink_dir(target, link)
    }
    #[cfg(unix)]
    {
        std::os::unix::fs::symlink(target, link)
    }
    #[cfg(not(any(windows, unix)))]
    {
        let _ = (target, link);
        Err(std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            "directory symlinks are unsupported on this platform",
        ))
    }
}

fn sha256(bytes: &[u8]) -> String {
    let mut digest = Sha256::new();
    digest.update(bytes);
    hex::encode(digest.finalize())
}

fn refresh_manifest_entry(directory: &Path, field: &str, file_name: &str) {
    let manifest_path = directory.join("manifest.json");
    let mut manifest: serde_json::Value =
        serde_json::from_slice(&fs::read(&manifest_path).unwrap()).unwrap();
    let bytes = fs::read(directory.join(file_name)).unwrap();
    manifest[field]["bytes"] = (bytes.len() as u64).into();
    manifest[field]["sha256"] = sha256(&bytes).into();
    fs::write(manifest_path, serde_json::to_vec(&manifest).unwrap()).unwrap();
}

#[test]
fn preview_writer_publishes_exactly_four_entries_and_cleans_staging() {
    let root = tempfile::tempdir().unwrap();
    let destination = root.path().join("preview-000001");

    let manifest = write_preview_artifact(&destination, &preview()).unwrap();

    let mut names = fs::read_dir(&destination)
        .unwrap()
        .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
        .collect::<Vec<_>>();
    names.sort();
    assert_eq!(
        names,
        vec![
            "manifest.json",
            PREVIEW_MOUTH_ROI_FILE_NAME,
            PREVIEW_PREDICTION_FILE_NAME,
            PREVIEW_TARGET_FILE_NAME,
        ]
    );
    assert_eq!(
        fs::read(destination.join("manifest.json")).unwrap(),
        serde_json::to_vec(&manifest).unwrap()
    );
    assert!(staging_entries(root.path()).is_empty());
}

#[test]
fn existing_preview_destination_is_preserved_without_staging() {
    let root = tempfile::tempdir().unwrap();
    let destination = root.path().join("preview-000001");
    fs::create_dir(&destination).unwrap();
    fs::write(destination.join("sentinel"), b"old preview").unwrap();

    let error = write_preview_artifact(&destination, &preview()).unwrap_err();
    assert!(matches!(error, TrainingError::CheckpointDirectory(_)));
    assert_eq!(
        fs::read(destination.join("sentinel")).unwrap(),
        b"old preview"
    );
    assert!(staging_entries(root.path()).is_empty());
}

#[test]
fn invalid_preview_is_rejected_before_staging() {
    let root = tempfile::tempdir().unwrap();
    let destination = root.path().join("preview-000001");
    let invalid = PreviewArtifact::new(
        4,
        9,
        2,
        17,
        "original-unet",
        "a".repeat(64),
        "training",
        vec![0.25; 10],
        vec![0.25; PREVIEW_TENSOR_ELEMENTS],
        vec![0.25; PREVIEW_TENSOR_ELEMENTS],
    )
    .unwrap_err();
    assert!(matches!(invalid, TrainingError::InvalidCheckpoint(_)));
    assert!(!destination.exists());
    assert!(staging_entries(root.path()).is_empty());
}

#[test]
fn metrics_round_trip_and_existing_destination_are_atomic() {
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("metrics.json");
    let original = metrics();

    write_training_metrics(&path, &original).unwrap();
    assert_eq!(read_training_metrics(&path).unwrap(), original);
    let before = fs::read(&path).unwrap();

    let error = write_training_metrics(&path, &metrics()).unwrap_err();
    assert!(matches!(error, TrainingError::CheckpointDirectory(_)));
    assert_eq!(fs::read(&path).unwrap(), before);
}

#[test]
fn oversized_metrics_json_is_rejected_before_deserialization() {
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("metrics.json");
    fs::write(&path, vec![b' '; 64 * 1024 + 1]).unwrap();

    let error = read_training_metrics(&path).unwrap_err();
    assert!(matches!(error, TrainingError::CheckpointDirectory(_)));
}

#[test]
fn metrics_unknown_fields_are_rejected() {
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("metrics.json");
    let mut value = serde_json::to_value(metrics()).unwrap();
    value["unexpected"] = true.into();
    fs::write(&path, serde_json::to_vec(&value).unwrap()).unwrap();

    let error = read_training_metrics(&path).unwrap_err();
    assert!(matches!(error, TrainingError::InvalidCheckpoint(_)));
}

#[test]
fn manifest_contains_actual_byte_counts_and_hashes() {
    let root = tempfile::tempdir().unwrap();
    let destination = root.path().join("preview-000001");
    write_preview_artifact(&destination, &preview()).unwrap();
    let manifest: serde_json::Value =
        serde_json::from_slice(&fs::read(destination.join("manifest.json")).unwrap()).unwrap();

    for (field, file_name) in [
        ("prediction", PREVIEW_PREDICTION_FILE_NAME),
        ("target", PREVIEW_TARGET_FILE_NAME),
        ("mouth_roi", PREVIEW_MOUTH_ROI_FILE_NAME),
    ] {
        let bytes = fs::read(destination.join(file_name)).unwrap();
        assert_eq!(
            manifest[field]["bytes"].as_u64().unwrap(),
            bytes.len() as u64
        );
        assert_eq!(manifest[field]["sha256"].as_str().unwrap(), sha256(&bytes));
    }
}

#[test]
fn manifest_byte_count_tampering_is_rejected() {
    let root = tempfile::tempdir().unwrap();
    let destination = root.path().join("preview-000001");
    write_preview_artifact(&destination, &preview()).unwrap();
    let manifest_path = destination.join("manifest.json");
    let mut manifest: serde_json::Value =
        serde_json::from_slice(&fs::read(&manifest_path).unwrap()).unwrap();
    manifest["prediction"]["bytes"] = 1_u64.into();
    fs::write(manifest_path, serde_json::to_vec(&manifest).unwrap()).unwrap();

    let error = read_preview_artifact(&destination, "original-unet", &"a".repeat(64)).unwrap_err();
    assert!(matches!(error, TrainingError::InvalidCheckpoint(_)));
}

#[test]
fn malformed_header_with_refreshed_hash_is_rejected_during_decode() {
    let root = tempfile::tempdir().unwrap();
    let destination = root.path().join("preview-000001");
    write_preview_artifact(&destination, &preview()).unwrap();
    let path = destination.join(PREVIEW_TARGET_FILE_NAME);
    let mut bytes = fs::read(&path).unwrap();
    bytes[0] ^= 1;
    fs::write(&path, bytes).unwrap();
    refresh_manifest_entry(&destination, "target", PREVIEW_TARGET_FILE_NAME);

    let error = read_preview_artifact(&destination, "original-unet", &"a".repeat(64)).unwrap_err();
    assert!(matches!(error, TrainingError::InvalidCheckpoint(_)));
}

#[test]
fn non_finite_payload_with_refreshed_hash_is_rejected() {
    let root = tempfile::tempdir().unwrap();
    let destination = root.path().join("preview-000001");
    write_preview_artifact(&destination, &preview()).unwrap();
    let path = destination.join(PREVIEW_PREDICTION_FILE_NAME);
    let mut bytes = fs::read(&path).unwrap();
    bytes[32..36].copy_from_slice(&f32::NAN.to_le_bytes());
    fs::write(&path, bytes).unwrap();
    refresh_manifest_entry(&destination, "prediction", PREVIEW_PREDICTION_FILE_NAME);

    let error = read_preview_artifact(&destination, "original-unet", &"a".repeat(64)).unwrap_err();
    assert!(matches!(error, TrainingError::InvalidCheckpoint(_)));
}

#[test]
fn oversized_preview_entry_is_rejected() {
    let root = tempfile::tempdir().unwrap();
    let destination = root.path().join("preview-000001");
    write_preview_artifact(&destination, &preview()).unwrap();
    let path = destination.join(PREVIEW_TARGET_FILE_NAME);
    let mut bytes = fs::read(&path).unwrap();
    bytes.resize(1024 * 1024 + 1, 0);
    fs::write(path, bytes).unwrap();

    let error = read_preview_artifact(&destination, "original-unet", &"a".repeat(64)).unwrap_err();
    assert!(matches!(error, TrainingError::InvalidCheckpoint(_)));
}

#[test]
fn symlinked_preview_entry_is_rejected_when_supported() {
    let root = tempfile::tempdir().unwrap();
    let destination = root.path().join("preview-000001");
    write_preview_artifact(&destination, &preview()).unwrap();
    let path = destination.join(PREVIEW_TARGET_FILE_NAME);
    let target = root.path().join("target.f32");
    fs::rename(&path, &target).unwrap();
    if let Err(error) = create_file_symlink(&target, &path) {
        if matches!(
            error.kind(),
            std::io::ErrorKind::PermissionDenied | std::io::ErrorKind::Unsupported
        ) {
            eprintln!("skipping symlink assertion: {error}");
            return;
        }
        panic!("unable to create test symlink: {error}");
    }

    let error = read_preview_artifact(&destination, "original-unet", &"a".repeat(64)).unwrap_err();
    assert!(matches!(error, TrainingError::CheckpointDirectory(_)));
}

#[test]
fn symlinked_preview_parent_is_rejected_without_writing_through_it() {
    let root = tempfile::tempdir().unwrap();
    let real_parent = root.path().join("real-parent");
    fs::create_dir(&real_parent).unwrap();
    let linked_parent = root.path().join("linked-parent");
    if let Err(error) = create_directory_symlink(&real_parent, &linked_parent) {
        if matches!(
            error.kind(),
            std::io::ErrorKind::PermissionDenied | std::io::ErrorKind::Unsupported
        ) {
            eprintln!("skipping symlink assertion: {error}");
            return;
        }
        panic!("unable to create test symlink: {error}");
    }

    let destination = linked_parent.join("preview-000001");
    let error = write_preview_artifact(&destination, &preview()).unwrap_err();
    assert!(matches!(error, TrainingError::CheckpointDirectory(_)));
    assert!(!real_parent.join("preview-000001").exists());
    assert!(staging_entries(root.path()).is_empty());
}

#[test]
fn symlinked_preview_read_parent_is_rejected() {
    let root = tempfile::tempdir().unwrap();
    let real_parent = root.path().join("real-parent");
    fs::create_dir(&real_parent).unwrap();
    let real_destination = real_parent.join("preview-000001");
    write_preview_artifact(&real_destination, &preview()).unwrap();

    let linked_parent = root.path().join("linked-parent");
    if let Err(error) = create_directory_symlink(&real_parent, &linked_parent) {
        if matches!(
            error.kind(),
            std::io::ErrorKind::PermissionDenied | std::io::ErrorKind::Unsupported
        ) {
            eprintln!("skipping symlink assertion: {error}");
            return;
        }
        panic!("unable to create test directory symlink: {error}");
    }

    let linked_destination = linked_parent.join("preview-000001");
    let error =
        read_preview_artifact(&linked_destination, "original-unet", &"a".repeat(64)).unwrap_err();
    assert!(matches!(error, TrainingError::CheckpointDirectory(_)));
}

#[test]
fn symlinked_preview_directory_is_rejected_when_supported() {
    let root = tempfile::tempdir().unwrap();
    let real = root.path().join("real-preview");
    fs::create_dir(&real).unwrap();
    for name in [
        "manifest.json",
        PREVIEW_PREDICTION_FILE_NAME,
        PREVIEW_TARGET_FILE_NAME,
        PREVIEW_MOUTH_ROI_FILE_NAME,
    ] {
        fs::write(real.join(name), b"placeholder").unwrap();
    }
    let linked = root.path().join("linked-preview");
    if let Err(error) = create_directory_symlink(&real, &linked) {
        if matches!(
            error.kind(),
            std::io::ErrorKind::PermissionDenied | std::io::ErrorKind::Unsupported
        ) {
            eprintln!("skipping symlink assertion: {error}");
            return;
        }
        panic!("unable to create test symlink: {error}");
    }

    let error = read_preview_artifact(&linked, "original-unet", &"a".repeat(64)).unwrap_err();
    assert!(matches!(error, TrainingError::CheckpointDirectory(_)));
}

#[test]
fn missing_preview_directory_is_reported_as_directory_error() {
    let root = tempfile::tempdir().unwrap();
    let missing = root.path().join("missing-preview");
    let error = read_preview_artifact(&missing, "original-unet", &"a".repeat(64)).unwrap_err();
    assert!(matches!(error, TrainingError::CheckpointDirectory(_)));
}

#[test]
fn symlinked_metrics_file_and_parent_are_rejected() {
    let root = tempfile::tempdir().unwrap();
    let real_metrics = root.path().join("real-metrics.json");
    write_training_metrics(&real_metrics, &metrics()).unwrap();
    let linked_metrics = root.path().join("linked-metrics.json");
    if let Err(error) = create_file_symlink(&real_metrics, &linked_metrics) {
        if matches!(
            error.kind(),
            std::io::ErrorKind::PermissionDenied | std::io::ErrorKind::Unsupported
        ) {
            eprintln!("skipping symlink assertion: {error}");
        } else {
            panic!("unable to create test symlink: {error}");
        }
    } else {
        let error = read_training_metrics(&linked_metrics).unwrap_err();
        assert!(matches!(error, TrainingError::CheckpointDirectory(_)));
    }

    let real_parent = root.path().join("real-metrics-parent");
    fs::create_dir(&real_parent).unwrap();
    let linked_parent = root.path().join("linked-metrics-parent");
    if let Err(error) = create_directory_symlink(&real_parent, &linked_parent) {
        if matches!(
            error.kind(),
            std::io::ErrorKind::PermissionDenied | std::io::ErrorKind::Unsupported
        ) {
            eprintln!("skipping symlink assertion: {error}");
            return;
        }
        panic!("unable to create test directory symlink: {error}");
    }
    let error = read_training_metrics(linked_parent.join("metrics.json")).unwrap_err();
    assert!(matches!(error, TrainingError::CheckpointDirectory(_)));
}

#[test]
fn metrics_schema_version_constant_is_one() {
    assert_eq!(TRAINING_METRICS_SCHEMA_VERSION, 1);
}
