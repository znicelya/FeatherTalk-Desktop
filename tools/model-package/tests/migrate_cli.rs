use std::{fs, io, path::Path, process::Command};

use feathertalk_audio::read_feature_file;
use ndarray::{Array2, Array3};
use zip::ZipArchive;

fn binary() -> Command {
    Command::new(env!("CARGO_BIN_EXE_feathertalk-model-package"))
}

#[test]
fn migrate_help_lists_model_and_feature_commands() {
    let output = binary().args(["migrate", "--help"]).output().unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("model"));
    assert!(stdout.contains("features"));
}

#[test]
fn valid_rank_three_f32_npy_converts_to_versioned_features() {
    let root = tempfile::tempdir().unwrap();
    let source = root.path().join("aud_hu.npy");
    let destination = root.path().join("features.f32");
    let values = (0..4096).map(|value| value as f32).collect::<Vec<_>>();
    ndarray_npy::write_npy(
        &source,
        &Array3::from_shape_vec((2, 2, 1024), values.clone()).unwrap(),
    )
    .unwrap();

    let output = binary()
        .args([
            "migrate",
            "features",
            "--source",
            source.to_str().unwrap(),
            "--destination",
            destination.to_str().unwrap(),
        ])
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let matrix = read_feature_file(&destination).unwrap();
    assert_eq!(matrix.tokens(), 4);
    assert_eq!(matrix.dims(), 1024);
    assert_eq!(matrix.values(), values);
    let report: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(report["source_shape"], serde_json::json!([2, 2, 1024]));
    assert_eq!(report["tokens"], 4);
    assert_eq!(report["dims"], 1024);
    assert_eq!(report["sha256"].as_str().unwrap().len(), 64);
}

#[test]
fn feature_migration_rejects_wrong_dtype_and_rank() {
    let root = tempfile::tempdir().unwrap();
    let f64_source = root.path().join("f64.npy");
    let rank_two_source = root.path().join("rank2.npy");
    ndarray_npy::write_npy(&f64_source, &Array3::<f64>::zeros((1, 2, 1024))).unwrap();
    ndarray_npy::write_npy(&rank_two_source, &Array2::<f32>::zeros((2, 1024))).unwrap();

    let f64_output = run_feature_migration(&f64_source, &root.path().join("f64.f32"));
    assert!(!f64_output.status.success());
    assert!(String::from_utf8_lossy(&f64_output.stderr).contains("f32"));

    let rank_output = run_feature_migration(&rank_two_source, &root.path().join("rank.f32"));
    assert!(!rank_output.status.success());
    assert!(String::from_utf8_lossy(&rank_output.stderr).contains("rank 3"));
}

#[test]
fn feature_migration_rejects_truncated_npy() {
    let root = tempfile::tempdir().unwrap();
    let source = root.path().join("truncated.npy");
    let valid = root.path().join("valid.npy");
    ndarray_npy::write_npy(&valid, &Array3::<f32>::zeros((1, 2, 1024))).unwrap();
    let mut bytes = fs::read(valid).unwrap();
    bytes.truncate(bytes.len() - 1);
    fs::write(&source, bytes).unwrap();

    let output = run_feature_migration(&source, &root.path().join("features.f32"));

    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("NPY"));
}

#[test]
fn feature_migration_refuses_existing_destination_before_reading_source() {
    let root = tempfile::tempdir().unwrap();
    let source = root.path().join("missing.npy");
    let destination = root.path().join("features.f32");
    fs::write(&destination, b"sentinel").unwrap();

    let output = run_feature_migration(&source, &destination);

    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("destination already exists"));
    assert_eq!(fs::read(destination).unwrap(), b"sentinel");
}

#[test]
fn feather_hubert_model_migration_builds_a_standard_package() {
    let root = tempfile::tempdir().unwrap();
    let source = root.path().join("feather_micro.pth");
    extract_golden("weights/feather_micro.pth", &source);
    let licenses = root.path().join("LICENSES.json");
    fs::write(
        &licenses,
        br#"{"schema_version":1,"entries":[{"component":"test","license_id":"LicenseRef-Test","source_url":"https://example.invalid","notice":"test only"}]}"#,
    )
    .unwrap();
    let destination = root.path().join("package");

    let output = binary()
        .args([
            "migrate",
            "model",
            "--kind",
            "feather-hubert",
            "--source",
            source.to_str().unwrap(),
            "--licenses",
            licenses.to_str().unwrap(),
            "--destination",
            destination.to_str().unwrap(),
            "--created-at",
            "2026-08-27T00:00:00Z",
            "--minimum-app-version",
            "0.1.0",
        ])
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let manifest: serde_json::Value =
        serde_json::from_slice(&fs::read(destination.join("manifest.json")).unwrap()).unwrap();
    assert_eq!(manifest["model_type"], "feather_hubert");
    assert_eq!(manifest["configuration"]["output_dim"], 64);
    assert_eq!(manifest["tensors"]["tensor_count"], 35);
}

fn run_feature_migration(source: &Path, destination: &Path) -> std::process::Output {
    binary()
        .args([
            "migrate",
            "features",
            "--source",
            source.to_str().unwrap(),
            "--destination",
            destination.to_str().unwrap(),
        ])
        .output()
        .unwrap()
}

fn extract_golden(member: &str, destination: &Path) {
    let archive_path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/golden/burn-feasibility-v1.zip");
    let archive = fs::File::open(archive_path).unwrap();
    let mut archive = ZipArchive::new(archive).unwrap();
    let mut source = archive.by_name(member).unwrap();
    let mut destination = fs::File::create(destination).unwrap();
    io::copy(&mut source, &mut destination).unwrap();
}
