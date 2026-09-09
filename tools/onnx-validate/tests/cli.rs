use std::{fs, process::Command};

use feathertalk_export::onnx::{OnnxModel, OnnxModelKind, serialize_model};

fn binary() -> Command {
    Command::new(env!("CARGO_BIN_EXE_feathertalk-onnx-validate"))
}

fn write_valid_model(root: &std::path::Path) -> std::path::PathBuf {
    let model = root.join("model.onnx");
    fs::write(
        &model,
        serialize_model(&OnnxModel::new(OnnxModelKind::FeatherHubert)).unwrap(),
    )
    .unwrap();
    model
}

#[test]
fn structural_mode_validates_without_provider() {
    let root = tempfile::tempdir().unwrap();
    let model = write_valid_model(root.path());
    let output = binary()
        .args([
            "--model",
            model.to_str().unwrap(),
            "--kind",
            "feather-hubert",
            "--structural-only",
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let value: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(value["provider"], "structural-only");
    assert_eq!(value["passed"], true);
}

#[test]
fn malformed_model_is_nonzero_and_does_not_require_input_files() {
    let root = tempfile::tempdir().unwrap();
    let model = root.path().join("malformed.onnx");
    fs::write(&model, b"not protobuf").unwrap();
    let output = binary()
        .args([
            "--model",
            model.to_str().unwrap(),
            "--kind",
            "feather-hubert",
            "--structural-only",
        ])
        .output()
        .unwrap();
    assert!(!output.status.success());
}

#[test]
fn runtime_mode_requires_both_fixture_paths() {
    let root = tempfile::tempdir().unwrap();
    let model = write_valid_model(root.path());
    let input = root.path().join("input.npy");
    let output = binary()
        .args([
            "--model",
            model.to_str().unwrap(),
            "--kind",
            "feather-hubert",
            "--input",
            input.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(
        String::from_utf8_lossy(&output.stderr)
            .contains("--input and --expected-output must be provided together")
    );
}

#[test]
fn structural_mode_rejects_runtime_fixture_arguments() {
    let root = tempfile::tempdir().unwrap();
    let model = write_valid_model(root.path());
    let input = root.path().join("input.npy");
    let expected = root.path().join("expected.npy");
    let output = binary()
        .args([
            "--model",
            model.to_str().unwrap(),
            "--kind",
            "feather-hubert",
            "--input",
            input.to_str().unwrap(),
            "--expected-output",
            expected.to_str().unwrap(),
            "--structural-only",
        ])
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(
        String::from_utf8_lossy(&output.stderr)
            .contains("--structural-only cannot be combined with runtime fixture arguments")
    );
}

#[test]
fn runtime_mode_requires_reference_fixtures() {
    let root = tempfile::tempdir().unwrap();
    let model = write_valid_model(root.path());
    let output = binary()
        .args([
            "--model",
            model.to_str().unwrap(),
            "--kind",
            "feather-hubert",
        ])
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(
        String::from_utf8_lossy(&output.stderr)
            .contains("runtime mode requires --input and --expected-output")
    );
}
