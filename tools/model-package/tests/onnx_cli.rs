use std::{fs, process::Command};

use feathertalk_export::onnx::{OnnxModel, OnnxModelKind, serialize_model};

fn binary() -> Command {
    Command::new(env!("CARGO_BIN_EXE_feathertalk-model-package"))
}

#[test]
fn onnx_help_lists_export_and_validation_commands() {
    let output = binary().args(["onnx", "--help"]).output().unwrap();
    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    for command in ["feather-hubert", "unet", "validate"] {
        assert!(
            stdout.contains(command),
            "missing command {command}: {stdout}"
        );
    }
}

#[test]
fn onnx_validate_prints_json_for_a_structurally_valid_model() {
    let root = tempfile::tempdir().unwrap();
    let source = root.path().join("model.onnx");
    fs::write(
        &source,
        serialize_model(&OnnxModel::new(OnnxModelKind::FeatherHubert)).unwrap(),
    )
    .unwrap();

    let output = binary()
        .args([
            "onnx",
            "validate",
            "--source",
            source.to_str().unwrap(),
            "--kind",
            "feather-hubert",
        ])
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let value: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(value["model_kind"], "feather_hubert");
    assert_eq!(value["opset"], 17);
    assert_eq!(value["sha256"].as_str().unwrap().len(), 64);
}

#[test]
fn onnx_export_refuses_existing_destination_before_reading_source() {
    let root = tempfile::tempdir().unwrap();
    let destination = root.path().join("existing.onnx");
    fs::write(&destination, b"keep-me").unwrap();

    let output = binary()
        .args([
            "onnx",
            "feather-hubert",
            "--source",
            root.path().join("missing.pth").to_str().unwrap(),
            "--destination",
            destination.to_str().unwrap(),
        ])
        .output()
        .unwrap();

    assert!(!output.status.success());
    assert_eq!(fs::read(&destination).unwrap(), b"keep-me");
    assert!(String::from_utf8_lossy(&output.stderr).contains("destination already exists"));
}
