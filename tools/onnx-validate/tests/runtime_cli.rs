#![cfg(feature = "ort-runtime")]

use std::{fs, process::Command};

use feathertalk_export::onnx::{
    OnnxModel, OnnxModelKind, OnnxModelProto, OnnxNodeProto, OnnxTensorProto, serialize_model,
};
use ndarray::{ArrayD, IxDyn};
use prost::Message;

fn binary() -> Command {
    Command::new(env!("CARGO_BIN_EXE_feathertalk-onnx-validate"))
}

fn write_valid_model(root: &std::path::Path, kind: OnnxModelKind) -> std::path::PathBuf {
    let model = root.join("model.onnx");
    fs::write(&model, serialize_model(&OnnxModel::new(kind)).unwrap()).unwrap();
    model
}

#[test]
fn malformed_input_npy_is_nonzero_before_runtime_initialization() {
    let root = tempfile::tempdir().unwrap();
    let model = write_valid_model(root.path(), OnnxModelKind::FeatherHubert);
    let input = root.path().join("input.npy");
    let expected = root.path().join("expected.npy");
    fs::write(&input, b"not npy").unwrap();
    fs::write(&expected, b"not npy either").unwrap();

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
        ])
        .output()
        .unwrap();

    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("input NPY"));
}

#[test]
fn unet_runtime_mode_requires_two_inputs() {
    let root = tempfile::tempdir().unwrap();
    let model = write_valid_model(root.path(), OnnxModelKind::OriginalUnet);
    let input = root.path().join("input.npy");
    let expected = root.path().join("expected.npy");

    let output = binary()
        .args([
            "--model",
            model.to_str().unwrap(),
            "--kind",
            "original-unet",
            "--input",
            input.to_str().unwrap(),
            "--expected-output",
            expected.to_str().unwrap(),
        ])
        .output()
        .unwrap();

    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("requires 2 --input values"));
}

#[test]
fn runtime_executes_a_reference_reshape_model_when_dylib_is_available() {
    let Some(runtime) = std::env::var_os("ORT_DYLIB_PATH") else {
        return;
    };
    if !std::path::Path::new(&runtime).is_file() {
        return;
    }

    let root = tempfile::tempdir().unwrap();
    let model = root.path().join("reshape.onnx");
    let mut proto = OnnxModelProto::decode(
        serialize_model(&OnnxModel::new(OnnxModelKind::FeatherHubert))
            .unwrap()
            .as_slice(),
    )
    .unwrap();
    let graph = proto.graph.as_mut().unwrap();
    graph.initializer.push(OnnxTensorProto {
        dims: vec![3],
        data_type: 7,
        name: "reshape.shape".to_owned(),
        raw_data: [1_i64, 1, 1024]
            .into_iter()
            .flat_map(i64::to_le_bytes)
            .collect(),
        doc_string: String::new(),
    });
    graph.node.push(OnnxNodeProto {
        input: vec!["waveform".to_owned(), "reshape.shape".to_owned()],
        output: vec!["hidden".to_owned()],
        name: "reshape".to_owned(),
        op_type: "Reshape".to_owned(),
        attribute: Vec::new(),
        doc_string: String::new(),
        domain: String::new(),
    });
    fs::write(&model, proto.encode_to_vec()).unwrap();

    let input = root.path().join("input.npy");
    let expected = root.path().join("expected.npy");
    ndarray_npy::write_npy(
        &input,
        &ArrayD::from_shape_vec(IxDyn(&[1, 1024]), vec![0.25_f32; 1024]).unwrap(),
    )
    .unwrap();
    ndarray_npy::write_npy(
        &expected,
        &ArrayD::from_shape_vec(IxDyn(&[1, 1, 1024]), vec![0.25_f32; 1024]).unwrap(),
    )
    .unwrap();

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
        ])
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let report: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(report["provider"], "CPUExecutionProvider");
    assert_eq!(report["max_absolute_error"], 0.0);
    assert_eq!(report["mean_absolute_error"], 0.0);
    assert_eq!(report["passed"], true);
}
