//! Publishing a serialised ONNX graph: validate first, then write once.

use feathertalk_export::onnx::{ONNX_OPSET_VERSION, OnnxModelKind, export_original_unet_onnx};
use feathertalk_export::{OnnxPublishError, publish_onnx_model};
use feathertalk_models::{backend::CpuBackend, unet::OriginalUnetConfig};
use sha2::{Digest, Sha256};

/// A real graph, small enough to build in a test. The declared interface is the
/// production one either way: `OnnxModel::new` takes the inputs and outputs from
/// `OnnxModelKind`, not from the configuration.
fn micro_original_unet() -> Vec<u8> {
    let config = OriginalUnetConfig::parity_micro();
    let device = Default::default();
    let model = config.init::<CpuBackend>(&device);
    export_original_unet_onnx(&model, &config).expect("the micro graph exports")
}

#[test]
fn a_validated_graph_is_published_with_its_digest() {
    let root = tempfile::tempdir().expect("a temporary directory is available");
    let destination = root.path().join("original_unet.onnx");
    let bytes = micro_original_unet();

    let artifact = publish_onnx_model(&destination, OnnxModelKind::OriginalUnet, &bytes)
        .expect("a contract-clean graph publishes");

    assert_eq!(artifact.kind(), OnnxModelKind::OriginalUnet);
    assert_eq!(artifact.opset(), ONNX_OPSET_VERSION);
    assert_eq!(
        artifact.bytes(),
        u64::try_from(bytes.len()).expect("the size fits u64")
    );
    assert_eq!(artifact.sha256(), hex::encode(Sha256::digest(&bytes)));
    assert_eq!(
        std::fs::read(&destination).expect("the file is published"),
        bytes
    );
}

#[test]
fn a_graph_that_fails_the_contract_publishes_nothing() {
    let root = tempfile::tempdir().expect("a temporary directory is available");
    let destination = root.path().join("original_unet.onnx");

    let error = publish_onnx_model(
        &destination,
        OnnxModelKind::OriginalUnet,
        b"not an onnx model",
    )
    .expect_err("bytes that are not a contract-clean graph are refused");

    assert!(matches!(error, OnnxPublishError::Contract(_)), "{error}");
    assert!(!destination.exists());
}

#[test]
fn a_second_publication_to_the_same_path_is_refused() {
    let root = tempfile::tempdir().expect("a temporary directory is available");
    let destination = root.path().join("original_unet.onnx");
    let bytes = micro_original_unet();
    publish_onnx_model(&destination, OnnxModelKind::OriginalUnet, &bytes)
        .expect("the first publication succeeds");

    let error = publish_onnx_model(&destination, OnnxModelKind::OriginalUnet, &bytes)
        .expect_err("the destination is published once");

    assert!(matches!(error, OnnxPublishError::Io(_)), "{error}");
    assert_eq!(
        std::fs::read(&destination).expect("the file survives"),
        bytes
    );
    // The staging file is cleaned up, so the directory holds the model alone.
    assert_eq!(
        std::fs::read_dir(root.path())
            .expect("the directory is readable")
            .count(),
        1
    );
}

#[test]
fn a_destination_without_a_directory_parent_is_refused() {
    let root = tempfile::tempdir().expect("a temporary directory is available");
    let destination = root.path().join("missing").join("original_unet.onnx");

    let error = publish_onnx_model(
        &destination,
        OnnxModelKind::OriginalUnet,
        &micro_original_unet(),
    )
    .expect_err("a missing parent is refused");

    assert!(
        matches!(error, OnnxPublishError::InvalidRequest(_)),
        "{error}"
    );
    assert!(error.to_string().contains("parent"), "{error}");
    assert!(!destination.exists());
}
