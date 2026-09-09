//! The `export_onnx` command: a published package becomes an ONNX model.

use std::{fs, path::Path};

use feathertalk_domain::{
    ErrorCode, ExportOnnxParams, OnnxExportKind, Progress, Request, TaskStage,
};
use feathertalk_export::{
    onnx::{ONNX_OPSET_VERSION, OnnxModelKind, validate_model_contract},
    read_package_manifest,
};
use feathertalk_media::CancellationToken;
use feathertalk_training::CheckpointDescriptor;
use feathertalk_worker::{CommandOutcome, NoReporter, WorkerConfig, execute, execute_export_onnx};

#[path = "support/mod.rs"]
mod support;

use support::{
    Recorder, published_mobileone_package, published_onnx_hubert_package, published_package,
    published_unet_package, write_checkpoint,
};

/// The version `published_package` publishes under, which is also the version
/// `WorkerConfig` reports.
const WORKER_VERSION: &str = "0.1.0";

fn params(source: &Path, kind: OnnxExportKind, destination: &Path) -> ExportOnnxParams {
    ExportOnnxParams {
        source: source.to_path_buf(),
        kind,
        destination: destination.to_path_buf(),
    }
}

#[test]
fn a_micro_feather_hubert_package_becomes_an_onnx_model() {
    let root = tempfile::tempdir().expect("a temporary directory is available");
    let source = published_onnx_hubert_package(root.path(), "hubert");
    let destination = root.path().join("feather_hubert.onnx");
    let recorder = Recorder::new();

    let payload = execute_export_onnx(
        &params(&source, OnnxExportKind::FeatherHubert, &destination),
        &CancellationToken::new(),
        &recorder,
    )
    .expect("a micro package exports");

    let manifest = read_package_manifest(&source).expect("the package is readable");
    assert_eq!(payload["kind"], "export_onnx");
    assert_eq!(payload["model_kind"], "feather_hubert");
    assert_eq!(
        payload["architecture_version"],
        manifest.architecture_version.as_str()
    );
    assert_eq!(payload["source"], source.display().to_string());
    assert_eq!(payload["destination"], destination.display().to_string());
    assert_eq!(payload["opset"], ONNX_OPSET_VERSION);
    assert_eq!(
        payload["source_model_sha256"],
        manifest.model.sha256.as_str()
    );
    assert_eq!(
        payload["sha256"]
            .as_str()
            .expect("the digest is text")
            .len(),
        64
    );
    let published = fs::read(&destination).expect("the model is published");
    assert_eq!(
        payload["bytes"].as_u64().expect("the size is numeric"),
        u64::try_from(published.len()).expect("the size fits u64")
    );
    // The published contract, read back from the file the client will ship.
    validate_model_contract(&published, &OnnxModelKind::FeatherHubert.public_contract())
        .expect("the published graph honours its contract");

    assert_eq!(
        recorder.events(),
        vec![
            (TaskStage::Preparing, None),
            (
                TaskStage::Exporting,
                Some(Progress {
                    completed: 0,
                    total: Some(1),
                }),
            ),
            (
                TaskStage::Exporting,
                Some(Progress {
                    completed: 1,
                    total: Some(1),
                }),
            ),
        ]
    );
}

#[test]
fn a_micro_original_unet_package_becomes_an_onnx_model() {
    let root = tempfile::tempdir().expect("a temporary directory is available");
    let source = published_unet_package(root.path(), "original_unet_v1");
    let destination = root.path().join("original_unet.onnx");

    let payload = execute_export_onnx(
        &params(&source, OnnxExportKind::OriginalUnet, &destination),
        &CancellationToken::new(),
        &NoReporter,
    )
    .expect("a micro package exports");

    let manifest = read_package_manifest(&source).expect("the package is readable");
    assert_eq!(payload["model_kind"], "original_unet");
    assert_eq!(payload["opset"], ONNX_OPSET_VERSION);
    assert_eq!(
        payload["source_model_sha256"],
        manifest.model.sha256.as_str()
    );
    let published = fs::read(&destination).expect("the model is published");
    validate_model_contract(&published, &OnnxModelKind::OriginalUnet.public_contract())
        .expect("the published graph honours its contract");
}

/// The fused package: the graph is loaded as inference weights and serialised.
/// The branched shape has no test because it has no package -- see
/// `published_mobileone_package`.
#[test]
fn a_fused_mobileone_package_becomes_an_onnx_model() {
    let root = tempfile::tempdir().expect("a temporary directory is available");
    let source = published_mobileone_package(root.path(), "mobileone_unet_v1");
    let destination = root.path().join("mobileone_unet.onnx");

    let payload = execute_export_onnx(
        &params(&source, OnnxExportKind::MobileOneUnet, &destination),
        &CancellationToken::new(),
        &NoReporter,
    )
    .expect("a fused package exports");

    assert_eq!(payload["model_kind"], "mobileone_unet");
    let published = fs::read(&destination).expect("the model is published");
    validate_model_contract(&published, &OnnxModelKind::MobileOneUnet.public_contract())
        .expect("the published graph honours its contract");
}

#[test]
fn export_onnx_rejects_a_relative_source() {
    let root = tempfile::tempdir().expect("a temporary directory is available");
    let destination = root.path().join("model.onnx");

    let error = execute_export_onnx(
        &params(
            Path::new("models/hubert"),
            OnnxExportKind::FeatherHubert,
            &destination,
        ),
        &CancellationToken::new(),
        &NoReporter,
    )
    .expect_err("a relative source is refused");

    assert!(error.to_string().contains("absolute"), "{error}");
    assert!(!destination.exists());
}

#[test]
fn export_onnx_rejects_a_training_checkpoint_as_source() {
    let root = tempfile::tempdir().expect("a temporary directory is available");
    let checkpoint = root.path().join("epoch-1");
    write_checkpoint(
        &checkpoint,
        CheckpointDescriptor::new("original_unet", "original-unet-burn-v1", "0".repeat(64)),
    );
    let destination = root.path().join("model.onnx");

    let error = execute_export_onnx(
        &params(&checkpoint, OnnxExportKind::OriginalUnet, &destination),
        &CancellationToken::new(),
        &NoReporter,
    )
    .expect_err("a checkpoint is not an ONNX source");

    assert!(error.to_string().contains("model package"), "{error}");
    assert!(!destination.exists());
}

#[test]
fn export_onnx_rejects_a_kind_the_package_does_not_hold() {
    let root = tempfile::tempdir().expect("a temporary directory is available");
    let source = published_package(root.path(), "hubert", WORKER_VERSION);
    let destination = root.path().join("model.onnx");

    let error = execute_export_onnx(
        &params(&source, OnnxExportKind::OriginalUnet, &destination),
        &CancellationToken::new(),
        &NoReporter,
    )
    .expect_err("the requested kind has to be the kind the package holds");

    assert!(error.to_string().contains("feather_hubert"), "{error}");
    assert!(error.to_string().contains("original_unet"), "{error}");
    assert!(!destination.exists());
}

#[test]
fn export_onnx_reports_a_configuration_the_graph_cannot_carry() {
    let root = tempfile::tempdir().expect("a temporary directory is available");
    // A parity-micro package: 64 hidden channels, which the public contract's
    // 1024-wide `hidden` output cannot describe.
    let source = published_package(root.path(), "hubert", WORKER_VERSION);
    let destination = root.path().join("model.onnx");

    let error = execute_export_onnx(
        &params(&source, OnnxExportKind::FeatherHubert, &destination),
        &CancellationToken::new(),
        &NoReporter,
    )
    .expect_err("a configuration the graph cannot carry is refused");

    assert!(error.to_string().contains("output_dim"), "{error}");
    assert_eq!(error.stage(), TaskStage::Exporting);
    assert!(!destination.exists());
}

#[test]
fn export_onnx_rejects_an_existing_destination() {
    let root = tempfile::tempdir().expect("a temporary directory is available");
    let source = published_package(root.path(), "hubert", WORKER_VERSION);
    let destination = root.path().join("model.onnx");
    fs::write(&destination, b"occupied").expect("the destination is occupied");

    let error = execute_export_onnx(
        &params(&source, OnnxExportKind::FeatherHubert, &destination),
        &CancellationToken::new(),
        &NoReporter,
    )
    .expect_err("an occupied destination is refused");

    assert!(
        error.to_string().contains("must not already exist"),
        "{error}"
    );
    assert_eq!(
        fs::read(&destination).expect("the file survives"),
        b"occupied"
    );
}

#[test]
fn export_onnx_rejects_a_destination_without_a_directory_parent() {
    let root = tempfile::tempdir().expect("a temporary directory is available");
    let source = published_package(root.path(), "hubert", WORKER_VERSION);
    let destination = root.path().join("missing").join("model.onnx");

    let error = execute_export_onnx(
        &params(&source, OnnxExportKind::FeatherHubert, &destination),
        &CancellationToken::new(),
        &NoReporter,
    )
    .expect_err("a missing destination parent is refused");

    assert!(error.to_string().contains("parent"), "{error}");
    assert!(!destination.exists());
}

#[test]
fn export_onnx_honours_cancellation_before_it_starts() {
    let root = tempfile::tempdir().expect("a temporary directory is available");
    let source = published_package(root.path(), "hubert", WORKER_VERSION);
    let destination = root.path().join("model.onnx");
    let token = CancellationToken::new();
    token.cancel();

    let error = execute_export_onnx(
        &params(&source, OnnxExportKind::FeatherHubert, &destination),
        &token,
        &NoReporter,
    )
    .expect_err("a cancelled export does not publish");

    assert!(error.is_cancelled(), "{error}");
    assert_eq!(error.stage(), TaskStage::Preparing);
    assert!(!destination.exists());
}

#[test]
fn a_cancel_during_the_export_publishes_nothing() {
    let root = tempfile::tempdir().expect("a temporary directory is available");
    let source = published_package(root.path(), "hubert", WORKER_VERSION);
    let destination = root.path().join("model.onnx");
    let token = CancellationToken::new();
    // The events are `Preparing` and `Exporting 0/1`, so the cancel lands in the
    // export stage rather than in admission.
    let recorder = Recorder::cancelling_after(2, token.clone());

    let error = execute_export_onnx(
        &params(&source, OnnxExportKind::FeatherHubert, &destination),
        &token,
        &recorder,
    )
    .expect_err("a cancelled export does not publish");

    assert!(error.is_cancelled(), "{error}");
    assert_eq!(error.stage(), TaskStage::Exporting);
    assert!(!destination.exists());
}

#[test]
fn the_command_maps_an_onnx_failure_to_model_incompatible() {
    let root = tempfile::tempdir().expect("a temporary directory is available");
    let destination = root.path().join("model.onnx");

    let outcome = execute(
        &Request::ExportOnnx(params(
            Path::new("models/hubert"),
            OnnxExportKind::FeatherHubert,
            &destination,
        )),
        &WorkerConfig::from_values(None, None, None),
        &CancellationToken::new(),
        &NoReporter,
    );

    let CommandOutcome::Failed(error) = outcome else {
        panic!("a relative source is a task failure, got {outcome:?}");
    };
    assert_eq!(error.code, ErrorCode::ModelIncompatible);
    assert_eq!(error.summary, "ONNX 导出失败");
    assert_eq!(error.stage, TaskStage::Preparing);
    assert!(error.detail.contains("absolute"), "{}", error.detail);
    assert!(!destination.exists());
}

#[test]
fn the_command_reports_a_cancelled_onnx_export_as_cancelled() {
    let root = tempfile::tempdir().expect("a temporary directory is available");
    let source = published_package(root.path(), "hubert", WORKER_VERSION);
    let destination = root.path().join("model.onnx");
    let token = CancellationToken::new();
    token.cancel();

    let outcome = execute(
        &Request::ExportOnnx(params(&source, OnnxExportKind::FeatherHubert, &destination)),
        &WorkerConfig::from_values(None, None, None),
        &token,
        &NoReporter,
    );

    assert!(
        matches!(outcome, CommandOutcome::Cancelled),
        "got {outcome:?}"
    );
    assert!(!destination.exists());
}
