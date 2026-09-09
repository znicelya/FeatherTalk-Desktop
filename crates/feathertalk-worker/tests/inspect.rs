use std::{fs, path::Path};

use feathertalk_domain::{ErrorCode, InspectModelParams, TaskError, TaskStage};
use feathertalk_media::CancellationToken;
use feathertalk_training::CheckpointDescriptor;
use feathertalk_worker::{
    CommandOutcome, WorkerConfig, checkpoint_descriptor, execute_inspect_model, render_variant,
};
use serde_json::Value;

#[path = "support/mod.rs"]
mod support;

use support::{published_package, write_checkpoint};

/// Inspection needs no toolchain at all, which is the point of the command.
fn config() -> WorkerConfig {
    WorkerConfig::from_values(None, None, None)
}

fn params(source: &Path) -> InspectModelParams {
    InspectModelParams {
        source: source.to_path_buf(),
    }
}

fn failed(outcome: CommandOutcome) -> TaskError {
    match outcome {
        CommandOutcome::Failed(error) => error,
        other => panic!("expected a failure, got {other:?}"),
    }
}

fn completed(outcome: CommandOutcome) -> Value {
    match outcome {
        CommandOutcome::Completed(Some(payload)) => payload,
        other => panic!("expected a payload, got {other:?}"),
    }
}

#[test]
fn a_relative_source_is_refused_before_anything_is_read() {
    let outcome = execute_inspect_model(
        &params(Path::new("models/hubert")),
        &config(),
        &CancellationToken::new(),
    );
    let error = failed(outcome);
    assert_eq!(error.code, ErrorCode::MediaInvalid);
    assert_eq!(error.summary, "模型目录必须是绝对路径");
    assert_eq!(error.stage, TaskStage::Preparing);
}

#[test]
fn a_cancelled_token_stops_before_the_first_read() {
    let root = tempfile::tempdir().expect("the temporary root is created");
    let token = CancellationToken::new();
    token.cancel();
    // The source need not even exist: nothing is read.
    let outcome = execute_inspect_model(&params(&root.path().join("absent")), &config(), &token);
    assert!(matches!(outcome, CommandOutcome::Cancelled));
}

#[test]
fn a_real_package_is_inspected() {
    let root = tempfile::tempdir().expect("the temporary root is created");
    let dir = published_package(root.path(), "package", "0.1.0");
    let payload = completed(execute_inspect_model(
        &params(&dir),
        &config(),
        &CancellationToken::new(),
    ));
    assert_eq!(payload["source_kind"], "model_package");
    assert_eq!(payload["model_kind"], "feather_hubert");
    assert_eq!(payload["compatible"], true);
    assert_eq!(
        payload["files"]
            .as_array()
            .expect("files is an array")
            .len(),
        2
    );
}

#[test]
fn a_real_checkpoint_is_inspected() {
    let root = tempfile::tempdir().expect("the temporary root is created");
    let dir = root.path().join("checkpoint");
    let variant = render_variant("original_unet").expect("the kind is known");
    write_checkpoint(
        &dir,
        checkpoint_descriptor(&variant.configuration()).expect("the descriptor is computed"),
    );
    let payload = completed(execute_inspect_model(
        &params(&dir),
        &config(),
        &CancellationToken::new(),
    ));
    assert_eq!(payload["source_kind"], "training_checkpoint");
    assert_eq!(payload["model_kind"], "original_unet");
    assert_eq!(payload["global_step"], 2);
    assert_eq!(payload["compatible"], true);
}

#[test]
fn an_incompatible_checkpoint_is_still_reported() {
    let root = tempfile::tempdir().expect("the temporary root is created");
    let dir = root.path().join("checkpoint");
    write_checkpoint(
        &dir,
        CheckpointDescriptor::new("legacy_unet", "v1", "0".repeat(64)),
    );
    let payload = completed(execute_inspect_model(
        &params(&dir),
        &config(),
        &CancellationToken::new(),
    ));
    // An unusable model is an answer, not an error (design section 4).
    assert_eq!(payload["compatible"], false);
    assert_eq!(
        payload["incompatibilities"],
        serde_json::json!(["model_kind"])
    );
}

#[test]
fn a_directory_that_is_neither_layout_is_refused() {
    let root = tempfile::tempdir().expect("the temporary root is created");
    let dir = root.path().join("mystery");
    fs::create_dir_all(&dir).expect("the directory is created");
    fs::write(dir.join("weights.pth"), b"x").expect("the placeholder is written");
    let error = failed(execute_inspect_model(
        &params(&dir),
        &config(),
        &CancellationToken::new(),
    ));
    assert_eq!(error.code, ErrorCode::ModelIncompatible);
    assert_eq!(error.summary, "无法识别的模型目录");
}

#[test]
fn a_broken_manifest_is_a_model_error() {
    let root = tempfile::tempdir().expect("the temporary root is created");
    let dir = published_package(root.path(), "package", "0.1.0");
    fs::write(dir.join("manifest.json"), b"not json").expect("the manifest is overwritten");
    let error = failed(execute_inspect_model(
        &params(&dir),
        &config(),
        &CancellationToken::new(),
    ));
    assert_eq!(error.code, ErrorCode::ModelIncompatible);
    assert_eq!(error.stage, TaskStage::Preparing);
}
