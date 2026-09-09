//! The `export_model_package` command: a checkpoint becomes a published package.

use std::{fs, path::Path};

use feathertalk_domain::{ErrorCode, ExportModelPackageParams, Progress, Request, TaskStage};
use feathertalk_export::{
    LICENSE_FILE_NAME, LicenseBundle, LicenseEntry, MANIFEST_FILE_NAME, MODEL_FILE_NAME,
    ModelConfiguration, TrainingMode as PackageTrainingMode, read_package_manifest,
};
use feathertalk_media::CancellationToken;
use feathertalk_models::unet::OriginalUnetConfig;
use feathertalk_training::{
    CHECKPOINT_MODEL_FILE_NAME, CheckpointDescriptor, read_training_checkpoint,
};
use feathertalk_worker::{
    CommandOutcome, NoReporter, RenderVariant, WorkerConfig, checkpoint_descriptor, execute,
    execute_export_model_package, export_plan, publish_checkpoint_package,
};

#[path = "support/mod.rs"]
mod support;

use support::{Recorder, published_package, write_checkpoint};

/// The version the seam publishes under. `WorkerConfig` reports the crate's own
/// version, and this is it, so a test that goes through the command and a test
/// that goes through the seam agree.
const WORKER_VERSION: &str = "0.1.0";

fn params(source: &Path, destination: &Path) -> ExportModelPackageParams {
    ExportModelPackageParams {
        source: source.to_path_buf(),
        destination: destination.to_path_buf(),
    }
}

/// The descriptor the micro fixture checkpoint carries.
fn micro_descriptor() -> CheckpointDescriptor {
    checkpoint_descriptor(&ModelConfiguration::original_unet(
        &OriginalUnetConfig::parity_micro(),
    ))
    .expect("the configuration serialises")
}

/// The variant the micro fixture was written with, which is what the seam takes
/// instead of resolving the production configuration.
fn micro_variant() -> RenderVariant {
    RenderVariant::OriginalUnet(OriginalUnetConfig::parity_micro())
}

/// Writes the license bundle an export reads from beside the checkpoint.
fn write_licenses(directory: &Path) {
    let bundle = LicenseBundle {
        schema_version: 1,
        entries: vec![LicenseEntry {
            component: "synthetic UNet fixture".to_owned(),
            license_id: "LicenseRef-Test".to_owned(),
            source_url: "https://example.invalid/original-unet".to_owned(),
            notice: "test-only local record".to_owned(),
        }],
    };
    fs::write(
        directory.join(LICENSE_FILE_NAME),
        serde_json::to_vec(&bundle).expect("the bundle serialises"),
    )
    .expect("the licenses fixture is written");
}

#[test]
fn export_rejects_a_relative_source_before_reading_it() {
    let root = tempfile::tempdir().expect("a temporary directory is available");
    let destination = root.path().join("package");
    let error = execute_export_model_package(
        &params(Path::new("checkpoints/epoch-1"), &destination),
        &WorkerConfig::from_values(None, None, None),
        &CancellationToken::new(),
        &NoReporter,
    )
    .expect_err("a relative source is refused");
    assert!(error.to_string().contains("absolute"), "{error}");
    assert!(!destination.exists());
}

#[test]
fn export_rejects_a_published_package_as_source() {
    let root = tempfile::tempdir().expect("a temporary directory is available");
    let source = published_package(root.path(), "hubert", WORKER_VERSION);
    write_licenses(root.path());
    let destination = root.path().join("package");
    let error = execute_export_model_package(
        &params(&source, &destination),
        &WorkerConfig::from_values(None, None, None),
        &CancellationToken::new(),
        &NoReporter,
    )
    .expect_err("a published package is not an export source");
    assert!(error.to_string().contains("checkpoint"), "{error}");
    assert!(!destination.exists());
}

#[test]
fn export_rejects_a_checkpoint_without_a_license_bundle_beside_it() {
    let root = tempfile::tempdir().expect("a temporary directory is available");
    let checkpoint = root.path().join("epoch-1");
    write_checkpoint(&checkpoint, micro_descriptor());
    let destination = root.path().join("package");
    let error = execute_export_model_package(
        &params(&checkpoint, &destination),
        &WorkerConfig::from_values(None, None, None),
        &CancellationToken::new(),
        &NoReporter,
    )
    .expect_err("a missing license bundle is refused");
    assert!(error.to_string().contains(LICENSE_FILE_NAME), "{error}");
    assert!(!destination.exists());
}

#[test]
fn export_rejects_an_existing_destination() {
    let root = tempfile::tempdir().expect("a temporary directory is available");
    let checkpoint = root.path().join("epoch-1");
    write_checkpoint(&checkpoint, micro_descriptor());
    write_licenses(root.path());
    let destination = root.path().join("package");
    fs::create_dir(&destination).expect("the destination is occupied");
    let error = execute_export_model_package(
        &params(&checkpoint, &destination),
        &WorkerConfig::from_values(None, None, None),
        &CancellationToken::new(),
        &NoReporter,
    )
    .expect_err("an occupied destination is refused");
    assert!(
        error.to_string().contains("must not already exist"),
        "{error}"
    );
    assert!(
        fs::read_dir(&destination)
            .expect("the destination survives")
            .next()
            .is_none()
    );
}

#[test]
fn export_rejects_an_unknown_model_kind() {
    let root = tempfile::tempdir().expect("a temporary directory is available");
    let checkpoint = root.path().join("epoch-1");
    write_checkpoint(
        &checkpoint,
        CheckpointDescriptor::new("mystery_net", "mystery-v1", "0".repeat(64)),
    );
    write_licenses(root.path());
    let destination = root.path().join("package");
    let error = execute_export_model_package(
        &params(&checkpoint, &destination),
        &WorkerConfig::from_values(None, None, None),
        &CancellationToken::new(),
        &NoReporter,
    )
    .expect_err("an unknown model kind is refused");
    assert!(error.to_string().contains("mystery_net"), "{error}");
    assert!(!destination.exists());
}

#[test]
fn export_refuses_a_configuration_the_worker_does_not_ship() {
    let root = tempfile::tempdir().expect("a temporary directory is available");
    let checkpoint = root.path().join("epoch-1");
    write_checkpoint(&checkpoint, micro_descriptor());
    write_licenses(root.path());
    let destination = root.path().join("package");
    // The kind is `original_unet`, so the command resolves the production
    // configuration; the micro fixture's configuration digest is not that one.
    let error = execute_export_model_package(
        &params(&checkpoint, &destination),
        &WorkerConfig::from_values(None, None, None),
        &CancellationToken::new(),
        &NoReporter,
    )
    .expect_err("a non-production configuration is refused");
    assert!(
        error.to_string().contains("descriptor does not match"),
        "{error}"
    );
    assert!(!destination.exists());
}

#[test]
fn export_honours_cancellation_before_the_load() {
    let root = tempfile::tempdir().expect("a temporary directory is available");
    let checkpoint = root.path().join("epoch-1");
    write_checkpoint(&checkpoint, micro_descriptor());
    write_licenses(root.path());
    let destination = root.path().join("package");
    let token = CancellationToken::new();
    token.cancel();
    let error = execute_export_model_package(
        &params(&checkpoint, &destination),
        &WorkerConfig::from_values(None, None, None),
        &token,
        &NoReporter,
    )
    .expect_err("a cancelled export does not publish");
    assert!(error.is_cancelled(), "{error}");
    assert!(!destination.exists());
}

#[test]
fn a_cancel_during_the_export_publishes_nothing() {
    let root = tempfile::tempdir().expect("a temporary directory is available");
    let checkpoint = root.path().join("epoch-1");
    write_checkpoint(&checkpoint, micro_descriptor());
    write_licenses(root.path());
    let destination = root.path().join("package");
    let metadata = read_training_checkpoint(&checkpoint).expect("the fixture is a checkpoint");
    let plan = export_plan(
        &params(&checkpoint, &destination),
        &metadata,
        WORKER_VERSION,
    )
    .expect("the plan is derived from the checkpoint");
    let token = CancellationToken::new();
    // The first event is `Exporting 0/1`, so the cancel lands inside the
    // publication rather than before it was ever entered.
    let recorder = Recorder::cancelling_after(1, token.clone());
    let error = publish_checkpoint_package(&plan, &micro_variant(), &token, &recorder)
        .expect_err("a cancelled publication does not publish");
    assert!(error.is_cancelled(), "{error}");
    assert_eq!(error.stage(), TaskStage::Exporting);
    assert!(!destination.exists());
}

#[test]
fn a_micro_checkpoint_becomes_a_published_package() {
    let root = tempfile::tempdir().expect("a temporary directory is available");
    let checkpoint = root.path().join("epoch-1");
    write_checkpoint(&checkpoint, micro_descriptor());
    write_licenses(root.path());
    let destination = root.path().join("package");
    let metadata = read_training_checkpoint(&checkpoint).expect("the fixture is a checkpoint");
    let plan = export_plan(
        &params(&checkpoint, &destination),
        &metadata,
        WORKER_VERSION,
    )
    .expect("the plan is derived from the checkpoint");
    let recorder = Recorder::new();

    let payload = publish_checkpoint_package(
        &plan,
        &micro_variant(),
        &CancellationToken::new(),
        &recorder,
    )
    .expect("the package is published");

    assert_eq!(payload["kind"], "export_model_package");
    assert_eq!(payload["model_kind"], "original_unet");
    assert_eq!(payload["source"], checkpoint.display().to_string());
    assert_eq!(payload["destination"], destination.display().to_string());
    assert_eq!(payload["epoch"], 1);
    assert_eq!(payload["global_step"], 2);
    assert_eq!(payload["training_mode"], "baseline");
    assert_eq!(
        payload["source_sha256"],
        metadata.manifest.model.sha256.as_str()
    );
    assert!(
        payload["tensor_count"]
            .as_u64()
            .expect("the tensor count is numeric")
            > 0
    );
    assert!(
        payload["total_elements"]
            .as_u64()
            .expect("the element count is numeric")
            > 0
    );

    // The published package, as the reader sees it.
    assert!(destination.join(MANIFEST_FILE_NAME).is_file());
    assert!(destination.join(MODEL_FILE_NAME).is_file());
    assert!(destination.join(LICENSE_FILE_NAME).is_file());
    let manifest = read_package_manifest(&destination).expect("the package is readable");
    assert_eq!(manifest.model_type, "original_unet");
    assert_eq!(payload["model_sha256"], manifest.model.sha256.as_str());
    assert_eq!(
        payload["architecture_version"],
        manifest.architecture_version.as_str()
    );
    assert_eq!(manifest.minimum_app_version, WORKER_VERSION);
    // The manifest records the recipe the weights were trained under, not
    // `inference`: the mode and the four loss weights come from the checkpoint.
    assert_eq!(manifest.training.mode, PackageTrainingMode::Baseline);
    assert_eq!(manifest.training.mouth_weight, 4.0);
    assert_eq!(manifest.training.temporal_weight, 0.5);
    assert_eq!(manifest.training.temporal_mouth_weight, 4.0);
    assert_eq!(manifest.training.perceptual_weight, 0.01);
    // The source is the checkpoint's own record, at the epoch it was written in.
    assert_eq!(manifest.source.format, "feathertalk-training-checkpoint");
    assert_eq!(manifest.source.identifier, "original_unet");
    assert_eq!(manifest.source.version, "epoch-1-step-2");
    assert_eq!(manifest.source.file_name, CHECKPOINT_MODEL_FILE_NAME);
    assert_eq!(manifest.source.sha256, metadata.manifest.model.sha256);
    assert!(manifest.source.url.is_none());
    // A published package is not a resume point.
    assert!(manifest.optimizer.is_none());
    assert!(manifest.training_state.is_none());

    assert_eq!(
        recorder.events(),
        vec![
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
fn the_command_maps_an_export_failure_to_model_incompatible() {
    let root = tempfile::tempdir().expect("a temporary directory is available");
    let destination = root.path().join("package");
    let outcome = execute(
        &Request::ExportModelPackage(params(Path::new("checkpoints/epoch-1"), &destination)),
        &WorkerConfig::from_values(None, None, None),
        &CancellationToken::new(),
        &NoReporter,
    );
    let CommandOutcome::Failed(error) = outcome else {
        panic!("an unreadable source is a task failure, got {outcome:?}");
    };
    assert_eq!(error.code, ErrorCode::ModelIncompatible);
    assert_eq!(error.summary, "模型导出失败");
    assert_eq!(error.stage, TaskStage::Preparing);
    assert!(error.detail.contains("absolute"), "{}", error.detail);
    assert!(!destination.exists());
}

#[test]
fn the_command_reports_a_cancelled_export_as_cancelled() {
    let root = tempfile::tempdir().expect("a temporary directory is available");
    let checkpoint = root.path().join("epoch-1");
    write_checkpoint(&checkpoint, micro_descriptor());
    write_licenses(root.path());
    let destination = root.path().join("package");
    let token = CancellationToken::new();
    token.cancel();
    let outcome = execute(
        &Request::ExportModelPackage(params(&checkpoint, &destination)),
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
