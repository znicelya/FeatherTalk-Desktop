use std::path::Path;

use feathertalk_export::read_package_manifest;
use feathertalk_training::{CheckpointDescriptor, read_training_checkpoint};
use feathertalk_worker::{
    InspectSummary, InspectedModel, ModelSourceKind, checkpoint_descriptor, checkpoint_files,
    inspect_to_json, package_files, render_variant,
};
use serde_json::Value;

#[path = "support/mod.rs"]
mod support;

use support::{published_package, write_checkpoint};

/// Every key the payload promises, sorted. Both layouts answer with all of them.
const KEYS: [&str; 18] = [
    "architecture_version",
    "compatible",
    "created_at",
    "epoch",
    "files",
    "global_step",
    "incompatibilities",
    "inputs",
    "minimum_app_version",
    "model_config_sha256",
    "model_kind",
    "outputs",
    "parameter_count",
    "schema_version",
    "source_kind",
    "source_path",
    "tensor_count",
    "training_mode",
];

fn keys(value: &Value) -> Vec<String> {
    let mut names: Vec<String> = value
        .as_object()
        .expect("the payload is an object")
        .keys()
        .cloned()
        .collect();
    names.sort();
    names
}

#[test]
fn a_package_payload_answers_every_key() {
    let root = tempfile::tempdir().expect("the temporary root is created");
    let dir = published_package(root.path(), "package", "0.1.0");
    let manifest = read_package_manifest(&dir).expect("the package is readable");
    let files = package_files(&dir, &manifest);
    let payload = inspect_to_json(&InspectSummary {
        source_kind: ModelSourceKind::ModelPackage,
        source_path: &dir,
        model: InspectedModel::Package(&manifest),
        files: &files,
        incompatibilities: &[],
    });

    assert_eq!(keys(&payload), KEYS);
    assert_eq!(payload["source_kind"], "model_package");
    assert_eq!(payload["source_path"], dir.display().to_string());
    assert_eq!(payload["schema_version"], 1);
    assert_eq!(payload["model_kind"], "feather_hubert");
    assert_eq!(payload["training_mode"], "inference");
    // Counting a checkpoint's parameters would mean reading its record, so only a
    // package -- whose manifest states them -- reports these two.
    assert!(
        payload["parameter_count"]
            .as_u64()
            .expect("a package states its parameter count")
            > 0
    );
    assert!(
        payload["tensor_count"]
            .as_u64()
            .expect("a package states its tensor count")
            > 0
    );
    assert_eq!(payload["inputs"][0]["name"], "waveform");
    assert_eq!(payload["outputs"][0]["name"], "hidden");
    assert!(payload["inputs"][0]["shape"].is_array());
    assert_eq!(payload["minimum_app_version"], "0.1.0");
    assert_eq!(payload["created_at"], "2026-08-27T00:00:00Z");
    assert!(payload["model_config_sha256"].is_null());
    assert!(payload["epoch"].is_null());
    assert!(payload["global_step"].is_null());
    let entries = payload["files"].as_array().expect("files is an array");
    assert_eq!(entries.len(), 2);
    assert_eq!(entries[0]["file_name"], "model.safetensors");
    assert_eq!(entries[0]["bytes"], entries[0]["bytes_on_disk"]);
    assert_eq!(payload["compatible"], true);
    assert_eq!(
        payload["incompatibilities"]
            .as_array()
            .expect("incompatibilities is an array")
            .len(),
        0
    );
}

#[test]
fn a_checkpoint_payload_answers_the_same_keys() {
    let root = tempfile::tempdir().expect("the temporary root is created");
    let dir = root.path().join("checkpoint");
    let variant = render_variant("original_unet").expect("the kind is known");
    write_checkpoint(
        &dir,
        checkpoint_descriptor(&variant.configuration()).expect("the descriptor is computed"),
    );
    let checkpoint = read_training_checkpoint(&dir).expect("the checkpoint is readable");
    let files = checkpoint_files(&dir, &checkpoint.manifest);
    let payload = inspect_to_json(&InspectSummary {
        source_kind: ModelSourceKind::TrainingCheckpoint,
        source_path: &dir,
        model: InspectedModel::Checkpoint(&checkpoint),
        files: &files,
        incompatibilities: &[],
    });

    assert_eq!(keys(&payload), KEYS);
    assert_eq!(payload["source_kind"], "training_checkpoint");
    assert_eq!(payload["model_kind"], "original_unet");
    assert_eq!(payload["training_mode"], "baseline");
    assert_eq!(payload["epoch"], 1);
    assert_eq!(payload["global_step"], 2);
    assert!(payload["model_config_sha256"].is_string());
    assert!(payload["parameter_count"].is_null());
    assert!(payload["tensor_count"].is_null());
    assert!(payload["created_at"].is_null());
    assert!(payload["minimum_app_version"].is_null());
    assert_eq!(
        payload["inputs"]
            .as_array()
            .expect("inputs is an array")
            .len(),
        0
    );
    assert_eq!(
        payload["outputs"]
            .as_array()
            .expect("outputs is an array")
            .len(),
        0
    );
    assert_eq!(
        payload["files"]
            .as_array()
            .expect("files is an array")
            .len(),
        3
    );
    assert_eq!(payload["compatible"], true);
}

#[test]
fn any_reason_makes_the_model_incompatible() {
    let root = tempfile::tempdir().expect("the temporary root is created");
    let dir = root.path().join("checkpoint");
    write_checkpoint(
        &dir,
        CheckpointDescriptor::new("legacy_unet", "v1", "0".repeat(64)),
    );
    let checkpoint = read_training_checkpoint(&dir).expect("the checkpoint is readable");
    let files = checkpoint_files(&dir, &checkpoint.manifest);
    let payload = inspect_to_json(&InspectSummary {
        source_kind: ModelSourceKind::TrainingCheckpoint,
        source_path: Path::new("/models/legacy"),
        model: InspectedModel::Checkpoint(&checkpoint),
        files: &files,
        incompatibilities: &["model_kind"],
    });

    assert_eq!(payload["compatible"], false);
    assert_eq!(
        payload["incompatibilities"],
        serde_json::json!(["model_kind"])
    );
}
