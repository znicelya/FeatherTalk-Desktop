use std::{
    fs,
    path::{Path, PathBuf},
};

use feathertalk_domain::{ErrorCode, TaskStage};
use feathertalk_export::read_package_manifest;
use feathertalk_training::{CheckpointDescriptor, read_training_checkpoint};
use feathertalk_worker::{
    ModelSourceKind, checkpoint_descriptor, checkpoint_files, checkpoint_incompatibilities,
    model_source_kind, package_files, package_incompatibilities, render_variant,
};

#[path = "support/mod.rs"]
mod support;

use support::{published_package, write_checkpoint};

/// Classification never opens a file, so every fixture is a one-byte placeholder.
fn touch(path: &Path) {
    fs::write(path, b"x").expect("the placeholder is written");
}

fn directory(root: &Path, name: &str) -> PathBuf {
    let dir = root.join(name);
    fs::create_dir_all(&dir).expect("the directory is created");
    dir
}

#[test]
fn a_directory_with_a_safetensors_model_is_a_package() {
    let root = tempfile::tempdir().expect("the temporary root is created");
    let dir = directory(root.path(), "package");
    touch(&dir.join("model.safetensors"));
    touch(&dir.join("manifest.json"));
    assert_eq!(
        model_source_kind(&dir).expect("the layout is recognized"),
        ModelSourceKind::ModelPackage
    );
}

#[test]
fn a_directory_with_a_binary_model_is_a_checkpoint() {
    let root = tempfile::tempdir().expect("the temporary root is created");
    let dir = directory(root.path(), "checkpoint");
    touch(&dir.join("model.bin"));
    touch(&dir.join("manifest.json"));
    assert_eq!(
        model_source_kind(&dir).expect("the layout is recognized"),
        ModelSourceKind::TrainingCheckpoint
    );
}

#[test]
fn a_directory_holding_both_model_files_is_refused() {
    let root = tempfile::tempdir().expect("the temporary root is created");
    let dir = directory(root.path(), "both");
    touch(&dir.join("model.safetensors"));
    touch(&dir.join("model.bin"));
    let error = model_source_kind(&dir).expect_err("two layouts at once is no layout");
    assert_eq!(error.code, ErrorCode::ModelIncompatible);
    assert_eq!(error.summary, "无法识别的模型目录");
    assert_eq!(error.stage, TaskStage::Preparing);
}

#[test]
fn a_directory_holding_neither_model_file_is_refused() {
    let root = tempfile::tempdir().expect("the temporary root is created");
    let dir = directory(root.path(), "empty");
    touch(&dir.join("manifest.json"));
    let error = model_source_kind(&dir).expect_err("a manifest alone is no layout");
    assert_eq!(error.code, ErrorCode::ModelIncompatible);
}

#[test]
fn a_relative_source_is_refused_before_any_probe() {
    let error = model_source_kind(Path::new("model")).expect_err("a relative source is refused");
    assert_eq!(error.code, ErrorCode::MediaInvalid);
    assert_eq!(error.summary, "模型目录必须是绝对路径");
}

#[test]
fn a_file_instead_of_a_directory_is_refused() {
    let root = tempfile::tempdir().expect("the temporary root is created");
    let file = root.path().join("model.safetensors");
    touch(&file);
    let error = model_source_kind(&file).expect_err("a file is not a model directory");
    assert_eq!(error.code, ErrorCode::MediaInvalid);
    assert_eq!(error.summary, "模型目录不可用");
}

#[test]
fn a_missing_source_is_refused() {
    let root = tempfile::tempdir().expect("the temporary root is created");
    let error =
        model_source_kind(&root.path().join("absent")).expect_err("a missing source is refused");
    assert_eq!(error.code, ErrorCode::MediaInvalid);
    assert_eq!(error.summary, "模型目录不可用");
}

#[test]
fn both_kinds_have_a_stable_slug() {
    assert_eq!(ModelSourceKind::ModelPackage.as_slug(), "model_package");
    assert_eq!(
        ModelSourceKind::TrainingCheckpoint.as_slug(),
        "training_checkpoint"
    );
}

#[test]
fn a_package_this_build_satisfies_is_compatible() {
    let root = tempfile::tempdir().expect("the temporary root is created");
    let dir = published_package(root.path(), "package", "0.1.0");
    let manifest = read_package_manifest(&dir).expect("the package is readable");
    let files = package_files(&dir, &manifest);
    assert_eq!(files.len(), 2);
    assert_eq!(files[0].file_name, "model.safetensors");
    assert_eq!(files[1].file_name, "LICENSES.json");
    assert!(files.iter().all(|file| file.agrees()));
    assert!(package_incompatibilities(&manifest, &files, "0.1.0").is_empty());
}

#[test]
fn a_package_that_wants_a_newer_app_is_incompatible() {
    let root = tempfile::tempdir().expect("the temporary root is created");
    let dir = published_package(root.path(), "package", "9.9.9");
    let manifest = read_package_manifest(&dir).expect("the package is readable");
    let files = package_files(&dir, &manifest);
    assert_eq!(
        package_incompatibilities(&manifest, &files, "0.1.0"),
        vec!["minimum_app_version"]
    );
}

#[test]
fn an_unparseable_worker_version_is_incompatible() {
    let root = tempfile::tempdir().expect("the temporary root is created");
    let dir = published_package(root.path(), "package", "0.1.0");
    let manifest = read_package_manifest(&dir).expect("the package is readable");
    let files = package_files(&dir, &manifest);
    // Refusing to guess is the honest answer: a version this code cannot read is
    // not a version it can claim to satisfy.
    assert_eq!(
        package_incompatibilities(&manifest, &files, "0.1"),
        vec!["minimum_app_version"]
    );
}

#[test]
fn a_truncated_model_file_is_reported_by_size() {
    let root = tempfile::tempdir().expect("the temporary root is created");
    let dir = published_package(root.path(), "package", "0.1.0");
    let manifest = read_package_manifest(&dir).expect("the package is readable");
    fs::write(dir.join("model.safetensors"), b"truncated").expect("the model is truncated");
    let files = package_files(&dir, &manifest);
    assert!(!files[0].agrees());
    assert_eq!(
        package_incompatibilities(&manifest, &files, "0.1.0"),
        vec!["file_size"]
    );
}

#[test]
fn a_checkpoint_this_build_can_rebuild_is_compatible() {
    let root = tempfile::tempdir().expect("the temporary root is created");
    let dir = root.path().join("checkpoint");
    let variant = render_variant("original_unet").expect("the kind is known");
    let descriptor =
        checkpoint_descriptor(&variant.configuration()).expect("the descriptor is computed");
    write_checkpoint(&dir, descriptor);
    let checkpoint = read_training_checkpoint(&dir).expect("the checkpoint is readable");
    let files = checkpoint_files(&dir, &checkpoint.manifest);
    assert_eq!(files.len(), 3);
    assert_eq!(files[0].file_name, "model.bin");
    assert_eq!(files[1].file_name, "optimizer.bin");
    assert_eq!(files[2].file_name, "training-state.json");
    assert!(checkpoint_incompatibilities(&checkpoint.manifest, &files).is_empty());
}

#[test]
fn a_checkpoint_of_an_unknown_kind_is_incompatible() {
    let root = tempfile::tempdir().expect("the temporary root is created");
    let dir = root.path().join("checkpoint");
    write_checkpoint(
        &dir,
        CheckpointDescriptor::new("legacy_unet", "v1", "0".repeat(64)),
    );
    let checkpoint = read_training_checkpoint(&dir).expect("the checkpoint is readable");
    let files = checkpoint_files(&dir, &checkpoint.manifest);
    // The kind decides which configuration to compare against, so an unknown kind
    // is the only reason reported: the rest cannot be computed.
    assert_eq!(
        checkpoint_incompatibilities(&checkpoint.manifest, &files),
        vec!["model_kind"]
    );
}

#[test]
fn a_checkpoint_of_another_architecture_is_incompatible() {
    let root = tempfile::tempdir().expect("the temporary root is created");
    let dir = root.path().join("checkpoint");
    let variant = render_variant("original_unet").expect("the kind is known");
    let mine = checkpoint_descriptor(&variant.configuration()).expect("the descriptor is computed");
    write_checkpoint(
        &dir,
        CheckpointDescriptor::new("original_unet", "unet-burn-v0", &mine.model_config_sha256),
    );
    let checkpoint = read_training_checkpoint(&dir).expect("the checkpoint is readable");
    let files = checkpoint_files(&dir, &checkpoint.manifest);
    assert_eq!(
        checkpoint_incompatibilities(&checkpoint.manifest, &files),
        vec!["architecture_version"]
    );
}

#[test]
fn a_checkpoint_of_another_configuration_is_incompatible() {
    let root = tempfile::tempdir().expect("the temporary root is created");
    let dir = root.path().join("checkpoint");
    let variant = render_variant("original_unet").expect("the kind is known");
    let mine = checkpoint_descriptor(&variant.configuration()).expect("the descriptor is computed");
    write_checkpoint(
        &dir,
        CheckpointDescriptor::new("original_unet", &mine.architecture_version, "0".repeat(64)),
    );
    let checkpoint = read_training_checkpoint(&dir).expect("the checkpoint is readable");
    let files = checkpoint_files(&dir, &checkpoint.manifest);
    assert_eq!(
        checkpoint_incompatibilities(&checkpoint.manifest, &files),
        vec!["model_config_sha256"]
    );
}
