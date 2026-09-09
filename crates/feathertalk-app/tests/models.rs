//! Model tool requests and local admission without a desktop window.

use std::fs;
use std::path::{Path, PathBuf};

use feathertalk_app::facts::FactValue;
use feathertalk_app::models::{ModelForm, ModelKind, ModelOperation, result_facts};
use feathertalk_domain::{
    ExportModelPackageParams, ExportOnnxParams, ImportLegacyModelParams, InspectModelParams,
    LegacyModelKind, MigrateLegacyFeaturesParams, OnnxExportKind, Request, TaskKind,
};
use serde_json::json;

fn form(operation: ModelOperation, source: &Path, destination: Option<&Path>) -> ModelForm {
    let mut form = ModelForm::default();
    form.set_operation(operation);
    form.select_source(source.to_path_buf());
    if let Some(destination) = destination {
        form.select_destination(destination.to_path_buf());
    }
    form
}

#[test]
fn inspect_only_submits_its_source_directory() {
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("checkpoint");
    let form = form(ModelOperation::Inspect, &source, None);
    assert_eq!(
        form.request().unwrap(),
        Request::InspectModel(InspectModelParams { source })
    );
}

#[test]
fn legacy_import_maps_both_supported_kinds_to_the_worker() {
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("legacy.pth.tar");
    let destination = dir.path().join("package");
    for (choice, expected) in [
        (ModelKind::FeatherHubert, LegacyModelKind::FeatherHubert),
        (ModelKind::OriginalUnet, LegacyModelKind::OriginalUnet),
    ] {
        let mut form = form(ModelOperation::ImportLegacy, &source, Some(&destination));
        form.set_kind(choice);
        assert_eq!(
            form.request().unwrap(),
            Request::ImportLegacyModel(ImportLegacyModelParams {
                source: source.clone(),
                kind: expected,
                destination: destination.clone(),
            })
        );
    }
}

#[test]
fn legacy_import_cannot_submit_a_kind_rejected_by_the_worker() {
    let dir = tempfile::tempdir().unwrap();
    let mut form = form(
        ModelOperation::ImportLegacy,
        &dir.path().join("legacy.pth"),
        Some(&dir.path().join("package")),
    );
    form.set_kind(ModelKind::MobileOneUnet);
    assert_eq!(
        form.request().unwrap_err().key,
        "models.error.unsupported_kind"
    );
}

#[test]
fn package_export_does_not_invent_a_model_kind_parameter() {
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("checkpoint");
    let destination = dir.path().join("package");
    let form = form(ModelOperation::ExportPackage, &source, Some(&destination));
    assert_eq!(
        form.request().unwrap(),
        Request::ExportModelPackage(ExportModelPackageParams {
            source,
            destination
        })
    );
}

#[test]
fn onnx_export_maps_all_three_kinds_and_keeps_the_file_path() {
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("package");
    let destination = dir.path().join("face.onnx");
    for (choice, expected) in [
        (ModelKind::FeatherHubert, OnnxExportKind::FeatherHubert),
        (ModelKind::OriginalUnet, OnnxExportKind::OriginalUnet),
        (ModelKind::MobileOneUnet, OnnxExportKind::MobileOneUnet),
    ] {
        let mut form = form(ModelOperation::ExportOnnx, &source, Some(&destination));
        form.set_kind(choice);
        assert_eq!(
            form.request().unwrap(),
            Request::ExportOnnx(ExportOnnxParams {
                source: source.clone(),
                kind: expected,
                destination: destination.clone(),
            })
        );
    }
}

#[test]
fn feature_migration_keeps_the_numpy_source_and_output_file() {
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("features.npy");
    let destination = dir.path().join("feather_hubert.f32");
    let form = form(ModelOperation::MigrateFeatures, &source, Some(&destination));
    assert_eq!(
        form.request().unwrap(),
        Request::MigrateLegacyFeatures(MigrateLegacyFeaturesParams {
            source,
            destination
        })
    );
}

#[test]
fn every_operation_requires_a_source() {
    for operation in ModelOperation::ALL {
        let mut form = ModelForm::default();
        form.set_operation(operation);
        assert_eq!(form.request().unwrap_err().key, "models.error.no_source");
        form.select_source(PathBuf::new());
        assert_eq!(form.request().unwrap_err().key, "models.error.no_source");
    }
}

#[test]
fn every_write_operation_requires_a_destination() {
    let dir = tempfile::tempdir().unwrap();
    for (operation, source) in [
        (ModelOperation::ImportLegacy, "legacy.pth"),
        (ModelOperation::ExportPackage, "checkpoint"),
        (ModelOperation::ExportOnnx, "package"),
        (ModelOperation::MigrateFeatures, "features.npy"),
    ] {
        let mut form = form(operation, &dir.path().join(source), None);
        assert_eq!(
            form.request().unwrap_err().key,
            "models.error.no_destination"
        );
        form.select_destination(PathBuf::new());
        assert_eq!(
            form.request().unwrap_err().key,
            "models.error.no_destination"
        );
    }
}

#[test]
fn relative_paths_are_refused_before_the_worker_starts() {
    let dir = tempfile::tempdir().unwrap();
    let mut form = form(ModelOperation::ExportOnnx, Path::new("package"), None);
    assert_eq!(
        form.request().unwrap_err().key,
        "models.error.source_absolute"
    );
    form.select_source(dir.path().join("package"));
    form.select_destination(PathBuf::from("model.onnx"));
    assert_eq!(
        form.request().unwrap_err().key,
        "models.error.destination_absolute"
    );
}

#[test]
fn write_operations_never_target_their_own_source() {
    let dir = tempfile::tempdir().unwrap();
    for (operation, source) in [
        (ModelOperation::ImportLegacy, "legacy.pth"),
        (ModelOperation::ExportPackage, "checkpoint"),
        (ModelOperation::ExportOnnx, "package"),
        (ModelOperation::MigrateFeatures, "features.npy"),
    ] {
        let source = dir.path().join(source);
        let form = form(operation, &source, Some(&source));
        assert_eq!(form.request().unwrap_err().key, "models.error.same_path");
    }
}

#[test]
fn equivalent_paths_cannot_bypass_source_protection() {
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("package");
    let destination = dir.path().join("temporary").join("..").join("package");
    let form = form(ModelOperation::ExportOnnx, &source, Some(&destination));
    assert_eq!(form.request().unwrap_err().key, "models.error.same_path");
}

#[test]
fn exporting_inside_the_source_directory_cannot_invalidate_its_manifest() {
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("source");
    for (operation, filename) in [
        (ModelOperation::ExportPackage, "nested-package"),
        (ModelOperation::ExportOnnx, "model.onnx"),
    ] {
        let form = form(operation, &source, Some(&source.join(filename)));
        assert_eq!(
            form.request().unwrap_err().key,
            "models.error.destination_inside_source"
        );
    }
}

#[cfg(windows)]
#[test]
fn windows_case_differences_cannot_bypass_source_protection() {
    let source = Path::new("C:/Models/Package");
    let destination = Path::new("c:/models/package");
    let form = form(ModelOperation::ExportOnnx, source, Some(destination));
    assert_eq!(form.request().unwrap_err().key, "models.error.same_path");
}

#[test]
fn legacy_inputs_use_the_extensions_the_worker_accepts() {
    let dir = tempfile::tempdir().unwrap();
    for (operation, source, key) in [
        (
            ModelOperation::ImportLegacy,
            "legacy.safetensors",
            "models.error.legacy_extension",
        ),
        (
            ModelOperation::MigrateFeatures,
            "features.NPY",
            "models.error.features_extension",
        ),
    ] {
        let form = form(
            operation,
            &dir.path().join(source),
            Some(&dir.path().join("output")),
        );
        assert_eq!(form.request().unwrap_err().key, key);
    }
}

#[test]
fn switching_tools_clears_paths_and_irrelevant_model_kind() {
    let dir = tempfile::tempdir().unwrap();
    let mut form = form(
        ModelOperation::ExportOnnx,
        &dir.path().join("package"),
        Some(&dir.path().join("model.onnx")),
    );
    form.set_kind(ModelKind::MobileOneUnet);
    assert!(form.set_operation(ModelOperation::ImportLegacy));
    assert!(form.source().is_none());
    assert!(form.destination().is_none());
    assert_eq!(form.kind(), ModelKind::FeatherHubert);
    assert_eq!(form.request().unwrap_err().key, "models.error.no_source");
}

#[test]
fn reselecting_the_current_tool_keeps_the_form() {
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("package");
    let mut form = form(ModelOperation::Inspect, &source, None);
    assert!(!form.set_operation(ModelOperation::Inspect));
    assert_eq!(form.source(), Some(source.as_path()));
}

#[test]
fn cancelling_a_native_dialog_retains_the_existing_paths() {
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("package");
    let destination = dir.path().join("model.onnx");
    let mut form = form(ModelOperation::ExportOnnx, &source, Some(&destination));
    assert!(!form.apply_source_pick(ModelOperation::ExportOnnx, None));
    assert!(!form.apply_destination_pick(ModelOperation::ExportOnnx, None));
    assert_eq!(form.source(), Some(source.as_path()));
    assert_eq!(form.destination(), Some(destination.as_path()));
}

#[test]
fn a_dialog_for_a_previous_tool_cannot_fill_the_new_tool() {
    let dir = tempfile::tempdir().unwrap();
    let mut form = ModelForm::default();
    form.set_operation(ModelOperation::ImportLegacy);
    assert!(!form.apply_source_pick(ModelOperation::Inspect, Some(dir.path().join("package"))));
    assert!(!form.apply_destination_pick(ModelOperation::Inspect, Some(dir.path().join("output"))));
    assert!(form.source().is_none());
    assert!(form.destination().is_none());
}

#[test]
fn native_dialog_modes_match_worker_file_and_directory_contracts() {
    for (operation, source_file, destination_directory) in [
        (ModelOperation::Inspect, false, false),
        (ModelOperation::ImportLegacy, true, true),
        (ModelOperation::ExportPackage, false, true),
        (ModelOperation::ExportOnnx, false, false),
        (ModelOperation::MigrateFeatures, true, false),
    ] {
        assert_eq!(operation.source_is_file(), source_file);
        assert_eq!(operation.destination_is_directory(), destination_directory);
    }
}

#[test]
fn selecting_a_package_parent_names_a_new_child_without_creating_it() {
    let dir = tempfile::tempdir().unwrap();
    let mut form = form(
        ModelOperation::ExportPackage,
        &dir.path().join("checkpoint-42"),
        None,
    );
    form.select_destination_parent(dir.path());
    let destination = dir.path().join("checkpoint-42-package");
    assert_eq!(form.destination(), Some(destination.as_path()));
    assert!(!destination.exists());
}

#[test]
fn local_preflight_reports_the_wrong_source_type() {
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("not-a-directory");
    fs::write(&source, b"weights").unwrap();
    let form = form(ModelOperation::Inspect, &source, None);
    assert_eq!(
        form.preflight().unwrap_err().key,
        "models.error.source_directory"
    );

    let source = dir.path().join("features.npy");
    fs::create_dir(&source).unwrap();
    let form = crate::form(
        ModelOperation::MigrateFeatures,
        &source,
        Some(&dir.path().join("features.f32")),
    );
    assert_eq!(
        form.preflight().unwrap_err().key,
        "models.error.source_file"
    );
}

#[test]
fn local_preflight_rejects_an_existing_destination_without_modifying_it() {
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("package");
    fs::create_dir(&source).unwrap();
    let destination = dir.path().join("model.onnx");
    fs::write(&destination, b"keep this model").unwrap();
    let form = form(ModelOperation::ExportOnnx, &source, Some(&destination));
    assert_eq!(
        form.preflight().unwrap_err().key,
        "models.error.destination_exists"
    );
    assert_eq!(fs::read(destination).unwrap(), b"keep this model");
}

#[test]
fn local_preflight_requires_an_existing_output_parent() {
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("package");
    fs::create_dir(&source).unwrap();
    let form = form(
        ModelOperation::ExportOnnx,
        &source,
        Some(&dir.path().join("missing").join("model.onnx")),
    );
    assert_eq!(
        form.preflight().unwrap_err().key,
        "models.error.destination_parent"
    );
}

#[test]
fn local_preflight_leaves_model_format_verification_to_the_worker() {
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("package");
    fs::create_dir(&source).unwrap();
    let form = form(
        ModelOperation::ExportOnnx,
        &source,
        Some(&dir.path().join("model.onnx")),
    );
    assert!(matches!(form.preflight().unwrap(), Request::ExportOnnx(_)));
}

#[test]
fn checkpoint_results_never_turn_unknown_parameter_counts_into_zeroes() {
    let facts = result_facts(
        TaskKind::InspectModel,
        &json!({
            "source_kind": "training_checkpoint", "source_path": "checkpoint-42",
            "model_kind": "original_unet", "compatible": true,
            "parameter_count": null, "tensor_count": null, "epoch": 12, "global_step": 2400
        }),
    );
    assert!(
        facts.iter().any(|fact| fact.label == "models.result.epoch"
            && fact.value == FactValue::Text("12".into()))
    );
    assert!(!facts.iter().any(|fact| matches!(
        fact.label,
        "models.result.parameters" | "models.result.tensors"
    )));
}

#[test]
fn inspection_reports_incompatibility_even_when_the_task_completed() {
    let facts = result_facts(
        TaskKind::InspectModel,
        &json!({
            "model_kind": "original_unet", "compatible": false, "incompatibilities": ["file_size"]
        }),
    );
    assert!(
        facts
            .iter()
            .any(|fact| fact.label == "models.result.compatibility"
                && fact.value == FactValue::Key("models.result.incompatible"))
    );
    assert!(
        facts
            .iter()
            .any(|fact| fact.value == FactValue::Key("models.reason.file_size"))
    );
}

#[test]
fn migration_summary_uses_only_the_returned_dimensions_and_byte_count() {
    let facts = result_facts(
        TaskKind::MigrateLegacyFeatures,
        &json!({
            "destination": "features.f32", "tokens": 84, "dims": 1024, "bytes": 344080
        }),
    );
    for (label, value) in [
        ("models.result.tokens", "84"),
        ("models.result.dimensions", "1024"),
        ("models.result.bytes", "344080"),
    ] {
        assert!(
            facts
                .iter()
                .any(|fact| fact.label == label && fact.value == FactValue::Text(value.into()))
        );
    }
}
