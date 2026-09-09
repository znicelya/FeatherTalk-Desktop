//! The JSON payload an inspected model returns.

use std::path::Path;

use feathertalk_export::{ModelPackageManifest, TensorSpec, TrainingMode as PackageTrainingMode};
use feathertalk_training::{TrainingCheckpointMetadata, TrainingMode as CheckpointTrainingMode};
use serde_json::{Value, json};

use crate::{InspectedFile, ModelSourceKind};

/// Which of the two layouts was read, with the manifest that came out of it.
#[derive(Debug)]
pub enum InspectedModel<'a> {
    Package(&'a ModelPackageManifest),
    Checkpoint(&'a TrainingCheckpointMetadata),
}

/// What an inspected model has to say for itself.
///
/// A struct rather than five positional arguments, for the reason
/// `train_result.rs` gives: most of these are strings, so a wrong order would
/// type-check and report the wrong model without a word.
#[derive(Debug)]
pub struct InspectSummary<'a> {
    pub source_kind: ModelSourceKind,
    pub source_path: &'a Path,
    pub model: InspectedModel<'a>,
    pub files: &'a [InspectedFile],
    /// The reason codes from `inspecting`; empty means usable by this build.
    pub incompatibilities: &'a [&'static str],
}

/// Shapes the payload the `completed` event of an inspect task carries.
///
/// Both arms are written out in full rather than merged from a common base: the
/// key set is the contract (design section 6), and spelling it twice is what makes
/// a missing key a diff instead of a runtime surprise. Where a layout has no
/// answer the value is `null`, never a zero -- a checkpoint reporting
/// `parameter_count: 0` would read as an empty model.
pub fn inspect_to_json(summary: &InspectSummary<'_>) -> Value {
    let compatible = summary.incompatibilities.is_empty();
    let source_path = path_text(summary.source_path);
    let files = files_json(summary.files);
    match summary.model {
        InspectedModel::Package(manifest) => json!({
            "source_kind": summary.source_kind.as_slug(),
            "source_path": source_path,
            "schema_version": manifest.schema_version,
            "model_kind": manifest.model_type.as_str(),
            "architecture_version": manifest.architecture_version.as_str(),
            // A package is a published artifact, not a resume point: it carries no
            // configuration digest to match a checkpoint against.
            "model_config_sha256": Value::Null,
            "parameter_count": manifest.tensors.total_elements,
            "tensor_count": manifest.tensors.tensor_count,
            "inputs": specs_json(&manifest.inputs),
            "outputs": specs_json(&manifest.outputs),
            "training_mode": package_mode_slug(manifest.training.mode),
            "epoch": Value::Null,
            "global_step": Value::Null,
            "created_at": manifest.created_at.as_str(),
            "minimum_app_version": manifest.minimum_app_version.as_str(),
            "files": files,
            "compatible": compatible,
            "incompatibilities": summary.incompatibilities,
        }),
        InspectedModel::Checkpoint(checkpoint) => json!({
            "source_kind": summary.source_kind.as_slug(),
            "source_path": source_path,
            "schema_version": checkpoint.manifest.schema_version,
            "model_kind": checkpoint.manifest.model_kind.as_str(),
            "architecture_version": checkpoint.manifest.architecture_version.as_str(),
            "model_config_sha256": checkpoint.manifest.model_config_sha256.as_str(),
            // Counting parameters means reading the record, which this command
            // does not do (design section 3).
            "parameter_count": Value::Null,
            "tensor_count": Value::Null,
            "inputs": Value::Array(Vec::new()),
            "outputs": Value::Array(Vec::new()),
            "training_mode": checkpoint_mode_slug(checkpoint.state.training_config.mode),
            "epoch": checkpoint.state.epoch,
            "global_step": checkpoint.state.global_step,
            // A checkpoint manifest records the toolchain that wrote it, not a
            // timestamp or an app floor.
            "created_at": Value::Null,
            "minimum_app_version": Value::Null,
            "files": files,
            "compatible": compatible,
            "incompatibilities": summary.incompatibilities,
        }),
    }
}

fn specs_json(specs: &[TensorSpec]) -> Value {
    Value::Array(
        specs
            .iter()
            .map(|spec| {
                json!({
                    "name": spec.name.as_str(),
                    // A dynamic axis is -1 in the manifest and stays -1 here.
                    "shape": spec.shape,
                    "dtype": spec.dtype.as_str(),
                })
            })
            .collect(),
    )
}

fn files_json(files: &[InspectedFile]) -> Value {
    Value::Array(
        files
            .iter()
            .map(|file| {
                json!({
                    "file_name": file.file_name.as_str(),
                    "bytes": file.bytes,
                    "sha256": file.sha256.as_str(),
                    "bytes_on_disk": file.bytes_on_disk,
                })
            })
            .collect(),
    )
}

/// Matched exhaustively rather than serialised, so a fifth mode is a compile error
/// here instead of a surprise string in the payload.
pub(crate) fn package_mode_slug(mode: PackageTrainingMode) -> &'static str {
    match mode {
        PackageTrainingMode::Inference => "inference",
        PackageTrainingMode::Baseline => "baseline",
        PackageTrainingMode::MouthRoi => "mouth_roi",
        PackageTrainingMode::MouthRoiTemporal => "mouth_roi_temporal",
    }
}

/// The checkpoint enum has no `Inference`: a checkpoint is by definition
/// mid-training.
fn checkpoint_mode_slug(mode: CheckpointTrainingMode) -> &'static str {
    match mode {
        CheckpointTrainingMode::Baseline => "baseline",
        CheckpointTrainingMode::MouthRoi => "mouth_roi",
        CheckpointTrainingMode::MouthRoiTemporal => "mouth_roi_temporal",
    }
}

fn path_text(path: &Path) -> String {
    path.display().to_string()
}
