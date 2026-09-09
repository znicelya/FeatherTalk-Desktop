//! The `inspect_model` command: what a model directory says about itself.

use feathertalk_domain::{InspectModelParams, TaskStage};
use feathertalk_export::read_package_manifest;
use feathertalk_media::CancellationToken;
use feathertalk_training::read_training_checkpoint;

use crate::{
    CommandOutcome, InspectSummary, InspectedModel, ModelSourceKind, WorkerConfig,
    checkpoint_files, checkpoint_incompatibilities,
    error_map::{package_task_error, training_task_error},
    inspect_result::inspect_to_json,
    inspecting::model_source_kind,
    package_files, package_incompatibilities,
};

/// Reads the manifests of a model directory and reports what is in it.
///
/// No reporter and no progress events: three manifests and a handful of
/// `symlink_metadata` calls have no phase a client could act on (design section
/// 7). The token is checked twice all the same -- before the readers and before
/// the payload -- so a cancel that arrives during a slow directory walk still
/// lands.
pub fn execute_inspect_model(
    params: &InspectModelParams,
    config: &WorkerConfig,
    token: &CancellationToken,
) -> CommandOutcome {
    if token.is_cancelled() {
        return CommandOutcome::Cancelled;
    }
    let source = params.source.as_path();
    let kind = match model_source_kind(source) {
        Ok(kind) => kind,
        Err(error) => return CommandOutcome::Failed(error),
    };
    let payload = match kind {
        ModelSourceKind::ModelPackage => {
            let manifest = match read_package_manifest(source) {
                Ok(manifest) => manifest,
                Err(error) => return CommandOutcome::Failed(package_task_error(&error)),
            };
            let files = package_files(source, &manifest);
            let reasons = package_incompatibilities(&manifest, &files, config.worker_version());
            if token.is_cancelled() {
                return CommandOutcome::Cancelled;
            }
            inspect_to_json(&InspectSummary {
                source_kind: kind,
                source_path: source,
                model: InspectedModel::Package(&manifest),
                files: &files,
                incompatibilities: &reasons,
            })
        }
        ModelSourceKind::TrainingCheckpoint => {
            let checkpoint = match read_training_checkpoint(source) {
                Ok(checkpoint) => checkpoint,
                Err(error) => {
                    return CommandOutcome::Failed(training_task_error(
                        &error,
                        TaskStage::Preparing,
                    ));
                }
            };
            let files = checkpoint_files(source, &checkpoint.manifest);
            let reasons = checkpoint_incompatibilities(&checkpoint.manifest, &files);
            if token.is_cancelled() {
                return CommandOutcome::Cancelled;
            }
            inspect_to_json(&InspectSummary {
                source_kind: kind,
                source_path: source,
                model: InspectedModel::Checkpoint(&checkpoint),
                files: &files,
                incompatibilities: &reasons,
            })
        }
    };
    CommandOutcome::Completed(Some(payload))
}
