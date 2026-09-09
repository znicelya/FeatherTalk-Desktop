use feathertalk_domain::{ErrorCode, Progress, Request, TaskError, TaskKind, TaskStage};
use feathertalk_export::read_package_manifest;
use feathertalk_media::{
    CancellableProcessRunner, CancellationToken, MediaError, MediaInput, NormalizationSpec,
    NormalizePhase, ProcessRunner, normalize_media_observed, probe_media_with_runner,
    validate_input,
};
use feathertalk_project::validate_project_dir;

use crate::{
    TaskReporter, WorkerConfig, execute_export_model_package, execute_export_onnx,
    execute_import_legacy_model, execute_inspect_model, execute_lock_asset_package,
    execute_migrate_legacy_features, export_task_error, is_media_cancellation,
    legacy_feature_task_error, legacy_task_error, media_task_error, normalize_to_json,
    onnx_task_error, package_task_error, probe_to_json, project_task_error,
};

/// How many progress steps `normalize_media` reports. Verification and the
/// commit are bounded and short, so they end the count rather than extend it.
const NORMALIZE_STEPS: u64 = 3;

#[derive(Debug)]
pub enum CommandOutcome {
    /// The command finished. `Some` carries the JSON object a `completed` event
    /// reports; `None` means the command has no result payload.
    Completed(Option<serde_json::Value>),
    Cancelled,
    Failed(TaskError),
}

pub fn execute(
    request: &Request,
    config: &WorkerConfig,
    token: &CancellationToken,
    reporter: &dyn TaskReporter,
) -> CommandOutcome {
    let runner = CancellableProcessRunner::new(token.clone());
    execute_with_runner(request, config, token, reporter, &runner)
}

pub fn execute_with_runner<R: ProcessRunner + ?Sized>(
    request: &Request,
    config: &WorkerConfig,
    token: &CancellationToken,
    reporter: &dyn TaskReporter,
    runner: &R,
) -> CommandOutcome {
    if token.is_cancelled() {
        return CommandOutcome::Cancelled;
    }
    match request {
        Request::ValidateProject(params) => match validate_project_dir(&params.project_dir) {
            // Project validation is filesystem-bound and has no interrupt hook,
            // so cancellation is honoured at this boundary: the work is thrown
            // away rather than reported as a completed task.
            Ok(_) if token.is_cancelled() => CommandOutcome::Cancelled,
            Ok(_) => CommandOutcome::Completed(None),
            Err(error) => CommandOutcome::Failed(project_task_error(&error)),
        },
        Request::ProbeMedia(params) => {
            let Some(toolchain) = config.media() else {
                // Unreachable through the runtime, which rejects `probe_media`
                // when no toolchain is configured. Kept so a direct caller
                // cannot get a panic instead of an error.
                return CommandOutcome::Failed(unsupported(request.kind()));
            };
            let input = match validate_input(&MediaInput {
                source: params.input.clone(),
            }) {
                Ok(input) => input,
                Err(error) => return media_failure(&error),
            };
            match probe_media_with_runner(&input, toolchain, runner) {
                Ok(probe) => CommandOutcome::Completed(Some(probe_to_json(&probe))),
                Err(error) => media_failure(&error),
            }
        }
        Request::NormalizeMedia(params) => {
            let Some(toolchain) = config.media() else {
                return CommandOutcome::Failed(unsupported(request.kind()));
            };
            let input = match validate_input(&MediaInput {
                source: params.input.clone(),
            }) {
                Ok(input) => input,
                Err(error) => return media_failure(&error),
            };
            // The targets are fixed by the asset contract, and
            // `validate_normalization` rejects anything else, so there is
            // nothing here for a caller to configure.
            let spec = NormalizationSpec {
                target_video_fps: 25,
                target_audio_sample_rate: 16_000,
                target_audio_channels: 1,
                output_dir: params.output_dir.clone(),
            };
            match normalize_media_observed(&input, &spec, toolchain, runner, &|phase| {
                report_phase(reporter, phase)
            }) {
                Ok(normalized) => CommandOutcome::Completed(Some(normalize_to_json(&normalized))),
                Err(error) => media_failure(&error),
            }
        }
        Request::Train(_)
        | Request::Render(_)
        | Request::ExtractFrames(_)
        | Request::ExtractFeatures(_) => {
            crate::compute_commands::execute_compute(request, config, token, reporter, runner)
        }
        Request::LockAssetPackage(params) => {
            let Some(features) = config.features() else {
                return CommandOutcome::Failed(unsupported(request.kind()));
            };
            // Only the package manifest is needed: the command runs no
            // inference, so mapping the safetensors weights into memory to
            // read one string would be pure waste. `read_package_manifest`
            // still runs `validate_package_directory` and
            // `manifest.validate()`, so a broken package is caught here.
            let manifest = match read_package_manifest(features.hubert_dir()) {
                Ok(manifest) => manifest,
                Err(error) => return CommandOutcome::Failed(package_task_error(&error)),
            };
            execute_lock_asset_package(params, token, reporter, &manifest.model.sha256)
        }
        // No toolchain guard: inspection reads manifests, so the handshake
        // announces it unconditionally and there is nothing to reject on.
        Request::InspectModel(params) => execute_inspect_model(params, config, token),
        Request::ImportLegacyModel(params) => {
            match execute_import_legacy_model(params, config, token, reporter) {
                Ok(payload) => CommandOutcome::Completed(Some(payload)),
                Err(error) if error.is_cancelled() => CommandOutcome::Cancelled,
                Err(error) => CommandOutcome::Failed(legacy_task_error(&error, error.stage())),
            }
        }
        Request::MigrateLegacyFeatures(params) => {
            match execute_migrate_legacy_features(params, token, reporter) {
                Ok(payload) => CommandOutcome::Completed(Some(payload)),
                Err(error) if error.is_cancelled() => CommandOutcome::Cancelled,
                Err(error) => {
                    CommandOutcome::Failed(legacy_feature_task_error(&error, error.stage()))
                }
            }
        }
        // No toolchain guard either: the export reads a checkpoint and writes a
        // package directory, so the handshake announces it unconditionally.
        Request::ExportModelPackage(params) => {
            match execute_export_model_package(params, config, token, reporter) {
                Ok(payload) => CommandOutcome::Completed(Some(payload)),
                Err(error) if error.is_cancelled() => CommandOutcome::Cancelled,
                Err(error) => CommandOutcome::Failed(export_task_error(&error, error.stage())),
            }
        }
        // Nor here: the graph is serialised in process, and the reference runtime
        // that validates it lives in `tools/onnx-validate` rather than in the
        // worker.
        Request::ExportOnnx(params) => match execute_export_onnx(params, token, reporter) {
            Ok(payload) => CommandOutcome::Completed(Some(payload)),
            Err(error) if error.is_cancelled() => CommandOutcome::Cancelled,
            Err(error) => CommandOutcome::Failed(onnx_task_error(&error, error.stage())),
        },
    }
}

/// Map a normalization phase onto the protocol stage that names it.
///
/// Protocol version 2 has no stage for media normalization, so the two passes
/// that dominate wall time borrow the stages that describe their output.
/// Verification and the commit report nothing: giving them a stage would mean
/// moving the label backwards to `preparing`, which reads as a bug.
fn report_phase(reporter: &dyn TaskReporter, phase: NormalizePhase) {
    let (stage, completed) = match phase {
        NormalizePhase::Probing => (TaskStage::Preparing, 1),
        NormalizePhase::NormalizingVideo => (TaskStage::ExtractingFrames, 2),
        NormalizePhase::NormalizingAudio => (TaskStage::ExtractingAudio, 3),
        NormalizePhase::Verifying | NormalizePhase::Committing => return,
    };
    reporter.report(
        stage,
        Some(Progress {
            completed,
            total: Some(NORMALIZE_STEPS),
        }),
    );
}

pub(crate) fn media_failure(error: &MediaError) -> CommandOutcome {
    if is_media_cancellation(error) {
        CommandOutcome::Cancelled
    } else {
        CommandOutcome::Failed(media_task_error(error))
    }
}

pub(crate) fn unsupported(kind: TaskKind) -> TaskError {
    TaskError::new(
        ErrorCode::WorkerCrashed,
        "当前 worker 不支持该命令",
        &format!(
            "command {} is not supported by this worker build",
            kind.as_slug()
        ),
        TaskStage::Preparing,
    )
}
