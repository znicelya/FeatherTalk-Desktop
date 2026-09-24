//! Routes model computation through the exact device selected at admission.

use std::panic::{AssertUnwindSafe, catch_unwind};

use burn::tensor::Device;
use feathertalk_domain::{AdapterInfo, Backend, Metrics, Progress, Request, TaskError, TaskStage};
use feathertalk_frame_pipeline::SystemProcessRunner as FrameProcessRunner;
use feathertalk_inference::{JpegFrameReader, SystemRawVideoSinkFactory};
use feathertalk_media::{CancellationToken, ProcessRunner};

use crate::{
    CommandOutcome, FeatureModel, FrameModels, GpuContext, GpuFailure, TaskReporter, WorkerConfig,
    execute_extract_features, execute_extract_frames, execute_render_on, execute_train_on,
    execution_name, package_task_error, pipeline_task_error,
};
use crate::{commands::unsupported, error_map::panic_task_error, reporter::TrackedReporter};

pub(crate) fn execute_compute<R: ProcessRunner + ?Sized>(
    request: &Request,
    config: &WorkerConfig,
    token: &CancellationToken,
    reporter: &dyn TaskReporter,
    runner: &R,
) -> CommandOutcome {
    let adapter = match config.compute_adapter() {
        Ok(adapter) => adapter,
        Err(reason) => {
            return CommandOutcome::Failed(
                GpuFailure::Unavailable(reason).task_error(TaskStage::Preparing),
            );
        }
    };
    match adapter.backend {
        Backend::Cpu => {
            let device = crate::cpu_device();
            with_device_metadata(
                execute_on::<R>(request, config, token, reporter, runner, &device, None),
                &adapter,
                execution_name(&device),
                None,
            )
        }
        Backend::Wgpu => {
            let context = match config.compute().open_wgpu(&adapter.id) {
                Ok(context) => context,
                Err(error) => {
                    return CommandOutcome::Failed(error.task_error(TaskStage::Preparing));
                }
            };
            let device = context.device.clone();
            execute_gpu::<R>(
                request,
                config,
                token,
                reporter,
                runner,
                &adapter,
                &device,
                GpuContext::Wgpu(context),
            )
        }
        #[cfg(any(target_os = "windows", target_os = "linux"))]
        Backend::Cuda => {
            let context = match config.compute().open_cuda(&adapter.id) {
                Ok(context) => context,
                Err(error) => {
                    return CommandOutcome::Failed(error.task_error(TaskStage::Preparing));
                }
            };
            let device = context.device.clone();
            execute_gpu::<R>(
                request,
                config,
                token,
                reporter,
                runner,
                &adapter,
                &device,
                GpuContext::Cuda(context),
            )
        }
        #[cfg(target_os = "linux")]
        Backend::Rocm => {
            let context = match config.compute().open_rocm(&adapter.id) {
                Ok(context) => context,
                Err(error) => {
                    return CommandOutcome::Failed(error.task_error(TaskStage::Preparing));
                }
            };
            let device = context.device.clone();
            execute_gpu::<R>(
                request,
                config,
                token,
                reporter,
                runner,
                &adapter,
                &device,
                GpuContext::Rocm(context),
            )
        }
        _ => CommandOutcome::Failed(
            GpuFailure::Unavailable(format!(
                "Backend {:?} cannot execute on this platform",
                adapter.backend
            ))
            .task_error(TaskStage::Preparing),
        ),
    }
}

#[allow(clippy::too_many_arguments)]
fn execute_gpu<R: ProcessRunner + ?Sized>(
    request: &Request,
    config: &WorkerConfig,
    token: &CancellationToken,
    reporter: &dyn TaskReporter,
    runner: &R,
    adapter: &AdapterInfo,
    device: &Device,
    context: GpuContext,
) -> CommandOutcome {
    let tracked = TrackedReporter::new(reporter);
    tracked.report(TaskStage::Preparing, None);
    if let Err(error) = context.check() {
        return CommandOutcome::Failed(error.task_error(tracked.stage()));
    }
    let guarded = ComputeReporter {
        inner: &tracked,
        context: &context,
    };
    let outcome = catch_unwind(AssertUnwindSafe(|| {
        execute_on::<R>(
            request,
            config,
            token,
            &guarded,
            runner,
            device,
            Some(&context),
        )
    }));
    // Successful commands checked before publishing. Do not replay a task on
    // another backend after it has modified training state or published data.
    if !matches!(&outcome, Ok(CommandOutcome::Completed(_)))
        && let Err(error) = context.check()
    {
        return CommandOutcome::Failed(error.task_error(tracked.stage()));
    }
    match outcome {
        Ok(outcome) => with_device_metadata(
            outcome,
            adapter,
            execution_name(device),
            Some(context.graphics_api()),
        ),
        Err(payload) => CommandOutcome::Failed(panic_task_error(payload.as_ref(), tracked.stage())),
    }
}

fn execute_on<R: ProcessRunner + ?Sized>(
    request: &Request,
    config: &WorkerConfig,
    token: &CancellationToken,
    reporter: &dyn TaskReporter,
    runner: &R,
    device: &Device,
    context: Option<&GpuContext>,
) -> CommandOutcome {
    if token.is_cancelled() {
        return CommandOutcome::Cancelled;
    }
    match request {
        Request::Train(params) => {
            let Some(training) = config.training() else {
                return CommandOutcome::Failed(unsupported(request.kind()));
            };
            execute_train_on(params, token, reporter, training, device)
        }
        Request::Render(params) => {
            let Some(media) = config.media() else {
                return CommandOutcome::Failed(unsupported(request.kind()));
            };
            execute_render_on(
                params,
                token,
                reporter,
                media,
                &JpegFrameReader::default(),
                &SystemRawVideoSinkFactory,
                device,
            )
        }
        Request::ExtractFrames(params) => {
            let Some(models) = config.models() else {
                return CommandOutcome::Failed(unsupported(request.kind()));
            };
            reporter.report(TaskStage::Preparing, None);
            let models = match FrameModels::load_checked(models, device.clone(), context.cloned()) {
                Ok(models) => models,
                Err(error) => return CommandOutcome::Failed(pipeline_task_error(&error)),
            };
            // The frame pipeline checks cancellation between chunks and bounds
            // each extractor process with its existing timeout.
            execute_extract_frames(
                params,
                config,
                token,
                reporter,
                runner,
                &FrameProcessRunner,
                models.decoder(),
                models.detector(),
                models.predictor(),
            )
        }
        Request::ExtractFeatures(params) => {
            let Some(features) = config.features() else {
                return CommandOutcome::Failed(unsupported(request.kind()));
            };
            reporter.report(TaskStage::Preparing, None);
            let model = match FeatureModel::load_on(features, device.clone()) {
                Ok(model) => model,
                Err(error) => return CommandOutcome::Failed(package_task_error(&error)),
            };
            let (mut encoder, model_sha256) = model.into_parts();
            execute_extract_features(params, token, reporter, &mut encoder, &model_sha256)
        }
        _ => CommandOutcome::Failed(unsupported(request.kind())),
    }
}

/// Combines progress observation with the final GPU check while keeping the
/// public CPU command helpers independent of a native graphics context.
struct ComputeReporter<'a> {
    inner: &'a TrackedReporter<'a>,
    context: &'a GpuContext,
}

impl TaskReporter for ComputeReporter<'_> {
    fn report(&self, stage: TaskStage, progress: Option<Progress>) {
        self.inner.report(stage, progress);
    }

    fn report_metrics(&self, stage: TaskStage, progress: Option<Progress>, metrics: Metrics) {
        self.inner.report_metrics(stage, progress, metrics);
    }

    fn before_publish(&self) -> Result<(), TaskError> {
        self.context
            .check()
            .map_err(|error| error.task_error(self.inner.stage()))?;
        self.inner.before_publish()
    }
}

fn with_device_metadata(
    mut outcome: CommandOutcome,
    adapter: &AdapterInfo,
    backend: &str,
    graphics_api: Option<&str>,
) -> CommandOutcome {
    if let CommandOutcome::Completed(Some(serde_json::Value::Object(result))) = &mut outcome {
        result.insert("backend".to_owned(), backend.into());
        result.insert("adapter".to_owned(), serde_json::json!(adapter));
        result.insert("graphics_api".to_owned(), serde_json::json!(graphics_api));
        result.insert("used_cpu_fallback".to_owned(), false.into());
    }
    outcome
}
