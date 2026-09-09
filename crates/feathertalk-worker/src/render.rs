//! The render loop and the progress it reports.

use std::{
    cell::RefCell,
    sync::atomic::{AtomicU64, Ordering},
};

use burn::{backend::Autodiff, module::AutodiffModule, tensor::Device};
use feathertalk_domain::{ErrorCode, Progress, RenderParams, TaskError, TaskStage};
use feathertalk_inference::{
    BgrFrame, CommandSpec, FrameReader, InferenceError, RawVideoSink, RawVideoSinkFactory,
    execute_offline_render,
};
use feathertalk_media::{CancellationToken, MediaToolchain};
use feathertalk_models::unet::TalkingHeadModel;
use feathertalk_project::validate_project_dir;
use feathertalk_training::{load_training_checkpoint_model, read_training_checkpoint};

use crate::admission::check_project_dir;
use crate::render_result::render_to_json_on;
use crate::{
    CommandOutcome, RenderBackend, RenderDevice, RenderJob, RenderSummary, RenderVariant,
    TaskReporter, WorkerBackend, check_max_output_frames, check_render_paths,
    checkpoint_descriptor, is_inference_cancellation, project_task_error, render_job,
    render_task_error, render_variant, training_task_error,
};

/// Wraps the caller's sink factory so the render reports one progress event per
/// written frame and stops at the next frame once the task is cancelled.
///
/// The sink borrows the reporter, which is why `RawVideoSinkFactory` no longer
/// requires `Send + Sync`: a `TaskReporter` is not `Sync`, and the render loop
/// runs on this thread anyway.
struct ObservedSinkFactory<'a, F: ?Sized> {
    inner: &'a F,
    reporter: &'a dyn TaskReporter,
    token: &'a CancellationToken,
    /// The frames the render will write, from the locked manifest and the cap.
    total: u64,
    /// The frames it has written so far. Interior mutability because `start` and
    /// `write_frame` only ever get a shared borrow of the factory.
    frames: AtomicU64,
    /// Kept typed while the sink's error unwinds inference's staging guard.
    health_failure: RefCell<Option<TaskError>>,
}

impl<F: ?Sized> ObservedSinkFactory<'_, F> {
    /// Counts one written frame and reports it.
    fn observe(&self) {
        let frame = self
            .frames
            .fetch_add(1, Ordering::Relaxed)
            .saturating_add(1);
        self.reporter.report(
            TaskStage::Rendering {
                frame,
                total: self.total,
            },
            Some(Progress {
                completed: frame.min(self.total),
                total: Some(self.total),
            }),
        );
    }

    /// The stage a failure happened at: `Preparing` while nothing has been
    /// written, the last reported frame once the loop is running.
    fn stage(&self) -> TaskStage {
        let frame = self.frames.load(Ordering::Relaxed);
        if frame == 0 {
            TaskStage::Preparing
        } else {
            TaskStage::Rendering {
                frame,
                total: self.total,
            }
        }
    }
}

struct ObservedSink<'a, F: ?Sized> {
    inner: Box<dyn RawVideoSink + 'a>,
    observer: &'a ObservedSinkFactory<'a, F>,
}

impl<F: RawVideoSinkFactory + ?Sized> RawVideoSinkFactory for ObservedSinkFactory<'_, F> {
    fn start(&self, command: &CommandSpec) -> Result<Box<dyn RawVideoSink + '_>, InferenceError> {
        Ok(Box::new(ObservedSink {
            inner: self.inner.start(command)?,
            observer: self,
        }))
    }
}

impl<F: RawVideoSinkFactory + ?Sized> RawVideoSink for ObservedSink<'_, F> {
    fn write_frame(&mut self, frame: &BgrFrame) -> Result<(), InferenceError> {
        // Checked before the write, so a cancelled task stops at a frame
        // boundary rather than half a frame into the encoder's stdin.
        if self.observer.token.is_cancelled() {
            return Err(InferenceError::Cancelled {
                operation: "render",
            });
        }
        self.inner.write_frame(frame)?;
        self.observer.observe();
        Ok(())
    }

    fn finish(self: Box<Self>) -> Result<(), InferenceError> {
        // Destructured rather than moved out field by field: `finish` consumes
        // the boxed inner sink, and `Self` is not `Copy`.
        let ObservedSink { inner, observer } = *self;
        inner.finish()?;
        if let Err(error) = observer.reporter.before_publish() {
            let message = error.detail.clone();
            *observer.health_failure.borrow_mut() = Some(error);
            return Err(InferenceError::SinkFinish { message });
        }
        Ok(())
    }
}

/// Renders every planned frame, reporting one progress event per frame.
///
/// Inference owns the loop, so the wrapper sink is the only place cancellation
/// can act: a failing `write_frame` is how a caller stops it, and inference's
/// own guard then removes the staging file (design section 6).
pub fn run_render<B, M, R, F>(
    job: &RenderJob,
    model: &M,
    device: &Device<B>,
    token: &CancellationToken,
    reporter: &dyn TaskReporter,
    frame_reader: &R,
    sink_factory: &F,
) -> CommandOutcome
where
    B: WorkerBackend,
    M: TalkingHeadModel<B>,
    R: FrameReader + ?Sized,
    F: RawVideoSinkFactory + ?Sized,
{
    // A task cancelled before the first frame never starts the encoder.
    if token.is_cancelled() {
        return CommandOutcome::Cancelled;
    }
    let observed = ObservedSinkFactory {
        inner: sink_factory,
        reporter,
        token,
        total: job.progress_total,
        frames: AtomicU64::new(0),
        health_failure: RefCell::new(None),
    };
    let rendered = execute_offline_render::<B, M, R, ObservedSinkFactory<'_, F>>(
        model,
        device,
        &job.request,
        frame_reader,
        &observed,
    );
    if let Some(error) = observed.health_failure.borrow_mut().take() {
        return CommandOutcome::Failed(error);
    }
    match rendered {
        Ok(result) => CommandOutcome::Completed(Some(render_to_json_on(
            &RenderSummary {
                result: &result,
                descriptor: &job.descriptor,
                checkpoint_dir: &job.checkpoint_dir,
                checkpoint_epoch: job.checkpoint_epoch,
                checkpoint_global_step: job.checkpoint_global_step,
                source_frame_count: job.source_frame_count,
                max_output_frames: job.max_output_frames,
            },
            B::EXECUTION_NAME,
        ))),
        // A cancelled render is not a failure: no error event, no output file.
        Err(error) if is_inference_cancellation(&error) => CommandOutcome::Cancelled,
        Err(error) => CommandOutcome::Failed(render_task_error(&error, observed.stage())),
    }
}

/// Renders a locked project with one of its own training checkpoints.
///
/// The order is deliberate (design section 10): the cheap rejections first, then
/// the project manifest, then the checkpoint's own manifest -- which is where the
/// model variant is written, so the template cannot be built any earlier.
pub fn execute_render<R, F>(
    params: &RenderParams,
    token: &CancellationToken,
    reporter: &dyn TaskReporter,
    toolchain: &MediaToolchain,
    frame_reader: &R,
    sink_factory: &F,
) -> CommandOutcome
where
    R: FrameReader + ?Sized,
    F: RawVideoSinkFactory + ?Sized,
{
    execute_render_on::<RenderBackend, R, F>(
        params,
        token,
        reporter,
        toolchain,
        frame_reader,
        sink_factory,
        &RenderDevice::default(),
    )
}

/// Loads backend-neutral checkpoint records directly onto the selected device.
pub fn execute_render_on<B, R, F>(
    params: &RenderParams,
    token: &CancellationToken,
    reporter: &dyn TaskReporter,
    toolchain: &MediaToolchain,
    frame_reader: &R,
    sink_factory: &F,
    device: &Device<B>,
) -> CommandOutcome
where
    B: WorkerBackend,
    R: FrameReader + ?Sized,
    F: RawVideoSinkFactory + ?Sized,
{
    // Reading a checkpoint is the most expensive thing this command does before
    // the first frame, so a task cancelled while it waited does not pay for it.
    if token.is_cancelled() {
        return CommandOutcome::Cancelled;
    }
    // Everything below reads something off the disk, so the stage goes out first.
    reporter.report(TaskStage::Preparing, None);
    if let Err(error) = check_project_dir(&params.project_dir) {
        return CommandOutcome::Failed(error);
    }
    if let Err(error) = check_render_paths(params) {
        return CommandOutcome::Failed(error);
    }
    if let Err(error) = check_max_output_frames(params.max_output_frames) {
        return CommandOutcome::Failed(error);
    }
    let project = match validate_project_dir(&params.project_dir) {
        Ok(project) => project,
        Err(error) => return CommandOutcome::Failed(project_task_error(&error)),
    };
    // The locked manifest owns the frame count. The feature file is read by
    // inference, but it never decides how many frames the client was promised.
    let frame_count = project.asset_package().manifest().frame_count;
    let metadata = match read_training_checkpoint(&params.checkpoint) {
        Ok(metadata) => metadata,
        Err(error) => {
            return CommandOutcome::Failed(training_task_error(&error, TaskStage::Preparing));
        }
    };
    let Some(variant) = render_variant(&metadata.manifest.model_kind) else {
        return CommandOutcome::Failed(TaskError::new(
            ErrorCode::ModelIncompatible,
            "检查点的模型类型不受支持",
            &format!("unsupported model_kind: {}", metadata.manifest.model_kind),
            TaskStage::Preparing,
        ));
    };
    let descriptor = match checkpoint_descriptor(&variant.configuration()) {
        Ok(descriptor) => descriptor,
        Err(error) => {
            return CommandOutcome::Failed(training_task_error(&error, TaskStage::Preparing));
        }
    };
    let job = match render_job(
        params,
        frame_count,
        toolchain.ffmpeg(),
        descriptor,
        metadata.state.epoch,
        metadata.state.global_step,
    ) {
        Ok(job) => job,
        Err(error) => return CommandOutcome::Failed(error),
    };

    // Read the autodiff record on this device, then discard the autodiff shell
    // with `valid`; checkpoints carry tensor values rather than device handles.
    match variant {
        RenderVariant::OriginalUnet(configuration) => {
            let template = configuration.init::<Autodiff<B>>(device);
            let restored = match load_training_checkpoint_model::<Autodiff<B>, _>(
                &params.checkpoint,
                &template,
                device,
                &job.descriptor,
            ) {
                Ok(restored) => restored,
                Err(error) => {
                    return CommandOutcome::Failed(training_task_error(
                        &error,
                        TaskStage::Preparing,
                    ));
                }
            };
            // Loading half a gigabyte of weights is the longest step before the
            // first frame, so the token is checked again on the far side of it.
            if token.is_cancelled() {
                return CommandOutcome::Cancelled;
            }
            let model = restored.model.valid();
            run_render(
                &job,
                &model,
                device,
                token,
                reporter,
                frame_reader,
                sink_factory,
            )
        }
        RenderVariant::MobileOneUnet(configuration) => {
            let template = configuration.init::<Autodiff<B>>(device);
            let restored = match load_training_checkpoint_model::<Autodiff<B>, _>(
                &params.checkpoint,
                &template,
                device,
                &job.descriptor,
            ) {
                Ok(restored) => restored,
                Err(error) => {
                    return CommandOutcome::Failed(training_task_error(
                        &error,
                        TaskStage::Preparing,
                    ));
                }
            };
            if token.is_cancelled() {
                return CommandOutcome::Cancelled;
            }
            // Inference fuses the multi-branch blocks; training needed them
            // separate, which is why the descriptor above describes the unfused
            // shape.
            let model = restored.model.valid().reparameterize();
            run_render(
                &job,
                &model,
                device,
                token,
                reporter,
                frame_reader,
                sink_factory,
            )
        }
    }
}
