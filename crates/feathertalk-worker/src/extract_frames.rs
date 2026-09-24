use feathertalk_domain::{ExtractFramesParams, Progress, TaskKind, TaskStage};
use feathertalk_frame_pipeline::{
    FaceDetector, FrameDecoder, FrameExtractor, FramePipelineSpec, LandmarkPredictor,
    PipelineError, PipelineObserver, PipelinePhase, evaluate_frames_observed,
    extract_frames_observed, publish_frame_artifacts,
};
use feathertalk_media::{
    CancellationToken, MediaInput, MediaToolchain, probe_video_with_runner, validate_input,
};

use crate::{
    CommandOutcome, GpuFailure, TaskReporter, WorkerConfig,
    admission::{check_project_dir, invalid_request},
    commands::{media_failure, unsupported},
    is_pipeline_cancellation, pipeline_task_error, quality_task_error, quality_to_json,
};

/// The frame rate `normalize_media` fixes for a project video.
const TARGET_FRAME_RATE: (u32, u32) = (25, 1);

/// Extract every frame of a normalised video into a project's asset directory.
///
/// The three models arrive as trait objects so a caller can drive the command
/// without loading weights; `FrameModels` in this crate supplies the real ones.
#[allow(clippy::too_many_arguments)]
pub fn execute_extract_frames<M, F>(
    params: &ExtractFramesParams,
    config: &WorkerConfig,
    token: &CancellationToken,
    reporter: &dyn TaskReporter,
    media_runner: &M,
    frame_runner: &F,
    decoder: &dyn FrameDecoder,
    detector: &dyn FaceDetector,
    predictor: &dyn LandmarkPredictor,
) -> CommandOutcome
where
    M: feathertalk_media::ProcessRunner + ?Sized,
    F: feathertalk_frame_pipeline::ProcessRunner + ?Sized,
{
    let cuda_device = match config.cuda_device_index() {
        Ok(device) => device,
        Err(reason) => {
            return CommandOutcome::Failed(
                GpuFailure::Unavailable(reason).task_error(TaskStage::Preparing),
            );
        }
    };
    let Some(media) = config.media() else {
        return CommandOutcome::Failed(unsupported(TaskKind::ExtractFrames));
    };
    // One stage before the probe: probing and model loading take seconds, and
    // the CLI would otherwise print nothing until the first chunk lands.
    reporter.report(TaskStage::Preparing, None);
    let spec = match frame_spec(params, media, media_runner) {
        Ok(spec) => spec,
        Err(outcome) => return outcome,
    };
    let extractor = match FrameExtractor::new(media.ffmpeg().to_owned(), media.timeout()) {
        Ok(extractor) => extractor.with_cuda_device(cuda_device),
        Err(error) => return pipeline_failure(&error),
    };
    let observer = FrameProgress { reporter, token };
    let mut batch = match extract_frames_observed(&spec, &extractor, frame_runner, &observer) {
        Ok(batch) => batch,
        Err(error) => return pipeline_failure(&error),
    };
    let evaluation = match evaluate_frames_observed(&batch, decoder, detector, predictor, &observer)
    {
        Ok(evaluation) => evaluation,
        Err(error) => return pipeline_failure(&error),
    };
    if !evaluation.is_success() {
        // Staging is still armed, so the batch destructor removes the frames
        // this run wrote and the project keeps whatever it had before.
        return CommandOutcome::Failed(quality_task_error(evaluation.anomalies()));
    }
    if let Err(error) = reporter.before_publish() {
        return CommandOutcome::Failed(error);
    }
    match publish_frame_artifacts(&spec, &mut batch, &evaluation) {
        Ok(report) => CommandOutcome::Completed(Some(quality_to_json(&spec, &report))),
        Err(error) => pipeline_failure(&error),
    }
}

/// Bridges the pipeline's observer onto the worker's reporter and token.
struct FrameProgress<'a> {
    reporter: &'a dyn TaskReporter,
    token: &'a CancellationToken,
}

impl PipelineObserver for FrameProgress<'_> {
    fn phase(&self, phase: PipelinePhase) {
        let (stage, completed, total) = match phase {
            PipelinePhase::Extracting { completed, total } => {
                (TaskStage::ExtractingFrames, completed, total)
            }
            PipelinePhase::Evaluating { completed, total } => {
                (TaskStage::DetectingFaces, completed, total)
            }
        };
        self.reporter.report(
            stage,
            Some(Progress {
                completed,
                total: Some(total),
            }),
        );
    }

    fn is_cancelled(&self) -> bool {
        self.token.is_cancelled()
    }
}

/// Admission plus the probe: everything that has to hold before the first
/// ffmpeg invocation.
fn frame_spec<M>(
    params: &ExtractFramesParams,
    media: &MediaToolchain,
    runner: &M,
) -> Result<FramePipelineSpec, CommandOutcome>
where
    M: feathertalk_media::ProcessRunner + ?Sized,
{
    check_project_dir(&params.project_dir).map_err(CommandOutcome::Failed)?;
    if !params.video.is_absolute() {
        return Err(CommandOutcome::Failed(invalid_request(
            "输入文件必须是绝对路径",
            format!("video {} is not absolute", params.video.display()),
        )));
    }
    let input = validate_input(&MediaInput {
        source: params.video.clone(),
    })
    .map_err(|error| media_failure(&error))?;
    // A video-only probe, because that is what `normalize_media` produces: the
    // audio lives in `audio_16k_mono.wav`, and the audio/video probe would
    // refuse the very artifact this command exists to read.
    let probe =
        probe_video_with_runner(&input, media, runner).map_err(|error| media_failure(&error))?;
    let Some(video) = probe.video() else {
        return Err(CommandOutcome::Failed(invalid_request(
            "输入文件不含视频流",
            format!("{} has no video stream", params.video.display()),
        )));
    };
    let rate = video.frame_rate();
    if (rate.numerator(), rate.denominator()) != TARGET_FRAME_RATE {
        return Err(CommandOutcome::Failed(invalid_request(
            "抽帧要求 25fps 的归一化视频",
            format!(
                "video frame rate is {}/{}, expected {}/{}",
                rate.numerator(),
                rate.denominator(),
                TARGET_FRAME_RATE.0,
                TARGET_FRAME_RATE.1
            ),
        )));
    }
    FramePipelineSpec::new(
        params.video.clone(),
        params.project_dir.join("assets"),
        video.frame_count(),
        video.width(),
        video.height(),
    )
    .map_err(|error| CommandOutcome::Failed(pipeline_task_error(&error)))
}

/// Cancellation is not a failure: the pipeline reports it as an error and the
/// runtime needs it back as `Cancelled`.
fn pipeline_failure(error: &PipelineError) -> CommandOutcome {
    if is_pipeline_cancellation(error) {
        CommandOutcome::Cancelled
    } else {
        CommandOutcome::Failed(pipeline_task_error(error))
    }
}
