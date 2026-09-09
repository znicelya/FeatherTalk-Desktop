mod commands;
mod error;
mod evaluate;
mod extraction;
mod landmark;
mod model;
mod observe;
mod process;
mod publish;

pub use commands::{CommandSpec, frame_command};
pub use error::PipelineError;
pub use evaluate::{
    AcceptedFrame, BLUR_VARIANCE_THRESHOLD, DecodedFrame, FACE_CONFIDENCE_THRESHOLD, FaceDetector,
    FrameDecoder, FrameEvaluation, LandmarkPredictor, MIN_BBOX_INTERSECTION_RATIO,
    NMS_IOU_THRESHOLD, evaluate_frames_observed, evaluate_frames_with_models,
};
pub use extraction::{
    ExtractedFrame, FRAME_CHUNK, FrameBatch, extract_frames, extract_frames_observed,
    extract_frames_with_runner,
};
pub use landmark::{LANDMARK_POINTS, MAX_LANDMARK_FILE_BYTES, read_landmark_file};
pub use model::{
    AnomalyCode, FaceDetection, FrameAnomaly, FramePipelineSpec, FrameQuality, QualityReport,
    RecoveryAction,
};
pub use observe::{NoObserver, PipelineObserver, PipelinePhase};
pub use process::{
    FrameExtractor, MAX_CAPTURE_BYTES, MAX_FRAME_BYTES, ProcessOutput, ProcessRunner,
    SystemProcessRunner,
};
pub use publish::{publish_frame_artifacts, read_quality_report};

pub fn run_frame_pipeline_with_runner<R, D, F, L>(
    spec: &FramePipelineSpec,
    extractor: &FrameExtractor,
    runner: &R,
    decoder: &D,
    detector: &F,
    predictor: &L,
) -> Result<QualityReport, PipelineError>
where
    R: ProcessRunner + ?Sized,
    D: FrameDecoder + ?Sized,
    F: FaceDetector + ?Sized,
    L: LandmarkPredictor + ?Sized,
{
    let mut batch = extract_frames_with_runner(spec, extractor, runner)?;
    let evaluation = evaluate_frames_with_models(&batch, decoder, detector, predictor)?;
    publish_frame_artifacts(spec, &mut batch, &evaluation)
}
