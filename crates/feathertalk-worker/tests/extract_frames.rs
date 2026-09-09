#[path = "support/health.rs"]
mod health;

use std::{ffi::OsStr, fs, path::Path, sync::Mutex, time::Duration};

use feathertalk_domain::{ErrorCode, ExtractFramesParams, Progress, TaskError, TaskStage};
use feathertalk_frame_pipeline::{
    self as pipeline, DecodedFrame, FaceDetection, FaceDetector, FrameDecoder, LandmarkPredictor,
    PipelineError,
};
use feathertalk_media::{CancellationToken, CommandSpec, MediaError, ProcessOutput, ProcessRunner};
use feathertalk_pfld::{CropGeometry, PFLDLandmarks, decode_landmarks};
use feathertalk_worker::{
    CommandOutcome, NoReporter, TaskReporter, WorkerConfig, execute_extract_frames,
};
use tempfile::TempDir;

/// Answers every ffprobe call with the same report, so a test only has to say
/// how many frames the video has and what frame rate it claims.
struct MediaRunner {
    probe: Vec<u8>,
    calls: Mutex<usize>,
}

impl MediaRunner {
    fn new(frame_count: u64, frame_rate: &str) -> Self {
        // A normalized video carries no audio: `normalize_media` writes the
        // sound to `audio_16k_mono.wav` and verifies `video_25fps.mp4` as
        // video-only. The fixture has to have the same shape, or these tests
        // pass against a file the command can never be given.
        let probe = serde_json::json!({
            "format": { "format_name": "mov,mp4", "duration": "2.0" },
            "streams": [
                {
                    "codec_type": "video",
                    "codec_name": "h264",
                    "pix_fmt": "yuv420p",
                    "width": 640,
                    "height": 480,
                    "avg_frame_rate": frame_rate,
                    "nb_read_frames": frame_count.to_string(),
                    "duration": "2.0"
                }
            ]
        });
        Self {
            probe: probe.to_string().into_bytes(),
            calls: Mutex::new(0),
        }
    }

    fn call_count(&self) -> usize {
        *self.calls.lock().unwrap()
    }
}

impl ProcessRunner for MediaRunner {
    fn run(&self, _command: &CommandSpec, _timeout: Duration) -> Result<ProcessOutput, MediaError> {
        *self.calls.lock().unwrap() += 1;
        Ok(ProcessOutput::new(Some(0), self.probe.clone(), Vec::new()))
    }
}

/// Stands in for ffmpeg's `image2` muxer: the chunk command ends with a
/// `%06d.jpg` pattern, and `-start_number` plus `-frames:v` say which indices
/// that pattern expands to.
struct FfmpegRunner {
    chunks: Mutex<Vec<(u64, u64)>>,
}

impl FfmpegRunner {
    fn new() -> Self {
        Self {
            chunks: Mutex::new(Vec::new()),
        }
    }

    /// One `(first_index, count)` pair per invocation.
    fn chunks(&self) -> Vec<(u64, u64)> {
        self.chunks.lock().unwrap().clone()
    }
}

impl pipeline::ProcessRunner for FfmpegRunner {
    fn run(
        &self,
        command: &pipeline::CommandSpec,
        _timeout: Duration,
    ) -> Result<pipeline::ProcessOutput, PipelineError> {
        let pattern = Path::new(command.arguments().last().unwrap());
        let directory = pattern.parent().unwrap().to_owned();
        let first = flag_number(command, "-start_number");
        let count = flag_number(command, "-frames:v");
        for index in first..first + count {
            fs::write(directory.join(format!("{index:06}.jpg")), b"jpeg").unwrap();
        }
        self.chunks.lock().unwrap().push((first, count));
        Ok(pipeline::ProcessOutput::new(
            Some(0),
            Vec::new(),
            Vec::new(),
        ))
    }
}

/// The same helper `frame-pipeline/tests/support/mod.rs` owns; a test binary in
/// this crate cannot reach the other crate's test module.
fn flag_number(command: &pipeline::CommandSpec, flag: &str) -> u64 {
    let arguments = command.arguments();
    let position = arguments
        .iter()
        .position(|argument| argument.as_os_str() == OsStr::new(flag))
        .unwrap_or_else(|| panic!("{flag} is missing from the frame command"));
    arguments[position + 1]
        .to_str()
        .unwrap_or_else(|| panic!("{flag} carries non-UTF-8 text"))
        .parse()
        .unwrap_or_else(|_| panic!("{flag} must carry a number"))
}

struct Decoder {
    blur: f64,
}

impl FrameDecoder for Decoder {
    fn decode(&self, _index: u64, path: &Path) -> Result<DecodedFrame, PipelineError> {
        Ok(DecodedFrame::new(path.to_owned(), 640, 480, self.blur).unwrap())
    }
}

struct Detector {
    detections: Vec<FaceDetection>,
}

impl FaceDetector for Detector {
    fn detect(&self, _frame: &DecodedFrame) -> Result<Vec<FaceDetection>, PipelineError> {
        Ok(self.detections.clone())
    }
}

struct Predictor {
    landmarks: PFLDLandmarks,
}

impl LandmarkPredictor for Predictor {
    fn predict(
        &self,
        _frame: &DecodedFrame,
        _face: &FaceDetection,
    ) -> Result<PFLDLandmarks, PipelineError> {
        Ok(self.landmarks.clone())
    }
}

/// A blur variance well above `BLUR_VARIANCE_THRESHOLD`.
fn decoder() -> Decoder {
    Decoder { blur: 30.0 }
}

/// One face, scored above `FACE_CONFIDENCE_THRESHOLD`, well inside a 640x480
/// frame.
fn detector() -> Detector {
    Detector {
        detections: vec![FaceDetection {
            bbox: [12.0, 12.0, 400.0, 350.0],
            score: 0.9,
            keypoints: [[0.0, 0.0]; 5],
        }],
    }
}

fn predictor() -> Predictor {
    Predictor {
        landmarks: decode_landmarks(
            &vec![0.5; 220],
            &vec![0.0; 220],
            CropGeometry {
                width: 640,
                height: 480,
                offset_x: 0,
                offset_y: 0,
            },
        )
        .unwrap(),
    }
}

struct Recorder {
    events: Mutex<Vec<(TaskStage, Option<Progress>)>>,
}

impl Recorder {
    fn new() -> Self {
        Self {
            events: Mutex::new(Vec::new()),
        }
    }

    fn events(&self) -> Vec<(TaskStage, Option<Progress>)> {
        self.events.lock().unwrap().clone()
    }
}

impl TaskReporter for Recorder {
    fn report(&self, stage: TaskStage, progress: Option<Progress>) {
        self.events.lock().unwrap().push((stage, progress));
    }
}

/// A project directory that passes admission: a real directory holding
/// `project.json`, with the normalised video inside `assets/`.
fn project() -> (TempDir, ExtractFramesParams) {
    let root = tempfile::tempdir().unwrap();
    let project_dir = root.path().join("project");
    let assets = project_dir.join("assets");
    fs::create_dir_all(&assets).unwrap();
    // Only the file's presence is checked; nothing here parses the manifest.
    fs::write(project_dir.join("project.json"), b"{}").unwrap();
    let video = assets.join("video_25fps.mp4");
    fs::write(&video, b"video").unwrap();
    (root, ExtractFramesParams { project_dir, video })
}

/// The media half of the configuration. Neither binary is ever executed --
/// both runners are fakes -- and the models arrive as arguments, so the model
/// directories play no part here.
fn config() -> WorkerConfig {
    let root = std::env::current_dir().unwrap();
    WorkerConfig::from_values(
        Some(root.join("ffprobe-test").display().to_string()),
        Some(root.join("ffmpeg-test").display().to_string()),
        Some("10000".to_owned()),
    )
}

/// Everything a test varies, with a working default for each.
struct Case {
    config: WorkerConfig,
    token: CancellationToken,
    media: MediaRunner,
    frames: FfmpegRunner,
    detector: Detector,
}

impl Case {
    fn new() -> Self {
        Self {
            config: config(),
            token: CancellationToken::new(),
            media: MediaRunner::new(3, "25/1"),
            frames: FfmpegRunner::new(),
            detector: detector(),
        }
    }

    fn run(&self, params: &ExtractFramesParams, reporter: &dyn TaskReporter) -> CommandOutcome {
        execute_extract_frames(
            params,
            &self.config,
            &self.token,
            reporter,
            &self.media,
            &self.frames,
            &decoder(),
            &self.detector,
            &predictor(),
        )
    }
}

fn progress(completed: u64, total: u64) -> Option<Progress> {
    Some(Progress {
        completed,
        total: Some(total),
    })
}

fn file_count(directory: &Path) -> usize {
    fs::read_dir(directory).unwrap().count()
}

fn expect_failure(outcome: CommandOutcome) -> TaskError {
    match outcome {
        CommandOutcome::Failed(error) => error,
        other => panic!("expected a failure, got {other:?}"),
    }
}

#[test]
fn a_device_fault_before_frame_publication_discards_the_staged_batch() {
    let (_root, params) = project();
    let case = Case::new();
    let error = expect_failure(case.run(&params, &health::PublicationFault::on_check(1)));
    assert_eq!(error.code, ErrorCode::GpuDeviceLost);
    assert_eq!(error.stage, TaskStage::DetectingFaces);
    assert!(!params.project_dir.join("assets/frames").exists());
    assert!(!params.project_dir.join("assets/landmarks").exists());
}

#[test]
fn three_frames_are_published_and_every_stage_is_reported() {
    let (_root, params) = project();
    let case = Case::new();
    let recorder = Recorder::new();

    let result = match case.run(&params, &recorder) {
        CommandOutcome::Completed(Some(result)) => result,
        other => panic!("expected a completed command, got {other:?}"),
    };

    let assets = params.project_dir.join("assets");
    let frames = assets.join("frames");
    let landmarks = assets.join("landmarks");
    assert_eq!(result["output_dir"], assets.display().to_string());
    assert_eq!(result["frames_dir"], frames.display().to_string());
    assert_eq!(result["landmarks_dir"], landmarks.display().to_string());
    assert_eq!(
        result["quality_report"],
        assets.join("quality.json").display().to_string()
    );
    assert_eq!(result["frame_count"], 3);
    assert_eq!(result["frame_width"], 640);
    assert_eq!(result["frame_height"], 480);
    assert_eq!(file_count(&frames), 3);
    assert_eq!(file_count(&landmarks), 3);
    assert!(assets.join("quality.json").is_file());
    // `FRAME_CHUNK` is 250, so three frames are a single invocation.
    assert_eq!(case.frames.chunks(), vec![(0, 3)]);
    assert_eq!(
        recorder.events(),
        vec![
            (TaskStage::Preparing, None),
            (TaskStage::ExtractingFrames, progress(3, 3)),
            (TaskStage::DetectingFaces, progress(0, 3)),
            (TaskStage::DetectingFaces, progress(1, 3)),
            (TaskStage::DetectingFaces, progress(2, 3)),
        ]
    );
}

#[test]
fn a_video_that_is_not_25fps_is_rejected_before_ffmpeg_runs() {
    let (_root, params) = project();
    let mut case = Case::new();
    case.media = MediaRunner::new(3, "30/1");

    let error = expect_failure(case.run(&params, &NoReporter));

    assert_eq!(error.code, ErrorCode::MediaInvalid);
    assert_eq!(error.summary, "抽帧要求 25fps 的归一化视频");
    assert!(error.detail.contains("30/1"), "detail was {}", error.detail);
    assert_eq!(case.media.call_count(), 1);
    assert!(case.frames.chunks().is_empty());
}

#[test]
fn a_missing_project_directory_is_rejected_before_ffprobe_runs() {
    let (root, params) = project();
    let params = ExtractFramesParams {
        project_dir: root.path().join("absent"),
        ..params
    };
    let case = Case::new();

    let error = expect_failure(case.run(&params, &NoReporter));

    assert_eq!(error.code, ErrorCode::MediaInvalid);
    assert_eq!(error.summary, "工程目录不可用");
    assert_eq!(case.media.call_count(), 0);
}

#[test]
fn a_project_directory_without_a_manifest_is_rejected() {
    let (_root, params) = project();
    fs::remove_file(params.project_dir.join("project.json")).unwrap();
    let case = Case::new();

    let error = expect_failure(case.run(&params, &NoReporter));

    assert_eq!(error.code, ErrorCode::MediaInvalid);
    assert_eq!(error.summary, "工程目录缺少 project.json");
    assert_eq!(case.media.call_count(), 0);
}

#[test]
fn an_existing_frames_directory_stops_the_run_and_is_left_untouched() {
    let (_root, params) = project();
    let frames = params.project_dir.join("assets").join("frames");
    fs::create_dir_all(&frames).unwrap();
    fs::write(frames.join("000000.jpg"), b"old").unwrap();
    let case = Case::new();

    let error = expect_failure(case.run(&params, &NoReporter));

    assert_eq!(error.code, ErrorCode::MediaInvalid);
    assert_eq!(error.summary, "素材目录已存在抽帧结果");
    assert_eq!(
        fs::read(frames.join("000000.jpg")).unwrap(),
        b"old".to_vec()
    );
    assert!(case.frames.chunks().is_empty());
}

#[test]
fn a_cancelled_token_stops_the_run_before_the_first_chunk() {
    let (_root, params) = project();
    let case = Case::new();
    case.token.cancel();

    let outcome = case.run(&params, &NoReporter);

    assert!(
        matches!(outcome, CommandOutcome::Cancelled),
        "expected a cancelled run, got {outcome:?}"
    );
    assert!(case.frames.chunks().is_empty());
}

#[test]
fn a_frame_without_a_face_fails_the_run_and_publishes_nothing() {
    let (_root, params) = project();
    let mut case = Case::new();
    case.detector = Detector {
        detections: Vec::new(),
    };

    let error = expect_failure(case.run(&params, &NoReporter));

    assert_eq!(error.code, ErrorCode::FaceNotFound);
    assert_eq!(error.summary, "有帧未检测到人脸");
    assert!(
        error.detail.contains("3 frame(s) rejected"),
        "detail was {}",
        error.detail
    );
    // The batch destructor removes the staging directory, so the project keeps
    // whatever it had before the run.
    assert!(!params.project_dir.join("assets").join("frames").exists());
}

#[test]
fn a_worker_without_a_media_toolchain_reports_the_command_as_unsupported() {
    let (_root, params) = project();
    let mut case = Case::new();
    case.config = WorkerConfig::from_values(None, None, None);

    let error = expect_failure(case.run(&params, &NoReporter));

    assert_eq!(error.code, ErrorCode::WorkerCrashed);
    assert_eq!(error.summary, "当前 worker 不支持该命令");
    assert_eq!(case.media.call_count(), 0);
}
