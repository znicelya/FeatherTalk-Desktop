use std::{collections::VecDeque, path::PathBuf, sync::Mutex, time::Duration};

use feathertalk_domain::{
    ErrorCode, ExtractFeaturesParams, ExtractFramesParams, NormalizeMediaParams, ProbeMediaParams,
    Progress, ProjectDirParams, Recovery, RenderParams, Request, TaskStage, TrainParams,
    TrainingMode, UnetVariant,
};
use feathertalk_media::{CancellationToken, CommandSpec, MediaError, ProcessOutput, ProcessRunner};
use feathertalk_project::{
    AssetManifest, AssetPackageState, FeatureType, ModelSelection, ProjectManifest,
    TaskHistoryEntry, TaskHistoryStatus, lock_asset_package, write_project_manifest_atomic,
};
use feathertalk_worker::{
    CommandOutcome, NoReporter, TaskReporter, WorkerConfig, execute_with_runner,
};

struct FakeRunner {
    outputs: Mutex<VecDeque<Result<ProcessOutput, MediaError>>>,
    commands: Mutex<Vec<CommandSpec>>,
}

impl FakeRunner {
    fn new(outputs: Vec<Result<ProcessOutput, MediaError>>) -> Self {
        Self {
            outputs: Mutex::new(outputs.into_iter().collect()),
            commands: Mutex::new(Vec::new()),
        }
    }

    fn call_count(&self) -> usize {
        self.commands.lock().unwrap().len()
    }
}

impl ProcessRunner for FakeRunner {
    fn run(&self, command: &CommandSpec, _timeout: Duration) -> Result<ProcessOutput, MediaError> {
        self.commands.lock().unwrap().push(command.clone());
        self.outputs.lock().unwrap().pop_front().unwrap()
    }
}

fn bare_config() -> WorkerConfig {
    WorkerConfig::from_values(None, None, None)
}

#[test]
fn every_compute_command_refuses_an_unavailable_gpu_before_touching_inputs() {
    let root = std::env::current_dir().unwrap();
    let requests = [
        Request::Train(TrainParams {
            project_dir: root.clone(),
            mode: TrainingMode::Baseline,
            variant: UnetVariant::OriginalUnet,
            epochs: 1,
            batch_size: 1,
            resume: false,
        }),
        Request::Render(RenderParams {
            project_dir: root.clone(),
            checkpoint: root.join("missing-checkpoint"),
            audio: root.join("missing.wav"),
            output: root.join("unused.mp4"),
            max_output_frames: None,
        }),
        Request::ExtractFrames(ExtractFramesParams {
            project_dir: root.clone(),
            video: root.join("missing.mp4"),
        }),
        Request::ExtractFeatures(ExtractFeaturesParams {
            project_dir: root.clone(),
            audio: root.join("missing.wav"),
        }),
        Request::NormalizeMedia(NormalizeMediaParams {
            input: root.join("missing.mp4"),
            output_dir: root.join("unused-media"),
        }),
    ];
    let runner = FakeRunner::new(vec![]);
    for backend in ["wgpu", "cuda"] {
        let config = media_config().with_compute_selection(Some(backend), Some("missing-gpu"));
        for request in &requests {
            let outcome = execute_with_runner(
                &request,
                &config,
                &CancellationToken::new(),
                &NoReporter,
                &runner,
            );
            let CommandOutcome::Failed(error) = outcome else {
                panic!("an unavailable GPU cannot execute {request:?}");
            };
            assert_eq!(error.recovery, Recovery::SelectDifferentAdapter);
            assert!(error.detail.contains("missing-gpu"), "{error:?}");
            assert_eq!(error.stage, TaskStage::Preparing);
        }
    }
    assert_eq!(runner.call_count(), 0);
}

fn media_config() -> WorkerConfig {
    let root = std::env::current_dir().unwrap();
    WorkerConfig::from_values(
        Some(root.join("ffprobe-test").display().to_string()),
        Some(root.join("ffmpeg-test").display().to_string()),
        Some("10000".to_owned()),
    )
}

fn valid_probe() -> Vec<u8> {
    br#"{
      "format":{"format_name":"mov,mp4","duration":"2.0"},
      "streams":[
        {"codec_type":"video","codec_name":"h264","pix_fmt":"yuv420p","width":640,"height":480,"avg_frame_rate":"25/1","nb_read_frames":"50","duration":"2.0"},
        {"codec_type":"audio","codec_name":"aac","sample_fmt":"fltp","sample_rate":"48000","channels":2,"duration":"2.0"}
      ]
    }"#
    .to_vec()
}

fn probe_request(input: PathBuf) -> Request {
    Request::ProbeMedia(ProbeMediaParams { input })
}

fn media_file() -> (tempfile::TempDir, PathBuf) {
    let temp = tempfile::tempdir().unwrap();
    let source = temp.path().join("input.mov");
    std::fs::write(&source, b"media").unwrap();
    (temp, source)
}

fn valid_project() -> ProjectManifest {
    ProjectManifest {
        schema_version: 1,
        project_id: "demo".to_owned(),
        display_name: "Demo".to_owned(),
        asset_package: "assets/assets.json".to_owned(),
        default_model: ModelSelection::OriginalUnet,
        task_history: vec![TaskHistoryEntry {
            task_id: "task-1".to_owned(),
            kind: "preprocess".to_owned(),
            status: TaskHistoryStatus::Completed,
            updated_at: "2026-08-20T10:00:00Z".to_owned(),
        }],
    }
}

fn locked_manifest() -> AssetManifest {
    AssetManifest {
        schema_version: 1,
        state: AssetPackageState::Locked,
        video_fps: 25,
        audio_sample_rate: 16_000,
        audio_channels: 1,
        frame_count: 12,
        frame_width: 160,
        frame_height: 160,
        feature_type: FeatureType::FeatherHubert,
        feature_shape: [12, 2, 1024],
        landmark_model_sha256: "a".repeat(64),
        feature_model_sha256: "b".repeat(64),
    }
}

fn complete_project() -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(dir.path().join("assets/frames")).unwrap();
    std::fs::create_dir_all(dir.path().join("assets/landmarks")).unwrap();
    std::fs::create_dir_all(dir.path().join("assets/features")).unwrap();
    for file in [
        "assets/video_25fps.mp4",
        "assets/audio_16k_mono.wav",
        "assets/features/feather_hubert.f32",
    ] {
        std::fs::write(dir.path().join(file), b"x").unwrap();
    }
    write_project_manifest_atomic(&dir.path().join("project.json"), &valid_project()).unwrap();
    lock_asset_package(dir.path(), locked_manifest()).unwrap();
    dir
}

#[test]
fn validating_a_complete_project_completes_without_a_result() {
    let dir = complete_project();
    let request = Request::ValidateProject(ProjectDirParams {
        project_dir: dir.path().to_path_buf(),
    });
    let runner = FakeRunner::new(vec![]);
    let outcome = execute_with_runner(
        &request,
        &bare_config(),
        &CancellationToken::new(),
        &NoReporter,
        &runner,
    );
    assert!(
        matches!(outcome, CommandOutcome::Completed(None)),
        "{outcome:?}"
    );
    assert_eq!(runner.call_count(), 0);
}

#[test]
fn validating_a_missing_project_fails_with_a_wire_error() {
    let dir = tempfile::tempdir().unwrap();
    let request = Request::ValidateProject(ProjectDirParams {
        project_dir: dir.path().join("nope"),
    });
    let runner = FakeRunner::new(vec![]);
    let CommandOutcome::Failed(error) = execute_with_runner(
        &request,
        &bare_config(),
        &CancellationToken::new(),
        &NoReporter,
        &runner,
    ) else {
        panic!("a missing project must fail");
    };
    error.validate().unwrap();
    assert!(!error.summary.is_empty());
}

#[test]
fn probing_media_completes_with_the_probe_result() {
    let (_temp, source) = media_file();
    let runner = FakeRunner::new(vec![Ok(ProcessOutput::new(
        Some(0),
        valid_probe(),
        Vec::new(),
    ))]);
    let config = media_config();
    let CommandOutcome::Completed(Some(result)) = execute_with_runner(
        &probe_request(source),
        &config,
        &CancellationToken::new(),
        &NoReporter,
        &runner,
    ) else {
        panic!("a successful probe must carry a result");
    };
    assert!(result.is_object());
    assert_eq!(result["format"]["format_name"], "mov,mp4");
    assert_eq!(result["format"]["duration_seconds"], 2.0);
    assert_eq!(result["video"]["codec_name"], "h264");
    assert_eq!(result["video"]["pixel_format"], "yuv420p");
    assert_eq!(result["video"]["width"], 640);
    assert_eq!(result["video"]["height"], 480);
    assert_eq!(result["video"]["frame_rate"]["numerator"], 25);
    assert_eq!(result["video"]["frame_rate"]["denominator"], 1);
    assert_eq!(result["video"]["frame_count"], 50);
    assert_eq!(result["audio"]["codec_name"], "aac");
    assert_eq!(result["audio"]["sample_format"], "fltp");
    assert_eq!(result["audio"]["sample_rate"], 48_000);
    assert_eq!(result["audio"]["channels"], 2);
    assert_eq!(runner.call_count(), 1);
    assert_eq!(result.get("input"), None, "the result must not leak paths");
}

#[test]
fn probing_a_missing_file_fails_before_the_tool_runs() {
    let temp = tempfile::tempdir().unwrap();
    let runner = FakeRunner::new(vec![]);
    let config = media_config();
    let CommandOutcome::Failed(error) = execute_with_runner(
        &probe_request(temp.path().join("absent.mov")),
        &config,
        &CancellationToken::new(),
        &NoReporter,
        &runner,
    ) else {
        panic!("a missing input must fail");
    };
    assert_eq!(error.code, ErrorCode::MediaInvalid);
    error.validate().unwrap();
    assert_eq!(runner.call_count(), 0);
}

#[test]
fn a_cancelled_tool_reports_cancellation_not_failure() {
    let (_temp, source) = media_file();
    let runner = FakeRunner::new(vec![Err(MediaError::ToolCancelled { operation: "probe" })]);
    let config = media_config();
    let outcome = execute_with_runner(
        &probe_request(source),
        &config,
        &CancellationToken::new(),
        &NoReporter,
        &runner,
    );
    assert!(matches!(outcome, CommandOutcome::Cancelled), "{outcome:?}");
}

#[test]
fn an_already_cancelled_token_runs_nothing() {
    let (_temp, source) = media_file();
    let runner = FakeRunner::new(vec![]);
    let config = media_config();
    let token = CancellationToken::new();
    token.cancel();
    let outcome = execute_with_runner(
        &probe_request(source),
        &config,
        &token,
        &NoReporter,
        &runner,
    );
    assert!(matches!(outcome, CommandOutcome::Cancelled), "{outcome:?}");
    assert_eq!(runner.call_count(), 0);
}

#[test]
fn probing_without_a_toolchain_is_refused() {
    let (_temp, source) = media_file();
    let runner = FakeRunner::new(vec![]);
    let CommandOutcome::Failed(error) = execute_with_runner(
        &probe_request(source),
        &bare_config(),
        &CancellationToken::new(),
        &NoReporter,
        &runner,
    ) else {
        panic!("probing without a toolchain must fail");
    };
    assert_eq!(error.code, ErrorCode::WorkerCrashed);
    error.validate().unwrap();
}

#[test]
fn an_unsupported_command_is_refused_with_its_slug() {
    let request = Request::Train(TrainParams {
        project_dir: PathBuf::from("C:/tmp/project"),
        mode: TrainingMode::Baseline,
        variant: UnetVariant::OriginalUnet,
        epochs: 1,
        batch_size: 1,
        resume: false,
    });
    let runner = FakeRunner::new(vec![]);
    let CommandOutcome::Failed(error) = execute_with_runner(
        &request,
        &bare_config(),
        &CancellationToken::new(),
        &NoReporter,
        &runner,
    ) else {
        panic!("an unsupported command must fail");
    };
    assert_eq!(error.code, ErrorCode::WorkerCrashed);
    assert!(error.detail.contains("train"), "{}", error.detail);
    error.validate().unwrap();
}

#[test]
fn extract_features_reports_a_package_failure_as_a_model_incompatibility() {
    let temp = tempfile::tempdir().unwrap();
    let request = Request::ExtractFeatures(ExtractFeaturesParams {
        project_dir: temp.path().join("project"),
        audio: temp.path().join("project/assets/audio_16k_mono.wav"),
    });
    let config = WorkerConfig::from_values_with_toolchains(
        None,
        None,
        None,
        None,
        None,
        Some(temp.path().display().to_string()),
    );
    let runner = FakeRunner::new(vec![]);
    let CommandOutcome::Failed(error) = execute_with_runner(
        &request,
        &config,
        &CancellationToken::new(),
        &NoReporter,
        &runner,
    ) else {
        panic!("loading a package out of an empty directory must fail");
    };
    assert_eq!(error.code, ErrorCode::ModelIncompatible);
    assert_eq!(error.summary, "特征模型加载失败");
    assert!(
        error.detail.contains("FEATHERTALK_WORKER_HUBERT_DIR"),
        "{}",
        error.detail
    );
    error.validate().unwrap();
}

#[test]
fn extract_features_without_a_model_directory_is_refused_with_its_slug() {
    let request = Request::ExtractFeatures(ExtractFeaturesParams {
        project_dir: PathBuf::from("C:/tmp/project"),
        audio: PathBuf::from("C:/tmp/project/assets/audio_16k_mono.wav"),
    });
    let runner = FakeRunner::new(vec![]);
    let CommandOutcome::Failed(error) = execute_with_runner(
        &request,
        &bare_config(),
        &CancellationToken::new(),
        &NoReporter,
        &runner,
    ) else {
        panic!("extract_features without a model directory must fail");
    };
    assert_eq!(error.code, ErrorCode::WorkerCrashed);
    assert_eq!(error.summary, "当前 worker 不支持该命令");
    assert!(
        error.detail.contains("extract_features"),
        "{}",
        error.detail
    );
    error.validate().unwrap();
}

#[test]
fn lock_asset_package_reports_a_package_failure_as_a_model_incompatibility() {
    let temp = tempfile::tempdir().unwrap();
    let request = Request::LockAssetPackage(ProjectDirParams {
        project_dir: temp.path().join("project"),
    });
    let config = WorkerConfig::from_values_with_toolchains(
        None,
        None,
        None,
        None,
        None,
        Some(temp.path().display().to_string()),
    );
    let runner = FakeRunner::new(vec![]);
    let CommandOutcome::Failed(error) = execute_with_runner(
        &request,
        &config,
        &CancellationToken::new(),
        &NoReporter,
        &runner,
    ) else {
        panic!("locking with a broken package must fail");
    };
    // The package is read before the project is admitted, so an empty model
    // directory is reported as a model problem even though the project
    // directory does not exist either.
    assert_eq!(error.code, ErrorCode::ModelIncompatible);
    assert_eq!(error.summary, "特征模型加载失败");
    assert!(
        error.detail.contains("FEATHERTALK_WORKER_HUBERT_DIR"),
        "{}",
        error.detail
    );
    error.validate().unwrap();
}

#[test]
fn lock_asset_package_without_a_model_directory_is_refused_with_its_slug() {
    let request = Request::LockAssetPackage(ProjectDirParams {
        project_dir: PathBuf::from("C:/tmp/project"),
    });
    let runner = FakeRunner::new(vec![]);
    let CommandOutcome::Failed(error) = execute_with_runner(
        &request,
        &bare_config(),
        &CancellationToken::new(),
        &NoReporter,
        &runner,
    ) else {
        panic!("lock_asset_package without a model directory must fail");
    };
    assert_eq!(error.code, ErrorCode::WorkerCrashed);
    assert_eq!(error.summary, "当前 worker 不支持该命令");
    assert!(
        error.detail.contains("lock_asset_package"),
        "{}",
        error.detail
    );
    error.validate().unwrap();
}

/// A runner that scripts probe output and writes the bytes `ffmpeg` would have
/// written, so the normalization pipeline can verify and commit them.
struct NormalizeRunner {
    outputs: Mutex<VecDeque<Result<ProcessOutput, MediaError>>>,
    commands: Mutex<Vec<CommandSpec>>,
}

impl NormalizeRunner {
    fn new(outputs: Vec<Result<ProcessOutput, MediaError>>) -> Self {
        Self {
            outputs: Mutex::new(outputs.into_iter().collect()),
            commands: Mutex::new(Vec::new()),
        }
    }
}

impl ProcessRunner for NormalizeRunner {
    fn run(&self, command: &CommandSpec, _timeout: Duration) -> Result<ProcessOutput, MediaError> {
        self.commands.lock().unwrap().push(command.clone());
        let output = self.outputs.lock().unwrap().pop_front().unwrap()?;
        if matches!(command.operation(), "normalize_video" | "normalize_audio") {
            let path = PathBuf::from(command.arguments().last().unwrap());
            std::fs::write(path, b"normalized-bytes").unwrap();
        }
        Ok(output)
    }
}

/// Records everything a command reports.
#[derive(Default)]
struct RecordingReporter {
    reports: Mutex<Vec<(String, Option<Progress>)>>,
}

impl TaskReporter for RecordingReporter {
    fn report(&self, stage: TaskStage, progress: Option<Progress>) {
        self.reports
            .lock()
            .unwrap()
            .push((stage.as_slug().to_owned(), progress));
    }
}

fn normalized_video_probe() -> Vec<u8> {
    br#"{"format":{"format_name":"mp4","duration":"2.0"},"streams":[{"codec_type":"video","codec_name":"mpeg4","pix_fmt":"yuv420p","width":640,"height":480,"avg_frame_rate":"25/1","nb_read_frames":"50","duration":"2.0"}]}"#.to_vec()
}

fn normalized_audio_probe() -> Vec<u8> {
    br#"{"format":{"format_name":"wav","duration":"2.0"},"streams":[{"codec_type":"audio","codec_name":"pcm_s16le","sample_fmt":"s16","sample_rate":"16000","channels":1,"duration":"2.0"}]}"#.to_vec()
}

fn normalize_request(input: PathBuf, output_dir: PathBuf) -> Request {
    Request::NormalizeMedia(NormalizeMediaParams { input, output_dir })
}

fn normalize_outputs() -> Vec<Result<ProcessOutput, MediaError>> {
    vec![
        Ok(ProcessOutput::new(Some(0), valid_probe(), Vec::new())),
        Ok(ProcessOutput::new(Some(0), Vec::new(), Vec::new())),
        Ok(ProcessOutput::new(Some(0), Vec::new(), Vec::new())),
        Ok(ProcessOutput::new(
            Some(0),
            normalized_video_probe(),
            Vec::new(),
        )),
        Ok(ProcessOutput::new(
            Some(0),
            normalized_audio_probe(),
            Vec::new(),
        )),
    ]
}

#[test]
fn normalizing_media_reports_paths_sizes_and_hashes() {
    let (temp, source) = media_file();
    let output_dir = temp.path().join("assets");
    let runner = NormalizeRunner::new(normalize_outputs());
    let reporter = RecordingReporter::default();

    let CommandOutcome::Completed(Some(result)) = execute_with_runner(
        &normalize_request(source, output_dir.clone()),
        &media_config(),
        &CancellationToken::new(),
        &reporter,
        &runner,
    ) else {
        panic!("a scripted normalization completes with a result");
    };

    assert_eq!(result["video"]["codec_name"], "mpeg4");
    assert_eq!(result["video"]["frame_rate"]["numerator"], 25);
    assert_eq!(result["audio"]["sample_rate"], 16_000);
    assert_eq!(result["audio"]["channels"], 1);
    assert_eq!(result["video"]["bytes"], b"normalized-bytes".len());
    assert_eq!(
        result["video"]["sha256"].as_str().unwrap().len(),
        64,
        "{result}"
    );
    // The source probe is reported under `source`, in the probe payload shape.
    assert_eq!(result["source"]["video"]["codec_name"], "h264");
    // The committed paths are what a later task has to open.
    let video_path = PathBuf::from(result["video"]["path"].as_str().unwrap());
    assert!(video_path.is_file(), "{}", video_path.display());
    assert_eq!(video_path.file_name().unwrap(), "video_25fps.mp4");
    let audio_path = PathBuf::from(result["audio"]["path"].as_str().unwrap());
    assert_eq!(audio_path.file_name().unwrap(), "audio_16k_mono.wav");
    assert!(
        result["output_dir"].as_str().unwrap().ends_with("assets"),
        "{result}"
    );
}

#[test]
fn normalizing_media_reports_three_progress_steps() {
    let (temp, source) = media_file();
    let output_dir = temp.path().join("assets");
    let runner = NormalizeRunner::new(normalize_outputs());
    let reporter = RecordingReporter::default();

    execute_with_runner(
        &normalize_request(source, output_dir),
        &media_config(),
        &CancellationToken::new(),
        &reporter,
        &runner,
    );

    assert_eq!(
        *reporter.reports.lock().unwrap(),
        vec![
            (
                "preparing".to_owned(),
                Some(Progress {
                    completed: 1,
                    total: Some(3)
                })
            ),
            (
                "extracting_frames".to_owned(),
                Some(Progress {
                    completed: 2,
                    total: Some(3)
                })
            ),
            (
                "extracting_audio".to_owned(),
                Some(Progress {
                    completed: 3,
                    total: Some(3)
                })
            ),
        ]
    );
}

#[test]
fn normalizing_without_a_toolchain_is_unsupported() {
    let (temp, source) = media_file();
    let runner = NormalizeRunner::new(vec![]);
    let CommandOutcome::Failed(error) = execute_with_runner(
        &normalize_request(source, temp.path().join("assets")),
        &bare_config(),
        &CancellationToken::new(),
        &NoReporter,
        &runner,
    ) else {
        panic!("no toolchain means the command cannot run");
    };
    assert_eq!(error.code, ErrorCode::WorkerCrashed);
    assert!(error.detail.contains("normalize_media"), "{}", error.detail);
}

#[test]
fn a_source_without_audio_fails_before_any_output_is_written() {
    let (temp, source) = media_file();
    let output_dir = temp.path().join("assets");
    let video_only = br#"{"format":{"format_name":"mov,mp4","duration":"2.0"},"streams":[{"codec_type":"video","codec_name":"h264","pix_fmt":"yuv420p","width":640,"height":480,"avg_frame_rate":"25/1","nb_read_frames":"50","duration":"2.0"}]}"#.to_vec();
    let runner = NormalizeRunner::new(vec![Ok(ProcessOutput::new(
        Some(0),
        video_only,
        Vec::new(),
    ))]);
    let CommandOutcome::Failed(error) = execute_with_runner(
        &normalize_request(source, output_dir.clone()),
        &media_config(),
        &CancellationToken::new(),
        &NoReporter,
        &runner,
    ) else {
        panic!("a source without audio cannot be normalized");
    };
    assert_eq!(error.code, ErrorCode::MediaInvalid);
    assert!(!output_dir.join("video_25fps.mp4").exists());
}

#[test]
fn a_cancelled_normalization_reports_cancelled() {
    let (temp, source) = media_file();
    let runner = NormalizeRunner::new(vec![Err(MediaError::ToolCancelled {
        operation: "ffprobe",
    })]);
    let outcome = execute_with_runner(
        &normalize_request(source, temp.path().join("assets")),
        &media_config(),
        &CancellationToken::new(),
        &NoReporter,
        &runner,
    );
    assert!(matches!(outcome, CommandOutcome::Cancelled), "{outcome:?}");
}
