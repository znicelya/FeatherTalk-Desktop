//! The asset lock, driven through the worker's public entry point.

use std::{
    fs,
    path::{Path, PathBuf},
    sync::Mutex,
};

use feathertalk_audio::{FeatureMatrix, read_feature_file, write_feature_file_no_clobber};
use feathertalk_domain::{ErrorCode, Progress, ProjectDirParams, TaskError, TaskStage};
use feathertalk_frame_pipeline::{
    AnomalyCode, FrameAnomaly, FrameQuality, QualityReport, RecoveryAction,
};
use feathertalk_media::CancellationToken;
use feathertalk_pfld::PFLD_MODEL_SHA256;
use feathertalk_worker::{CommandOutcome, NoReporter, TaskReporter, execute_lock_asset_package};
use serde_json::{Value, json};
use tempfile::TempDir;

/// Stands in for the digest Task 9 reads out of the FeatherHuBERT package
/// manifest.
const MODEL_SHA256: &str = "a1b2c3d4a1b2c3d4a1b2c3d4a1b2c3d4a1b2c3d4a1b2c3d4a1b2c3d4a1b2c3d4";

/// The per-frame digests the quality report carries. The lock verifies
/// structure and never re-hashes a frame, so any 64-hex string does.
const SHA256: &str = "1111111111111111111111111111111111111111111111111111111111111111";

/// Three frames is the smallest fixture that still proves the walk iterates.
const FRAME_COUNT: u64 = 3;

/// FeatherHuBERT's output width; `commit_feature_artifact` accepts no other.
const DIMS: usize = 1_024;

#[derive(Default)]
struct Recorder {
    events: Mutex<Vec<(TaskStage, Option<Progress>)>>,
}

impl Recorder {
    fn events(&self) -> Vec<(TaskStage, Option<Progress>)> {
        self.events
            .lock()
            .expect("the recorder must not be poisoned")
            .clone()
    }
}

impl TaskReporter for Recorder {
    fn report(&self, stage: TaskStage, progress: Option<Progress>) {
        self.events
            .lock()
            .expect("the recorder must not be poisoned")
            .push((stage, progress));
    }
}

/// The checked-in 1280x720 frame, borrowed from the adapters crate.
fn fixture() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../feathertalk-frame-adapters/tests/fixtures/demo_frame_v1/frame.jpg")
}

/// 110 points, all inside the fixture frame.
fn landmark_text() -> String {
    let mut text = String::new();
    for index in 0..110 {
        text.push_str(&format!("{index} {}\n", index * 2));
    }
    text
}

/// A complete, unlocked package: a project manifest, the two normalised media
/// files, `FRAME_COUNT` frames with landmarks, a feature file of exactly
/// `2 * FRAME_COUNT` tokens, and a clean quality report. The `TempDir` is
/// returned so it outlives the test body.
fn project() -> (TempDir, ProjectDirParams) {
    let root = TempDir::new().expect("a temporary directory must be available");
    let project_dir = root.path().join("project");
    let assets = project_dir.join("assets");
    fs::create_dir_all(assets.join("frames")).unwrap();
    fs::create_dir_all(assets.join("landmarks")).unwrap();
    fs::create_dir_all(assets.join("features")).unwrap();
    // Only its presence is checked; nothing here parses the project manifest.
    fs::write(project_dir.join("project.json"), b"{}").unwrap();
    fs::write(assets.join("video_25fps.mp4"), b"video").unwrap();
    fs::write(assets.join("audio_16k_mono.wav"), b"audio").unwrap();
    write_features(&assets, 2 * FRAME_COUNT as usize);
    let source = fs::read(fixture()).expect("the fixture frame must be readable");
    for index in 0..FRAME_COUNT {
        fs::write(assets.join(format!("frames/{index:06}.jpg")), &source).unwrap();
        fs::write(
            assets.join(format!("landmarks/{index:06}.lms")),
            landmark_text(),
        )
        .unwrap();
    }
    let frames = frame_qualities(&assets);
    let report = QualityReport::new(FRAME_COUNT, frames, Vec::new()).unwrap();
    write_report(&assets, &report);
    (root, ProjectDirParams { project_dir })
}

/// The report entries for the frames `project` wrote.
fn frame_qualities(assets: &Path) -> Vec<FrameQuality> {
    (0..FRAME_COUNT)
        .map(|index| {
            let path = assets.join(format!("frames/{index:06}.jpg"));
            let frame_bytes = fs::metadata(&path).unwrap().len();
            FrameQuality::new(
                index,
                format!("frames/{index:06}.jpg"),
                format!("landmarks/{index:06}.lms"),
                frame_bytes,
                SHA256,
                SHA256,
                0.9,
                [0.0, 0.0, 64.0, 64.0],
                12.5,
            )
            .expect("the frame quality fixture must be valid")
        })
        .collect()
}

fn write_report(assets: &Path, report: &QualityReport) {
    let bytes = serde_json::to_vec_pretty(report).expect("the report must serialise");
    fs::write(assets.join("quality.json"), bytes).unwrap();
}

/// Replace the feature file with one of `tokens` tokens.
/// `write_feature_file_no_clobber` refuses to overwrite, so the old file goes
/// first.
fn write_features(assets: &Path, tokens: usize) {
    let path = assets.join("features").join("feather_hubert.f32");
    if path.exists() {
        fs::remove_file(&path).unwrap();
    }
    let matrix = FeatureMatrix::new(tokens, DIMS, vec![0.25; tokens * DIMS]).unwrap();
    write_feature_file_no_clobber(&path, &matrix).expect("the feature fixture must be writable");
}

fn run(
    params: &ProjectDirParams,
    token: &CancellationToken,
    reporter: &dyn TaskReporter,
) -> CommandOutcome {
    execute_lock_asset_package(params, token, reporter, MODEL_SHA256)
}

fn progress(completed: u64) -> Option<Progress> {
    Some(Progress {
        completed,
        total: Some(FRAME_COUNT),
    })
}

fn expect_completed(outcome: CommandOutcome) -> Value {
    match outcome {
        CommandOutcome::Completed(Some(result)) => result,
        other => panic!("expected a completed command, got {other:?}"),
    }
}

fn expect_failure(outcome: CommandOutcome) -> TaskError {
    match outcome {
        CommandOutcome::Failed(error) => error,
        other => panic!("expected a failure, got {other:?}"),
    }
}

fn manifest_path(params: &ProjectDirParams) -> PathBuf {
    params.project_dir.join("assets").join("assets.json")
}

#[test]
fn a_complete_package_is_locked_and_reports_its_manifest() {
    let (_root, params) = project();

    let result = expect_completed(run(&params, &CancellationToken::new(), &NoReporter));

    let manifest = manifest_path(&params);
    let feature_file = params
        .project_dir
        .join("assets")
        .join("features")
        .join("feather_hubert.f32");
    assert_eq!(
        result["project_dir"],
        params.project_dir.display().to_string()
    );
    assert_eq!(result["manifest_file"], manifest.display().to_string());
    assert_eq!(result["feature_file"], feature_file.display().to_string());
    assert_eq!(result["frame_count"], 3);
    assert_eq!(result["frame_width"], 1_280);
    assert_eq!(result["frame_height"], 720);
    assert_eq!(result["tokens"], 6);
    assert_eq!(result["dims"], 1_024);
    // The 44-byte header plus 6 * 1024 f32 values.
    assert_eq!(result["bytes"], 24_620);
    assert_eq!(result["token_adjustment"], 0);
    assert_eq!(result["landmark_model_sha256"], PFLD_MODEL_SHA256);
    assert_eq!(result["feature_model_sha256"], MODEL_SHA256);
    assert_eq!(result["sha256"].as_str().unwrap().len(), 64);

    let written: Value = serde_json::from_slice(&fs::read(&manifest).unwrap()).unwrap();
    assert_eq!(written["schema_version"], 1);
    assert_eq!(written["state"], "locked");
    assert_eq!(written["video_fps"], 25);
    assert_eq!(written["audio_sample_rate"], 16_000);
    assert_eq!(written["audio_channels"], 1);
    assert_eq!(written["frame_count"], 3);
    assert_eq!(written["frame_width"], 1_280);
    assert_eq!(written["frame_height"], 720);
    assert_eq!(written["feature_type"], "feather_hubert");
    assert_eq!(written["feature_shape"], json!([3, 2, 1_024]));
    assert_eq!(written["landmark_model_sha256"], PFLD_MODEL_SHA256);
    assert_eq!(written["feature_model_sha256"], MODEL_SHA256);
}

#[test]
fn every_frame_reports_progress_under_one_stage() {
    let (_root, params) = project();
    let recorder = Recorder::default();

    expect_completed(run(&params, &CancellationToken::new(), &recorder));

    // One `Preparing` with no progress while admission reads the report and
    // the feature file, then one per frame. Nothing else: the commit is a
    // rename, not a stage.
    assert_eq!(
        recorder.events(),
        vec![
            (TaskStage::Preparing, None),
            (TaskStage::Preparing, progress(1)),
            (TaskStage::Preparing, progress(2)),
            (TaskStage::Preparing, progress(3)),
        ]
    );
}

#[test]
fn a_relative_project_dir_is_rejected() {
    let relative = ProjectDirParams {
        project_dir: PathBuf::from("project"),
    };

    let error = expect_failure(run(&relative, &CancellationToken::new(), &NoReporter));

    assert_eq!(error.code, ErrorCode::MediaInvalid);
    assert_eq!(error.stage, TaskStage::Preparing);
    assert_eq!(error.summary, "工程目录必须是绝对路径");
}

#[test]
fn an_already_locked_package_is_refused() {
    let (_root, params) = project();
    expect_completed(run(&params, &CancellationToken::new(), &NoReporter));

    let error = expect_failure(run(&params, &CancellationToken::new(), &NoReporter));

    assert_eq!(error.code, ErrorCode::MediaInvalid);
    assert_eq!(error.stage, TaskStage::Preparing);
    assert_eq!(error.summary, "素材包已加锁");
    assert!(error.detail.contains("assets.json"), "{}", error.detail);
    error.validate().unwrap();
}

#[test]
fn a_corrupt_asset_manifest_is_not_a_crash() {
    let (_root, params) = project();
    fs::write(manifest_path(&params), b"{ not json").unwrap();

    let error = expect_failure(run(&params, &CancellationToken::new(), &NoReporter));

    // `feathertalk-project` owns the wording for a broken manifest. What this
    // pins is that a hand-edited file is a task failure, never a crash.
    assert_ne!(error.code, ErrorCode::WorkerCrashed);
    assert_eq!(error.stage, TaskStage::Preparing);
    error.validate().unwrap();
}

#[test]
fn a_report_with_anomalies_is_refused() {
    let (_root, params) = project();
    let assets = params.project_dir.join("assets");
    let mut frames = frame_qualities(&assets);
    // An anomaly and an accepted frame cannot share an index, so the excluded
    // frame leaves the accepted list.
    let excluded = frames.pop().unwrap();
    let anomaly = FrameAnomaly::new(
        excluded.index(),
        AnomalyCode::BlurredFrame,
        "画面模糊",
        "blur variance 3.1 is below the threshold",
        RecoveryAction::ExcludeFrame,
    )
    .unwrap();
    let report = QualityReport::new(FRAME_COUNT, frames, vec![anomaly]).unwrap();
    write_report(&assets, &report);

    let error = expect_failure(run(&params, &CancellationToken::new(), &NoReporter));

    assert_eq!(error.code, ErrorCode::MediaInvalid);
    assert_eq!(error.summary, "素材包仍有异常帧");
    assert!(!manifest_path(&params).exists());
}

#[test]
fn a_report_that_did_not_accept_every_frame_is_refused() {
    let (_root, params) = project();
    let assets = params.project_dir.join("assets");
    let mut frames = frame_qualities(&assets);
    frames.pop();
    // `QualityReport::new` derives `accepted_count` from the entries it is
    // given, so two entries against a frame count of three is exactly the
    // "a frame was dropped and never recovered" state.
    let report = QualityReport::new(FRAME_COUNT, frames, Vec::new()).unwrap();
    write_report(&assets, &report);

    let error = expect_failure(run(&params, &CancellationToken::new(), &NoReporter));

    assert_eq!(error.code, ErrorCode::MediaInvalid);
    assert_eq!(error.summary, "仍有帧未被接受");
    assert!(error.detail.contains("2 of 3"), "{}", error.detail);
}

#[test]
fn a_missing_media_file_is_refused_by_name() {
    let (_root, params) = project();
    let audio = params.project_dir.join("assets").join("audio_16k_mono.wav");
    fs::remove_file(&audio).unwrap();

    let error = expect_failure(run(&params, &CancellationToken::new(), &NoReporter));

    assert_eq!(error.code, ErrorCode::MediaInvalid);
    assert_eq!(error.summary, "素材包缺少必需文件");
    assert!(
        error.detail.contains("audio_16k_mono.wav"),
        "{}",
        error.detail
    );
}

#[test]
fn a_feature_file_from_another_take_is_refused() {
    let (_root, params) = project();
    // Six tokens are wanted; 57 is 51 away, one token past the fitting limit.
    write_features(&params.project_dir.join("assets"), 57);

    let error = expect_failure(run(&params, &CancellationToken::new(), &NoReporter));

    assert_eq!(error.code, ErrorCode::MediaInvalid);
    assert_eq!(error.summary, "特征令牌数与帧数不匹配");
    assert!(error.detail.contains("57 tokens"), "{}", error.detail);
    assert!(error.detail.contains("need 6"), "{}", error.detail);
}

#[test]
fn a_missing_frame_reaches_the_scan() {
    let (_root, params) = project();
    let frame = params.project_dir.join("assets").join("frames/000001.jpg");
    fs::remove_file(&frame).unwrap();

    let error = expect_failure(run(&params, &CancellationToken::new(), &NoReporter));

    // Task 7 owns this wording; seeing it here proves the command runs the
    // scan rather than trusting the report.
    assert_eq!(error.summary, "抽出的帧不可用");
    assert!(!manifest_path(&params).exists());
}

#[test]
fn a_cancelled_token_writes_no_manifest() {
    let (_root, params) = project();
    let token = CancellationToken::new();
    token.cancel();

    let outcome = run(&params, &token, &NoReporter);

    assert!(
        matches!(outcome, CommandOutcome::Cancelled),
        "expected a cancelled run, got {outcome:?}"
    );
    assert!(!manifest_path(&params).exists());
}

#[test]
fn a_feature_file_two_tokens_short_is_padded_before_the_commit() {
    let (_root, params) = project();
    let assets = params.project_dir.join("assets");
    write_features(&assets, 4);

    let result = expect_completed(run(&params, &CancellationToken::new(), &NoReporter));

    assert_eq!(result["tokens"], 6);
    assert_eq!(result["token_adjustment"], 2);
    let path = assets.join("features").join("feather_hubert.f32");
    let matrix = read_feature_file(&path).unwrap();
    assert_eq!(matrix.tokens(), 6);
    assert_eq!(matrix.dims(), DIMS);
    // Padding is zero vectors, so the tail is distinguishable from data.
    assert!(
        matrix.values()[4 * DIMS..]
            .iter()
            .all(|value| *value == 0.0)
    );
}
