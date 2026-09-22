#[path = "support/health.rs"]
mod health;

use std::{
    fs,
    path::{Path, PathBuf},
    sync::Mutex};

use feathertalk_audio::{AudioError, ChunkEncoder, expected_hubert_frames};
use feathertalk_domain::{ErrorCode, ExtractFeaturesParams, Progress, TaskError, TaskStage};
use feathertalk_media::CancellationToken;
use feathertalk_worker::{CommandOutcome, NoReporter, TaskReporter, execute_extract_features};
use tempfile::TempDir;

/// Stands in for the digest `FeatureModel` reads out of the package manifest.
const MODEL_SHA256: &str = "a1b2c3d4a1b2c3d4a1b2c3d4a1b2c3d4a1b2c3d4a1b2c3d4a1b2c3d4a1b2c3d4";

/// An encoder that records the chunks it was asked for and answers with the
/// token count the plan expects, so a test can pin the plumbing without
/// weights. `cancel_on_first_chunk` cancels from inside the first call, which
/// is the only way to reach the seam between two chunks.
struct FakeEncoder {
    dims: usize,
    chunks: Vec<usize>,
    cancel_on_first_chunk: Option<CancellationToken>}

impl FakeEncoder {
    fn new(dims: usize) -> Self {
        Self {
            dims,
            chunks: Vec::new(),
            cancel_on_first_chunk: None}
    }
}

impl ChunkEncoder for FakeEncoder {
    fn output_dim(&self) -> usize {
        self.dims
    }

    fn encode(&mut self, chunk_index: usize, samples: &[f32]) -> Result<Vec<f32>, AudioError> {
        self.chunks.push(chunk_index);
        if chunk_index == 0
            && let Some(token) = &self.cancel_on_first_chunk
        {
            token.cancel();
        }
        Ok(vec![0.5; expected_hubert_frames(samples.len()) * self.dims])
    }
}

struct Recorder {
    events: Mutex<Vec<(TaskStage, Option<Progress>)>>}

impl Recorder {
    fn new() -> Self {
        Self {
            events: Mutex::new(Vec::new())}
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
/// `project.json`, with the normalised wav inside `assets/`.
fn project(samples: &[i16]) -> (TempDir, ExtractFeaturesParams) {
    let root = tempfile::tempdir().unwrap();
    let project_dir = root.path().join("project");
    let assets = project_dir.join("assets");
    fs::create_dir_all(&assets).unwrap();
    // Only the file's presence is checked; nothing here parses the manifest.
    fs::write(project_dir.join("project.json"), b"{}").unwrap();
    let audio = assets.join("audio_16k_mono.wav");
    write_wav(&audio, samples);
    (root, ExtractFeaturesParams { project_dir, audio })
}

/// The canonical 44-byte header `normalize_media` produces, followed by the
/// samples: 16 kHz, mono, 16-bit PCM.
fn write_wav(path: &Path, samples: &[i16]) {
    let payload = samples.len() as u32 * 2;
    let mut bytes = Vec::with_capacity(44 + payload as usize);
    bytes.extend(b"RIFF");
    bytes.extend((36 + payload).to_le_bytes());
    bytes.extend(b"WAVEfmt ");
    bytes.extend(16u32.to_le_bytes());
    bytes.extend(1u16.to_le_bytes());
    bytes.extend(1u16.to_le_bytes());
    bytes.extend(16_000u32.to_le_bytes());
    bytes.extend(32_000u32.to_le_bytes());
    bytes.extend(2u16.to_le_bytes());
    bytes.extend(16u16.to_le_bytes());
    bytes.extend(b"data");
    bytes.extend(payload.to_le_bytes());
    bytes.extend(samples.iter().copied().flat_map(i16::to_le_bytes));
    fs::write(path, bytes).unwrap();
}

/// A ramp rather than a constant, because `normalize_waveform` refuses a
/// waveform with no dynamic range.
fn ramp(count: usize) -> Vec<i16> {
    (0..count)
        .map(|index| (index % 2_000) as i16 - 1_000)
        .collect()
}

fn run(
    params: &ExtractFeaturesParams,
    token: &CancellationToken,
    reporter: &dyn TaskReporter,
    encoder: &mut FakeEncoder,
) -> CommandOutcome {
    execute_extract_features(params, token, reporter, encoder, MODEL_SHA256)
}

fn progress(completed: u64, total: u64) -> Option<Progress> {
    Some(Progress {
        completed,
        total: Some(total)})
}

fn expect_failure(outcome: CommandOutcome) -> TaskError {
    match outcome {
        CommandOutcome::Failed(error) => error,
        other => panic!("expected a failure, got {other:?}")}
}

#[test]
fn a_device_fault_before_feature_publication_keeps_the_destination_absent() {
    let (_root, params) = project(&ramp(3_200));
    let error = expect_failure(run(
        &params,
        &CancellationToken::new(),
        &health::PublicationFault::on_check(1),
        &mut FakeEncoder::new(4),
    ));
    assert_eq!(error.code, ErrorCode::GpuDeviceLost);
    assert_eq!(error.stage, TaskStage::ExtractingFeatures);
    assert!(
        !params
            .project_dir
            .join("assets/features/feather_hubert.f32")
            .exists()
    );
}

#[test]
fn a_two_second_wav_becomes_an_even_token_feature_file() {
    // 32 000 samples at 16 kHz: (32 000 - 80) / 320 is 99 frames.
    let (_root, params) = project(&ramp(32_000));
    let token = CancellationToken::new();
    let recorder = Recorder::new();
    let mut encoder = FakeEncoder::new(4);

    let result = match run(&params, &token, &recorder, &mut encoder) {
        CommandOutcome::Completed(Some(result)) => result,
        other => panic!("expected a completed command, got {other:?}")};

    let features = params.project_dir.join("assets").join("features");
    let file = features.join("feather_hubert.f32");
    assert_eq!(result["output_dir"], features.display().to_string());
    assert_eq!(result["feature_file"], file.display().to_string());
    // 99 is odd, so the last token has no video frame to pair with.
    assert_eq!(result["tokens"], 98);
    assert_eq!(result["dims"], 4);
    assert_eq!(result["frame_count"], 49);
    // The 44-byte header plus 98 * 4 f32 values.
    assert_eq!(result["bytes"], 1_612);
    assert_eq!(result["model_sha256"], MODEL_SHA256);
    assert_eq!(fs::metadata(&file).unwrap().len(), 1_612);
    // `DEFAULT_CHUNK_SAMPLES` is 320 000, so two seconds are one chunk.
    assert_eq!(encoder.chunks, vec![0]);
    assert_eq!(
        recorder.events(),
        vec![
            (TaskStage::Preparing, None),
            (TaskStage::ExtractingFeatures, progress(1, 1)),
        ]
    );
}

#[test]
fn a_long_wav_reports_progress_for_every_chunk() {
    // 640 000 samples are two whole chunks and 1 999 frames.
    let (_root, params) = project(&ramp(640_000));
    let token = CancellationToken::new();
    let recorder = Recorder::new();
    let mut encoder = FakeEncoder::new(4);

    let result = match run(&params, &token, &recorder, &mut encoder) {
        CommandOutcome::Completed(Some(result)) => result,
        other => panic!("expected a completed command, got {other:?}")};

    assert_eq!(result["tokens"], 1_998);
    assert_eq!(result["frame_count"], 999);
    assert_eq!(result["bytes"], 32_012);
    // The chunks overlap by `HUBERT_KERNEL - HUBERT_STRIDE` samples, so the
    // first one is 320 080 samples long and the second one 320 000.
    assert_eq!(encoder.chunks, vec![0, 1]);
    assert_eq!(
        recorder.events(),
        vec![
            (TaskStage::Preparing, None),
            (TaskStage::ExtractingFeatures, progress(1, 2)),
            (TaskStage::ExtractingFeatures, progress(2, 2)),
        ]
    );
}

#[test]
fn a_cancel_between_chunks_leaves_no_feature_file() {
    let (_root, params) = project(&ramp(640_000));
    let token = CancellationToken::new();
    let mut encoder = FakeEncoder::new(4);
    encoder.cancel_on_first_chunk = Some(token.clone());

    let outcome = run(&params, &token, &NoReporter, &mut encoder);

    assert!(
        matches!(outcome, CommandOutcome::Cancelled),
        "expected a cancelled run, got {outcome:?}"
    );
    assert_eq!(encoder.chunks, vec![0]);
    assert!(!params.project_dir.join("assets").join("features").exists());
}

#[test]
fn relative_paths_are_rejected_before_anything_is_touched() {
    let (_root, params) = project(&ramp(32_000));
    let token = CancellationToken::new();
    let mut encoder = FakeEncoder::new(4);
    let relative_dir = ExtractFeaturesParams {
        project_dir: PathBuf::from("project"),
        audio: params.audio.clone()};
    let relative_audio = ExtractFeaturesParams {
        project_dir: params.project_dir.clone(),
        audio: PathBuf::from("assets/audio_16k_mono.wav")};

    let first = expect_failure(run(&relative_dir, &token, &NoReporter, &mut encoder));
    let second = expect_failure(run(&relative_audio, &token, &NoReporter, &mut encoder));

    assert_eq!(first.code, ErrorCode::MediaInvalid);
    assert_eq!(first.stage, TaskStage::Preparing);
    assert_eq!(first.summary, "工程目录必须是绝对路径");
    assert_eq!(second.code, ErrorCode::MediaInvalid);
    assert_eq!(second.stage, TaskStage::Preparing);
    assert_eq!(second.summary, "音频文件必须是绝对路径");
    assert!(encoder.chunks.is_empty());
}

#[test]
fn a_project_without_a_manifest_is_rejected_before_the_audio_is_read() {
    let (_root, params) = project(&ramp(32_000));
    fs::remove_file(params.project_dir.join("project.json")).unwrap();
    let token = CancellationToken::new();
    let mut encoder = FakeEncoder::new(4);

    let error = expect_failure(run(&params, &token, &NoReporter, &mut encoder));

    assert_eq!(error.code, ErrorCode::MediaInvalid);
    assert_eq!(error.stage, TaskStage::Preparing);
    assert_eq!(error.summary, "工程目录缺少 project.json");
    assert!(encoder.chunks.is_empty());
}

#[test]
fn a_short_audio_file_is_rejected_with_its_frame_count() {
    // 400 samples are exactly one frame, and one token has no pair.
    let (_root, params) = project(&ramp(400));
    let token = CancellationToken::new();
    let mut encoder = FakeEncoder::new(4);

    let error = expect_failure(run(&params, &token, &NoReporter, &mut encoder));

    assert_eq!(error.code, ErrorCode::MediaInvalid);
    assert_eq!(error.stage, TaskStage::Preparing);
    assert_eq!(error.summary, "音频太短，无法提取特征");
    assert!(
        error
            .detail
            .contains("400 samples yield 1 FeatherHuBERT frame"),
        "detail was {}",
        error.detail
    );
    assert!(encoder.chunks.is_empty());
}

#[test]
fn an_existing_feature_file_is_never_overwritten() {
    let (_root, params) = project(&ramp(32_000));
    let features = params.project_dir.join("assets").join("features");
    fs::create_dir_all(&features).unwrap();
    let existing = features.join("feather_hubert.f32");
    fs::write(&existing, b"old").unwrap();
    let token = CancellationToken::new();
    let mut encoder = FakeEncoder::new(4);

    let error = expect_failure(run(&params, &token, &NoReporter, &mut encoder));

    assert_eq!(error.code, ErrorCode::MediaInvalid);
    assert_eq!(error.stage, TaskStage::Preparing);
    assert_eq!(error.summary, "特征文件已存在");
    assert_eq!(fs::read(&existing).unwrap(), b"old".to_vec());
    assert!(encoder.chunks.is_empty());
}

#[test]
fn a_file_that_is_not_a_wav_is_rejected_as_invalid_media() {
    let (_root, params) = project(&ramp(32_000));
    fs::write(&params.audio, b"this is not a wav file").unwrap();
    let token = CancellationToken::new();
    let mut encoder = FakeEncoder::new(4);

    let error = expect_failure(run(&params, &token, &NoReporter, &mut encoder));

    assert_eq!(error.code, ErrorCode::MediaInvalid);
    assert_eq!(error.stage, TaskStage::Preparing);
    assert_eq!(error.summary, "音频文件不是有效的 WAV");
    assert!(encoder.chunks.is_empty());
}
