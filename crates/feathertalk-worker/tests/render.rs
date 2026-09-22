use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Duration;

use burn::optim::AdamConfig;
use feathertalk_domain::{
    ErrorCode, Progress, RenderParams, TaskError, TaskStage, TrainParams,
    TrainingMode as DomainTrainingMode, UnetVariant};
use feathertalk_export::ModelConfiguration;
use feathertalk_media::{CancellationToken, MediaToolchain};
use feathertalk_models::unet::OriginalUnetConfig;
use feathertalk_training::{
    CheckpointDescriptor, DATA_LOADER_STATE_SCHEMA_VERSION, DataLoaderConfig, DataLoaderState,
    Provenance, RandomAlgorithm, SamplingConfig, SamplingKind, TRAINING_STATE_SCHEMA_VERSION,
    TrainingCheckpointState, save_training_checkpoint};
use feathertalk_worker::{
    CommandOutcome, RenderDevice, RenderJob, TRAINING_SEED, TrainDevice,
    checkpoint_descriptor, execute_render, render_job, run_render, training_config};
use serde_json::Value;

#[path = "support/mod.rs"]
mod support;

use support::{
    MemorySinkFactory, Recorder, StubFrameReader, lock_render_tree, model, render_audio,
    render_model, render_tree};

/// The request a render command carries.
fn render_params(
    project_dir: &Path,
    checkpoint: PathBuf,
    output: PathBuf,
    max_output_frames: Option<u64>,
) -> RenderParams {
    RenderParams {
        project_dir: project_dir.to_path_buf(),
        checkpoint,
        audio: render_audio(project_dir),
        output,
        max_output_frames}
}

/// A job over a fixture project, carrying the checkpoint identity a real
/// training run would have written.
///
/// The test binary stands in for ffmpeg: the sink is a stub, so the path only has
/// to be an absolute file that exists.
fn job(
    project_dir: &Path,
    output: PathBuf,
    frame_count: u64,
    max_output_frames: Option<u64>,
) -> RenderJob {
    let params = render_params(
        project_dir,
        project_dir
            .join("models")
            .join("unet")
            .join("checkpoint-00000002"),
        output,
        max_output_frames,
    );
    let configuration = ModelConfiguration::original_unet(&OriginalUnetConfig::production());
    let descriptor = checkpoint_descriptor(&configuration).expect("the configuration serialises");
    let ffmpeg = std::env::current_exe().expect("the test binary knows its own path");
    render_job(&params, frame_count, &ffmpeg, descriptor, 1, 2).expect("the job is valid")
}

fn completed(outcome: CommandOutcome) -> Value {
    match outcome {
        CommandOutcome::Completed(Some(payload)) => payload,
        other => panic!("expected a completed outcome, got {other:?}")}
}

fn failed(outcome: CommandOutcome) -> TaskError {
    match outcome {
        CommandOutcome::Failed(error) => error,
        other => panic!("expected a failed outcome, got {other:?}")}
}

/// One reported frame, with the progress that goes with it.
fn frame_event(frame: u64, total: u64) -> (TaskStage, Option<Progress>) {
    (
        TaskStage::Rendering { frame, total },
        Some(Progress {
            completed: frame,
            total: Some(total)}),
    )
}

/// The staging file the factory was asked to write, once the render has ended.
fn staging(sinks: &MemorySinkFactory) -> Option<PathBuf> {
    sinks.staging.lock().expect("the factory is intact").clone()
}

fn written_frames(sinks: &MemorySinkFactory) -> usize {
    sinks.frames.lock().expect("the factory is intact").len()
}

#[test]
fn a_device_fault_after_encoding_discards_the_staged_video() {
    let (root, project) = render_tree(2, 2);
    let job = job(&project, root.path().join("render.mp4"), 2, Some(1));
    let device = RenderDevice::default();
    let sinks = MemorySinkFactory::default();
    let error = failed(run_render(
        &job,
        &render_model(&device),
        &device,
        &CancellationToken::new(),
        &support::health::PublicationFault::on_check(1),
        &StubFrameReader::default(),
        &sinks,
    ));
    assert_eq!(error.code, ErrorCode::GpuDeviceLost);
    assert_eq!(error.stage, TaskStage::Rendering { frame: 1, total: 1 });
    assert_eq!(written_frames(&sinks), 1);
    assert!(!job.request.output_path().exists());
    assert!(
        !staging(&sinks)
            .expect("the encoder finished its staging file")
            .exists()
    );
}

#[test]
fn a_render_reports_one_progress_event_per_frame() {
    let (root, project) = render_tree(2, 2);
    let job = job(&project, root.path().join("render.mp4"), 2, None);
    let device = RenderDevice::default();
    let model = render_model(&device);
    let token = CancellationToken::new();
    let reporter = Recorder::new();
    let reader = StubFrameReader::default();
    let sinks = MemorySinkFactory::default();

    let outcome = run_render(&job, &model, &device, &token, &reporter, &reader, &sinks);

    let payload = completed(outcome);
    assert_eq!(payload["frame_count"], 2);
    // One event per written frame, in order, with the total the job carries.
    let events = reporter.events();
    assert_eq!(
        events,
        vec![frame_event(1, 2), frame_event(2, 2)],
        "{events:?}"
    );
    assert!(job.request.output_path().is_file());
    assert_eq!(written_frames(&sinks), 2);
}

#[test]
fn a_cancelled_render_leaves_neither_a_video_nor_a_staging_file() {
    let (root, project) = render_tree(2, 2);
    let job = job(&project, root.path().join("render.mp4"), 2, None);
    let device = RenderDevice::default();
    let model = render_model(&device);
    let token = CancellationToken::new();
    // Cancelled once the first frame has been reported, so the second write is
    // the one that sees it.
    let reporter = Recorder::cancelling_after(1, token.clone());
    let reader = StubFrameReader::default();
    let sinks = MemorySinkFactory::default();

    let outcome = run_render(&job, &model, &device, &token, &reporter, &reader, &sinks);

    assert!(matches!(outcome, CommandOutcome::Cancelled), "{outcome:?}");
    assert_eq!(written_frames(&sinks), 1);
    assert!(!job.request.output_path().exists());
    // Inference's own guard removes what it staged; this is the assertion that
    // proves a cancelled render leaves no half-written file behind.
    let staged = staging(&sinks).expect("the encoder was started");
    assert!(!staged.exists(), "{}", staged.display());
}

#[test]
fn a_render_that_cannot_start_fails_before_the_first_frame() {
    let (root, project) = render_tree(2, 2);
    let output = root.path().join("render.mp4");
    let job = job(&project, output.clone(), 2, None);
    // Written after the job was assembled, so the executor's own destination
    // check is the one that rejects it.
    std::fs::write(&output, b"sentinel").expect("the destination is written");
    let device = RenderDevice::default();
    let model = render_model(&device);
    let token = CancellationToken::new();
    let reporter = Recorder::new();
    let reader = StubFrameReader::default();
    let sinks = MemorySinkFactory::default();

    let outcome = run_render(&job, &model, &device, &token, &reporter, &reader, &sinks);

    let error = failed(outcome);
    assert_eq!(error.code, ErrorCode::MediaInvalid);
    // Nothing was written, so the failure belongs to preparation.
    assert_eq!(error.stage, TaskStage::Preparing);
    assert!(reporter.events().is_empty(), "{:?}", reporter.events());
    // The file that was already there is untouched.
    assert_eq!(
        std::fs::read(&output).expect("the destination is readable"),
        b"sentinel"
    );
}

#[test]
fn a_failure_mid_render_reports_the_frame_it_reached() {
    let (root, project) = render_tree(2, 2);
    let job = job(&project, root.path().join("render.mp4"), 2, None);
    let device = RenderDevice::default();
    let model = render_model(&device);
    let token = CancellationToken::new();
    let reporter = Recorder::new();
    // The first output frame reuses the frame the executor prefetched, so the
    // second source read is the first one that can fail.
    let reader = StubFrameReader::failing_at(1);
    let sinks = MemorySinkFactory::default();

    let outcome = run_render(&job, &model, &device, &token, &reporter, &reader, &sinks);

    let error = failed(outcome);
    assert_eq!(error.code, ErrorCode::WorkerCrashed);
    // The stage of the last frame that got through, rather than `Preparing`.
    assert_eq!(error.stage, TaskStage::Rendering { frame: 1, total: 2 });
    assert!(!job.request.output_path().exists());
    let staged = staging(&sinks).expect("the encoder was started");
    assert!(!staged.exists(), "{}", staged.display());
}

#[test]
fn a_render_cancelled_before_it_starts_never_opens_the_encoder() {
    let (root, project) = render_tree(2, 2);
    let job = job(&project, root.path().join("render.mp4"), 2, None);
    let device = RenderDevice::default();
    let model = render_model(&device);
    let token = CancellationToken::new();
    token.cancel();
    let reporter = Recorder::new();
    let reader = StubFrameReader::default();
    let sinks = MemorySinkFactory::default();

    let outcome = run_render(&job, &model, &device, &token, &reporter, &reader, &sinks);

    assert!(matches!(outcome, CommandOutcome::Cancelled), "{outcome:?}");
    // The sink was never started, so there is nothing to clean up and nothing to
    // report.
    assert!(staging(&sinks).is_none());
    assert_eq!(written_frames(&sinks), 0);
    assert!(reporter.events().is_empty(), "{:?}", reporter.events());
    assert!(!job.request.output_path().exists());
}

#[test]
fn a_capped_render_counts_up_to_the_cap() {
    let (root, project) = render_tree(2, 2);
    // Two frames in the project, one of them asked for.
    let job = job(&project, root.path().join("render.mp4"), 2, Some(1));
    let device = RenderDevice::default();
    let model = render_model(&device);
    let token = CancellationToken::new();
    let reporter = Recorder::new();
    let reader = StubFrameReader::default();
    let sinks = MemorySinkFactory::default();

    let outcome = run_render(&job, &model, &device, &token, &reporter, &reader, &sinks);

    let payload = completed(outcome);
    assert_eq!(payload["frame_count"], 1);
    assert_eq!(payload["source_frame_count"], 2);
    assert_eq!(payload["max_output_frames"], 1);
    // The denominator is the cap, not the project: a progress bar that ended at
    // one half would be a lie.
    let events = reporter.events();
    assert_eq!(events, vec![frame_event(1, 1)], "{events:?}");
    assert_eq!(written_frames(&sinks), 1);
}

/// The kind no `render_variant` will ever resolve.
const UNKNOWN_MODEL_KIND: &str = "fancy_unet";

/// Neither tool ever runs here -- the sink is in memory -- so the toolchain only
/// has to hold two absolute paths.
fn toolchain(root: &Path) -> MediaToolchain {
    MediaToolchain::new(
        root.join("ffmpeg.exe"),
        root.join("ffprobe.exe"),
        Duration::from_secs(30),
    )
    .expect("two absolute paths make a toolchain")
}

/// A directory that gets past admission -- an absolute path holding a
/// `project.json` -- and no further: the manifest is deliberately not JSON.
fn bare_project(root: &Path) -> PathBuf {
    let project = root.join("bare");
    std::fs::create_dir_all(&project).expect("the project directory is created");
    std::fs::write(project.join("project.json"), b"not json").expect("the manifest is written");
    project
}

/// The training state a checkpoint carries.
///
/// Ported from `feathertalk-training/tests/checkpoint_atomicity.rs`: every field
/// here is one `TrainingCheckpointState::validate` insists on, and the three
/// values it cross-checks come from `training_config` rather than from a literal.
fn state(project_dir: &Path, frame_count: u64) -> TrainingCheckpointState {
    let params = TrainParams {
        project_dir: project_dir.to_path_buf(),
        mode: DomainTrainingMode::Baseline,
        variant: UnetVariant::OriginalUnet,
        epochs: 1,
        batch_size: 1,
        resume: false};
    let config = training_config(&params);
    let batch_size = config.batch_size;
    let temporal_stride = config.temporal_stride;
    TrainingCheckpointState {
        schema_version: TRAINING_STATE_SCHEMA_VERSION,
        epoch: 1,
        global_step: 2,
        random_seed: TRAINING_SEED,
        data_loader: DataLoaderState {
            schema_version: DATA_LOADER_STATE_SCHEMA_VERSION,
            random_algorithm: RandomAlgorithm::Splitmix64FisherYatesV1,
            config: DataLoaderConfig {
                batch_size,
                seed: TRAINING_SEED,
                sampling: SamplingConfig {
                    kind: SamplingKind::SingleFrame,
                    temporal_stride}},
            frame_count,
            epoch: 1,
            next_position: 0},
        training_config: config,
        asset_provenance: Provenance {
            entries: BTreeMap::new()},
        model_provenance: Provenance {
            entries: BTreeMap::new()}}
}

/// Writes a real checkpoint whose manifest names a model kind this worker cannot
/// build.
///
/// The real writer rather than four hand-written files: the manifest declares the
/// size and digest of everything beside it, and only the writer knows how to make
/// those agree.
fn unknown_kind_checkpoint(directory: &Path, project_dir: &Path, frame_count: u64) {
    let device = TrainDevice::default();
    save_training_checkpoint::<_, _>(
        directory,
        &model(&device),
        &AdamConfig::new().init(),
        CheckpointDescriptor::new(UNKNOWN_MODEL_KIND, "v1", "0".repeat(64)),
        state(project_dir, frame_count),
    )
    .expect("the checkpoint is written");
}

#[test]
fn a_relative_checkpoint_is_refused_before_anything_is_read() {
    let root = tempfile::tempdir().expect("the temporary root is created");
    let project = bare_project(root.path());
    let params = render_params(
        &project,
        PathBuf::from("models/unet/checkpoint-00000002"),
        root.path().join("render.mp4"),
        None,
    );
    let token = CancellationToken::new();
    let reporter = Recorder::new();
    let sinks = MemorySinkFactory::default();

    let outcome = execute_render(
        &params,
        &token,
        &reporter,
        &toolchain(root.path()),
        &StubFrameReader::default(),
        &sinks,
    );

    let error = failed(outcome);
    assert_eq!(error.code, ErrorCode::MediaInvalid);
    assert_eq!(error.stage, TaskStage::Preparing);
    // The command announced the stage it was in and then stopped: nothing was
    // rendered, so nothing reported a frame.
    let events = reporter.events();
    assert_eq!(events, vec![(TaskStage::Preparing, None)], "{events:?}");
    assert!(staging(&sinks).is_none());
}

#[test]
fn a_project_that_is_not_locked_is_refused_by_the_project_crate() {
    let root = tempfile::tempdir().expect("the temporary root is created");
    let project = bare_project(root.path());
    let params = render_params(
        &project,
        project.join("checkpoint-00000002"),
        root.path().join("render.mp4"),
        None,
    );
    let token = CancellationToken::new();
    let reporter = Recorder::new();
    let sinks = MemorySinkFactory::default();

    let outcome = execute_render(
        &params,
        &token,
        &reporter,
        &toolchain(root.path()),
        &StubFrameReader::default(),
        &sinks,
    );

    let error = failed(outcome);
    assert_eq!(error.code, ErrorCode::MediaInvalid);
    assert_eq!(error.stage, TaskStage::Preparing);
    // The wording belongs to `project_task_error`, which is how this asserts the
    // rejection came from the project crate rather than from inference.
    assert_eq!(error.summary, "项目清单 JSON 格式错误");
    assert!(staging(&sinks).is_none());
}

#[test]
fn a_checkpoint_of_an_unknown_model_kind_is_refused_by_name() {
    let (root, project) = render_tree(2, 2);
    lock_render_tree(&project, 2);
    let checkpoint = project
        .join("models")
        .join("unet")
        .join("checkpoint-00000002");
    unknown_kind_checkpoint(&checkpoint, &project, 2);
    let params = render_params(&project, checkpoint, root.path().join("render.mp4"), None);
    let token = CancellationToken::new();
    let reporter = Recorder::new();
    let sinks = MemorySinkFactory::default();

    let outcome = execute_render(
        &params,
        &token,
        &reporter,
        &toolchain(root.path()),
        &StubFrameReader::default(),
        &sinks,
    );

    let error = failed(outcome);
    assert_eq!(error.code, ErrorCode::ModelIncompatible);
    assert_eq!(error.stage, TaskStage::Preparing);
    // The kind that was read is in the detail, so the operator learns which name
    // this worker could not build.
    assert!(
        error.detail.contains(UNKNOWN_MODEL_KIND),
        "{}",
        error.detail
    );
    assert!(staging(&sinks).is_none());
}

#[test]
fn a_zero_frame_cap_is_refused() {
    let root = tempfile::tempdir().expect("the temporary root is created");
    let project = bare_project(root.path());
    let params = render_params(
        &project,
        project.join("checkpoint-00000002"),
        root.path().join("render.mp4"),
        Some(0),
    );
    let token = CancellationToken::new();
    let reporter = Recorder::new();
    let sinks = MemorySinkFactory::default();

    let outcome = execute_render(
        &params,
        &token,
        &reporter,
        &toolchain(root.path()),
        &StubFrameReader::default(),
        &sinks,
    );

    let error = failed(outcome);
    assert_eq!(error.code, ErrorCode::MediaInvalid);
    // Refused before the project manifest was read, which is why an unlocked
    // project still reports the cap rather than the manifest.
    assert_eq!(error.summary, "最大输出帧数必须大于 0");
    assert!(staging(&sinks).is_none());
}

#[test]
fn a_cancelled_command_reads_no_checkpoint() {
    let root = tempfile::tempdir().expect("the temporary root is created");
    let project = bare_project(root.path());
    let params = render_params(
        &project,
        project.join("checkpoint-00000002"),
        root.path().join("render.mp4"),
        None,
    );
    let token = CancellationToken::new();
    token.cancel();
    let reporter = Recorder::new();
    let sinks = MemorySinkFactory::default();

    let outcome = execute_render(
        &params,
        &token,
        &reporter,
        &toolchain(root.path()),
        &StubFrameReader::default(),
        &sinks,
    );

    assert!(matches!(outcome, CommandOutcome::Cancelled), "{outcome:?}");
    // Not even a stage: a task cancelled in the queue never entered one.
    assert!(reporter.events().is_empty(), "{:?}", reporter.events());
    assert!(staging(&sinks).is_none());
}
