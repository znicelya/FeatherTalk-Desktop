use std::{
    fs,
    io::{self, BufRead, Read, Write},
    path::PathBuf,
    sync::{
        Arc, Mutex,
        mpsc::{self, Receiver, Sender},
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

use feathertalk_domain::{
    CancelFrame, ClientFrame, DomainError, ErrorCode, Event, ExtractFeaturesParams,
    ExtractFramesParams, NormalizeMediaParams, PROTOCOL_VERSION, ProbeMediaParams, Progress,
    ProjectDirParams, RenderParams, Request, ServerFrame, ShutdownFrame, StartFrame, TaskId,
    TaskKind, TaskStage, TrainParams, TrainingMode, UnetVariant, decode_line, encode_line,
};
use feathertalk_media::{CancellationToken, CommandSpec, MediaError, ProcessOutput, ProcessRunner};
use feathertalk_worker::{
    CPU_ADAPTER_ID, CommandOutcome, JobExecutor, NoReporter, TaskReporter, WorkerConfig,
    execute_with_runner, serve_with_executor,
};

/// An output sink the test can read while the worker is still writing to it.
#[derive(Clone, Default)]
struct SharedSink(Arc<Mutex<Vec<u8>>>);

impl Write for SharedSink {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.0.lock().unwrap().extend_from_slice(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

impl SharedSink {
    /// Decode every frame that has already been fully written.
    ///
    /// A trailing fragment without its `\n` is skipped rather than decoded, so
    /// polling never races a half-written line. Every decoded frame is
    /// validated here, which is what asserts the runtime never emits a frame
    /// that fails `ServerFrame::validate`.
    fn frames(&self) -> Vec<ServerFrame> {
        let bytes = self.0.lock().unwrap().clone();
        let text = String::from_utf8(bytes).unwrap();
        let complete = match text.rfind('\n') {
            Some(index) => &text[..=index],
            None => "",
        };
        complete
            .lines()
            .filter(|line| !line.trim().is_empty())
            .map(|line| {
                let frame: ServerFrame = decode_line(line).unwrap();
                frame.validate().unwrap();
                frame
            })
            .collect()
    }
}

/// A `BufRead` whose bytes arrive from the test thread. Dropping the sender is
/// end-of-stream, which is how the tests model a closed stdin.
struct ChannelReader {
    receiver: Receiver<Vec<u8>>,
    buffer: Vec<u8>,
    cursor: usize,
    closed: bool,
}

impl ChannelReader {
    fn new(receiver: Receiver<Vec<u8>>) -> Self {
        Self {
            receiver,
            buffer: Vec::new(),
            cursor: 0,
            closed: false,
        }
    }
}

impl BufRead for ChannelReader {
    fn fill_buf(&mut self) -> io::Result<&[u8]> {
        while self.cursor >= self.buffer.len() {
            if self.closed {
                return Ok(&[]);
            }
            match self.receiver.recv() {
                Ok(chunk) => {
                    self.buffer = chunk;
                    self.cursor = 0;
                }
                Err(_) => {
                    self.closed = true;
                    return Ok(&[]);
                }
            }
        }
        Ok(&self.buffer[self.cursor..])
    }

    fn consume(&mut self, amount: usize) {
        self.cursor += amount;
    }
}

impl Read for ChannelReader {
    fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
        let available = self.fill_buf()?;
        let count = available.len().min(out.len());
        out[..count].copy_from_slice(&available[..count]);
        self.consume(count);
        Ok(count)
    }
}

struct Harness {
    input: Option<Sender<Vec<u8>>>,
    sink: SharedSink,
    worker: Option<JoinHandle<Result<(), DomainError>>>,
}

impl Harness {
    fn start(config: WorkerConfig, executor: JobExecutor) -> Self {
        let (input, receiver) = mpsc::channel::<Vec<u8>>();
        let sink = SharedSink::default();
        let worker_sink = sink.clone();
        let worker = thread::spawn(move || {
            serve_with_executor(ChannelReader::new(receiver), worker_sink, &config, executor)
        });
        Self {
            input: Some(input),
            sink,
            worker: Some(worker),
        }
    }

    fn send(&self, frame: &ClientFrame) {
        let mut line = encode_line(frame).unwrap().into_bytes();
        line.push(b'\n');
        self.input.as_ref().unwrap().send(line).unwrap();
    }

    fn send_raw(&self, line: &str) {
        self.input
            .as_ref()
            .unwrap()
            .send(format!("{line}\n").into_bytes())
            .unwrap();
    }

    fn frames(&self) -> Vec<ServerFrame> {
        self.sink.frames()
    }

    fn wait_for(
        &self,
        description: &str,
        predicate: impl Fn(&[ServerFrame]) -> bool,
    ) -> Vec<ServerFrame> {
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            let frames = self.frames();
            if predicate(&frames) {
                return frames;
            }
            assert!(
                Instant::now() < deadline,
                "timed out waiting for {description}; frames so far: {frames:?}"
            );
            thread::sleep(Duration::from_millis(10));
        }
    }

    /// Close the input stream, wait for the worker to exit, and return every
    /// frame it wrote.
    fn finish(mut self) -> Vec<ServerFrame> {
        self.input = None;
        self.worker.take().unwrap().join().unwrap().unwrap();
        self.frames()
    }
}

fn task(suffix: &str) -> TaskId {
    TaskId::parse(&format!("1787900000000-{suffix}")).unwrap()
}

fn start(task_id: &TaskId, request: Request) -> ClientFrame {
    ClientFrame::Start(StartFrame {
        protocol_version: PROTOCOL_VERSION,
        task_id: task_id.clone(),
        request,
    })
}

fn cancel(task_id: &TaskId) -> ClientFrame {
    ClientFrame::Cancel(CancelFrame {
        protocol_version: PROTOCOL_VERSION,
        task_id: task_id.clone(),
    })
}

fn shutdown() -> ClientFrame {
    ClientFrame::Shutdown(ShutdownFrame {
        protocol_version: PROTOCOL_VERSION,
    })
}

fn validate_project(dir: &str) -> Request {
    Request::ValidateProject(ProjectDirParams {
        project_dir: PathBuf::from(dir),
    })
}

fn train_request() -> Request {
    Request::Train(TrainParams {
        project_dir: PathBuf::from("C:/tmp/project"),
        mode: TrainingMode::Baseline,
        variant: UnetVariant::OriginalUnet,
        epochs: 1,
        batch_size: 1,
        resume: false,
    })
}

fn render_request() -> Request {
    Request::Render(RenderParams {
        project_dir: PathBuf::from("C:/tmp/project"),
        checkpoint: PathBuf::from("C:/tmp/project/models/unet/checkpoint-00000004"),
        audio: PathBuf::from("C:/tmp/project/assets/audio_16k_mono.wav"),
        output: PathBuf::from("C:/tmp/preview.mp4"),
        max_output_frames: None,
    })
}

fn extract_frames_request() -> Request {
    Request::ExtractFrames(ExtractFramesParams {
        project_dir: PathBuf::from("C:/tmp/project"),
        video: PathBuf::from("C:/tmp/project/assets/video_25fps.mp4"),
    })
}

fn extract_features_request() -> Request {
    Request::ExtractFeatures(ExtractFeaturesParams {
        project_dir: PathBuf::from("C:/tmp/project"),
        audio: PathBuf::from("C:/tmp/project/assets/audio_16k_mono.wav"),
    })
}

fn lock_asset_package_request() -> Request {
    Request::LockAssetPackage(ProjectDirParams {
        project_dir: PathBuf::from("C:/tmp/project"),
    })
}

fn absolute(name: &str) -> String {
    std::env::current_dir()
        .unwrap()
        .join(name)
        .display()
        .to_string()
}

/// A configuration whose media toolchain is accepted, so `probe_media` is
/// supported. The paths never have to exist: `MediaToolchain::new` only
/// requires absolute paths.
fn media_config() -> WorkerConfig {
    WorkerConfig::from_values(
        Some(absolute("ffprobe-test")),
        Some(absolute("ffmpeg-test")),
        None,
    )
}

/// No media environment at all: `probe_media` is unsupported and there is no
/// rejection reason to report.
fn bare_config() -> WorkerConfig {
    WorkerConfig::from_values(None, None, None)
}

/// Relative paths are rejected by `MediaToolchain::new`, so this configuration
/// carries a rejection reason.
fn broken_config() -> WorkerConfig {
    WorkerConfig::from_values(Some("ffprobe".to_owned()), Some("ffmpeg".to_owned()), None)
}

/// Media and models both resolve, so `extract_frames` reaches the executor.
fn full_config() -> WorkerConfig {
    WorkerConfig::from_values_with_models(
        Some(absolute("ffprobe-test")),
        Some(absolute("ffmpeg-test")),
        None,
        Some(absolute("scrfd-test")),
        Some(absolute("pfld-test")),
    )
}

/// Every toolchain resolves, so `extract_features` and `lock_asset_package`
/// reach the executor as well.
fn every_toolchain_config() -> WorkerConfig {
    WorkerConfig::from_values_with_toolchains(
        Some(absolute("ffprobe-test")),
        Some(absolute("ffmpeg-test")),
        None,
        Some(absolute("scrfd-test")),
        Some(absolute("pfld-test")),
        Some(absolute("hubert-test")),
    )
}

fn instant_executor() -> JobExecutor {
    Box::new(|_request, _config, _token, _reporter| CommandOutcome::Completed(None))
}

/// Reports that the job started, then runs until it is cancelled.
fn blocking_executor(started: Sender<TaskId>) -> JobExecutor {
    Box::new(move |request, _config, token, _reporter| {
        let _ = request;
        started.send(task("0000000f")).unwrap();
        while !token.is_cancelled() {
            thread::sleep(Duration::from_millis(5));
        }
        CommandOutcome::Cancelled
    })
}

/// Reports that the job started, then waits for the test to release it. A
/// dropped release channel or a cancelled token ends the job as cancelled.
fn gated_executor(started: Sender<()>, release: Receiver<()>) -> JobExecutor {
    Box::new(move |_request, _config, token, _reporter| {
        started.send(()).unwrap();
        if release.recv().is_err() || token.is_cancelled() {
            return CommandOutcome::Cancelled;
        }
        CommandOutcome::Completed(None)
    })
}

/// A process runner that behaves like an external tool killed by the
/// cancellation token: it blocks while the token is clear and then reports the
/// cancellation the real `CancellableProcessRunner` reports after a kill.
struct BlockingRunner {
    started: Mutex<Sender<()>>,
    token: CancellationToken,
}

impl ProcessRunner for BlockingRunner {
    fn run(&self, _spec: &CommandSpec, _timeout: Duration) -> Result<ProcessOutput, MediaError> {
        self.started.lock().unwrap().send(()).unwrap();
        while !self.token.is_cancelled() {
            thread::sleep(Duration::from_millis(5));
        }
        Err(MediaError::ToolCancelled {
            operation: "ffprobe",
        })
    }
}

fn blocking_probe_executor(started: Sender<()>) -> JobExecutor {
    Box::new(move |request, config, token, reporter| {
        let runner = BlockingRunner {
            started: Mutex::new(started.clone()),
            token: token.clone(),
        };
        execute_with_runner(request, config, token, reporter, &runner)
    })
}

/// Reports two progress events, then completes.
fn reporting_executor() -> JobExecutor {
    Box::new(|_request, _config, _token, reporter| {
        reporter.report(
            TaskStage::ExtractingFrames,
            Some(Progress {
                completed: 1,
                total: Some(2),
            }),
        );
        reporter.report(
            TaskStage::ExtractingAudio,
            Some(Progress {
                completed: 2,
                total: Some(2),
            }),
        );
        CommandOutcome::Completed(None)
    })
}

fn events(frames: &[ServerFrame]) -> Vec<&Event> {
    frames
        .iter()
        .filter_map(|frame| match frame {
            ServerFrame::Event(event) => Some(event),
            _ => None,
        })
        .collect()
}

fn stages(frames: &[ServerFrame]) -> Vec<(&str, &str)> {
    events(frames)
        .into_iter()
        .map(|event| (event.task_id.as_str(), event.stage.as_slug()))
        .collect()
}

fn rejections(frames: &[ServerFrame]) -> Vec<&str> {
    frames
        .iter()
        .filter_map(|frame| match frame {
            ServerFrame::Rejected(rejected) => Some(rejected.reason.as_str()),
            _ => None,
        })
        .collect()
}

/// One task at a time: no task may report a non-terminal stage while another
/// task is still in flight. A task that is cancelled while queued terminates
/// without ever being in flight, which is the one allowed exception.
fn assert_serialized(frames: &[ServerFrame]) {
    let mut in_flight: Option<TaskId> = None;
    for event in events(frames) {
        match (in_flight.as_ref(), event.stage.is_terminal()) {
            (Some(active), true) if *active == event.task_id => in_flight = None,
            (_, true) => assert_eq!(
                event.stage,
                TaskStage::Cancelled,
                "only a queued task may terminate without running: {frames:?}"
            ),
            (None, false) => in_flight = Some(event.task_id.clone()),
            (Some(_), false) => panic!("two tasks were in flight at once: {frames:?}"),
        }
    }
}

#[test]
fn ready_is_the_first_frame_and_reports_the_cpu_adapter() {
    let frames = Harness::start(bare_config(), instant_executor()).finish();

    let ServerFrame::Ready(ready) = &frames[0] else {
        panic!("the first frame must be ready: {frames:?}");
    };
    assert_eq!(ready.protocol_version, PROTOCOL_VERSION);
    assert_eq!(ready.adapters.len(), 1);
    assert_eq!(ready.adapters[0].id, CPU_ADAPTER_ID);
    assert_eq!(frames.len(), 1, "an idle worker emits nothing else");
}

#[test]
fn a_panicking_job_reports_its_stage_and_releases_the_adapter_for_the_next_job() {
    let harness = Harness::start(
        bare_config(),
        Box::new(|request, _config, _token, reporter| {
            if let Request::ValidateProject(params) = request
                && params.project_dir == std::path::Path::new("panic")
            {
                reporter.report(TaskStage::ExtractingFeatures, None);
                panic!("device execution failed");
            }
            CommandOutcome::Completed(None)
        }),
    );
    let first = task("00000071");
    let second = task("00000072");
    harness.send(&start(&first, validate_project("panic")));
    harness.send(&start(&second, validate_project("next")));
    harness.wait_for("the next task to finish after the panic", |frames| {
        events(frames)
            .iter()
            .any(|event| event.task_id == second && event.stage == TaskStage::Completed)
    });
    let frames = harness.finish();
    let first_terminal: Vec<_> = events(&frames)
        .into_iter()
        .filter(|event| event.task_id == first && event.stage.is_terminal())
        .collect();
    assert_eq!(first_terminal.len(), 1, "{frames:?}");
    let error = first_terminal[0].error.as_ref().unwrap();
    assert_eq!(error.code, ErrorCode::WorkerCrashed);
    assert_eq!(error.stage, TaskStage::ExtractingFeatures);
    assert!(error.detail.contains("device execution failed"));
    error.validate().unwrap();
    assert!(
        events(&frames)
            .iter()
            .any(|event| { event.task_id == second && event.stage == TaskStage::Completed })
    );
}

#[test]
fn an_unavailable_gpu_is_rejected_before_execution_and_cpu_commands_still_work() {
    let config = every_toolchain_config().with_compute_selection(Some("wgpu"), None);
    let harness = Harness::start(config, instant_executor());
    let gpu_task = task("00000073");
    let cpu_task = task("00000074");
    harness.send(&start(&gpu_task, extract_features_request()));
    harness.wait_for("the invalid GPU selection to be rejected", |frames| {
        !rejections(frames).is_empty()
    });
    harness.send(&start(&cpu_task, validate_project("project")));
    harness.wait_for("the CPU operation to complete", |frames| {
        events(frames)
            .iter()
            .any(|event| event.task_id == cpu_task && event.stage == TaskStage::Completed)
    });
    let frames = harness.finish();
    assert_eq!(rejections(&frames).len(), 1);
    assert!(rejections(&frames)[0].contains("FEATHERTALK_WORKER_BACKEND"));
    assert!(
        !events(&frames)
            .iter()
            .any(|event| event.task_id == gpu_task)
    );
}

#[test]
fn gpu_memory_metrics_reach_the_wire_alongside_progress() {
    let harness = Harness::start(
        bare_config(),
        Box::new(|_, _, _, reporter| {
            reporter.report_metrics(
                TaskStage::ExtractingFeatures,
                Some(Progress {
                    completed: 1,
                    total: Some(2),
                }),
                feathertalk_domain::Metrics {
                    vram_bytes: Some(4096),
                    ..Default::default()
                },
            );
            CommandOutcome::Completed(None)
        }),
    );
    let task_id = task("00000075");
    harness.send(&start(&task_id, validate_project("project")));
    harness.wait_for("completion", |frames| {
        events(frames)
            .iter()
            .any(|event| event.stage == TaskStage::Completed)
    });
    let frames = harness.finish();
    let all = events(&frames);
    let event = all
        .iter()
        .find(|event| event.stage == TaskStage::ExtractingFeatures)
        .unwrap();
    assert_eq!(event.metrics.vram_bytes, Some(4096));
    assert_eq!(
        event.progress,
        Some(Progress {
            completed: 1,
            total: Some(2)
        })
    );
}

#[test]
fn a_usable_media_toolchain_enables_probe_media_in_the_handshake() {
    let frames = Harness::start(media_config(), instant_executor()).finish();

    let ServerFrame::Ready(ready) = &frames[0] else {
        panic!("the first frame must be ready: {frames:?}");
    };
    assert_eq!(
        ready.supported_commands,
        vec![
            TaskKind::ValidateProject,
            TaskKind::InspectModel,
            TaskKind::ImportLegacyModel,
            TaskKind::MigrateLegacyFeatures,
            TaskKind::ExportModelPackage,
            TaskKind::ExportOnnx,
            TaskKind::ProbeMedia,
            TaskKind::NormalizeMedia,
            TaskKind::Render
        ]
    );
}

#[test]
fn a_rejected_media_configuration_leaves_probe_media_out_of_the_handshake() {
    let frames = Harness::start(broken_config(), instant_executor()).finish();

    let ServerFrame::Ready(ready) = &frames[0] else {
        panic!("the first frame must be ready: {frames:?}");
    };
    assert_eq!(
        ready.supported_commands,
        vec![
            TaskKind::ValidateProject,
            TaskKind::InspectModel,
            TaskKind::ImportLegacyModel,
            TaskKind::MigrateLegacyFeatures,
            TaskKind::ExportModelPackage,
            TaskKind::ExportOnnx,
        ]
    );
}

#[test]
fn an_unsupported_command_is_rejected_without_creating_a_task() {
    let harness = Harness::start(media_config(), instant_executor());
    harness.send(&start(&task("0000000a"), train_request()));
    let frames = harness.finish();

    let reasons = rejections(&frames);
    assert_eq!(reasons.len(), 1, "{frames:?}");
    assert!(reasons[0].contains("train"), "{}", reasons[0]);
    // The reason has to name the variable an operator can fix.
    assert!(
        reasons[0].contains("FEATHERTALK_WORKER_VGG19_DIR"),
        "{}",
        reasons[0]
    );
    assert!(
        events(&frames).is_empty(),
        "a rejected start creates no task"
    );
}

#[test]
fn a_rejected_training_configuration_explains_itself() {
    let config = WorkerConfig::from_values_with_training(
        None,
        None,
        None,
        None,
        None,
        None,
        Some("vgg19".to_owned()),
    );
    let harness = Harness::start(config, instant_executor());
    harness.send(&start(&task("0000000a"), train_request()));
    let frames = harness.finish();

    let reasons = rejections(&frames);
    assert_eq!(reasons.len(), 1, "{frames:?}");
    assert!(reasons[0].contains("train"), "{}", reasons[0]);
    // A relative path is a rejection, not an absence, so the reason quotes it.
    assert!(reasons[0].contains("absolute"), "{}", reasons[0]);
}

#[test]
fn a_render_request_without_media_tools_names_the_missing_variable() {
    let harness = Harness::start(bare_config(), instant_executor());
    harness.send(&start(&task("0000000b"), render_request()));
    let frames = harness.finish();

    let reasons = rejections(&frames);
    assert_eq!(reasons.len(), 1, "{frames:?}");
    assert!(reasons[0].contains("render"), "{}", reasons[0]);
    // The reason names the wall an operator hits first, which is the tool the
    // configuration reports as missing.
    assert!(
        reasons[0].contains("FEATHERTALK_WORKER_FFPROBE"),
        "{}",
        reasons[0]
    );
    assert!(
        events(&frames).is_empty(),
        "a rejected start creates no task"
    );
}

#[test]
fn a_render_request_with_only_ffprobe_names_ffmpeg() {
    // `MediaToolchain::new` wants both tools, so once ffprobe resolves the
    // second one becomes the wall.
    let config = WorkerConfig::from_values(Some(absolute("ffprobe-test")), None, None);
    let harness = Harness::start(config, instant_executor());
    harness.send(&start(&task("0000000b"), render_request()));
    let frames = harness.finish();

    let reasons = rejections(&frames);
    assert_eq!(reasons.len(), 1, "{frames:?}");
    assert!(reasons[0].contains("render"), "{}", reasons[0]);
    assert!(
        reasons[0].contains("FEATHERTALK_WORKER_FFMPEG"),
        "{}",
        reasons[0]
    );
}

/// The executor thread must be the named, big-stack one: burn's autodiff graph
/// for a 160x160 step does not fit in the 2 MiB a bare `thread::spawn` gives.
fn thread_name_executor(name: Sender<Option<String>>) -> JobExecutor {
    Box::new(move |_request, _config, _token, _reporter| {
        let _ = name.send(thread::current().name().map(str::to_owned));
        CommandOutcome::Completed(None)
    })
}

#[test]
fn commands_run_on_the_named_execution_thread() {
    let (name_tx, name_rx) = mpsc::channel();
    let harness = Harness::start(bare_config(), thread_name_executor(name_tx));
    harness.send(&start(
        &task("0000000a"),
        Request::ValidateProject(ProjectDirParams {
            project_dir: PathBuf::from("C:/tmp/project"),
        }),
    ));
    let observed = name_rx
        .recv_timeout(Duration::from_secs(5))
        .expect("the executor ran");
    harness.finish();

    assert_eq!(observed.as_deref(), Some("execution"));
}

#[test]
fn probe_media_is_rejected_when_the_media_toolchain_is_unavailable() {
    let harness = Harness::start(broken_config(), instant_executor());
    let request = Request::ProbeMedia(ProbeMediaParams {
        input: PathBuf::from("C:/tmp/input.mp4"),
    });
    harness.send(&start(&task("0000000a"), request));
    let frames = harness.finish();

    let reasons = rejections(&frames);
    assert_eq!(reasons.len(), 1, "{frames:?}");
    assert!(reasons[0].contains("probe_media"), "{}", reasons[0]);
    assert!(events(&frames).is_empty());
}

#[test]
fn normalize_media_is_rejected_when_the_media_toolchain_is_unavailable() {
    let harness = Harness::start(broken_config(), instant_executor());
    let request = Request::NormalizeMedia(NormalizeMediaParams {
        input: PathBuf::from("C:/tmp/clip.mp4"),
        output_dir: PathBuf::from("C:/tmp/assets"),
    });
    harness.send(&start(&task("00000023"), request));
    let frames = harness.finish();

    let reasons = rejections(&frames);
    assert_eq!(reasons.len(), 1, "{frames:?}");
    assert!(reasons[0].contains("normalize_media"), "{}", reasons[0]);
    // The reason has to name the variable an operator can fix.
    assert!(
        reasons[0].contains("FEATHERTALK_WORKER_FFPROBE"),
        "{}",
        reasons[0]
    );
    assert!(events(&frames).is_empty());
}

#[test]
fn a_protocol_version_mismatch_is_rejected() {
    let harness = Harness::start(bare_config(), instant_executor());
    harness.send(&ClientFrame::Start(StartFrame {
        protocol_version: PROTOCOL_VERSION - 1,
        task_id: task("0000000a"),
        request: validate_project("C:/tmp/project"),
    }));
    let frames = harness.finish();

    let reasons = rejections(&frames);
    assert_eq!(reasons.len(), 1, "{frames:?}");
    assert!(
        reasons[0].contains(&PROTOCOL_VERSION.to_string()),
        "{}",
        reasons[0]
    );
    assert!(events(&frames).is_empty());
}

#[test]
fn an_undecodable_line_is_rejected_without_ending_the_session() {
    let harness = Harness::start(bare_config(), instant_executor());
    harness.send_raw("{ not json");
    harness.send(&start(
        &task("0000000a"),
        validate_project("C:/tmp/project"),
    ));
    let frames = harness.finish();

    assert_eq!(rejections(&frames).len(), 1, "{frames:?}");
    assert_eq!(
        stages(&frames),
        vec![
            ("1787900000000-0000000a", "preparing"),
            ("1787900000000-0000000a", "completed"),
        ]
    );
}

#[test]
fn queued_tasks_run_one_after_another_in_arrival_order() {
    let (started_tx, started_rx) = mpsc::channel::<()>();
    let (release_tx, release_rx) = mpsc::channel::<()>();
    let harness = Harness::start(bare_config(), gated_executor(started_tx, release_rx));

    let first = task("0000000a");
    let second = task("0000000b");
    harness.send(&start(&first, validate_project("C:/tmp/first")));
    harness.send(&start(&second, validate_project("C:/tmp/second")));

    started_rx.recv().unwrap();
    let frames = harness.wait_for("the first task to start", |frames| {
        !stages(frames).is_empty()
    });
    assert_eq!(
        stages(&frames),
        vec![("1787900000000-0000000a", "preparing")],
        "the second task must stay queued while the first runs"
    );

    release_tx.send(()).unwrap();
    started_rx.recv().unwrap();
    release_tx.send(()).unwrap();
    let frames = harness.finish();

    assert_eq!(
        stages(&frames),
        vec![
            ("1787900000000-0000000a", "preparing"),
            ("1787900000000-0000000a", "completed"),
            ("1787900000000-0000000b", "preparing"),
            ("1787900000000-0000000b", "completed"),
        ]
    );
    assert_serialized(&frames);
}

#[test]
fn only_one_task_holds_the_cpu_adapter_at_a_time() {
    let (started_tx, started_rx) = mpsc::channel::<()>();
    let (release_tx, release_rx) = mpsc::channel::<()>();
    let harness = Harness::start(bare_config(), gated_executor(started_tx, release_rx));

    for suffix in ["0000000a", "0000000b", "0000000c"] {
        harness.send(&start(&task(suffix), validate_project("C:/tmp/project")));
    }
    for _ in 0..3 {
        started_rx.recv().unwrap();
        release_tx.send(()).unwrap();
    }
    let frames = harness.finish();

    assert_eq!(events(&frames).len(), 6, "{frames:?}");
    assert_serialized(&frames);
}

#[test]
fn a_duplicate_task_id_is_rejected() {
    let (started_tx, started_rx) = mpsc::channel::<()>();
    let (release_tx, release_rx) = mpsc::channel::<()>();
    let harness = Harness::start(bare_config(), gated_executor(started_tx, release_rx));

    let task_id = task("0000000a");
    harness.send(&start(&task_id, validate_project("C:/tmp/project")));
    started_rx.recv().unwrap();
    harness.send(&start(&task_id, validate_project("C:/tmp/project")));
    harness.wait_for("the duplicate to be rejected", |frames| {
        !rejections(frames).is_empty()
    });
    release_tx.send(()).unwrap();
    // Wait for the terminal event before closing stdin: a drain cancels the
    // running task, so finishing first would race the executor's own return.
    harness.wait_for("the released task to complete", |frames| {
        stages(frames)
            .iter()
            .any(|(_, stage)| *stage == "completed")
    });
    let frames = harness.finish();

    let reasons = rejections(&frames);
    assert_eq!(reasons.len(), 1, "{frames:?}");
    assert!(reasons[0].contains(task_id.as_str()), "{}", reasons[0]);
    assert_eq!(
        stages(&frames),
        vec![
            ("1787900000000-0000000a", "preparing"),
            ("1787900000000-0000000a", "completed"),
        ],
        "the duplicate must not affect the running task"
    );
}

#[test]
fn cancelling_a_queued_task_ends_it_before_it_ever_runs() {
    let (started_tx, started_rx) = mpsc::channel::<()>();
    let (release_tx, release_rx) = mpsc::channel::<()>();
    let harness = Harness::start(bare_config(), gated_executor(started_tx, release_rx));

    let running = task("0000000a");
    let queued = task("0000000b");
    harness.send(&start(&running, validate_project("C:/tmp/first")));
    started_rx.recv().unwrap();
    harness.send(&start(&queued, validate_project("C:/tmp/second")));
    harness.send(&cancel(&queued));
    harness.wait_for("the queued task to be cancelled", |frames| {
        stages(frames).contains(&("1787900000000-0000000b", "cancelled"))
    });
    release_tx.send(()).unwrap();
    // Same hazard as the duplicate-id test: closing stdin drains the session and
    // cancels the running task, so wait for its own terminal event first.
    harness.wait_for("the running task to complete", |frames| {
        stages(frames).contains(&("1787900000000-0000000a", "completed"))
    });
    let frames = harness.finish();

    assert_eq!(
        stages(&frames),
        vec![
            ("1787900000000-0000000a", "preparing"),
            ("1787900000000-0000000b", "cancelled"),
            ("1787900000000-0000000a", "completed"),
        ],
        "a cancelled queued task never reports preparing"
    );
    assert!(
        started_rx.try_recv().is_err(),
        "a cancelled queued task must never be handed to the executor"
    );
}

#[test]
fn cancelling_a_running_task_emits_exactly_one_cancelled_event() {
    let (started_tx, started_rx) = mpsc::channel::<TaskId>();
    let harness = Harness::start(bare_config(), blocking_executor(started_tx));

    let task_id = task("0000000a");
    harness.send(&start(&task_id, validate_project("C:/tmp/project")));
    started_rx.recv().unwrap();
    harness.send(&cancel(&task_id));
    harness.send(&cancel(&task_id));
    let frames = harness.finish();

    assert_eq!(
        stages(&frames),
        vec![
            ("1787900000000-0000000a", "preparing"),
            ("1787900000000-0000000a", "cancelled"),
        ]
    );
    assert!(rejections(&frames).is_empty(), "cancel is idempotent");
}

#[test]
fn a_cancelled_external_process_becomes_one_cancelled_event() {
    let input_dir = tempfile::tempdir().unwrap();
    let input = input_dir.path().join("input.mp4");
    fs::write(&input, b"not a real video").unwrap();

    let (started_tx, started_rx) = mpsc::channel::<()>();
    let harness = Harness::start(media_config(), blocking_probe_executor(started_tx));

    let task_id = task("0000000a");
    harness.send(&start(
        &task_id,
        Request::ProbeMedia(ProbeMediaParams { input }),
    ));
    started_rx.recv().unwrap();
    harness.send(&cancel(&task_id));
    let frames = harness.finish();

    assert_eq!(
        stages(&frames),
        vec![
            ("1787900000000-0000000a", "preparing"),
            ("1787900000000-0000000a", "cancelled"),
        ]
    );
    let cancelled = events(&frames)[1];
    assert!(
        cancelled.error.is_none() && cancelled.result.is_none(),
        "{cancelled:?}"
    );
}

#[test]
fn cancelling_an_unknown_or_finished_task_is_silently_accepted() {
    let harness = Harness::start(bare_config(), instant_executor());
    let task_id = task("0000000a");

    harness.send(&cancel(&task("0000000f")));
    harness.send(&start(&task_id, validate_project("C:/tmp/project")));
    harness.wait_for("the task to complete", |frames| {
        stages(frames).contains(&("1787900000000-0000000a", "completed"))
    });
    harness.send(&cancel(&task_id));
    let frames = harness.finish();

    assert!(rejections(&frames).is_empty(), "{frames:?}");
    assert_eq!(
        stages(&frames),
        vec![
            ("1787900000000-0000000a", "preparing"),
            ("1787900000000-0000000a", "completed"),
        ],
        "a late cancel must not add a second terminal event"
    );
}

#[test]
fn a_failing_command_reports_a_failed_event_with_its_error() {
    let harness = Harness::start(
        bare_config(),
        Box::new(|_request, _config, _token, _reporter| {
            CommandOutcome::Failed(feathertalk_domain::TaskError::new(
                ErrorCode::MediaInvalid,
                "项目目录缺少必需文件",
                "project directory is missing assets/assets.json",
                TaskStage::Preparing,
            ))
        }),
    );
    harness.send(&start(
        &task("0000000a"),
        validate_project("C:/tmp/project"),
    ));
    let frames = harness.finish();

    let failed = events(&frames)[1];
    assert_eq!(failed.stage.as_slug(), "failed");
    assert_eq!(
        failed.error.as_ref().map(|error| error.code),
        Some(ErrorCode::MediaInvalid)
    );
}

#[test]
fn shutdown_cancels_queued_tasks_and_waits_for_the_running_one() {
    let (started_tx, started_rx) = mpsc::channel::<TaskId>();
    let harness = Harness::start(bare_config(), blocking_executor(started_tx));

    let running = task("0000000a");
    let queued = task("0000000b");
    harness.send(&start(&running, validate_project("C:/tmp/first")));
    started_rx.recv().unwrap();
    harness.send(&start(&queued, validate_project("C:/tmp/second")));
    harness.send(&shutdown());
    harness.send(&start(&task("0000000c"), validate_project("C:/tmp/third")));
    let frames = harness.finish();

    let observed = stages(&frames);
    assert!(
        observed.contains(&("1787900000000-0000000b", "cancelled")),
        "{frames:?}"
    );
    assert!(
        observed.contains(&("1787900000000-0000000a", "cancelled")),
        "{frames:?}"
    );
    assert!(
        !observed
            .iter()
            .any(|(task_id, _)| *task_id == "1787900000000-0000000c"),
        "a start after shutdown must not create a task: {frames:?}"
    );
    assert_serialized(&frames);
}

#[test]
fn closing_the_input_stream_shuts_the_worker_down() {
    let (started_tx, started_rx) = mpsc::channel::<TaskId>();
    let harness = Harness::start(bare_config(), blocking_executor(started_tx));

    let task_id = task("0000000a");
    harness.send(&start(&task_id, validate_project("C:/tmp/project")));
    started_rx.recv().unwrap();
    let frames = harness.finish();

    assert_eq!(
        stages(&frames),
        vec![
            ("1787900000000-0000000a", "preparing"),
            ("1787900000000-0000000a", "cancelled"),
        ],
        "a closed stdin cancels the running task and exits"
    );
}

#[test]
fn reported_progress_reaches_the_wire_in_order() {
    let harness = Harness::start(bare_config(), reporting_executor());
    let id = task("00000021");
    harness.send(&start(&id, validate_project("C:/tmp/project")));
    let frames = harness.wait_for("the task to complete", |frames| {
        stages(frames)
            .iter()
            .any(|(_, stage)| *stage == "completed")
    });

    assert_eq!(
        stages(&frames)
            .into_iter()
            .map(|(_, stage)| stage)
            .collect::<Vec<_>>(),
        vec![
            "preparing",
            "extracting_frames",
            "extracting_audio",
            "completed"
        ]
    );
    let progressed = events(&frames)
        .into_iter()
        .filter_map(|event| event.progress)
        .collect::<Vec<_>>();
    assert_eq!(
        progressed,
        vec![
            Progress {
                completed: 1,
                total: Some(2)
            },
            Progress {
                completed: 2,
                total: Some(2)
            }
        ]
    );
    // Progress events carry no metrics.
    for event in events(&frames) {
        assert_eq!(event.metrics, feathertalk_domain::Metrics::empty());
    }
    harness.finish();
}

#[test]
fn a_cancelled_task_keeps_the_progress_it_already_reported() {
    let (started, started_rx) = mpsc::channel::<()>();
    // Reports one progress event, then waits to be cancelled.
    let executor: JobExecutor = Box::new(move |_request, _config, token, reporter| {
        reporter.report(
            TaskStage::ExtractingFrames,
            Some(Progress {
                completed: 1,
                total: Some(2),
            }),
        );
        started.send(()).unwrap();
        while !token.is_cancelled() {
            thread::sleep(Duration::from_millis(5));
        }
        CommandOutcome::Cancelled
    });
    let harness = Harness::start(bare_config(), executor);
    let id = task("00000022");
    harness.send(&start(&id, validate_project("C:/tmp/project")));
    started_rx.recv().unwrap();
    harness.send(&cancel(&id));
    let frames = harness.finish();

    assert_eq!(
        stages(&frames)
            .into_iter()
            .map(|(_, stage)| stage)
            .collect::<Vec<_>>(),
        vec!["preparing", "extracting_frames", "cancelled"]
    );
}

#[test]
fn the_noop_reporter_emits_nothing() {
    // `NoReporter` is what a direct caller of `execute` passes; it must be safe
    // to call and must not panic.
    NoReporter.report(TaskStage::Preparing, None);
    NoReporter.report(
        TaskStage::ExtractingAudio,
        Some(Progress {
            completed: 1,
            total: None,
        }),
    );
}

#[test]
fn a_fully_configured_worker_enables_extract_frames_in_the_handshake() {
    let frames = Harness::start(full_config(), instant_executor()).finish();

    let ServerFrame::Ready(ready) = &frames[0] else {
        panic!("the first frame must be ready: {frames:?}");
    };
    assert_eq!(
        ready.supported_commands,
        vec![
            TaskKind::ValidateProject,
            TaskKind::InspectModel,
            TaskKind::ImportLegacyModel,
            TaskKind::MigrateLegacyFeatures,
            TaskKind::ExportModelPackage,
            TaskKind::ExportOnnx,
            TaskKind::ProbeMedia,
            TaskKind::NormalizeMedia,
            TaskKind::Render,
            TaskKind::ExtractFrames
        ]
    );
}

#[test]
fn extract_frames_reaches_the_executor_once_the_models_resolve() {
    let harness = Harness::start(full_config(), instant_executor());
    harness.send(&start(&task("0000002c"), extract_frames_request()));
    let frames = harness.finish();

    assert!(rejections(&frames).is_empty(), "{frames:?}");
    assert_eq!(
        stages(&frames),
        vec![
            ("1787900000000-0000002c", "preparing"),
            ("1787900000000-0000002c", "completed"),
        ]
    );
}

#[test]
fn extract_frames_is_rejected_when_the_models_are_unavailable() {
    let harness = Harness::start(media_config(), instant_executor());
    harness.send(&start(&task("0000002d"), extract_frames_request()));
    let frames = harness.finish();

    let reasons = rejections(&frames);
    assert_eq!(reasons.len(), 1, "{frames:?}");
    assert!(reasons[0].contains("extract_frames"), "{}", reasons[0]);
    // The reason has to name the variable an operator can fix.
    assert!(
        reasons[0].contains("FEATHERTALK_WORKER_SCRFD_DIR"),
        "{}",
        reasons[0]
    );
    assert!(events(&frames).is_empty());
}

#[test]
fn extract_frames_names_the_media_toolchain_before_the_models() {
    let harness = Harness::start(bare_config(), instant_executor());
    harness.send(&start(&task("0000002e"), extract_frames_request()));
    let frames = harness.finish();

    let reasons = rejections(&frames);
    assert_eq!(reasons.len(), 1, "{frames:?}");
    assert!(
        reasons[0].contains("FEATHERTALK_WORKER_FFPROBE"),
        "{}",
        reasons[0]
    );
}

#[test]
fn extract_features_reaches_the_executor_once_the_model_directory_resolves() {
    let harness = Harness::start(every_toolchain_config(), instant_executor());
    harness.send(&start(&task("0000002f"), extract_features_request()));
    let frames = harness.finish();

    assert!(rejections(&frames).is_empty(), "{frames:?}");
    assert_eq!(
        stages(&frames),
        vec![
            ("1787900000000-0000002f", "preparing"),
            ("1787900000000-0000002f", "completed"),
        ]
    );
}

#[test]
fn extract_features_is_rejected_with_the_hubert_variable() {
    let harness = Harness::start(full_config(), instant_executor());
    harness.send(&start(&task("00000030"), extract_features_request()));
    let frames = harness.finish();

    let reasons = rejections(&frames);
    assert_eq!(reasons.len(), 1, "{frames:?}");
    assert!(reasons[0].contains("extract_features"), "{}", reasons[0]);
    assert!(
        reasons[0].contains("FEATHERTALK_WORKER_HUBERT_DIR"),
        "{}",
        reasons[0]
    );
    assert!(events(&frames).is_empty());
}

#[test]
fn extract_features_never_asks_for_the_media_toolchain() {
    let harness = Harness::start(bare_config(), instant_executor());
    harness.send(&start(&task("00000031"), extract_features_request()));
    let frames = harness.finish();

    let reasons = rejections(&frames);
    assert_eq!(reasons.len(), 1, "{frames:?}");
    assert!(
        reasons[0].contains("FEATHERTALK_WORKER_HUBERT_DIR"),
        "{}",
        reasons[0]
    );
    assert!(
        !reasons[0].contains("FEATHERTALK_WORKER_FFPROBE"),
        "{}",
        reasons[0]
    );
}

#[test]
fn lock_asset_package_reaches_the_executor_once_the_model_directory_resolves() {
    let harness = Harness::start(every_toolchain_config(), instant_executor());
    harness.send(&start(&task("00000032"), lock_asset_package_request()));
    let frames = harness.finish();

    assert!(rejections(&frames).is_empty(), "{frames:?}");
    assert_eq!(
        stages(&frames),
        vec![
            ("1787900000000-00000032", "preparing"),
            ("1787900000000-00000032", "completed"),
        ]
    );
}

#[test]
fn lock_asset_package_is_rejected_with_the_hubert_variable() {
    let harness = Harness::start(full_config(), instant_executor());
    harness.send(&start(&task("00000033"), lock_asset_package_request()));
    let frames = harness.finish();

    let reasons = rejections(&frames);
    assert_eq!(reasons.len(), 1, "{frames:?}");
    assert!(reasons[0].contains("lock_asset_package"), "{}", reasons[0]);
    assert!(
        reasons[0].contains("FEATHERTALK_WORKER_HUBERT_DIR"),
        "{}",
        reasons[0]
    );
    assert!(events(&frames).is_empty());
}
