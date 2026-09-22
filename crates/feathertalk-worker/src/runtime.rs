use std::{
    collections::{BTreeMap, VecDeque},
    io::{BufRead, Write},
    panic::{AssertUnwindSafe, catch_unwind},
    sync::mpsc::{self, Receiver, Sender},
    thread};

use feathertalk_domain::{
    ClientFrame, DomainError, Event, FrameReader, FrameWriter, Metrics, PROTOCOL_VERSION, Progress,
    RejectedFrame, Request, ServerFrame, TaskId, TaskKind, TaskLifecycle, TaskStage};
use feathertalk_media::CancellationToken;
use time::{OffsetDateTime, format_description::well_known::Rfc3339};

use crate::error_map::panic_task_error;
use crate::reporter::TrackedReporter;

use crate::{
    AdapterLockError, AdapterLocks, CommandOutcome, ENV_FFMPEG, ENV_FFPROBE, ENV_HUBERT_DIR,
    ENV_PFLD_DIR, ENV_SCRFD_DIR, ENV_VGG19_DIR, TaskReporter, WorkerConfig, execute, ready_frame,
    supported_commands};

/// The executor thread's stack. A 160x160 training step builds a deep autodiff
/// graph and overflows the 2 MiB default in a debug build; 64 MiB is the size
/// `feathertalk-pfld` and `feathertalk-training-run` already settled on.
const EXECUTION_STACK_BYTES: usize = 64 * 1024 * 1024;

/// How the runtime reaches command execution.
///
/// Production callers use [`serve`], which passes [`crate::execute`]. Tests
/// pass a closure so queueing, cancellation, and shutdown can be observed
/// without a real external tool.
pub type JobExecutor = Box<
    dyn Fn(&Request, &WorkerConfig, &CancellationToken, &dyn TaskReporter) -> CommandOutcome
        + Send
        + 'static,
>;

/// The execution thread's reporter: one per job, holding that job's id and a
/// clone of the control channel.
///
/// A closed channel is ignored. Losing a progress event while the runtime is
/// already shutting down is not a reason to fail a task.
struct ChannelReporter {
    task_id: TaskId,
    control_tx: Sender<ControlMessage>}

impl TaskReporter for ChannelReporter {
    fn report(&self, stage: TaskStage, progress: Option<Progress>) {
        self.report_metrics(stage, progress, Metrics::empty());
    }

    fn report_metrics(&self, stage: TaskStage, progress: Option<Progress>, metrics: Metrics) {
        let mut event = Event::new(self.task_id.clone(), &now_rfc3339(), stage);
        event.progress = progress;
        event.metrics = metrics;
        let _ = self.control_tx.send(ControlMessage::Emit(event));
    }
}

/// Everything the control loop receives. It is the only owner of task state,
/// so every other thread talks to it through this one channel.
enum ControlMessage {
    Client(ClientFrame),
    ClientError(DomainError),
    InputClosed,
    Emit(Event),
    Finished { task_id: TaskId, adapter_id: String }}

/// One unit of work handed to the execution thread.
struct Job {
    task_id: TaskId,
    request: Request,
    token: CancellationToken,
    adapter_id: String}

struct TaskState {
    lifecycle: TaskLifecycle,
    token: CancellationToken,
    /// True until the job is handed to the execution thread. A queued task can
    /// be cancelled without ever running.
    queued: bool}

/// Serve one client session over `input`/`output` until shutdown or EOF.
pub fn serve<R, W>(input: R, output: W, config: &WorkerConfig) -> Result<(), DomainError>
where
    R: BufRead + Send + 'static,
    W: Write,
{
    serve_with_executor(input, output, config, Box::new(execute))
}

/// [`serve`] with an injected executor.
pub fn serve_with_executor<R, W>(
    input: R,
    output: W,
    config: &WorkerConfig,
    executor: JobExecutor,
) -> Result<(), DomainError>
where
    R: BufRead + Send + 'static,
    W: Write,
{
    // Enable CubeCL's compiled-kernel cache before any device is initialized;
    // device certification inside ready_frame() is the first such point, and the
    // runtime config freezes on first read.
    crate::install_runtime_config();

    let mut writer = FrameWriter::new(output);
    // The handshake goes out before a single byte is read, so a desktop that
    // sees an incompatible version never has to send a request first.
    write_frame(&mut writer, &ServerFrame::Ready(ready_frame(config)))?;

    let (control_tx, control_rx) = mpsc::channel::<ControlMessage>();
    let (job_tx, job_rx) = mpsc::channel::<Job>();

    let input_tx = control_tx.clone();
    // The input thread is deliberately detached. After `shutdown` the control
    // loop stops reading, and a thread blocked on a still-open stdin must not
    // keep the process alive; it dies with the process instead.
    let _ = thread::spawn(move || read_input(input, &input_tx));

    let execution_tx = control_tx;
    // The executor thread needs the whole configuration, and `WorkerConfig` is
    // a handful of small fields, so it gets its own clone instead of a shared
    // borrow.
    let execution_config = config.clone();
    // No `DomainError` variant describes a thread failure, and adding one would
    // widen the wire protocol for something an operator cannot act on, so the
    // spawn failure follows the same route as the final flush below.
    let execution = thread::Builder::new()
        .name("execution".to_owned())
        .stack_size(EXECUTION_STACK_BYTES)
        .spawn(move || run_jobs(&job_rx, &execution_tx, execution_config, executor))
        .map_err(|error| DomainError::MalformedFrame {
            reason: format!("cannot start the execution thread: {error}")})?;

    let result = control_loop(&control_rx, &mut writer, &job_tx, config);

    // Dropping the sender ends the execution thread's receive loop.
    drop(job_tx);
    let _ = execution.join();
    let mut output = writer.into_inner();
    output
        .flush()
        .map_err(|error| DomainError::MalformedFrame {
            reason: error.to_string()})?;
    result
}

fn read_input<R: BufRead>(input: R, control_tx: &Sender<ControlMessage>) {
    let mut reader = FrameReader::new(input);
    while let Some(decoded) = reader.read_frame::<ClientFrame>() {
        // `FrameReader` is syntax-only, so semantic validation happens here.
        // `ClientFrame::validate` includes the protocol-version check.
        let message = match decoded {
            Ok(frame) => match frame.validate() {
                Ok(()) => ControlMessage::Client(frame),
                Err(error) => ControlMessage::ClientError(error)},
            Err(error) => ControlMessage::ClientError(error)};
        if control_tx.send(message).is_err() {
            return;
        }
    }
    let _ = control_tx.send(ControlMessage::InputClosed);
}

fn run_jobs(
    job_rx: &Receiver<Job>,
    control_tx: &Sender<ControlMessage>,
    config: WorkerConfig,
    executor: JobExecutor,
) {
    while let Ok(job) = job_rx.recv() {
        let channel_reporter = ChannelReporter {
            task_id: job.task_id.clone(),
            control_tx: control_tx.clone()};
        let reporter = TrackedReporter::new(&channel_reporter);
        // Burn reports some failures by unwinding. The control loop still owes
        // the client a terminal event and must release the adapter afterwards.
        let outcome = catch_unwind(AssertUnwindSafe(|| {
            executor(&job.request, &config, &job.token, &reporter)
        }))
        .unwrap_or_else(|payload| {
            CommandOutcome::Failed(panic_task_error(payload.as_ref(), reporter.stage()))
        });
        let event = match outcome {
            CommandOutcome::Completed(result) => {
                let mut event =
                    Event::new(job.task_id.clone(), &now_rfc3339(), TaskStage::Completed);
                event.result = result;
                event
            }
            CommandOutcome::Cancelled => {
                Event::new(job.task_id.clone(), &now_rfc3339(), TaskStage::Cancelled)
            }
            CommandOutcome::Failed(error) => {
                let stage = TaskStage::Failed {
                    code: error.code,
                    message: error.summary.clone()};
                let mut event = Event::new(job.task_id.clone(), &now_rfc3339(), stage);
                event.error = Some(error);
                event
            }
        };
        let _ = control_tx.send(ControlMessage::Emit(event));
        // The adapter is released only after the event, so the next task never
        // starts before the previous one is reported.
        let _ = control_tx.send(ControlMessage::Finished {
            task_id: job.task_id,
            adapter_id: job.adapter_id});
    }
}

fn control_loop<W: Write>(
    control_rx: &Receiver<ControlMessage>,
    writer: &mut FrameWriter<W>,
    job_tx: &Sender<Job>,
    config: &WorkerConfig,
) -> Result<(), DomainError> {
    let supported = supported_commands(config);
    let mut tasks: BTreeMap<TaskId, TaskState> = BTreeMap::new();
    let mut pending: VecDeque<Job> = VecDeque::new();
    let mut locks = AdapterLocks::new(
        config
            .compute()
            .adapters()
            .iter()
            .map(|adapter| adapter.id.clone()),
    );
    let mut active: Option<TaskId> = None;
    let mut draining = false;

    while let Ok(message) = control_rx.recv() {
        match message {
            ControlMessage::Client(ClientFrame::Start(frame)) => {
                if draining {
                    reject(
                        writer,
                        "worker 正在关闭，请等待进程退出后重新启动任务。".to_owned(),
                    )?;
                } else if !supported.contains(&frame.request.kind()) {
                    reject(writer, unsupported_reason(&frame.request, config))?;
                } else if tasks.contains_key(&frame.task_id) {
                    reject(
                        writer,
                        format!(
                            "task_id {} 已存在，请为新任务生成新的 task_id。",
                            frame.task_id.as_str()
                        ),
                    )?;
                } else {
                    let adapter = match config.adapter_for(frame.request.kind()) {
                        Ok(adapter) => adapter,
                        Err(reason) => {
                            reject(writer, format!("计算设备不可用，请重新选择设备。{reason}"))?;
                            continue;
                        }
                    };
                    let token = CancellationToken::new();
                    tasks.insert(
                        frame.task_id.clone(),
                        TaskState {
                            lifecycle: TaskLifecycle::new(),
                            token: token.clone(),
                            queued: true},
                    );
                    pending.push_back(Job {
                        task_id: frame.task_id,
                        request: frame.request,
                        token,
                        adapter_id: adapter.id});
                }
            }
            ControlMessage::Client(ClientFrame::Cancel(frame)) => {
                // Cancel is idempotent: an unknown or already terminal task is
                // accepted silently.
                let cancel_queued = match tasks.get_mut(&frame.task_id) {
                    Some(state) if !state.lifecycle.is_terminal() => {
                        state.token.cancel();
                        let queued = state.queued;
                        state.queued = false;
                        queued
                    }
                    _ => false};
                if cancel_queued {
                    pending.retain(|job| job.task_id != frame.task_id);
                    let event = Event::new(frame.task_id, &now_rfc3339(), TaskStage::Cancelled);
                    emit(writer, &mut tasks, event)?;
                }
            }
            ControlMessage::Client(ClientFrame::Shutdown(_)) | ControlMessage::InputClosed => {
                draining = true;
                begin_drain(writer, &mut tasks, &mut pending, active.as_ref())?;
            }
            ControlMessage::ClientError(error) => reject(writer, client_error_reason(&error))?,
            ControlMessage::Emit(event) => emit(writer, &mut tasks, event)?,
            ControlMessage::Finished {
                task_id,
                adapter_id} => {
                locks.release(&adapter_id).map_err(lock_failure)?;
                if active.as_ref() == Some(&task_id) {
                    active = None;
                }
            }
        }

        if draining {
            if active.is_none() {
                break;
            }
        } else {
            dispatch(
                writer,
                &mut tasks,
                &mut pending,
                &mut locks,
                &mut active,
                job_tx,
            )?;
        }
    }
    Ok(())
}

/// Hand the next queued job to the execution thread if the runtime is idle and
/// its adapter is free.
fn dispatch<W: Write>(
    writer: &mut FrameWriter<W>,
    tasks: &mut BTreeMap<TaskId, TaskState>,
    pending: &mut VecDeque<Job>,
    locks: &mut AdapterLocks,
    active: &mut Option<TaskId>,
    job_tx: &Sender<Job>,
) -> Result<(), DomainError> {
    if active.is_some() {
        return Ok(());
    }
    let Some(job) = pending.pop_front() else {
        return Ok(());
    };
    if !locks.is_free(&job.adapter_id) {
        pending.push_front(job);
        return Ok(());
    }
    locks
        .acquire(&job.adapter_id, job.task_id.clone())
        .map_err(lock_failure)?;
    if let Some(state) = tasks.get_mut(&job.task_id) {
        state.queued = false;
    }
    *active = Some(job.task_id.clone());
    let event = Event::new(job.task_id.clone(), &now_rfc3339(), TaskStage::Preparing);
    emit(writer, tasks, event)?;
    job_tx.send(job).map_err(|_| DomainError::MalformedFrame {
        reason: "execution thread stopped before the task was dispatched".to_owned()})
}

/// Stop accepting work: cancel every queued task with its own `cancelled`
/// event and ask the running task to stop.
fn begin_drain<W: Write>(
    writer: &mut FrameWriter<W>,
    tasks: &mut BTreeMap<TaskId, TaskState>,
    pending: &mut VecDeque<Job>,
    active: Option<&TaskId>,
) -> Result<(), DomainError> {
    for job in std::mem::take(pending) {
        if let Some(state) = tasks.get_mut(&job.task_id) {
            state.token.cancel();
            state.queued = false;
        }
        let event = Event::new(job.task_id, &now_rfc3339(), TaskStage::Cancelled);
        emit(writer, tasks, event)?;
    }
    if let Some(task_id) = active
        && let Some(state) = tasks.get(task_id)
    {
        state.token.cancel();
    }
    Ok(())
}

/// Advance the task lifecycle and write the event.
///
/// A task whose lifecycle is already terminal silently drops the event. That is
/// what guarantees at most one terminal event per task even when a cancel and a
/// natural completion race each other.
fn emit<W: Write>(
    writer: &mut FrameWriter<W>,
    tasks: &mut BTreeMap<TaskId, TaskState>,
    event: Event,
) -> Result<(), DomainError> {
    let Some(state) = tasks.get_mut(&event.task_id) else {
        return Ok(());
    };
    if state.lifecycle.advance(event.stage.clone()).is_err() {
        return Ok(());
    }
    write_frame(writer, &ServerFrame::Event(event))
}

fn reject<W: Write>(writer: &mut FrameWriter<W>, reason: String) -> Result<(), DomainError> {
    write_frame(
        writer,
        &ServerFrame::Rejected(RejectedFrame {
            protocol_version: PROTOCOL_VERSION,
            reason}),
    )
}

fn write_frame<W: Write>(
    writer: &mut FrameWriter<W>,
    frame: &ServerFrame,
) -> Result<(), DomainError> {
    frame.validate()?;
    writer.write_frame(frame)
}

fn unsupported_reason(request: &Request, config: &WorkerConfig) -> String {
    let slug = request.kind().as_slug();
    match request.kind() {
        // Both media commands need the same two binaries, so they share the
        // reason that names what to fix.
        TaskKind::ProbeMedia | TaskKind::NormalizeMedia => media_reason(slug, config),
        // Extraction needs both halves. Media first: it probes the video before
        // it loads a model, so that is the wall an operator would hit next.
        TaskKind::ExtractFrames if config.media().is_none() => media_reason(slug, config),
        TaskKind::ExtractFrames => model_reason(slug, config),
        // Feature extraction needs no media tools, so its only wall is the
        // FeatherHuBERT directory.
        TaskKind::ExtractFeatures => feature_reason(slug, config),
        // The lock reads files the earlier commands already wrote, so the
        // package directory is its only wall too.
        TaskKind::LockAssetPackage => feature_reason(slug, config),
        // Training reads a locked project off disk, so the perceptual-loss
        // package is its only wall.
        TaskKind::Train => training_reason(slug, config),
        // Rendering needs the media toolchain and nothing else, so it shares the
        // media commands' reason: both tools, both variable names.
        TaskKind::Render => media_reason(slug, config),
        // Listing `supported_commands` instead of a hard-coded set keeps this
        // message correct as later commands land.
        _ => format!(
            "此 worker 不支持命令 {slug}，当前支持：{}。",
            supported_commands(config)
                .iter()
                .copied()
                .map(TaskKind::as_slug)
                .collect::<Vec<_>>()
                .join("、")
        )}
}

fn media_reason(slug: &str, config: &WorkerConfig) -> String {
    match config.media_rejection() {
        Some(rejection) => format!(
            "命令 {slug} 需要可用的媒体工具链，当前配置被拒绝：{rejection}。修正后重启 worker。"
        ),
        None => format!(
            "命令 {slug} 需要媒体工具链，请设置 {ENV_FFPROBE} 与 {ENV_FFMPEG} 后重启 worker。"
        )}
}

fn model_reason(slug: &str, config: &WorkerConfig) -> String {
    match config.model_rejection() {
        Some(rejection) => format!(
            "命令 {slug} 需要可用的模型目录，当前配置被拒绝：{rejection}。修正后重启 worker。"
        ),
        None => format!(
            "命令 {slug} 需要人脸与关键点模型，请设置 {ENV_SCRFD_DIR} 与 {ENV_PFLD_DIR} 后重启 worker。"
        )}
}

fn feature_reason(slug: &str, config: &WorkerConfig) -> String {
    match config.feature_rejection() {
        Some(rejection) => format!(
            "命令 {slug} 需要可用的特征模型目录，当前配置被拒绝：{rejection}。修正后重启 worker。"
        ),
        None => format!(
            "命令 {slug} 需要 FeatherHuBERT 特征模型，请设置 {ENV_HUBERT_DIR} 后重启 worker。"
        )}
}

fn training_reason(slug: &str, config: &WorkerConfig) -> String {
    match config.training_rejection() {
        Some(rejection) => format!(
            "命令 {slug} 需要可用的感知损失模型目录，当前配置被拒绝：{rejection}。修正后重启 worker。"
        ),
        None => {
            format!("命令 {slug} 需要 VGG19 感知损失模型，请设置 {ENV_VGG19_DIR} 后重启 worker。")
        }
    }
}

fn client_error_reason(error: &DomainError) -> String {
    match error {
        DomainError::ProtocolVersion { expected, actual } => {
            format!("协议版本不兼容：worker 使用 {expected}，收到 {actual}。请升级桌面端后重试。")
        }
        other => format!("无法解析请求帧：{other}。请检查帧格式后重试。")}
}

/// An adapter lock error at this point is an internal invariant violation: the
/// control loop is the only owner of the table.
fn lock_failure(error: AdapterLockError) -> DomainError {
    DomainError::InvalidField {
        field: "adapter_id",
        reason: error.to_string()}
}

fn now_rfc3339() -> String {
    OffsetDateTime::now_utc()
        .format(&Rfc3339)
        .expect("formatting a UTC timestamp as RFC 3339 cannot fail")
}
