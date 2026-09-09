//! The worker child process and the version 2 protocol.

use std::collections::VecDeque;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::process::{Child, ChildStderr, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{Receiver, RecvTimeoutError, channel};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use feathertalk_domain::{
    CancelFrame, ClientFrame, DomainError, Event, FrameWriter, MAX_FRAME_BYTES, PROTOCOL_VERSION,
    ReadyFrame, RejectedFrame, Request, ServerFrame, ShutdownFrame, StartFrame, TaskError, TaskId,
    TaskKind, TaskStage, decode_line,
};
use serde_json::Value;

use crate::{ClientError, ComputeError, ComputeOptions, SessionOptions, uses_compute_selection};

/// A decoded frame together with the exact line it was decoded from.
///
/// `--json` must reprint the worker's own bytes: this workspace compiles
/// `serde_json` without `preserve_order`, so any round trip through `Value`
/// would silently reorder object keys.
#[derive(Debug, Clone)]
pub struct FrameLine {
    pub raw: String,
    pub frame: ServerFrame,
}

/// What one bounded read of the frame channel produced.
enum FrameEvent {
    /// Boxed so the enum is not three hundred bytes wide on every timeout tick;
    /// the run loop polls this type at 100 ms for the life of a task.
    Frame(Box<FrameLine>),
    Timeout,
    Eof,
}

/// Read one newline-terminated line, refusing to buffer past `MAX_FRAME_BYTES`.
///
/// `FrameReader` is not usable here: it hands back a decoded frame and discards
/// the text. An over-long line is drained to its newline before the error is
/// reported, so the stream stays framed.
fn read_line_bounded<R: BufRead>(reader: &mut R) -> Option<Result<String, ClientError>> {
    let mut buffer: Vec<u8> = Vec::new();
    let mut discarded = false;
    loop {
        // The `fill_buf` borrow must end before `consume`, so copy inside.
        let (consumed, finished) = {
            let available = match reader.fill_buf() {
                Ok(available) => available,
                Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(error) => return Some(Err(ClientError::Io(error))),
            };
            if available.is_empty() {
                break;
            }
            let (chunk, consumed, finished) = match available.iter().position(|byte| *byte == b'\n')
            {
                Some(index) => (&available[..index], index + 1, true),
                None => (available, available.len(), false),
            };
            if buffer.len() + chunk.len() > MAX_FRAME_BYTES {
                discarded = true;
            }
            if !discarded {
                buffer.extend_from_slice(chunk);
            }
            (consumed, finished)
        };
        reader.consume(consumed);
        if finished {
            return Some(finish_line(buffer, discarded));
        }
    }
    if buffer.is_empty() && !discarded {
        return None;
    }
    Some(finish_line(buffer, discarded))
}

fn finish_line(buffer: Vec<u8>, discarded: bool) -> Result<String, ClientError> {
    if discarded {
        return Err(ClientError::Protocol(DomainError::FrameTooLong {
            limit: MAX_FRAME_BYTES,
        }));
    }
    Ok(String::from_utf8_lossy(&buffer)
        .trim_end_matches('\r')
        .to_string())
}

/// Move the worker's stdout onto its own thread.
///
/// The thread decodes but deliberately does **not** validate: validation errors
/// have to be attributed to a protocol phase, and only the main loop knows which
/// phase it is in. The thread stops after forwarding an error, because a stream
/// that has lost framing cannot be resynchronised.
fn spawn_reader(stdout: ChildStdout) -> (Receiver<Result<FrameLine, ClientError>>, JoinHandle<()>) {
    let (sender, receiver) = channel();
    let handle = std::thread::spawn(move || {
        let mut reader = BufReader::new(stdout);
        while let Some(line) = read_line_bounded(&mut reader) {
            let message = match line {
                Ok(raw) if raw.trim().is_empty() => continue,
                Ok(raw) => match decode_line::<ServerFrame>(&raw) {
                    Ok(frame) => Ok(FrameLine { raw, frame }),
                    Err(error) => Err(ClientError::Protocol(error)),
                },
                Err(error) => Err(error),
            };
            let fatal = message.is_err();
            if sender.send(message).is_err() || fatal {
                break;
            }
        }
    });
    (receiver, handle)
}

/// Drain the worker's stderr into a bounded ring so a failure report can quote
/// the last few lines. Without this pump a chatty worker would block on a full
/// pipe while the client waited for a frame that could never arrive.
fn spawn_stderr_pump(
    stderr: ChildStderr,
    tail: Arc<Mutex<VecDeque<String>>>,
    limit: usize,
) -> JoinHandle<()> {
    std::thread::spawn(move || {
        let reader = BufReader::new(stderr);
        for line in reader.lines() {
            let Ok(line) = line else {
                break;
            };
            if limit == 0 {
                continue;
            }
            let mut guard = tail.lock().expect("the stderr tail mutex is not poisoned");
            while guard.len() >= limit {
                guard.pop_front();
            }
            guard.push_back(line);
        }
    })
}

/// The child process and its three pipes.
struct Transport {
    child: Child,
    /// `None` once stdin has been closed or has failed; both mean the same to
    /// the worker, which exits on EOF.
    writer: Option<FrameWriter<ChildStdin>>,
    frames: Receiver<Result<FrameLine, ClientError>>,
    reader_thread: Option<JoinHandle<()>>,
    stderr_thread: Option<JoinHandle<()>>,
    stderr_tail: Arc<Mutex<VecDeque<String>>>,
    options: SessionOptions,
}

impl Transport {
    fn spawn(
        path: &Path,
        options: SessionOptions,
        env: &[(String, String)],
    ) -> Result<Self, ClientError> {
        let mut command = Command::new(path);
        command
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            // The desktop release has no console; the worker communicates via pipes.
            command.creation_flags(0x08000000); // CREATE_NO_WINDOW
        }
        for (key, value) in env {
            command.env(key, value);
        }
        let mut child = command.spawn().map_err(|source| ClientError::Spawn {
            path: path.to_path_buf(),
            source,
        })?;
        let stdin = child.stdin.take().expect("stdin was piped");
        let stdout = child.stdout.take().expect("stdout was piped");
        let stderr = child.stderr.take().expect("stderr was piped");
        let stderr_tail = Arc::new(Mutex::new(VecDeque::new()));
        let stderr_thread =
            spawn_stderr_pump(stderr, Arc::clone(&stderr_tail), options.stderr_tail_lines);
        let (frames, reader_thread) = spawn_reader(stdout);
        Ok(Self {
            child,
            writer: Some(FrameWriter::new(stdin)),
            frames,
            reader_thread: Some(reader_thread),
            stderr_thread: Some(stderr_thread),
            stderr_tail,
            options,
        })
    }

    fn stderr_tail(&self) -> Vec<String> {
        self.stderr_tail
            .lock()
            .map(|guard| guard.iter().cloned().collect())
            .unwrap_or_default()
    }

    /// Validate, then write. A write failure means the worker is gone, which is
    /// a more useful diagnosis than the raw broken-pipe error.
    fn write_frame(&mut self, frame: &ClientFrame) -> Result<(), ClientError> {
        frame.validate().map_err(ClientError::Protocol)?;
        if self.writer.is_none() {
            return Err(self.worker_gone());
        }
        let outcome = self
            .writer
            .as_mut()
            .expect("checked immediately above")
            .write_frame(frame);
        if outcome.is_err() {
            self.writer = None;
            return Err(self.worker_gone());
        }
        Ok(())
    }

    /// Build the `WorkerGone` report, giving the child a short grace period so
    /// the exit status is usually available.
    fn worker_gone(&mut self) -> ClientError {
        // Copy the deadline out first: `wait_for_exit` needs `&mut self`.
        let grace = self.options.shutdown_grace;
        let status = self.wait_for_exit(grace);
        ClientError::WorkerGone {
            status,
            stderr_tail: self.stderr_tail(),
        }
    }

    /// Poll for exit up to `timeout`. `None` means still running or unknown.
    fn wait_for_exit(&mut self, timeout: Duration) -> Option<i32> {
        let deadline = Instant::now() + timeout;
        loop {
            match self.child.try_wait() {
                Ok(Some(status)) => return status.code(),
                Ok(None) => {}
                Err(_) => return None,
            }
            if Instant::now() >= deadline {
                return None;
            }
            std::thread::sleep(Duration::from_millis(20));
        }
    }

    /// Close stdin. The worker treats EOF as `shutdown`.
    fn close_stdin(&mut self) {
        self.writer = None;
    }

    fn kill_and_reap(&mut self) -> Option<i32> {
        self.writer = None;
        let _ = self.child.kill();
        self.child.wait().ok().and_then(|status| status.code())
    }

    fn next_frame(&self, timeout: Duration) -> Result<FrameEvent, ClientError> {
        match self.frames.recv_timeout(timeout) {
            Ok(Ok(line)) => Ok(FrameEvent::Frame(Box::new(line))),
            Ok(Err(error)) => Err(error),
            Err(RecvTimeoutError::Timeout) => Ok(FrameEvent::Timeout),
            // The sender is dropped when the reader thread sees EOF.
            Err(RecvTimeoutError::Disconnected) => Ok(FrameEvent::Eof),
        }
    }
}

impl std::fmt::Debug for Transport {
    /// Hand-written because `FrameWriter` is not `Debug`. The pipes and the two
    /// threads say nothing useful in a failure report, so this prints only what
    /// identifies the child and whether it can still be written to.
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Transport")
            .field("pid", &self.child.id())
            .field("stdin_open", &self.writer.is_some())
            .finish_non_exhaustive()
    }
}

impl Drop for Transport {
    /// Never wait on a worker that may never exit: close stdin, kill, reap, then
    /// join the two threads, which end as soon as their pipes close.
    fn drop(&mut self) {
        self.writer = None;
        let _ = self.child.kill();
        let _ = self.child.wait();
        if let Some(handle) = self.reader_thread.take() {
            let _ = handle.join();
        }
        if let Some(handle) = self.stderr_thread.take() {
            let _ = handle.join();
        }
    }
}

/// One worker process that has completed the handshake.
///
/// A session runs at most one task. That is the protocol's rule, not a
/// limitation of this type.
#[derive(Debug)]
pub struct WorkerSession {
    transport: Transport,
    ready: ReadyFrame,
    ready_raw: String,
    compute: Result<ComputeOptions, ComputeError>,
    foreign_events: usize,
}

impl WorkerSession {
    pub fn spawn(path: &Path, options: SessionOptions) -> Result<Self, ClientError> {
        Self::spawn_with_env(path, options, &[])
    }

    /// Spawn with child-only environment overrides, inheriting other variables.
    /// CLI and desktop compute choices use this without changing the parent
    /// process's environment. An empty adapter override means unspecified.
    /// Compute errors are retained until `run`, so capability probes still work.
    pub fn spawn_with_env(
        path: &Path,
        options: SessionOptions,
        env: &[(String, String)],
    ) -> Result<Self, ClientError> {
        let handshake_timeout = options.handshake_timeout;
        let compute = ComputeOptions::from_child_env(env);
        let transport = Transport::spawn(path, options, env)?;
        let line = match transport.next_frame(handshake_timeout) {
            Ok(FrameEvent::Frame(line)) => *line,
            Ok(FrameEvent::Timeout) => {
                return Err(ClientError::Handshake {
                    reason: format!("no ready frame within {} ms", handshake_timeout.as_millis()),
                    stderr_tail: transport.stderr_tail(),
                });
            }
            Ok(FrameEvent::Eof) => {
                return Err(ClientError::Handshake {
                    reason: "the worker closed its output before sending a ready frame".to_string(),
                    stderr_tail: transport.stderr_tail(),
                });
            }
            Err(ClientError::Protocol(error)) => {
                return Err(ClientError::Handshake {
                    reason: format!("the first line was not a decodable frame: {error}"),
                    stderr_tail: transport.stderr_tail(),
                });
            }
            Err(error) => return Err(error),
        };
        // A partial move of `line.frame` is fine: `FrameLine` has no `Drop`.
        match line.frame {
            ServerFrame::Ready(ready) => {
                if let Err(error) = ready.validate() {
                    return Err(match error {
                        DomainError::ProtocolVersion { expected, actual } => {
                            ClientError::ProtocolVersion { expected, actual }
                        }
                        other => ClientError::Handshake {
                            reason: format!("the ready frame is not usable: {other}"),
                            stderr_tail: transport.stderr_tail(),
                        },
                    });
                }
                Ok(Self {
                    transport,
                    ready,
                    ready_raw: line.raw,
                    compute,
                    foreign_events: 0,
                })
            }
            ServerFrame::Rejected(rejected) => Err(ClientError::Rejected {
                reason: rejected.reason,
            }),
            ServerFrame::Event(event) => Err(ClientError::Handshake {
                reason: format!(
                    "the worker sent an event for task {} before its ready frame",
                    event.task_id.as_str()
                ),
                stderr_tail: transport.stderr_tail(),
            }),
        }
    }

    /// The validated handshake frame: backends, adapters, capabilities, and the
    /// command list the capability gate consults.
    pub fn ready(&self) -> &ReadyFrame {
        &self.ready
    }

    /// The handshake line exactly as the worker wrote it, for `--json`.
    pub fn ready_raw(&self) -> &str {
        &self.ready_raw
    }

    /// The worker's most recent stderr lines, oldest first.
    pub fn stderr_tail(&self) -> Vec<String> {
        self.transport.stderr_tail()
    }

    /// Run one task to completion. A session runs at most one.
    ///
    /// Session-level failures are returned as `SessionOutcome::SessionError`
    /// rather than `Err`, so the caller has a single total match over the four
    /// things that can happen.
    pub fn run(
        &mut self,
        task_id: TaskId,
        request: Request,
        cancel: &CancelToken,
        sink: &mut dyn EventSink,
    ) -> SessionOutcome {
        match self.run_inner(task_id, request, cancel, sink) {
            Ok(outcome) => outcome,
            Err(error) => SessionOutcome::SessionError(error),
        }
    }

    /// Refuse a command the worker did not advertise.
    ///
    /// The handshake is the only authority here. Sending a command the worker
    /// disclaimed would earn a rejection frame at best and confuse the user
    /// about whose fault it is at worst.
    fn ensure_supported(&self, request: &Request) -> Result<(), ClientError> {
        let requested = request.kind();
        if self.ready.supported_commands.contains(&requested) {
            return Ok(());
        }
        Err(ClientError::UnsupportedCommand {
            // `TaskKind::as_slug` takes `self` by value, so copy out of the slice.
            requested: requested.as_slug(),
            supported: self
                .ready
                .supported_commands
                .iter()
                .copied()
                .map(TaskKind::as_slug)
                .collect(),
        })
    }

    fn run_inner(
        &mut self,
        task_id: TaskId,
        request: Request,
        cancel: &CancelToken,
        sink: &mut dyn EventSink,
    ) -> Result<SessionOutcome, ClientError> {
        self.ensure_supported(&request)?;
        // Every fresh worker (including supervisor retries) must advertise the
        // requested device before receiving Start. An older CPU-only worker may
        // ignore compute environment variables, so worker-side checks alone
        // cannot prevent an unintended CPU task.
        if uses_compute_selection(request.kind()) {
            let validation = match &self.compute {
                Ok(compute) => compute.validate_for(request.kind(), &self.ready),
                Err(error) => Err(error.clone()),
            };
            validation.map_err(|error| ClientError::Rejected {
                reason: format!("compute configuration: {error}"),
            })?;
        }
        self.transport.write_frame(&ClientFrame::Start(StartFrame {
            protocol_version: PROTOCOL_VERSION,
            task_id: task_id.clone(),
            request,
        }))?;
        let mut cancel_state = CancelState::Idle;
        loop {
            // Checked before the bounded read, so a request registered while the
            // worker was quiet is acted on within one POLL_INTERVAL.
            if let Some(outcome) = self.service_cancel(cancel, &mut cancel_state, &task_id)? {
                return Ok(outcome);
            }
            match self.transport.next_frame(POLL_INTERVAL)? {
                // Quiet worker: keep waiting. The task has no deadline of its
                // own — training legitimately runs for hours.
                FrameEvent::Timeout => continue,
                FrameEvent::Eof => {
                    // A worker that exits after being asked to stop has stopped.
                    if !matches!(cancel_state, CancelState::Idle) {
                        return Ok(SessionOutcome::Cancelled);
                    }
                    return Err(self.transport.worker_gone());
                }
                FrameEvent::Frame(line) => {
                    // Validation happens here, not on the reader thread, so the
                    // error can be attributed to this phase of the protocol.
                    line.frame.validate().map_err(ClientError::Protocol)?;
                    match line.frame {
                        ServerFrame::Event(event) => {
                            if event.task_id.as_str() != task_id.as_str() {
                                // A chatty worker is allowed; count it so the
                                // ignore rule stays observable.
                                self.foreign_events += 1;
                                continue;
                            }
                            sink.on_event(&event, &line.raw);
                            if let Some(outcome) = terminal_outcome(event) {
                                return Ok(outcome);
                            }
                        }
                        ServerFrame::Rejected(rejected) => {
                            sink.on_rejected(&rejected, &line.raw);
                            return Err(ClientError::Rejected {
                                reason: rejected.reason,
                            });
                        }
                        ServerFrame::Ready(_) => {
                            return Err(ClientError::Protocol(DomainError::MalformedFrame {
                                reason: "the worker sent a second ready frame".to_string(),
                            }));
                        }
                    }
                }
            }
        }
    }

    /// How many events for other tasks were ignored.
    pub fn foreign_event_count(&self) -> usize {
        self.foreign_events
    }

    /// Act on the cancel token, and on the grace deadline if one is running.
    ///
    /// `Some(Cancelled)` means the session is over. Every branch here is
    /// bounded: the polite path has a deadline, and the deadline has a kill.
    fn service_cancel(
        &mut self,
        cancel: &CancelToken,
        state: &mut CancelState,
        task_id: &TaskId,
    ) -> Result<Option<SessionOutcome>, ClientError> {
        match (cancel.count(), *state) {
            // Nothing asked for, or already asked and still within the grace.
            (0, _) | (1, CancelState::Requested { .. }) => {}
            (1, CancelState::Idle) => {
                let grace = self.transport.options.cancel_grace;
                let frame = ClientFrame::Cancel(CancelFrame {
                    protocol_version: PROTOCOL_VERSION,
                    task_id: task_id.clone(),
                });
                if self.transport.write_frame(&frame).is_err() {
                    // The worker is already unreachable. The user asked for the
                    // task to stop, and it has: report that, not a transport
                    // failure they can do nothing about.
                    self.transport.kill_and_reap();
                    return Ok(Some(SessionOutcome::Cancelled));
                }
                *state = CancelState::Requested {
                    deadline: Instant::now() + grace,
                };
            }
            // Two or more requests: the user has asked twice, stop now.
            _ => {
                self.transport.kill_and_reap();
                return Ok(Some(SessionOutcome::Cancelled));
            }
        }
        if let CancelState::Requested { deadline } = *state
            && Instant::now() >= deadline
        {
            // The worker had its chance. Escalate: shutdown, then EOF, then kill.
            let grace = self.transport.options.shutdown_grace;
            let _ = self
                .transport
                .write_frame(&ClientFrame::Shutdown(ShutdownFrame {
                    protocol_version: PROTOCOL_VERSION,
                }));
            self.transport.close_stdin();
            if self.transport.wait_for_exit(grace).is_none() {
                self.transport.kill_and_reap();
            }
            return Ok(Some(SessionOutcome::Cancelled));
        }
        Ok(None)
    }

    /// Ask the worker to exit and wait out the shutdown grace.
    ///
    /// A best-effort write: if the worker is already gone the frame cannot be
    /// delivered and there is nothing to report. `Drop` kills any survivor, so
    /// this never leaks a process even when it returns `None`.
    pub fn shutdown(mut self) -> Option<i32> {
        let grace = self.transport.options.shutdown_grace;
        let _ = self
            .transport
            .write_frame(&ClientFrame::Shutdown(ShutdownFrame {
                protocol_version: PROTOCOL_VERSION,
            }));
        self.transport.close_stdin();
        self.transport.wait_for_exit(grace)
    }
}

/// How long one wait on the frame channel blocks. The loop has to wake up
/// regularly even when the worker is quiet, because that is when the cancel flag
/// is checked; 100 ms keeps Ctrl-C responsive without busy-waiting.
const POLL_INTERVAL: Duration = Duration::from_millis(100);

/// Where a session's events go. The CLI has two implementations, human and
/// JSON; the desktop shell will add a third that forwards to the UI.
///
/// `raw` is the worker's original line. Presenters that echo JSON must use it
/// rather than re-serialising `event`.
pub trait EventSink {
    fn on_event(&mut self, event: &Event, raw: &str);

    /// A mid-session rejection. Default: ignore it — the returned
    /// `ClientError::Rejected` already carries the reason.
    fn on_rejected(&mut self, rejected: &RejectedFrame, raw: &str) {
        let _ = (rejected, raw);
    }
}

/// How a session ended. The four variants are the four CLI exit codes: 0, 1, 2,
/// and 3, in this order.
#[derive(Debug)]
pub enum SessionOutcome {
    Completed { result: Option<Value> },
    Failed(TaskError),
    Cancelled,
    SessionError(ClientError),
}

/// A cancel request counter shared with a signal handler.
///
/// A counter rather than a flag: the first request asks politely, the second
/// kills. It is `Clone` so a handler can own one, and only ever counts up, so
/// there is no lost-wakeup case to reason about.
#[derive(Debug, Clone, Default)]
pub struct CancelToken(Arc<AtomicUsize>);

impl CancelToken {
    pub fn new() -> Self {
        Self(Arc::new(AtomicUsize::new(0)))
    }

    /// Signal-handler safe: one atomic add, no allocation, no locks.
    pub fn request(&self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }

    pub fn count(&self) -> usize {
        self.0.load(Ordering::SeqCst)
    }
}

/// Map a terminal stage onto an outcome. The three arms are exactly the stages
/// `TaskStage::is_terminal` reports; every other stage is progress.
fn terminal_outcome(event: Event) -> Option<SessionOutcome> {
    match event.stage {
        TaskStage::Completed => Some(SessionOutcome::Completed {
            result: event.result,
        }),
        TaskStage::Failed { .. } => {
            Some(SessionOutcome::Failed(event.error.expect(
                "Event::validate guarantees a failed stage carries its error",
            )))
        }
        TaskStage::Cancelled => Some(SessionOutcome::Cancelled),
        _ => None,
    }
}

/// Where the cancel escalation currently is.
///
/// Two states are enough: every kill path returns an outcome immediately, so
/// there is no "killed" state anyone could observe.
#[derive(Debug, Copy, Clone)]
enum CancelState {
    Idle,
    Requested { deadline: Instant },
}
