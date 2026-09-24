//! The one path from a worker process to the window.
//!
//! `WorkerSession::run` blocks until the task ends, and a training task can run
//! for hours, so every submission gets its own thread. Updates travel back over
//! an unbounded `async-channel`: unbounded because a bounded one would either
//! drop progress events or stall the worker's read loop, and `async-channel`
//! because GPUI's foreground executor awaits futures while the supervision thread
//! needs a blocking send.
//!
//! Nothing here touches `gpui`, so the whole pipeline is testable with a scripted
//! runner and no window.

use std::path::PathBuf;

use async_channel::{unbounded, Receiver, Sender};
use feathertalk_client::{CancelToken, EventSink};
use feathertalk_domain::{Event, Request, TaskId, TaskKind};
use feathertalk_supervisor::journal::TaskJournal;
use feathertalk_supervisor::policy::RestartPolicy;
use feathertalk_supervisor::runner::TaskRunner;
use feathertalk_supervisor::supervisor::{SupervisionReport, WorkerSupervisor};
use thiserror::Error;

/// One message from the supervision thread to the window.
///
/// Both variants are boxed: an `Event` and a `SupervisionReport` differ by an
/// order of magnitude in size, and every message in the channel would otherwise
/// pay for the larger one.
#[derive(Debug)]
pub enum TaskUpdate {
    /// A worker event, forwarded exactly as it arrived.
    Progress(Box<Event>),
    /// The supervised run ended; there is exactly one of these per submission.
    Finished(Box<SupervisionReport>),
}

/// Why a task could not be started.
#[derive(Debug, Error)]
pub enum PipelineError {
    #[error("the supervision thread could not be started: {0}")]
    Thread(#[from] std::io::Error),
}

/// Everything one supervised run needs.
///
/// A struct rather than six positional parameters: `submit` is called from inside
/// an event handler, where a six-argument call reads like nothing at all.
pub struct Job {
    pub task_id: TaskId,
    pub kind: TaskKind,
    pub request: Request,
    pub policy: RestartPolicy,
    /// Where crash logs go. The shell uses `<project>/logs`.
    pub log_dir: PathBuf,
    /// `None` skips the task history, which is what a run without a project does.
    pub journal: Option<TaskJournal>,
}

/// The window's half of a running task.
pub struct Submission {
    pub updates: Receiver<TaskUpdate>,
    /// Asking once is polite, twice kills the worker: the client crate's rule.
    pub cancel: CancelToken,
}

/// Forwards worker events into the channel.
pub struct ChannelSink {
    sender: Sender<TaskUpdate>,
}

impl ChannelSink {
    pub fn new(sender: Sender<TaskUpdate>) -> Self {
        Self { sender }
    }
}

impl EventSink for ChannelSink {
    fn on_event(&mut self, event: &Event, _raw: &str) {
        // A closed channel means the window is gone. The task keeps running:
        // losing a display is not a reason to abandon a checkpoint.
        let _ = self
            .sender
            .send_blocking(TaskUpdate::Progress(Box::new(event.clone())));
    }
}

/// Run `job` under supervision on a thread of its own.
pub fn submit<R>(runner: R, job: Job) -> Result<Submission, PipelineError>
where
    R: TaskRunner + Send + 'static,
{
    let (sender, updates) = unbounded();
    let cancel = CancelToken::new();
    let thread_cancel = cancel.clone();
    // Named so a crash report names the task rather than "thread '<unnamed>'".
    let name = format!("feathertalk-supervisor-{}", job.task_id.as_str());
    std::thread::Builder::new()
        .name(name)
        .spawn(move || run_job(runner, job, sender, thread_cancel))?;
    Ok(Submission { updates, cancel })
}

/// Supervise one task to its end, then send the report.
fn run_job<R: TaskRunner>(runner: R, job: Job, sender: Sender<TaskUpdate>, cancel: CancelToken) {
    let Job {
        task_id,
        kind,
        request,
        policy,
        log_dir,
        journal,
    } = job;
    let mut supervisor = WorkerSupervisor::new(runner, policy, log_dir);
    let mut sink = ChannelSink::new(sender.clone());
    let report = supervisor.run(
        &task_id,
        kind,
        request,
        journal.as_ref(),
        &cancel,
        &mut sink,
    );
    let _ = sender.send_blocking(TaskUpdate::Finished(Box::new(report)));
}
