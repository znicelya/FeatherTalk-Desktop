use std::path::PathBuf;

use feathertalk_client::{CancelToken, ClientError, EventSink, SessionOutcome};
use feathertalk_domain::{
    ErrorCode, Event, Recovery, RejectedFrame, Request, TaskError, TaskId, TaskKind, TaskStage,
    TaskStatus,
};
use serde_json::Value;
use time::OffsetDateTime;

use crate::crash_log::{CrashLogOutcome, CrashReport, write_crash_log};
use crate::journal::TaskJournal;
use crate::policy::{Disposition, RestartPolicy, classify};
use crate::runner::{Attempt, AttemptResult, TaskRunner};

/// What became of one supervised task.
///
/// `Crashed` and `Unavailable` carry English detail only: the desktop shell owns
/// the Chinese copy, and `TaskError`'s summary field is reserved for text the
/// worker itself wrote for the user.
#[derive(Debug)]
pub enum SupervisedOutcome {
    Completed {
        result: Option<Value>,
    },
    Failed(TaskError),
    Cancelled,
    /// The worker died, or never got far enough to speak the protocol.
    Crashed {
        stage: TaskStage,
        recovery: Recovery,
        detail: String,
    },
    /// Nothing about this installation will make the next attempt different.
    Unavailable(ClientError),
}

/// One supervised task, start to finish.
#[derive(Debug)]
pub struct SupervisionReport {
    pub task_id: TaskId,
    /// How many worker processes were spawned, including the first.
    pub attempts: u32,
    pub outcome: SupervisedOutcome,
    pub crash_logs: Vec<CrashLogOutcome>,
    /// Manifest bookkeeping that failed. The task's own result is never withheld
    /// because the history could not be written.
    pub journal_errors: Vec<String>,
}

/// Runs one task to a conclusion, restarting the worker when that can help.
pub struct WorkerSupervisor<R: TaskRunner> {
    runner: R,
    policy: RestartPolicy,
    log_dir: PathBuf,
    clock: fn() -> OffsetDateTime,
}

impl<R: TaskRunner> WorkerSupervisor<R> {
    pub fn new(runner: R, policy: RestartPolicy, log_dir: PathBuf) -> Self {
        Self {
            runner,
            policy,
            log_dir,
            clock: OffsetDateTime::now_utc,
        }
    }

    /// Test seam: a fixed clock keeps recorded timestamps deterministic.
    pub fn with_clock(mut self, clock: fn() -> OffsetDateTime) -> Self {
        self.clock = clock;
        self
    }

    pub fn runner(&self) -> &R {
        &self.runner
    }

    /// Run one task, restarting the worker while the policy allows it.
    ///
    /// The caller's `sink` sees exactly the events the worker emitted, including
    /// those from an attempt that later crashed: hiding them would make a restart
    /// look like a stall.
    pub fn run(
        &mut self,
        task_id: &TaskId,
        kind: TaskKind,
        request: Request,
        journal: Option<&TaskJournal>,
        cancel: &CancelToken,
        sink: &mut dyn EventSink,
    ) -> SupervisionReport {
        let mut journal_errors = Vec::new();
        let mut crash_logs = Vec::new();
        self.record(
            journal,
            task_id,
            kind,
            TaskStatus::Queued,
            &mut journal_errors,
        );

        let mut request = request;
        let mut tracker = StageTracker::new(sink);
        let mut recorded_running = false;
        let mut attempt: u32 = 0;

        let outcome = loop {
            attempt = attempt.saturating_add(1);
            let result = self.runner.run_attempt(
                Attempt {
                    task_id,
                    request: &request,
                    attempt,
                },
                cancel,
                &mut tracker,
            );

            if tracker.saw_progress && !recorded_running {
                recorded_running = true;
                self.record(
                    journal,
                    task_id,
                    kind,
                    TaskStatus::Running,
                    &mut journal_errors,
                );
            }

            let disposition = classify(&result.outcome, attempt, &self.policy);
            if worth_a_log(&result.outcome) {
                crash_logs.push(self.write_log(task_id, kind, attempt, &result, disposition));
            }

            match disposition {
                Disposition::Restart { resume } => {
                    if resume {
                        resume_from_checkpoint(&mut request);
                    }
                    if !self.policy.restart_delay.is_zero() {
                        std::thread::sleep(self.policy.restart_delay);
                    }
                }
                Disposition::Finish | Disposition::GiveUp => {
                    break conclude(result.outcome, tracker.last_stage.clone());
                }
            }
        };

        self.record(
            journal,
            task_id,
            kind,
            status_of(&outcome),
            &mut journal_errors,
        );
        SupervisionReport {
            task_id: task_id.clone(),
            attempts: attempt,
            outcome,
            crash_logs,
            journal_errors,
        }
    }

    fn record(
        &self,
        journal: Option<&TaskJournal>,
        task_id: &TaskId,
        kind: TaskKind,
        status: TaskStatus,
        errors: &mut Vec<String>,
    ) {
        let Some(journal) = journal else {
            return;
        };
        if let Err(error) = journal.record(task_id, kind, status, (self.clock)()) {
            errors.push(error.to_string());
        }
    }

    fn write_log(
        &self,
        task_id: &TaskId,
        kind: TaskKind,
        attempt: u32,
        result: &AttemptResult,
        disposition: Disposition,
    ) -> CrashLogOutcome {
        let report = CrashReport {
            task_id: task_id.clone(),
            kind,
            attempt,
            worker_path: result.worker_path.clone(),
            exit_status: result.exit_status,
            disposition,
            detail: detail_of(&result.outcome),
            stderr_tail: result.stderr_tail.clone(),
            observed_at: (self.clock)(),
        };
        write_crash_log(&self.log_dir, &report)
    }
}

/// Forwards every event and remembers the last stage seen.
///
/// The stage matters after a crash: the worker cannot report where it died, so
/// the last event it managed to send is the best answer available.
struct StageTracker<'a> {
    inner: &'a mut dyn EventSink,
    last_stage: Option<TaskStage>,
    saw_progress: bool,
}

impl<'a> StageTracker<'a> {
    fn new(inner: &'a mut dyn EventSink) -> Self {
        Self {
            inner,
            last_stage: None,
            saw_progress: false,
        }
    }
}

impl EventSink for StageTracker<'_> {
    fn on_event(&mut self, event: &Event, raw: &str) {
        if !event.stage.is_terminal() {
            self.saw_progress = true;
        }
        self.last_stage = Some(event.stage.clone());
        self.inner.on_event(event, raw);
    }

    fn on_rejected(&mut self, rejected: &RejectedFrame, raw: &str) {
        self.inner.on_rejected(rejected, raw);
    }
}

/// Turn the last attempt's outcome into the caller's answer.
fn conclude(outcome: SessionOutcome, last_stage: Option<TaskStage>) -> SupervisedOutcome {
    match outcome {
        SessionOutcome::Completed { result } => SupervisedOutcome::Completed { result },
        SessionOutcome::Failed(error) => SupervisedOutcome::Failed(error),
        SessionOutcome::Cancelled => SupervisedOutcome::Cancelled,
        SessionOutcome::SessionError(error) => {
            let detail = error.to_string();
            match crash_recovery(&error) {
                Some(recovery) => SupervisedOutcome::Crashed {
                    stage: last_stage.unwrap_or(TaskStage::Queued),
                    recovery,
                    detail,
                },
                None => SupervisedOutcome::Unavailable(error),
            }
        }
    }
}

/// Which client errors count as a crash, and what the interface should offer.
///
/// A worker that vanished mid-task may have left a checkpoint behind; one that
/// never finished starting has nothing to resume from, so a plain retry is the
/// honest suggestion.
fn crash_recovery(error: &ClientError) -> Option<Recovery> {
    match error {
        ClientError::WorkerGone { .. } => Some(Recovery::ResumeFromCheckpoint),
        ClientError::Handshake { .. } | ClientError::Spawn { .. } => Some(Recovery::Retry),
        ClientError::WorkerNotFound { .. }
        | ClientError::ProtocolVersion { .. }
        | ClientError::Rejected { .. }
        | ClientError::UnsupportedCommand { .. }
        | ClientError::Protocol(_)
        | ClientError::Io(_) => None,
    }
}

/// Whether this outcome deserves a saved log.
///
/// A worker that never existed leaves nothing to save; a task the worker rejected
/// on its own terms already explains itself in the report. What is worth keeping
/// is the output of a process that died or reported a crash-class failure.
fn worth_a_log(outcome: &SessionOutcome) -> bool {
    match outcome {
        SessionOutcome::SessionError(ClientError::WorkerNotFound { .. }) => false,
        SessionOutcome::SessionError(_) => true,
        SessionOutcome::Failed(error) => matches!(
            error.code,
            ErrorCode::GpuDeviceLost | ErrorCode::WorkerCrashed
        ),
        SessionOutcome::Completed { .. } | SessionOutcome::Cancelled => false,
    }
}

fn detail_of(outcome: &SessionOutcome) -> String {
    match outcome {
        SessionOutcome::SessionError(error) => error.to_string(),
        SessionOutcome::Failed(error) => format!("{}: {}", error.code.as_wire(), error.detail),
        SessionOutcome::Completed { .. } => "the task completed".to_owned(),
        SessionOutcome::Cancelled => "the task was cancelled".to_owned(),
    }
}

/// Ask the retry to continue from the last checkpoint.
///
/// Training is the only command with a checkpoint to resume from; every other
/// command re-runs from the start, which the project layout already makes safe.
fn resume_from_checkpoint(request: &mut Request) {
    if let Request::Train(params) = request {
        params.resume = true;
    }
}

fn status_of(outcome: &SupervisedOutcome) -> TaskStatus {
    match outcome {
        SupervisedOutcome::Completed { .. } => TaskStatus::Completed,
        SupervisedOutcome::Cancelled => TaskStatus::Cancelled,
        SupervisedOutcome::Failed(_)
        | SupervisedOutcome::Crashed { .. }
        | SupervisedOutcome::Unavailable(_) => TaskStatus::Failed,
    }
}
