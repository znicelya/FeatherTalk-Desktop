use std::path::PathBuf;

use feathertalk_client::{
    CancelToken, ClientError, EventSink, SessionOptions, SessionOutcome, WorkerLocator,
    WorkerSession,
};
use feathertalk_domain::{Request, TaskId};

/// The inputs of one attempt at a task.
#[derive(Debug)]
pub struct Attempt<'a> {
    pub task_id: &'a TaskId,
    pub request: &'a Request,
    /// One-based, so the first run is attempt 1.
    pub attempt: u32,
}

/// The outputs of one attempt, including what a crash log needs.
#[derive(Debug)]
pub struct AttemptResult {
    pub outcome: SessionOutcome,
    pub stderr_tail: Vec<String>,
    /// `None` when discovery failed and no executable was ever chosen.
    pub worker_path: Option<PathBuf>,
    pub exit_status: Option<i32>,
}

/// The process boundary of the supervisor.
///
/// Everything above this trait is policy — when to restart, what to keep, what to
/// tell the caller — and everything below it is a child process. Tests implement
/// the trait with a script, which is why the restart rules can be exercised
/// without spawning anything.
pub trait TaskRunner {
    fn run_attempt(
        &mut self,
        attempt: Attempt<'_>,
        cancel: &CancelToken,
        sink: &mut dyn EventSink,
    ) -> AttemptResult;
}

/// Runs one attempt as a real worker process: discover, spawn, hand shake, run
/// one task, then reap.
#[derive(Debug, Clone)]
pub struct WorkerRunner {
    locator: WorkerLocator,
    options: SessionOptions,
    env: Vec<(String, String)>,
}

impl WorkerRunner {
    pub fn new(locator: WorkerLocator, options: SessionOptions) -> Self {
        Self {
            locator,
            options,
            env: Vec::new(),
        }
    }

    /// Retain child-only configuration across every supervised attempt.
    pub fn with_env(mut self, env: Vec<(String, String)>) -> Self {
        self.env = env;
        self
    }
}

impl TaskRunner for WorkerRunner {
    fn run_attempt(
        &mut self,
        attempt: Attempt<'_>,
        cancel: &CancelToken,
        sink: &mut dyn EventSink,
    ) -> AttemptResult {
        let path = match self.locator.resolve() {
            Ok(path) => path,
            Err(error) => return before_the_task(error, None),
        };
        let mut session =
            match WorkerSession::spawn_with_env(&path, self.options.clone(), &self.env) {
                Ok(session) => session,
                Err(error) => return before_the_task(error, Some(path)),
            };

        // The session checks this fresh handshake against the effective child
        // compute configuration before Start, including on a restarted worker.
        // A rejected choice still follows the normal shutdown/reap path below.
        let outcome = session.run(
            attempt.task_id.clone(),
            attempt.request.clone(),
            cancel,
            sink,
        );
        // Read the tail before shutting down: `shutdown` consumes the session,
        // and a worker that died has already written whatever it is going to say.
        let stderr_tail = session.stderr_tail();
        let exit_status = session.shutdown();
        AttemptResult {
            outcome,
            stderr_tail,
            worker_path: Some(path),
            exit_status,
        }
    }
}

/// A failure before the task could start, reported through the same channel as a
/// mid-task failure so the supervisor has one classification path.
fn before_the_task(error: ClientError, worker_path: Option<PathBuf>) -> AttemptResult {
    let stderr_tail = error.stderr_tail().to_vec();
    AttemptResult {
        outcome: SessionOutcome::SessionError(error),
        stderr_tail,
        worker_path,
        exit_status: None,
    }
}
