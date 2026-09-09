use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use feathertalk_app::pipeline::{Job, Submission, TaskUpdate, submit};
use feathertalk_client::{CancelToken, ClientError, EventSink, SessionOutcome};
use feathertalk_domain::{Event, ProjectDirParams, Request, TaskId, TaskKind, TaskStage};
use feathertalk_supervisor::policy::RestartPolicy;
use feathertalk_supervisor::runner::{Attempt, AttemptResult, TaskRunner};
use feathertalk_supervisor::supervisor::SupervisedOutcome;

fn task_id() -> TaskId {
    TaskId::parse("1756000000000-00000001").expect("a well formed task id")
}

fn job(log_dir: PathBuf) -> Job {
    Job {
        task_id: task_id(),
        kind: TaskKind::ValidateProject,
        request: Request::ValidateProject(ProjectDirParams {
            project_dir: PathBuf::from("C:/projects/demo"),
        }),
        policy: RestartPolicy {
            max_attempts: 2,
            restart_delay: Duration::ZERO,
        },
        log_dir,
        // No task history: this test drives the channel, not the manifest.
        journal: None,
    }
}

fn crash() -> SessionOutcome {
    SessionOutcome::SessionError(ClientError::WorkerGone {
        status: Some(101),
        stderr_tail: vec!["thread 'main' panicked".to_owned()],
    })
}

/// One canned attempt: stages to emit, then the outcome to hand back.
struct Step {
    stages: Vec<TaskStage>,
    outcome: SessionOutcome,
}

impl Step {
    fn new(outcome: SessionOutcome) -> Self {
        Self {
            stages: Vec::new(),
            outcome,
        }
    }

    fn with_stages(mut self, stages: &[TaskStage]) -> Self {
        self.stages = stages.to_vec();
        self
    }
}

/// A runner that replays a script instead of spawning a process. Real processes
/// are covered by the supervisor crate's own end-to-end test; what matters here
/// is the thread and the channel.
struct ScriptedRunner {
    steps: VecDeque<Step>,
    /// Finished attempts, so a test can watch a thread it no longer listens to.
    attempts: Arc<AtomicUsize>,
}

impl ScriptedRunner {
    fn new(steps: Vec<Step>) -> (Self, Arc<AtomicUsize>) {
        let attempts = Arc::new(AtomicUsize::new(0));
        let runner = Self {
            steps: steps.into(),
            attempts: Arc::clone(&attempts),
        };
        (runner, attempts)
    }
}

impl TaskRunner for ScriptedRunner {
    fn run_attempt(
        &mut self,
        attempt: Attempt<'_>,
        _cancel: &CancelToken,
        sink: &mut dyn EventSink,
    ) -> AttemptResult {
        let step = self
            .steps
            .pop_front()
            .expect("the script has a step for this attempt");
        for stage in step.stages {
            let event = Event::new(attempt.task_id.clone(), "2026-09-05T13:00:00Z", stage);
            sink.on_event(&event, "{}");
        }
        self.attempts.fetch_add(1, Ordering::SeqCst);
        AttemptResult {
            outcome: step.outcome,
            stderr_tail: Vec::new(),
            worker_path: Some(PathBuf::from("C:/tools/feathertalk-worker.exe")),
            exit_status: Some(0),
        }
    }
}

/// A runner that waits for the cancel token instead of finishing on its own.
struct WaitingRunner;

impl TaskRunner for WaitingRunner {
    fn run_attempt(
        &mut self,
        _attempt: Attempt<'_>,
        cancel: &CancelToken,
        _sink: &mut dyn EventSink,
    ) -> AttemptResult {
        let deadline = Instant::now() + Duration::from_secs(5);
        while cancel.count() == 0 && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(10));
        }
        AttemptResult {
            outcome: SessionOutcome::Cancelled,
            stderr_tail: Vec::new(),
            worker_path: None,
            exit_status: None,
        }
    }
}

/// Collect every update until the supervision thread closes the channel.
fn drain(updates: &async_channel::Receiver<TaskUpdate>) -> Vec<TaskUpdate> {
    let mut collected = Vec::new();
    while let Ok(update) = updates.recv_blocking() {
        collected.push(update);
    }
    collected
}

/// The stages of every progress update, in the order they arrived.
fn stages(updates: &[TaskUpdate]) -> Vec<TaskStage> {
    updates
        .iter()
        .filter_map(|update| match update {
            TaskUpdate::Progress(event) => Some(event.stage.clone()),
            TaskUpdate::Finished(_) => None,
        })
        .collect()
}

/// The one report a finished run ends with.
fn report(updates: Vec<TaskUpdate>) -> Box<feathertalk_supervisor::SupervisionReport> {
    let last = updates
        .into_iter()
        .next_back()
        .expect("a finished run sends at least one update");
    match last {
        TaskUpdate::Finished(report) => report,
        TaskUpdate::Progress(event) => panic!("the last update is a progress event: {event:?}"),
    }
}

#[test]
fn a_completed_task_reports_its_events_then_one_report() {
    let logs = tempfile::tempdir().expect("a temporary directory");
    let (runner, _attempts) = ScriptedRunner::new(vec![
        Step::new(SessionOutcome::Completed { result: None })
            .with_stages(&[TaskStage::Preparing, TaskStage::Completed]),
    ]);

    let submission = submit(runner, job(logs.path().to_path_buf())).expect("a thread starts");

    let updates = drain(&submission.updates);
    assert_eq!(
        stages(&updates),
        vec![TaskStage::Preparing, TaskStage::Completed]
    );
    let report = report(updates);
    assert_eq!(report.attempts, 1);
    assert!(matches!(
        report.outcome,
        SupervisedOutcome::Completed { .. }
    ));
}

#[test]
fn a_crash_is_restarted_and_both_attempts_are_visible() {
    let logs = tempfile::tempdir().expect("a temporary directory");
    let (runner, _attempts) = ScriptedRunner::new(vec![
        Step::new(crash()).with_stages(&[TaskStage::Preparing]),
        Step::new(SessionOutcome::Completed { result: None })
            .with_stages(&[TaskStage::Preparing, TaskStage::Completed]),
    ]);

    let submission = submit(runner, job(logs.path().to_path_buf())).expect("a thread starts");

    let updates = drain(&submission.updates);
    // The window sees the restart as a stage that repeats, which is exactly what
    // happened: the first attempt got as far as preparing and died.
    assert_eq!(
        stages(&updates),
        vec![
            TaskStage::Preparing,
            TaskStage::Preparing,
            TaskStage::Completed
        ]
    );
    assert_eq!(report(updates).attempts, 2);
}

#[test]
fn a_cancelled_submission_reaches_the_worker() {
    let logs = tempfile::tempdir().expect("a temporary directory");

    let submission =
        submit(WaitingRunner, job(logs.path().to_path_buf())).expect("a thread starts");
    submission.cancel.request();

    let report = report(drain(&submission.updates));
    assert!(matches!(report.outcome, SupervisedOutcome::Cancelled));
    assert_eq!(report.attempts, 1);
}

#[test]
fn a_dropped_receiver_does_not_stop_the_run() {
    let logs = tempfile::tempdir().expect("a temporary directory");
    let (runner, attempts) = ScriptedRunner::new(vec![
        Step::new(SessionOutcome::Completed { result: None }).with_stages(&[TaskStage::Preparing]),
    ]);

    let Submission { updates, cancel } =
        submit(runner, job(logs.path().to_path_buf())).expect("a thread starts");
    // Losing the window is not a reason to abandon a checkpoint.
    drop(updates);

    let deadline = Instant::now() + Duration::from_secs(2);
    while attempts.load(Ordering::SeqCst) == 0 && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(10));
    }
    assert_eq!(attempts.load(Ordering::SeqCst), 1);
    assert_eq!(cancel.count(), 0, "nothing cancelled the run");
}
