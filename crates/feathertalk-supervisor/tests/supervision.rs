mod support;

use std::collections::VecDeque;
use std::path::PathBuf;
use std::time::Duration;

use feathertalk_client::{CancelToken, ClientError, EventSink, SessionOutcome};
use feathertalk_domain::{
    ErrorCode, Event, ProbeMediaParams, Request, TaskError, TaskKind, TaskStage, TrainParams,
    TrainingMode, UnetVariant,
};
use feathertalk_project::TaskHistoryStatus;
use feathertalk_supervisor::journal::TaskJournal;
use feathertalk_supervisor::policy::RestartPolicy;
use feathertalk_supervisor::runner::{Attempt, AttemptResult, TaskRunner};
use feathertalk_supervisor::supervisor::{SupervisedOutcome, WorkerSupervisor};
use support::{find_entry, project_with_history, read_manifest, task_id};
use time::OffsetDateTime;
use time::macros::datetime;

fn fixed_clock() -> OffsetDateTime {
    datetime!(2026-09-05 13:00 UTC)
}

fn fast_policy() -> RestartPolicy {
    RestartPolicy {
        max_attempts: 2,
        restart_delay: Duration::ZERO,
    }
}

fn train(resume: bool) -> Request {
    Request::Train(TrainParams {
        project_dir: PathBuf::from("C:/projects/demo"),
        mode: TrainingMode::Baseline,
        variant: UnetVariant::MobileOneUnet,
        epochs: 1,
        batch_size: 1,
        resume,
    })
}

fn probe() -> Request {
    Request::ProbeMedia(ProbeMediaParams {
        input: PathBuf::from("C:/media/clip.mp4"),
    })
}

fn crash() -> SessionOutcome {
    SessionOutcome::SessionError(ClientError::WorkerGone {
        status: Some(101),
        stderr_tail: vec!["thread 'main' panicked".to_owned()],
    })
}

/// One canned attempt: events to replay, then the outcome to hand back.
struct Step {
    events: Vec<TaskStage>,
    outcome: SessionOutcome,
}

impl Step {
    fn new(outcome: SessionOutcome) -> Self {
        Self {
            events: Vec::new(),
            outcome,
        }
    }

    fn with_events(mut self, stages: &[TaskStage]) -> Self {
        self.events = stages.to_vec();
        self
    }
}

/// A runner that replays a script instead of spawning a process, so the restart
/// rules can be tested without a worker binary.
struct ScriptedRunner {
    steps: VecDeque<Step>,
    seen: Vec<Request>,
}

impl ScriptedRunner {
    fn new(steps: Vec<Step>) -> Self {
        Self {
            steps: steps.into(),
            seen: Vec::new(),
        }
    }
}

impl TaskRunner for ScriptedRunner {
    fn run_attempt(
        &mut self,
        attempt: Attempt<'_>,
        _cancel: &CancelToken,
        sink: &mut dyn EventSink,
    ) -> AttemptResult {
        self.seen.push(attempt.request.clone());
        let step = self
            .steps
            .pop_front()
            .expect("the script has a step for this attempt");
        for stage in step.events {
            let event = Event::new(attempt.task_id.clone(), "2026-09-05T13:00:00Z", stage);
            sink.on_event(&event, "{}");
        }
        AttemptResult {
            outcome: step.outcome,
            stderr_tail: vec!["thread 'main' panicked".to_owned()],
            worker_path: Some(PathBuf::from("C:/tools/feathertalk-worker.exe")),
            exit_status: Some(101),
        }
    }
}

#[derive(Default)]
struct CollectingSink {
    stages: Vec<TaskStage>,
}

impl EventSink for CollectingSink {
    fn on_event(&mut self, event: &Event, _raw: &str) {
        self.stages.push(event.stage.clone());
    }
}

#[test]
fn a_crash_is_retried_and_the_second_attempt_wins() {
    let logs = tempfile::tempdir().expect("a temporary directory");
    let runner = ScriptedRunner::new(vec![
        Step::new(crash()),
        Step::new(SessionOutcome::Completed { result: None }),
    ]);
    let mut supervisor = WorkerSupervisor::new(runner, fast_policy(), logs.path().to_path_buf())
        .with_clock(fixed_clock);
    let id = task_id(1_756_000_000_000, 1);
    let mut sink = CollectingSink::default();

    let report = supervisor.run(
        &id,
        TaskKind::Train,
        train(false),
        None,
        &CancelToken::new(),
        &mut sink,
    );

    assert_eq!(report.attempts, 2);
    assert!(matches!(
        report.outcome,
        SupervisedOutcome::Completed { .. }
    ));
    assert_eq!(report.crash_logs.len(), 1);
    let written = std::fs::read_dir(logs.path())
        .expect("the log directory exists")
        .count();
    assert_eq!(written, 1);
}

#[test]
fn a_retried_training_task_asks_to_resume() {
    let logs = tempfile::tempdir().expect("a temporary directory");
    let runner = ScriptedRunner::new(vec![
        Step::new(crash()),
        Step::new(SessionOutcome::Completed { result: None }),
    ]);
    let mut supervisor = WorkerSupervisor::new(runner, fast_policy(), logs.path().to_path_buf())
        .with_clock(fixed_clock);
    let id = task_id(1_756_000_000_000, 2);

    supervisor.run(
        &id,
        TaskKind::Train,
        train(false),
        None,
        &CancelToken::new(),
        &mut CollectingSink::default(),
    );

    let seen = &supervisor.runner().seen;
    assert_eq!(seen.len(), 2);
    assert_eq!(seen[0], train(false));
    assert_eq!(
        seen[1],
        train(true),
        "the retry must resume the training run"
    );
}

#[test]
fn a_retried_command_without_checkpoints_is_resent_unchanged() {
    let logs = tempfile::tempdir().expect("a temporary directory");
    let runner = ScriptedRunner::new(vec![
        Step::new(crash()),
        Step::new(SessionOutcome::Completed { result: None }),
    ]);
    let mut supervisor = WorkerSupervisor::new(runner, fast_policy(), logs.path().to_path_buf())
        .with_clock(fixed_clock);
    let id = task_id(1_756_000_000_000, 3);

    supervisor.run(
        &id,
        TaskKind::ProbeMedia,
        probe(),
        None,
        &CancelToken::new(),
        &mut CollectingSink::default(),
    );

    let seen = &supervisor.runner().seen;
    assert_eq!(seen, &[probe(), probe()]);
}

#[test]
fn an_exhausted_budget_reports_a_crash() {
    let logs = tempfile::tempdir().expect("a temporary directory");
    let runner = ScriptedRunner::new(vec![
        Step::new(crash()).with_events(&[TaskStage::Preparing]),
        Step::new(crash()),
    ]);
    let mut supervisor = WorkerSupervisor::new(runner, fast_policy(), logs.path().to_path_buf())
        .with_clock(fixed_clock);
    let id = task_id(1_756_000_000_000, 4);

    let report = supervisor.run(
        &id,
        TaskKind::Train,
        train(false),
        None,
        &CancelToken::new(),
        &mut CollectingSink::default(),
    );

    assert_eq!(report.attempts, 2);
    match report.outcome {
        SupervisedOutcome::Crashed {
            stage,
            recovery,
            detail,
        } => {
            assert_eq!(stage, TaskStage::Preparing);
            assert_eq!(recovery, feathertalk_domain::Recovery::ResumeFromCheckpoint);
            assert!(detail.contains("worker"));
        }
        other => panic!("expected a crash, got {other:?}"),
    }
    assert_eq!(report.crash_logs.len(), 2);
}

#[test]
fn a_missing_worker_is_unavailable_and_never_retried() {
    let logs = tempfile::tempdir().expect("a temporary directory");
    let runner = ScriptedRunner::new(vec![Step::new(SessionOutcome::SessionError(
        ClientError::WorkerNotFound { probed: Vec::new() },
    ))]);
    let mut supervisor = WorkerSupervisor::new(runner, fast_policy(), logs.path().to_path_buf())
        .with_clock(fixed_clock);
    let id = task_id(1_756_000_000_000, 5);

    let report = supervisor.run(
        &id,
        TaskKind::Train,
        train(false),
        None,
        &CancelToken::new(),
        &mut CollectingSink::default(),
    );

    assert_eq!(report.attempts, 1);
    assert!(matches!(
        report.outcome,
        SupervisedOutcome::Unavailable(ClientError::WorkerNotFound { .. })
    ));
    assert!(
        report.crash_logs.is_empty(),
        "nothing crashed, so there is no last log to keep"
    );
    assert_eq!(supervisor.runner().seen.len(), 1);
}

#[test]
fn a_task_failure_from_the_worker_is_passed_through() {
    let logs = tempfile::tempdir().expect("a temporary directory");
    let failure = TaskError::new(
        ErrorCode::MediaInvalid,
        "素材无法解码",
        "ffprobe rejected the container",
        TaskStage::Preparing,
    );
    let runner = ScriptedRunner::new(vec![Step::new(SessionOutcome::Failed(failure))]);
    let mut supervisor = WorkerSupervisor::new(runner, fast_policy(), logs.path().to_path_buf())
        .with_clock(fixed_clock);
    let id = task_id(1_756_000_000_000, 6);

    let report = supervisor.run(
        &id,
        TaskKind::NormalizeMedia,
        probe(),
        None,
        &CancelToken::new(),
        &mut CollectingSink::default(),
    );

    assert_eq!(report.attempts, 1);
    match report.outcome {
        SupervisedOutcome::Failed(error) => assert_eq!(error.code, ErrorCode::MediaInvalid),
        other => panic!("expected the worker's own failure, got {other:?}"),
    }
    assert!(report.crash_logs.is_empty());
}

#[test]
fn the_journal_follows_the_task() {
    let logs = tempfile::tempdir().expect("a temporary directory");
    let project = project_with_history(Vec::new());
    let journal = TaskJournal::new(project.path());
    let runner = ScriptedRunner::new(vec![
        Step::new(crash()).with_events(&[TaskStage::Preparing]),
        Step::new(SessionOutcome::Completed { result: None }).with_events(&[TaskStage::Training {
            epoch: 1,
            step: 10,
            loss: 0.5,
        }]),
    ]);
    let mut supervisor = WorkerSupervisor::new(runner, fast_policy(), logs.path().to_path_buf())
        .with_clock(fixed_clock);
    let id = task_id(1_756_000_000_000, 7);

    let report = supervisor.run(
        &id,
        TaskKind::Train,
        train(false),
        Some(&journal),
        &CancelToken::new(),
        &mut CollectingSink::default(),
    );

    assert!(report.journal_errors.is_empty());
    let recorded = find_entry(&read_manifest(project.path()), id.as_str());
    assert_eq!(recorded.status, TaskHistoryStatus::Completed);
    assert_eq!(recorded.kind, "train");
}

#[test]
fn a_crash_leaves_the_task_failed_in_the_journal() {
    let logs = tempfile::tempdir().expect("a temporary directory");
    let project = project_with_history(Vec::new());
    let journal = TaskJournal::new(project.path());
    let runner = ScriptedRunner::new(vec![Step::new(crash()), Step::new(crash())]);
    let mut supervisor = WorkerSupervisor::new(runner, fast_policy(), logs.path().to_path_buf())
        .with_clock(fixed_clock);
    let id = task_id(1_756_000_000_000, 8);

    supervisor.run(
        &id,
        TaskKind::Train,
        train(false),
        Some(&journal),
        &CancelToken::new(),
        &mut CollectingSink::default(),
    );

    assert_eq!(
        find_entry(&read_manifest(project.path()), id.as_str()).status,
        TaskHistoryStatus::Failed
    );
}

#[test]
fn events_reach_the_caller_sink() {
    let logs = tempfile::tempdir().expect("a temporary directory");
    let runner = ScriptedRunner::new(vec![
        Step::new(SessionOutcome::Completed { result: None })
            .with_events(&[TaskStage::Preparing, TaskStage::ExtractingAudio]),
    ]);
    let mut supervisor = WorkerSupervisor::new(runner, fast_policy(), logs.path().to_path_buf())
        .with_clock(fixed_clock);
    let id = task_id(1_756_000_000_000, 9);
    let mut sink = CollectingSink::default();

    supervisor.run(
        &id,
        TaskKind::NormalizeMedia,
        probe(),
        None,
        &CancelToken::new(),
        &mut sink,
    );

    assert_eq!(
        sink.stages,
        vec![TaskStage::Preparing, TaskStage::ExtractingAudio]
    );
}
