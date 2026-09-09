//! Supervision across a real process boundary.
//!
//! The other tests in this crate replace the worker with a script, which is the
//! only practical way to cover a policy exhaustively. These two spawn a real
//! child instead, so the parts that exist only between processes — the
//! handshake, the exit status, the captured stderr, and the restart itself — are
//! proven by something other than a stub.

mod support;

use std::path::{Path, PathBuf};
use std::time::Duration;

use feathertalk_client::{
    CancelToken, ClientError, EventSink, SessionOptions, SessionOutcome, WorkerLocator,
};
use feathertalk_domain::{
    Event, ProjectDirParams, Request, TaskId, TaskKind, TrainParams, TrainingMode, UnetVariant,
};
use feathertalk_project::TaskHistoryStatus;
use feathertalk_supervisor::crash_log::CrashLogOutcome;
use feathertalk_supervisor::journal::TaskJournal;
use feathertalk_supervisor::policy::RestartPolicy;
use feathertalk_supervisor::runner::{Attempt, TaskRunner, WorkerRunner};
use feathertalk_supervisor::supervisor::{SupervisedOutcome, SupervisionReport, WorkerSupervisor};
use support::{find_entry, project_with_history, read_manifest, task_id};

/// Cargo builds the fake worker before this test binary and passes its path in.
const FAKE_WORKER: &str = env!("CARGO_BIN_EXE_feathertalk-supervised-fake-worker");

/// The file that tells the fake worker which script to follow. It lives in the
/// request's `project_dir`, so each test owns its own copy and the two can run
/// in parallel without a shared environment variable between them.
const SCENARIO_FILE: &str = "scenario.txt";

/// The two lines the crashing attempt writes to stderr before it dies.
const FIRST_STDERR_LINE: &str = "fake worker: about to abandon this attempt";
const SECOND_STDERR_LINE: &str = "fake worker: wgpu device lost while training";

fn train(project_dir: &Path) -> Request {
    Request::Train(TrainParams {
        project_dir: project_dir.to_path_buf(),
        mode: TrainingMode::Baseline,
        variant: UnetVariant::MobileOneUnet,
        epochs: 1,
        batch_size: 1,
        resume: false,
    })
}

/// Long enough to spawn a process on a loaded machine, short enough that a stuck
/// child fails this test rather than the whole run.
fn options() -> SessionOptions {
    SessionOptions {
        handshake_timeout: Duration::from_secs(5),
        cancel_grace: Duration::from_millis(500),
        shutdown_grace: Duration::from_secs(2),
        stderr_tail_lines: 20,
    }
}

fn policy() -> RestartPolicy {
    RestartPolicy {
        max_attempts: 2,
        restart_delay: Duration::ZERO,
    }
}

fn runner() -> WorkerRunner {
    WorkerRunner::new(
        WorkerLocator::from_parts(Some(PathBuf::from(FAKE_WORKER)), None, None),
        options(),
    )
    .with_env(vec![
        ("FEATHERTALK_WORKER_BACKEND".into(), "cpu".into()),
        ("FEATHERTALK_WORKER_ADAPTER".into(), "".into()),
    ])
}

#[derive(Default)]
struct CollectingSink {
    stages: Vec<String>,
}

impl EventSink for CollectingSink {
    fn on_event(&mut self, event: &Event, _raw: &str) {
        self.stages.push(event.stage.as_slug().to_owned());
    }
}

/// Run one supervised training task against the fake worker.
fn supervise(
    scenario: &str,
    project: &Path,
    logs: &Path,
    journal: Option<&TaskJournal>,
    id: &TaskId,
) -> (SupervisionReport, CollectingSink) {
    std::fs::write(project.join(SCENARIO_FILE), scenario).expect("the scenario file is writable");
    let mut supervisor = WorkerSupervisor::new(runner(), policy(), logs.to_path_buf());
    let mut sink = CollectingSink::default();
    let report = supervisor.run(
        id,
        TaskKind::Train,
        train(project),
        journal,
        &CancelToken::new(),
        &mut sink,
    );
    (report, sink)
}

#[test]
fn a_real_worker_completes_under_supervision() {
    let project = tempfile::tempdir().expect("a temporary directory");
    let logs = tempfile::tempdir().expect("a temporary directory");
    let id = task_id(1_756_000_000_000, 1);

    let (report, sink) = supervise("complete", project.path(), logs.path(), None, &id);

    assert_eq!(report.attempts, 1);
    let SupervisedOutcome::Completed { result } = report.outcome else {
        panic!("expected completion, got {:?}", report.outcome);
    };
    // The worker echoes what it was asked, which is how this test knows the
    // request survived serialisation rather than merely being accepted.
    assert_eq!(
        result,
        Some(serde_json::json!({ "attempt": 1, "resume": false }))
    );
    assert_eq!(sink.stages, vec!["preparing", "completed"]);
    assert!(
        report.crash_logs.is_empty(),
        "a clean run leaves no crash log: {:?}",
        report.crash_logs
    );
}

#[test]
fn a_real_crash_is_restarted_and_logged() {
    let project = project_with_history(Vec::new());
    let logs = tempfile::tempdir().expect("a temporary directory");
    let journal = TaskJournal::new(project.path());
    let id = task_id(1_756_000_000_001, 2);

    let (report, sink) = supervise(
        "crash-once",
        project.path(),
        logs.path(),
        Some(&journal),
        &id,
    );

    assert_eq!(report.attempts, 2);
    let SupervisedOutcome::Completed { result } = report.outcome else {
        panic!(
            "expected the second attempt to win, got {:?}",
            report.outcome
        );
    };
    // `resume` was false in the request this test handed over: the supervisor
    // flipped it, and the flag crossed a real pipe to get here.
    assert_eq!(
        result,
        Some(serde_json::json!({ "attempt": 2, "resume": true }))
    );
    // The doomed attempt's progress still reached the caller.
    assert_eq!(sink.stages, vec!["preparing", "preparing", "completed"]);

    let [CrashLogOutcome::Written(path)] = report.crash_logs.as_slice() else {
        panic!(
            "expected exactly one written crash log, got {:?}",
            report.crash_logs
        );
    };
    let log = std::fs::read_to_string(path).expect("the crash log reads back");
    for expected in [
        "command: train",
        "attempt: 1",
        "exit status: 1",
        "restart and resume from the last checkpoint",
        FIRST_STDERR_LINE,
        SECOND_STDERR_LINE,
    ] {
        assert!(log.contains(expected), "{expected} is missing from {log}");
    }

    let entry = find_entry(&read_manifest(project.path()), id.as_str());
    assert_eq!(entry.kind, "train");
    assert_eq!(entry.status, TaskHistoryStatus::Completed);
    assert!(
        report.journal_errors.is_empty(),
        "the manifest was writable: {:?}",
        report.journal_errors
    );
}

#[test]
fn the_selected_compute_environment_survives_a_supervisor_restart() {
    let project = tempfile::tempdir().expect("a temporary project");
    let logs = tempfile::tempdir().expect("a temporary log directory");
    std::fs::write(project.path().join(SCENARIO_FILE), "compute-crash-once").unwrap();
    let runner = runner().with_env(vec![
        ("FEATHERTALK_WORKER_BACKEND".into(), "wgpu".into()),
        ("FEATHERTALK_WORKER_ADAPTER".into(), "wgpu-selected".into()),
    ]);
    let mut supervisor = WorkerSupervisor::new(runner, policy(), logs.path().to_path_buf());
    let report = supervisor.run(
        &task_id(1_756_000_000_002, 3),
        TaskKind::Train,
        train(project.path()),
        None,
        &CancelToken::new(),
        &mut CollectingSink::default(),
    );
    assert!(matches!(
        report.outcome,
        SupervisedOutcome::Completed { .. }
    ));
    assert_eq!(report.attempts, 2);
    for attempt in [1, 2] {
        let received: serde_json::Value = serde_json::from_slice(
            &std::fs::read(
                project
                    .path()
                    .join(format!("compute-attempt-{attempt}.json")),
            )
            .unwrap(),
        )
        .unwrap();
        assert_eq!(
            received,
            serde_json::json!({
                "backend": "wgpu", "adapter": "wgpu-selected",
            })
        );
    }
}

fn protocol_env(directory: &Path, capabilities: &str) -> Vec<(String, String)> {
    vec![
        ("FT_SUPERVISED_CAPABILITIES".into(), capabilities.into()),
        (
            "FT_SUPERVISED_PROTOCOL_DIR".into(),
            directory.to_string_lossy().into_owned(),
        ),
    ]
}

#[test]
fn a_cpu_only_worker_rejects_a_gpu_attempt_without_starting_it() {
    let project = tempfile::tempdir().unwrap();
    let mut env = protocol_env(project.path(), "cpu-only");
    env.extend([
        ("FEATHERTALK_WORKER_BACKEND".into(), "wgpu".into()),
        ("FEATHERTALK_WORKER_ADAPTER".into(), "wgpu-selected".into()),
    ]);
    let mut runner = runner().with_env(env);
    let mut sink = CollectingSink::default();
    let result = runner.run_attempt(
        Attempt {
            task_id: &task_id(1_756_000_000_003, 4),
            request: &train(project.path()),
            attempt: 1,
        },
        &CancelToken::new(),
        &mut sink,
    );

    assert!(
        matches!(
            result.outcome,
            SessionOutcome::SessionError(ClientError::Rejected { .. })
        ),
        "{result:?}"
    );
    assert_eq!(result.exit_status, Some(0));
    assert!(sink.stages.is_empty());
    assert_eq!(
        std::fs::read_to_string(project.path().join("protocol-1.txt")).unwrap(),
        "shutdown\n"
    );
    assert!(!project.path().join("attempts.txt").exists());
}

#[test]
fn a_retry_revalidates_the_new_workers_compute_capabilities() {
    let project = tempfile::tempdir().unwrap();
    let logs = tempfile::tempdir().unwrap();
    std::fs::write(project.path().join(SCENARIO_FILE), "compute-crash-once").unwrap();
    let mut env = protocol_env(project.path(), "gpu-once");
    env.extend([
        ("FEATHERTALK_WORKER_BACKEND".into(), "wgpu".into()),
        ("FEATHERTALK_WORKER_ADAPTER".into(), "wgpu-selected".into()),
    ]);
    let mut supervisor =
        WorkerSupervisor::new(runner().with_env(env), policy(), logs.path().to_path_buf());
    let mut sink = CollectingSink::default();
    let report = supervisor.run(
        &task_id(1_756_000_000_004, 5),
        TaskKind::Train,
        train(project.path()),
        None,
        &CancelToken::new(),
        &mut sink,
    );

    assert!(
        matches!(
            report.outcome,
            SupervisedOutcome::Unavailable(ClientError::Rejected { .. })
        ),
        "{report:?}"
    );
    assert_eq!(report.attempts, 2);
    assert_eq!(sink.stages, vec!["preparing"]);
    assert_eq!(
        std::fs::read_to_string(project.path().join("protocol-1.txt")).unwrap(),
        "start\n"
    );
    assert_eq!(
        std::fs::read_to_string(project.path().join("protocol-2.txt")).unwrap(),
        "shutdown\n"
    );
    assert_eq!(
        std::fs::read_to_string(project.path().join("attempts.txt")).unwrap(),
        "1"
    );
    assert!(!project.path().join("compute-attempt-2.json").exists());
}

#[test]
fn noncompute_tasks_ignore_invalid_compute_environment() {
    let project = tempfile::tempdir().unwrap();
    let mut env = protocol_env(project.path(), "cpu-only");
    env.extend([
        (
            "FEATHERTALK_WORKER_BACKEND".into(),
            "invalid-backend".into(),
        ),
        (
            "FEATHERTALK_WORKER_ADAPTER".into(),
            "invalid-adapter".into(),
        ),
    ]);
    let mut runner = runner().with_env(env);
    let mut sink = CollectingSink::default();
    let result = runner.run_attempt(
        Attempt {
            task_id: &task_id(1_756_000_000_005, 6),
            request: &Request::ValidateProject(ProjectDirParams {
                project_dir: project.path().to_path_buf(),
            }),
            attempt: 1,
        },
        &CancelToken::new(),
        &mut sink,
    );

    assert!(
        matches!(result.outcome, SessionOutcome::Completed { .. }),
        "{result:?}"
    );
    assert_eq!(result.exit_status, Some(0));
    assert_eq!(sink.stages, vec!["preparing", "completed"]);
    assert_eq!(
        std::fs::read_to_string(project.path().join("protocol-1.txt")).unwrap(),
        "start\nshutdown\n"
    );
}

#[test]
fn inherited_and_partially_overridden_compute_environment_is_validated() {
    const CHILD: &str = "FT_SUPERVISED_INHERITED_COMPUTE_CHILD";
    if std::env::var_os(CHILD).is_none() {
        // A separate test process gives this runner inherited values without
        // mutating the environment shared by parallel Rust tests.
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "inherited_and_partially_overridden_compute_environment_is_validated",
                "--nocapture",
            ])
            .env(CHILD, "1")
            .env("FEATHERTALK_WORKER_BACKEND", "wgpu")
            .env("FEATHERTALK_WORKER_ADAPTER", "wgpu-selected")
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "stdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        return;
    }

    for (backend_override, expected_reason) in [
        (None, "not advertised"),
        (Some("cpu"), "CPU selection requires cpu-0"),
    ] {
        let project = tempfile::tempdir().unwrap();
        let mut env = protocol_env(project.path(), "cpu-only");
        if let Some(backend) = backend_override {
            env.push(("FEATHERTALK_WORKER_BACKEND".into(), backend.into()));
        }
        let mut runner = runner().with_env(env);
        let mut sink = CollectingSink::default();
        let result = runner.run_attempt(
            Attempt {
                task_id: &task_id(1_756_000_000_006, 7),
                request: &train(project.path()),
                attempt: 1,
            },
            &CancelToken::new(),
            &mut sink,
        );
        assert!(
            matches!(result.outcome, SessionOutcome::SessionError(ClientError::Rejected { ref reason })
                if reason.contains(expected_reason)),
            "{backend_override:?}: {result:?}"
        );
        assert_eq!(result.exit_status, Some(0));
        assert!(sink.stages.is_empty());
        assert_eq!(
            std::fs::read_to_string(project.path().join("protocol-1.txt")).unwrap(),
            "shutdown\n"
        );
    }
}
