#[path = "support/harness.rs"]
mod harness;

use std::path::PathBuf;

use feathertalk_client::{
    CancelToken, ClientError, EventSink, SessionOutcome, WorkerSession, generate_task_id,
};
use feathertalk_domain::{
    DomainError, ErrorCode, Event, ProbeMediaParams, ProjectDirParams, Request,
};

use harness::{fake_worker, fast_options, scenario};

/// Records what the session reported, in order.
#[derive(Default)]
struct Collected {
    stages: Vec<String>,
    raw: Vec<String>,
}

impl EventSink for Collected {
    fn on_event(&mut self, event: &Event, raw: &str) {
        self.stages.push(event.stage.as_slug().to_string());
        self.raw.push(raw.to_string());
    }
}

fn validate_project() -> Request {
    Request::ValidateProject(ProjectDirParams {
        project_dir: PathBuf::from("project-dir-the-fake-worker-never-reads"),
    })
}

fn probe_media() -> Request {
    Request::ProbeMedia(ProbeMediaParams {
        input: PathBuf::from("input-the-fake-worker-never-reads.mp4"),
    })
}

/// Spawn, run one task, and report the outcome, the sink, and the foreign count.
fn run_scenario(name: &str, request: Request) -> (SessionOutcome, Collected, usize) {
    let mut session =
        WorkerSession::spawn_with_env(&fake_worker(), fast_options(), &scenario(name))
            .expect("the handshake succeeds");
    let mut sink = Collected::default();
    let outcome = session.run(
        generate_task_id().expect("the generated id is valid"),
        request,
        &CancelToken::new(),
        &mut sink,
    );
    let foreign = session.foreign_event_count();
    (outcome, sink, foreign)
}

#[test]
fn a_completed_task_carries_the_workers_result() {
    let (outcome, sink, foreign) = run_scenario("ready-complete", validate_project());
    let SessionOutcome::Completed { result } = outcome else {
        panic!("expected completion, got {outcome:?}");
    };
    assert_eq!(result, Some(serde_json::json!({ "checked": true })));
    assert_eq!(sink.stages, vec!["preparing", "completed"]);
    assert_eq!(foreign, 0);
    assert!(
        sink.raw
            .iter()
            .all(|line| line.contains("\"frame\":\"event\"")),
        "the sink receives the worker's own bytes: {:?}",
        sink.raw
    );
}

#[test]
fn a_failed_task_carries_the_workers_error() {
    let (outcome, sink, _) = run_scenario("fail", validate_project());
    let SessionOutcome::Failed(error) = outcome else {
        panic!("expected a task failure, got {outcome:?}");
    };
    assert_eq!(error.code, ErrorCode::MediaInvalid);
    assert_eq!(error.summary, "输入文件无法解析，请确认它是完整的视频");
    assert_eq!(error.detail, "ffprobe exited with status 1");
    assert_eq!(sink.stages, vec!["failed"]);
}

#[test]
fn a_cancelled_stage_is_a_cancelled_outcome() {
    let (outcome, sink, _) = run_scenario("self-cancel", validate_project());
    assert!(
        matches!(outcome, SessionOutcome::Cancelled),
        "expected cancellation, got {outcome:?}"
    );
    assert_eq!(sink.stages, vec!["cancelled"]);
}

#[test]
fn an_unsupported_command_is_refused_before_it_is_sent() {
    let (outcome, sink, _) = run_scenario("only-validate", probe_media());
    let SessionOutcome::SessionError(ClientError::UnsupportedCommand {
        requested,
        supported,
    }) = outcome
    else {
        panic!("expected the capability gate to refuse, got {outcome:?}");
    };
    assert_eq!(requested, "probe_media");
    assert_eq!(supported, vec!["validate_project"]);
    assert!(sink.stages.is_empty(), "nothing was sent, so nothing ran");
}

#[test]
fn a_supported_command_passes_the_gate() {
    let (outcome, _, _) = run_scenario("only-probe", probe_media());
    assert!(
        matches!(outcome, SessionOutcome::Completed { .. }),
        "expected completion, got {outcome:?}"
    );
}

#[test]
fn an_event_for_another_task_is_ignored_and_counted() {
    let (outcome, sink, foreign) = run_scenario("foreign-event", validate_project());
    assert!(
        matches!(outcome, SessionOutcome::Completed { .. }),
        "expected completion, got {outcome:?}"
    );
    assert_eq!(sink.stages, vec!["completed"]);
    assert_eq!(foreign, 1);
}

#[test]
fn a_worker_that_exits_mid_task_is_reported_as_gone() {
    let (outcome, _, _) = run_scenario("die-after-ready", validate_project());
    assert!(
        matches!(
            outcome,
            SessionOutcome::SessionError(ClientError::WorkerGone { .. })
        ),
        "expected a lost worker, got {outcome:?}"
    );
}

#[test]
fn an_oversized_line_is_a_protocol_error() {
    let (outcome, _, _) = run_scenario("oversized-line", validate_project());
    assert!(
        matches!(
            outcome,
            SessionOutcome::SessionError(ClientError::Protocol(DomainError::FrameTooLong { .. }))
        ),
        "expected a frame bound violation, got {outcome:?}"
    );
}
