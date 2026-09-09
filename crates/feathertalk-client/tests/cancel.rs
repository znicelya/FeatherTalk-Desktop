#[path = "support/harness.rs"]
mod harness;

use std::path::PathBuf;
use std::time::{Duration, Instant};

use feathertalk_client::{CancelToken, EventSink, SessionOutcome, WorkerSession, generate_task_id};
use feathertalk_domain::{Event, ProjectDirParams, Request};

use harness::{fake_worker, fast_options, scenario};

/// The cancel tests only care about outcomes, not about event text.
struct Ignore;

impl EventSink for Ignore {
    fn on_event(&mut self, event: &Event, raw: &str) {
        let _ = (event, raw);
    }
}

fn validate_project() -> Request {
    Request::ValidateProject(ProjectDirParams {
        project_dir: PathBuf::from("project-dir-the-fake-worker-never-reads"),
    })
}

/// Run one task with `requests` cancel requests already registered.
///
/// Requesting before `run` rather than from a thread keeps these tests
/// deterministic: the token is checked at the top of every loop iteration, so a
/// pre-registered request is seen on the first one.
fn run_cancelled(name: &str, requests: usize) -> (SessionOutcome, Duration) {
    let mut session =
        WorkerSession::spawn_with_env(&fake_worker(), fast_options(), &scenario(name))
            .expect("the handshake succeeds");
    let cancel = CancelToken::new();
    for _ in 0..requests {
        cancel.request();
    }
    let started = Instant::now();
    let outcome = session.run(
        generate_task_id().expect("the generated id is valid"),
        validate_project(),
        &cancel,
        &mut Ignore,
    );
    (outcome, started.elapsed())
}

#[test]
fn a_worker_that_acknowledges_is_reported_cancelled() {
    let (outcome, elapsed) = run_cancelled("cancel-acks", 1);
    assert!(
        matches!(outcome, SessionOutcome::Cancelled),
        "expected cancellation, got {outcome:?}"
    );
    assert!(
        elapsed < Duration::from_secs(2),
        "a cooperative worker should not need the grace period: {elapsed:?}"
    );
}

#[test]
fn a_completion_in_flight_beats_a_cancel() {
    let (outcome, _) = run_cancelled("cancel-completes", 1);
    assert!(
        matches!(outcome, SessionOutcome::Completed { .. }),
        "work that finished must not be reported as cancelled, got {outcome:?}"
    );
}

#[test]
fn an_unresponsive_worker_is_stopped_when_the_grace_expires() {
    let (outcome, elapsed) = run_cancelled("cancel-ignored", 1);
    assert!(
        matches!(outcome, SessionOutcome::Cancelled),
        "expected cancellation, got {outcome:?}"
    );
    assert!(
        elapsed >= Duration::from_millis(200),
        "the worker must be given the full grace period first: {elapsed:?}"
    );
    assert!(
        elapsed < Duration::from_secs(5),
        "the escalation must be bounded: {elapsed:?}"
    );
}

#[test]
fn a_worker_that_exits_after_a_cancel_is_reported_cancelled() {
    let (outcome, _) = run_cancelled("die-on-cancel", 1);
    assert!(
        matches!(outcome, SessionOutcome::Cancelled),
        "EOF after a cancel is a cancellation, not a lost worker: got {outcome:?}"
    );
}

#[test]
fn a_second_request_kills_without_waiting() {
    let (outcome, elapsed) = run_cancelled("cancel-ignored", 2);
    assert!(
        matches!(outcome, SessionOutcome::Cancelled),
        "expected cancellation, got {outcome:?}"
    );
    assert!(
        elapsed < Duration::from_millis(200),
        "a second request must skip the grace period entirely: {elapsed:?}"
    );
}
