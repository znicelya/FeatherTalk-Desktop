use std::io;
use std::time::Duration;

use feathertalk_client::{ClientError, ProbedPath, SessionOutcome, WorkerPathSource};
use feathertalk_domain::{DomainError, ErrorCode, TaskError, TaskStage};
use feathertalk_supervisor::policy::{Disposition, RestartPolicy, classify};

fn failed(code: ErrorCode) -> SessionOutcome {
    SessionOutcome::Failed(TaskError::new(
        code,
        "任务失败",
        "detail for the test",
        TaskStage::Preparing,
    ))
}

fn session_error(error: ClientError) -> SessionOutcome {
    SessionOutcome::SessionError(error)
}

/// Every `ClientError` variant, so the matrix test cannot silently miss one.
fn every_client_error() -> Vec<(&'static str, ClientError)> {
    vec![
        (
            "worker_not_found",
            ClientError::WorkerNotFound {
                probed: vec![ProbedPath {
                    source: WorkerPathSource::EnvVar,
                    path: None,
                }],
            },
        ),
        (
            "spawn",
            ClientError::Spawn {
                path: "worker.exe".into(),
                source: io::Error::new(io::ErrorKind::PermissionDenied, "denied"),
            },
        ),
        (
            "handshake",
            ClientError::Handshake {
                reason: "no ready frame".to_owned(),
                stderr_tail: vec!["panicked at startup".to_owned()],
            },
        ),
        (
            "protocol_version",
            ClientError::ProtocolVersion {
                expected: 2,
                actual: 3,
            },
        ),
        (
            "rejected",
            ClientError::Rejected {
                reason: "unknown command".to_owned(),
            },
        ),
        (
            "unsupported_command",
            ClientError::UnsupportedCommand {
                requested: "train",
                supported: vec!["probe_media"],
            },
        ),
        (
            "protocol",
            ClientError::Protocol(DomainError::MalformedFrame {
                reason: "not json".to_owned(),
            }),
        ),
        ("io", ClientError::Io(io::Error::other("pipe broke"))),
        (
            "worker_gone",
            ClientError::WorkerGone {
                status: Some(101),
                stderr_tail: vec!["thread 'main' panicked".to_owned()],
            },
        ),
    ]
}

#[test]
fn a_terminal_session_finishes() {
    let policy = RestartPolicy::default();
    assert_eq!(
        classify(&SessionOutcome::Completed { result: None }, 1, &policy),
        Disposition::Finish
    );
    assert_eq!(
        classify(&SessionOutcome::Cancelled, 1, &policy),
        Disposition::Finish
    );
}

#[test]
fn only_the_two_crash_codes_restart() {
    let policy = RestartPolicy::default();
    for code in ErrorCode::ALL {
        let expected = match code {
            ErrorCode::GpuDeviceLost | ErrorCode::WorkerCrashed => {
                Disposition::Restart { resume: true }
            }
            _ => Disposition::Finish,
        };
        assert_eq!(
            classify(&failed(code), 1, &policy),
            expected,
            "{} was classified wrongly",
            code.as_wire()
        );
    }
}

#[test]
fn every_client_error_has_a_decided_disposition() {
    let policy = RestartPolicy::default();
    for (name, error) in every_client_error() {
        let expected = match name {
            "worker_gone" => Disposition::Restart { resume: true },
            "handshake" | "spawn" => Disposition::Restart { resume: false },
            _ => Disposition::GiveUp,
        };
        assert_eq!(
            classify(&session_error(error), 1, &policy),
            expected,
            "{name} was classified wrongly"
        );
    }
}

#[test]
fn the_attempt_budget_turns_restart_into_give_up() {
    let policy = RestartPolicy {
        max_attempts: 2,
        restart_delay: Duration::ZERO,
    };
    let crash = || {
        session_error(ClientError::WorkerGone {
            status: None,
            stderr_tail: Vec::new(),
        })
    };
    assert_eq!(
        classify(&crash(), 1, &policy),
        Disposition::Restart { resume: true }
    );
    assert_eq!(classify(&crash(), 2, &policy), Disposition::GiveUp);
    assert_eq!(
        classify(&failed(ErrorCode::GpuDeviceLost), 2, &policy),
        Disposition::GiveUp
    );
}

#[test]
fn a_single_attempt_budget_never_restarts() {
    let policy = RestartPolicy {
        max_attempts: 1,
        restart_delay: Duration::ZERO,
    };
    assert_eq!(
        classify(
            &session_error(ClientError::WorkerGone {
                status: None,
                stderr_tail: Vec::new(),
            }),
            1,
            &policy
        ),
        Disposition::GiveUp
    );
}

#[test]
fn the_default_policy_allows_exactly_one_restart() {
    let policy = RestartPolicy::default();
    assert_eq!(policy.max_attempts, 2);
    assert!(policy.restart_delay > Duration::ZERO);
}
