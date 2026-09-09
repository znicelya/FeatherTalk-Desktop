#[path = "support/harness.rs"]
mod harness;

use std::time::{Duration, Instant};

use feathertalk_client::{ClientError, WorkerSession};
use feathertalk_domain::TaskKind;

use harness::{fake_worker, fast_options, scenario};

fn spawn(name: &str) -> Result<WorkerSession, ClientError> {
    WorkerSession::spawn_with_env(&fake_worker(), fast_options(), &scenario(name))
}

#[test]
fn a_healthy_worker_completes_the_handshake() {
    let session = spawn("ready-complete").expect("the handshake succeeds");
    assert_eq!(session.ready().worker_version, "fake-0");
    assert_eq!(
        session.ready().supported_commands,
        vec![
            TaskKind::ValidateProject,
            TaskKind::ProbeMedia,
            TaskKind::NormalizeMedia
        ]
    );
    assert_eq!(session.ready().adapters.len(), 1);
    assert!(
        session.ready_raw().contains("\"frame\":\"ready\""),
        "the raw line is kept verbatim for --json: {}",
        session.ready_raw()
    );
}

#[test]
fn an_event_before_ready_fails_the_handshake() {
    let error = spawn("no-ready").expect_err("an event is not a handshake");
    let ClientError::Handshake { reason, .. } = error else {
        panic!("expected a handshake error, got {error:?}");
    };
    assert!(
        reason.contains("1787900000000-0000beef"),
        "the reason names the offending task: {reason}"
    );
}

#[test]
fn an_undecodable_first_line_fails_the_handshake() {
    let error = spawn("invalid-line").expect_err("a truncated frame is not a handshake");
    assert!(
        matches!(error, ClientError::Handshake { .. }),
        "expected a handshake error, got {error:?}"
    );
}

#[test]
fn a_rejected_frame_surfaces_the_worker_reason() {
    let error = spawn("rejected-handshake").expect_err("a rejection is not a handshake");
    let ClientError::Rejected { reason } = error else {
        panic!("expected a rejection, got {error:?}");
    };
    assert_eq!(reason, "工作进程当前无法接受新会话");
}

#[test]
fn a_future_protocol_version_is_reported_precisely() {
    let error = spawn("bad-version").expect_err("version 99 is not supported");
    assert!(
        matches!(
            error,
            ClientError::ProtocolVersion {
                expected: 2,
                actual: 99
            }
        ),
        "expected a protocol version mismatch, got {error:?}"
    );
}

#[test]
fn a_worker_with_no_commands_fails_the_handshake() {
    let error = spawn("empty-commands").expect_err("an empty command list is invalid");
    assert!(
        matches!(error, ClientError::Handshake { .. }),
        "expected a handshake error, got {error:?}"
    );
}

#[test]
fn a_silent_worker_times_out_instead_of_hanging() {
    let started = Instant::now();
    let error = spawn("silent").expect_err("silence is not a handshake");
    assert!(
        matches!(error, ClientError::Handshake { .. }),
        "expected a handshake error, got {error:?}"
    );
    assert!(
        started.elapsed() < Duration::from_secs(4),
        "the 800 ms handshake deadline was not honoured: {:?}",
        started.elapsed()
    );
}

#[test]
fn the_stderr_tail_is_bounded() {
    let session = spawn("noisy-stderr").expect("the handshake succeeds");
    let deadline = Instant::now() + Duration::from_secs(5);
    let mut tail = session.stderr_tail();
    while tail.len() < 20 && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(20));
        tail = session.stderr_tail();
    }
    assert_eq!(
        tail.len(),
        20,
        "the tail keeps exactly the configured bound"
    );
    assert_eq!(tail.first().map(String::as_str), Some("stderr line 180"));
    assert_eq!(tail.last().map(String::as_str), Some("stderr line 199"));
}

#[test]
fn dropping_a_session_reaps_a_hung_worker() {
    let started = Instant::now();
    let session = spawn("hang-after-ready").expect("the handshake succeeds");
    drop(session);
    assert!(
        started.elapsed() < Duration::from_secs(2),
        "drop must kill and join rather than wait for a worker that never exits: {:?}",
        started.elapsed()
    );
}
