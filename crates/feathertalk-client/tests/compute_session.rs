#[path = "support/harness.rs"]
mod harness;

use feathertalk_client::{
    CancelToken, ClientError, EventSink, SessionOutcome, WorkerSession, generate_task_id,
};
use feathertalk_domain::{ExtractFeaturesParams, Request};

struct Sink;
impl EventSink for Sink {
    fn on_event(&mut self, _event: &feathertalk_domain::Event, _raw: &str) {}
}

fn request() -> Request {
    Request::ExtractFeatures(ExtractFeaturesParams {
        project_dir: "project".into(),
        audio: "audio.wav".into(),
    })
}

#[test]
fn a_fresh_cpu_only_handshake_rejects_gpu_before_start_and_allows_clean_shutdown() {
    let directory = tempfile::tempdir().unwrap();
    let log = directory.path().join("protocol.txt");
    let env = vec![
        ("FT_FAKE_WORKER_SCENARIO".into(), "compute-cpu-only".into()),
        (
            "FT_FAKE_PROTOCOL_LOG".into(),
            log.to_string_lossy().into_owned(),
        ),
        ("FEATHERTALK_WORKER_BACKEND".into(), "wgpu".into()),
        ("FEATHERTALK_WORKER_ADAPTER".into(), "wgpu-test-0".into()),
    ];
    let mut session =
        WorkerSession::spawn_with_env(&harness::fake_worker(), harness::fast_options(), &env)
            .unwrap();
    let outcome = session.run(
        generate_task_id().unwrap(),
        request(),
        &CancelToken::new(),
        &mut Sink,
    );
    assert!(
        matches!(
            outcome,
            SessionOutcome::SessionError(ClientError::Rejected { .. })
        ),
        "{outcome:?}"
    );
    assert_eq!(session.shutdown(), Some(0));
    assert_eq!(std::fs::read_to_string(log).unwrap(), "shutdown\n");
}

#[test]
fn compute_validation_matches_the_last_child_override_including_empty_adapter() {
    let env = vec![
        ("FT_FAKE_WORKER_SCENARIO".into(), "compute-cpu-only".into()),
        ("FEATHERTALK_WORKER_BACKEND".into(), "wgpu".into()),
        ("FEATHERTALK_WORKER_ADAPTER".into(), "wgpu-test-0".into()),
        ("FEATHERTALK_WORKER_BACKEND".into(), "cpu".into()),
        ("FEATHERTALK_WORKER_ADAPTER".into(), "".into()),
    ];
    let mut session =
        WorkerSession::spawn_with_env(&harness::fake_worker(), harness::fast_options(), &env)
            .unwrap();
    let outcome = session.run(
        generate_task_id().unwrap(),
        request(),
        &CancelToken::new(),
        &mut Sink,
    );
    assert!(
        matches!(outcome, SessionOutcome::Completed { result: Some(ref value) }
        if *value == serde_json::json!({"backend": "cpu", "adapter": ""})),
        "{outcome:?}"
    );
    assert_eq!(session.shutdown(), Some(0));
}

#[cfg(windows)]
#[test]
fn compute_validation_uses_windows_case_insensitive_environment_keys() {
    let env = vec![
        ("FT_FAKE_WORKER_SCENARIO".into(), "compute-cpu-only".into()),
        ("FEATHERTALK_WORKER_BACKEND".into(), "wgpu".into()),
        ("FEATHERTALK_WORKER_ADAPTER".into(), "wgpu-test-0".into()),
        ("feathertalk_worker_backend".into(), "cpu".into()),
        ("feathertalk_worker_adapter".into(), "".into()),
    ];
    let mut session =
        WorkerSession::spawn_with_env(&harness::fake_worker(), harness::fast_options(), &env)
            .unwrap();
    let outcome = session.run(
        generate_task_id().unwrap(),
        request(),
        &CancelToken::new(),
        &mut Sink,
    );
    assert!(
        matches!(outcome, SessionOutcome::Completed { result: Some(ref value) }
        if *value == serde_json::json!({"backend": "cpu", "adapter": ""})),
        "{outcome:?}"
    );
    assert_eq!(session.shutdown(), Some(0));
}
