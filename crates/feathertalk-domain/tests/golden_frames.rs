use std::path::PathBuf;

use feathertalk_domain::{
    CancelFrame, ClientFrame, ErrorCode, Event, Metrics, PROTOCOL_VERSION, ProbeMediaParams,
    Progress, Recovery, Request, ServerFrame, ShutdownFrame, StartFrame, TaskError, TaskId,
    TaskStage, decode_line, encode_line,
};

const START_PROBE: &str = r#"{"frame":"start","data":{"protocol_version":2,"task_id":"1787900000000-0000000a","request":{"command":"probe_media","params":{"input":"a.mov"}}}}"#;

const CANCEL: &str =
    r#"{"frame":"cancel","data":{"protocol_version":2,"task_id":"1787900000000-0000000a"}}"#;

const SHUTDOWN: &str = r#"{"frame":"shutdown","data":{"protocol_version":2}}"#;

fn task_id() -> TaskId {
    TaskId::parse("1787900000000-0000000a").unwrap()
}

#[test]
fn client_frames_match_their_golden_lines_byte_for_byte() {
    let start = ClientFrame::Start(StartFrame {
        protocol_version: PROTOCOL_VERSION,
        task_id: task_id(),
        request: Request::ProbeMedia(ProbeMediaParams {
            input: PathBuf::from("a.mov"),
        }),
    });
    assert_eq!(encode_line(&start).unwrap(), START_PROBE);

    let cancel = ClientFrame::Cancel(CancelFrame {
        protocol_version: PROTOCOL_VERSION,
        task_id: task_id(),
    });
    assert_eq!(encode_line(&cancel).unwrap(), CANCEL);

    let shutdown = ClientFrame::Shutdown(ShutdownFrame {
        protocol_version: PROTOCOL_VERSION,
    });
    assert_eq!(encode_line(&shutdown).unwrap(), SHUTDOWN);
}

#[test]
fn golden_client_lines_still_decode() {
    for line in [START_PROBE, CANCEL, SHUTDOWN] {
        decode_line::<ClientFrame>(line).unwrap_or_else(|error| panic!("{line}: {error}"));
    }
}

const TRAINING_EVENT: &str = r#"{"frame":"event","data":{"protocol_version":2,"task_id":"1787900000000-0000000a","emitted_at":"2026-08-28T09:00:00Z","stage":{"stage":"training","data":{"epoch":3,"step":1200,"loss":0.0425}},"progress":{"completed":1200,"total":4000},"metrics":{"samples_per_second":12.5,"eta_seconds":90.0,"vram_bytes":3221225472},"error":null,"result":null}}"#;

const FAILED_EVENT: &str = r#"{"frame":"event","data":{"protocol_version":2,"task_id":"1787900000000-0000000a","emitted_at":"2026-08-28T09:00:00Z","stage":{"stage":"failed","data":{"code":"DISK_SPACE_LOW","message":"磁盘空间不足"}},"progress":null,"metrics":{"samples_per_second":null,"eta_seconds":null,"vram_bytes":null},"error":{"code":"DISK_SPACE_LOW","summary":"磁盘空间不足","detail":"needed 4 GiB","stage":{"stage":"exporting"},"recovery":"free_disk_space"},"result":null}}"#;

const COMPLETED_EVENT: &str = r#"{"frame":"event","data":{"protocol_version":2,"task_id":"1787900000000-0000000a","emitted_at":"2026-09-01T09:00:00Z","stage":{"stage":"completed"},"progress":null,"metrics":{"samples_per_second":null,"eta_seconds":null,"vram_bytes":null},"error":null,"result":{"format_name":"mov,mp4"}}}"#;

#[test]
fn a_completed_event_carries_its_command_result() {
    let mut event = Event::new(task_id(), "2026-09-01T09:00:00Z", TaskStage::Completed);
    event.result = Some(serde_json::json!({"format_name": "mov,mp4"}));
    event.validate().unwrap();
    assert_eq!(
        encode_line(&ServerFrame::Event(event)).unwrap(),
        COMPLETED_EVENT
    );
}

#[test]
fn a_training_event_matches_its_golden_line() {
    let mut event = Event::new(
        task_id(),
        "2026-08-28T09:00:00Z",
        TaskStage::Training {
            epoch: 3,
            step: 1200,
            loss: 0.0425,
        },
    );
    event.progress = Some(Progress {
        completed: 1200,
        total: Some(4000),
    });
    event.metrics = Metrics {
        samples_per_second: Some(12.5),
        eta_seconds: Some(90.0),
        vram_bytes: Some(3_221_225_472),
    };
    event.validate().unwrap();
    assert_eq!(
        encode_line(&ServerFrame::Event(event)).unwrap(),
        TRAINING_EVENT
    );
}

#[test]
fn a_failed_event_carries_summary_detail_stage_and_recovery() {
    let mut event = Event::new(
        task_id(),
        "2026-08-28T09:00:00Z",
        TaskStage::Failed {
            code: ErrorCode::DiskSpaceLow,
            message: "磁盘空间不足".to_owned(),
        },
    );
    event.error = Some(TaskError::new(
        ErrorCode::DiskSpaceLow,
        "磁盘空间不足",
        "needed 4 GiB",
        TaskStage::Exporting,
    ));
    event.validate().unwrap();
    assert_eq!(
        event.error.as_ref().unwrap().recovery,
        Recovery::FreeDiskSpace
    );
    assert_eq!(
        encode_line(&ServerFrame::Event(event)).unwrap(),
        FAILED_EVENT
    );
}

#[test]
fn golden_server_lines_still_decode() {
    for line in [TRAINING_EVENT, FAILED_EVENT, COMPLETED_EVENT] {
        let frame =
            decode_line::<ServerFrame>(line).unwrap_or_else(|error| panic!("{line}: {error}"));
        let ServerFrame::Event(event) = frame else {
            panic!("expected an event frame");
        };
        event.validate().unwrap();
    }
}
