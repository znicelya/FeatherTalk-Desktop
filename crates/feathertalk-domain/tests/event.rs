use feathertalk_domain::{
    DomainError, ErrorCode, Event, Metrics, PROTOCOL_VERSION, Progress, TaskError, TaskId,
    TaskStage,
};

fn task_id() -> TaskId {
    TaskId::parse("1787900000000-0000000a").unwrap()
}

const NOW: &str = "2026-08-28T09:00:00Z";

#[test]
fn a_new_event_carries_the_protocol_version_and_empty_metrics() {
    let event = Event::new(task_id(), NOW, TaskStage::Preparing);
    assert_eq!(event.protocol_version, PROTOCOL_VERSION);
    assert_eq!(event.metrics, Metrics::empty());
    assert_eq!(event.progress, None);
    assert_eq!(event.error, None);
    assert_eq!(event.result, None);
    event.validate().unwrap();
}

#[test]
fn a_failed_stage_requires_the_error_payload() {
    let mut event = Event::new(
        task_id(),
        NOW,
        TaskStage::Failed {
            code: ErrorCode::DiskSpaceLow,
            message: "磁盘空间不足".to_owned(),
        },
    );
    assert!(matches!(
        event.validate(),
        Err(DomainError::InvalidField { field: "error", .. })
    ));

    event.error = Some(TaskError::new(
        ErrorCode::DiskSpaceLow,
        "磁盘空间不足",
        "needed 4 GiB",
        TaskStage::Exporting,
    ));
    event.validate().unwrap();
}

#[test]
fn a_failed_stage_and_error_payload_must_use_the_same_code() {
    let mut event = Event::new(
        task_id(),
        NOW,
        TaskStage::Failed {
            code: ErrorCode::DiskSpaceLow,
            message: "纾佺洏绌洪棿涓嶈冻".to_owned(),
        },
    );
    event.error = Some(TaskError::new(
        ErrorCode::GpuDeviceLost,
        "鏄惧崱杩炴帴涓柇",
        "device lost",
        TaskStage::Exporting,
    ));

    assert!(matches!(
        event.validate(),
        Err(DomainError::InvalidField { field: "error", .. })
    ));
}

#[test]
fn a_non_failed_stage_must_not_carry_an_error_payload() {
    let mut event = Event::new(task_id(), NOW, TaskStage::Exporting);
    event.error = Some(TaskError::new(
        ErrorCode::DiskSpaceLow,
        "磁盘空间不足",
        "needed 4 GiB",
        TaskStage::Exporting,
    ));
    assert!(matches!(
        event.validate(),
        Err(DomainError::InvalidField { field: "error", .. })
    ));
}

#[test]
fn progress_rejects_a_completed_count_beyond_the_total() {
    let mut event = Event::new(task_id(), NOW, TaskStage::ExtractingFrames);
    event.progress = Some(Progress {
        completed: 5,
        total: Some(4),
    });
    assert!(matches!(
        event.validate(),
        Err(DomainError::InvalidField {
            field: "progress",
            ..
        })
    ));

    event.progress = Some(Progress {
        completed: 5,
        total: None,
    });
    event.validate().unwrap();
}

#[test]
fn validate_rejects_a_non_rfc3339_timestamp_and_a_foreign_protocol_version() {
    let mut event = Event::new(task_id(), "yesterday", TaskStage::Preparing);
    assert!(matches!(
        event.validate(),
        Err(DomainError::InvalidField {
            field: "emitted_at",
            ..
        })
    ));

    event = Event::new(task_id(), NOW, TaskStage::Preparing);
    event.protocol_version = PROTOCOL_VERSION + 1;
    assert!(matches!(
        event.validate(),
        Err(DomainError::ProtocolVersion { .. })
    ));
}

#[test]
fn events_round_trip_and_reject_unknown_fields() {
    let mut event = Event::new(
        task_id(),
        NOW,
        TaskStage::Training {
            epoch: 2,
            step: 40,
            loss: 0.1,
        },
    );
    event.metrics = Metrics {
        samples_per_second: Some(12.5),
        eta_seconds: Some(90.0),
        vram_bytes: Some(3_221_225_472),
    };
    let json = serde_json::to_string(&event).unwrap();
    assert_eq!(serde_json::from_str::<Event>(&json).unwrap(), event);

    let injected = json.replace(r#""metrics":{"#, r#""surprise":1,"metrics":{"#);
    assert!(serde_json::from_str::<Event>(&injected).is_err());
}

#[test]
fn a_completed_stage_may_carry_a_result_object() {
    let mut event = Event::new(task_id(), NOW, TaskStage::Completed);
    event.validate().unwrap();
    event.result = Some(serde_json::json!({"format_name": "mov,mp4"}));
    event.validate().unwrap();
}

#[test]
fn only_a_completed_stage_may_carry_a_result() {
    for stage in [
        TaskStage::Preparing,
        TaskStage::Rendering {
            frame: 1,
            total: 10,
        },
        TaskStage::Cancelled,
    ] {
        let mut event = Event::new(task_id(), NOW, stage);
        event.result = Some(serde_json::json!({}));
        assert!(matches!(
            event.validate(),
            Err(DomainError::InvalidField {
                field: "result",
                ..
            })
        ));
    }
}

#[test]
fn a_result_must_be_a_json_object() {
    for value in [
        serde_json::json!(7),
        serde_json::json!("mov,mp4"),
        serde_json::json!([1, 2]),
        serde_json::Value::Null,
    ] {
        let mut event = Event::new(task_id(), NOW, TaskStage::Completed);
        event.result = Some(value);
        assert!(matches!(
            event.validate(),
            Err(DomainError::InvalidField {
                field: "result",
                ..
            })
        ));
    }
}
