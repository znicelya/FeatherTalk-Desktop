use feathertalk_domain::{
    CancelFrame, ClientFrame, DomainError, MAX_FRAME_BYTES, PROTOCOL_VERSION, RejectedFrame,
    ServerFrame, ShutdownFrame, StartFrame, TaskId, check_protocol_version, decode_line,
    encode_line,
};

fn task_id() -> TaskId {
    TaskId::parse("1787900000000-0000000a").unwrap()
}

#[test]
fn every_client_frame_round_trips_on_one_line() {
    use feathertalk_domain::{ProbeMediaParams, Request};

    let frames = vec![
        ClientFrame::Start(StartFrame {
            protocol_version: PROTOCOL_VERSION,
            task_id: task_id(),
            request: Request::ProbeMedia(ProbeMediaParams {
                input: std::path::PathBuf::from("a.mov"),
            }),
        }),
        ClientFrame::Cancel(CancelFrame {
            protocol_version: PROTOCOL_VERSION,
            task_id: task_id(),
        }),
        ClientFrame::Shutdown(ShutdownFrame {
            protocol_version: PROTOCOL_VERSION,
        }),
    ];
    for frame in frames {
        let line = encode_line(&frame).unwrap();
        assert!(!line.contains('\n'), "encoded frame contains a newline");
        assert_eq!(decode_line::<ClientFrame>(&line).unwrap(), frame);
    }
}

#[test]
fn a_multiline_string_payload_still_encodes_to_one_line() {
    let frame = ServerFrame::Rejected(RejectedFrame {
        protocol_version: PROTOCOL_VERSION,
        reason: "line one\nline two".to_owned(),
    });
    let line = encode_line(&frame).unwrap();
    assert!(!line.contains('\n'));
    assert!(line.contains(r"line one\nline two"));
    assert_eq!(decode_line::<ServerFrame>(&line).unwrap(), frame);
}

#[test]
fn encoding_an_oversized_frame_is_refused() {
    let frame = ServerFrame::Rejected(RejectedFrame {
        protocol_version: PROTOCOL_VERSION,
        reason: "x".repeat(MAX_FRAME_BYTES),
    });
    assert!(matches!(
        encode_line(&frame),
        Err(DomainError::FrameTooLong {
            limit: MAX_FRAME_BYTES
        })
    ));
}

#[test]
fn decoding_an_oversized_line_is_refused_before_parsing() {
    let line = format!(r#"{{"frame":"{}"}}"#, "x".repeat(MAX_FRAME_BYTES));
    assert!(matches!(
        decode_line::<ServerFrame>(&line),
        Err(DomainError::FrameTooLong { .. })
    ));
}

#[test]
fn malformed_and_unknown_frames_are_refused() {
    for bad in [
        "",
        "   ",
        "not json",
        r#"{"frame":"greetings","data":{}}"#,
        r#"{"frame":"shutdown","data":{"protocol_version":2,"extra":true}}"#,
    ] {
        assert!(
            matches!(
                decode_line::<ClientFrame>(bad),
                Err(DomainError::MalformedFrame { .. })
            ),
            "expected rejection for {bad:?}"
        );
    }
}

#[test]
fn client_and_server_frames_reject_unknown_outer_fields() {
    let client = r#"{"frame":"shutdown","data":{"protocol_version":2},"extra":true}"#;
    assert!(serde_json::from_str::<ClientFrame>(client).is_err());

    let server = r#"{"frame":"rejected","data":{"protocol_version":2,"reason":"no"},"extra":true}"#;
    assert!(serde_json::from_str::<ServerFrame>(server).is_err());
}

#[test]
fn frame_validation_rejects_wrong_protocol_versions() {
    use feathertalk_domain::{ProbeMediaParams, Request};

    let wrong = PROTOCOL_VERSION + 1;
    let request = Request::ProbeMedia(ProbeMediaParams {
        input: std::path::PathBuf::from("a.mov"),
    });
    let start = StartFrame {
        protocol_version: wrong,
        task_id: task_id(),
        request,
    };
    assert!(matches!(
        start.validate(),
        Err(DomainError::ProtocolVersion { actual, .. }) if actual == wrong
    ));

    let cancel = CancelFrame {
        protocol_version: wrong,
        task_id: task_id(),
    };
    assert!(matches!(
        cancel.validate(),
        Err(DomainError::ProtocolVersion { actual, .. }) if actual == wrong
    ));

    let shutdown = ShutdownFrame {
        protocol_version: wrong,
    };
    assert!(matches!(
        shutdown.validate(),
        Err(DomainError::ProtocolVersion { actual, .. }) if actual == wrong
    ));

    let rejected = RejectedFrame {
        protocol_version: wrong,
        reason: "no".to_owned(),
    };
    assert!(matches!(
        rejected.validate(),
        Err(DomainError::ProtocolVersion { actual, .. }) if actual == wrong
    ));

    let mut ready = crate_ready();
    ready.protocol_version = wrong;
    assert!(matches!(
        ready.validate(),
        Err(DomainError::ProtocolVersion { actual, .. }) if actual == wrong
    ));
}

#[test]
fn frame_enum_validation_dispatches_and_event_validation_remains_valid() {
    use feathertalk_domain::{Event, Metrics, Progress, TaskStage};

    let wrong = ClientFrame::Shutdown(ShutdownFrame {
        protocol_version: PROTOCOL_VERSION + 1,
    });
    assert!(matches!(
        wrong.validate(),
        Err(DomainError::ProtocolVersion { .. })
    ));

    let event = Event {
        protocol_version: PROTOCOL_VERSION,
        task_id: task_id(),
        emitted_at: "2026-08-28T09:00:00Z".to_owned(),
        stage: TaskStage::Preparing,
        progress: Some(Progress {
            completed: 0,
            total: Some(1),
        }),
        metrics: Metrics::empty(),
        error: None,
        result: None,
    };
    assert!(ServerFrame::Event(event).validate().is_ok());
}

fn crate_ready() -> feathertalk_domain::ReadyFrame {
    feathertalk_domain::ReadyFrame {
        protocol_version: PROTOCOL_VERSION,
        worker_version: "0.1.0".to_owned(),
        backends: vec![feathertalk_domain::Backend::Cpu],
        adapters: vec![],
        supported_commands: vec![feathertalk_domain::TaskKind::ProbeMedia],
        capabilities: feathertalk_domain::Capabilities {
            training: false,
            wgpu_training: false,
            onnx_validation: false,
            ffmpeg: true,
        },
    }
}

#[test]
fn protocol_version_comparison_is_exact() {
    check_protocol_version(PROTOCOL_VERSION).unwrap();
    for wrong in [0, PROTOCOL_VERSION + 1, u32::MAX] {
        assert!(matches!(
            check_protocol_version(wrong),
            Err(DomainError::ProtocolVersion {
                expected: PROTOCOL_VERSION,
                ..
            })
        ));
    }
}
