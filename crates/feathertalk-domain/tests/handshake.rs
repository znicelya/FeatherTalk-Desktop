use feathertalk_domain::{
    AdapterInfo, AdapterKind, Backend, CancelFrame, Capabilities, ClientFrame, DomainError,
    PROTOCOL_VERSION, ReadyFrame, ServerFrame, TaskId, TaskKind,
};

fn adapters() -> Vec<AdapterInfo> {
    vec![
        AdapterInfo {
            id: "dx12:nvidia:0".to_owned(),
            name: "NVIDIA GeForce RTX 4090".to_owned(),
            backend: Backend::Wgpu,
            kind: AdapterKind::Discrete,
            certified: true,
            vram_bytes: Some(25_769_803_776),
        },
        AdapterInfo {
            id: "dx12:intel:0".to_owned(),
            name: "Intel UHD Graphics".to_owned(),
            backend: Backend::Wgpu,
            kind: AdapterKind::Integrated,
            certified: false,
            vram_bytes: None,
        },
    ]
}

fn ready() -> ReadyFrame {
    ReadyFrame {
        protocol_version: PROTOCOL_VERSION,
        worker_version: "0.1.0".to_owned(),
        backends: vec![Backend::Cpu, Backend::Wgpu],
        adapters: adapters(),
        supported_commands: vec![TaskKind::ValidateProject, TaskKind::ProbeMedia],
        capabilities: Capabilities {
            training: true,
            wgpu_training: true,
            onnx_validation: false,
            ffmpeg: true,
        },
    }
}

#[test]
fn uncertified_adapters_are_still_reported() {
    let params = ready();
    params.validate().unwrap();
    assert_eq!(params.adapters.len(), 2);
    assert!(params.adapters.iter().any(|adapter| !adapter.certified));
}

#[test]
fn adapter_ids_must_be_unique_and_non_empty() {
    let mut params = ready();
    params.adapters[1].id = params.adapters[0].id.clone();
    assert!(matches!(
        params.validate(),
        Err(DomainError::InvalidField {
            field: "adapters",
            ..
        })
    ));

    let mut params = ready();
    params.adapters[0].id = String::new();
    assert!(matches!(
        params.validate(),
        Err(DomainError::InvalidField {
            field: "adapters",
            ..
        })
    ));
}

#[test]
fn a_worker_reporting_no_backend_is_rejected() {
    let mut params = ready();
    params.backends.clear();
    assert!(matches!(
        params.validate(),
        Err(DomainError::InvalidField {
            field: "backends",
            ..
        })
    ));
}

#[test]
fn both_frame_directions_expose_the_protocol_version() {
    let cancel = ClientFrame::Cancel(CancelFrame {
        protocol_version: PROTOCOL_VERSION,
        task_id: TaskId::parse("1787900000000-0000000a").unwrap(),
    });
    assert_eq!(cancel.protocol_version(), PROTOCOL_VERSION);
    assert_eq!(
        ServerFrame::Ready(ready()).protocol_version(),
        PROTOCOL_VERSION
    );
}

#[test]
fn frames_use_adjacent_tagging_and_round_trip() {
    let frame = ServerFrame::Ready(ready());
    let json = serde_json::to_string(&frame).unwrap();
    assert!(json.starts_with(r#"{"frame":"ready","data":{"#), "{json}");
    assert_eq!(serde_json::from_str::<ServerFrame>(&json).unwrap(), frame);
}

#[test]
fn ready_frame_rejects_unknown_outer_fields() {
    let json = r#"{"frame":"ready","data":{"protocol_version":2,"worker_version":"0.1.0","backends":["cpu"],"adapters":[],"supported_commands":["probe_media"],"capabilities":{"training":false,"wgpu_training":false,"onnx_validation":false,"ffmpeg":true}},"extra":1}"#;
    assert!(serde_json::from_str::<ServerFrame>(json).is_err());
}

#[test]
fn the_protocol_version_is_two() {
    assert_eq!(PROTOCOL_VERSION, 2);
    assert_eq!(ready().protocol_version, 2);
}

#[test]
fn a_worker_reporting_no_supported_command_is_rejected() {
    let mut params = ready();
    params.supported_commands.clear();
    assert!(matches!(
        params.validate(),
        Err(DomainError::InvalidField {
            field: "supported_commands",
            ..
        })
    ));
}

#[test]
fn duplicate_supported_commands_are_rejected() {
    let mut params = ready();
    params.supported_commands = vec![TaskKind::ProbeMedia, TaskKind::ProbeMedia];
    assert!(matches!(
        params.validate(),
        Err(DomainError::InvalidField {
            field: "supported_commands",
            ..
        })
    ));
}

#[test]
fn supported_commands_travel_as_task_slugs() {
    let json = serde_json::to_string(&ready()).unwrap();
    assert!(
        json.contains(r#""supported_commands":["validate_project","probe_media"]"#),
        "{json}"
    );
    let restored: ReadyFrame = serde_json::from_str(&json).unwrap();
    assert_eq!(restored, ready());
}
