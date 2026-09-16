use feathertalk_client::ComputeOptions;
use feathertalk_domain::{
    AdapterInfo, AdapterKind, Backend, Capabilities, PROTOCOL_VERSION, ReadyFrame, TaskKind,
};

fn adapter(id: &str, backend: Backend, kind: AdapterKind, certified: bool) -> AdapterInfo {
    AdapterInfo {
        id: id.into(),
        name: id.into(),
        backend,
        kind,
        certified,
        vram_bytes: None,
    }
}

fn ready() -> ReadyFrame {
    ReadyFrame {
        protocol_version: PROTOCOL_VERSION,
        worker_version: "test".into(),
        backends: vec![Backend::Cpu, Backend::Wgpu],
        adapters: vec![
            adapter("experimental", Backend::Wgpu, AdapterKind::Discrete, false),
            adapter("software", Backend::Wgpu, AdapterKind::Cpu, true),
            adapter("wgpu-first", Backend::Wgpu, AdapterKind::Discrete, true),
            adapter("cpu-0", Backend::Cpu, AdapterKind::Cpu, true),
            adapter("wgpu-second", Backend::Wgpu, AdapterKind::Integrated, true),
        ],
        supported_commands: vec![TaskKind::Train, TaskKind::Render, TaskKind::ExtractFeatures],
        capabilities: Capabilities {
            training: true,
            wgpu_training: true,
            onnx_validation: false,
            ffmpeg: true,
        },
    }
}

#[test]
fn defaults_select_a_certified_gpu_before_cpu() {
    let frame = ready();
    assert_eq!(
        ComputeOptions::default()
            .resolve_adapter(&frame)
            .unwrap()
            .id,
        "wgpu-first"
    );
}

#[test]
fn wgpu_auto_selection_uses_the_first_certified_hardware_adapter() {
    let frame = ready();
    let options = ComputeOptions::new(Backend::Wgpu, None).unwrap();
    assert_eq!(options.resolve_adapter(&frame).unwrap().id, "wgpu-first");
}

#[test]
fn a_previously_valid_explicit_device_is_revalidated_after_a_restart() {
    let mut frame = ready();
    let options = ComputeOptions::new(Backend::Wgpu, Some("wgpu-first".into())).unwrap();
    assert!(options.resolve_adapter(&frame).is_ok());
    frame.adapters.retain(|device| device.id != "wgpu-first");
    let error = options.resolve_adapter(&frame).unwrap_err().to_string();
    assert!(error.contains("wgpu-first"), "{error}");
}

#[test]
fn explicit_uncertified_software_and_unadvertised_devices_are_rejected() {
    let frame = ready();
    for id in ["experimental", "software", "missing"] {
        let options = ComputeOptions::new(Backend::Wgpu, Some(id.into())).unwrap();
        assert!(options.resolve_adapter(&frame).is_err(), "{id}");
    }
    let mut frame = ready();
    frame.backends.retain(|backend| *backend != Backend::Wgpu);
    assert!(
        ComputeOptions::new(Backend::Wgpu, None)
            .unwrap()
            .resolve_adapter(&frame)
            .is_err()
    );
}

#[test]
fn task_eligibility_checks_the_advertised_command_and_wgpu_training_capability() {
    let mut frame = ready();
    let options = ComputeOptions::new(Backend::Wgpu, None).unwrap();
    assert!(options.validate_for(TaskKind::Train, &frame).is_ok());
    frame.capabilities.wgpu_training = false;
    assert!(options.validate_for(TaskKind::Train, &frame).is_err());
    assert!(options.validate_for(TaskKind::Render, &frame).is_ok());
    assert!(
        options
            .validate_for(TaskKind::ExtractFrames, &frame)
            .is_err()
    );
}

#[test]
fn environment_parsing_defaults_to_auto_and_keeps_invalid_values_as_errors() {
    let cpu = ComputeOptions::from_environment_values(None, None).unwrap();
    assert_eq!(serde_json::to_value(cpu.backend).unwrap(), "auto");
    assert!(cpu.adapter.is_none());
    for backend in ["", "   ", "unknown", "WGPU"] {
        assert!(ComputeOptions::from_environment_values(Some(backend), None).is_err());
    }
    assert!(ComputeOptions::from_environment_values(None, Some("wgpu-first")).is_ok());
    let options = ComputeOptions::from_environment_values(Some(" wgpu "), Some("  ")).unwrap();
    assert_eq!(options.backend, Backend::Wgpu);
    assert!(options.adapter.is_none());
}

#[test]
fn automatic_selection_prefers_cuda_and_falls_back_after_a_fresh_handshake() {
    let mut frame = ready();
    let cuda: Backend = serde_json::from_str(r#""cuda""#).unwrap();
    frame.backends.push(cuda);
    frame
        .adapters
        .push(adapter("cuda-z", cuda, AdapterKind::Discrete, true));
    frame
        .adapters
        .push(adapter("cuda-a", cuda, AdapterKind::Discrete, true));
    frame.adapters.push(adapter(
        "cuda-0-invalid",
        cuda,
        AdapterKind::Discrete,
        false,
    ));
    let automatic = ComputeOptions::default();
    for _ in 0..2 {
        assert_eq!(automatic.resolve_adapter(&frame).unwrap().id, "cuda-a");
        frame.adapters.reverse();
    }
    assert_eq!(
        ComputeOptions::new(Backend::Cpu, None)
            .unwrap()
            .resolve_adapter(&frame)
            .unwrap()
            .id,
        "cpu-0"
    );
    assert_eq!(
        ComputeOptions::new(Backend::Wgpu, None)
            .unwrap()
            .resolve_adapter(&frame)
            .unwrap()
            .id,
        "wgpu-first"
    );
    frame.backends.retain(|backend| *backend != cuda);
    frame.adapters.retain(|device| device.backend != cuda);
    assert_eq!(automatic.resolve_adapter(&frame).unwrap().id, "wgpu-first");
    frame
        .adapters
        .retain(|device| device.backend == Backend::Cpu);
    assert_eq!(automatic.resolve_adapter(&frame).unwrap().id, "cpu-0");
    assert!(
        ComputeOptions::new(cuda, None)
            .unwrap()
            .resolve_adapter(&frame)
            .is_err()
    );
}

#[test]
fn automatic_training_validates_the_backend_it_actually_selects() {
    let mut frame = ready();
    frame.capabilities.wgpu_training = false;
    assert!(
        ComputeOptions::default()
            .validate_for(TaskKind::Train, &frame)
            .is_err()
    );
    let cuda: Backend = serde_json::from_str(r#""cuda""#).unwrap();
    frame.backends.push(cuda);
    frame
        .adapters
        .push(adapter("cuda-a", cuda, AdapterKind::Discrete, true));
    assert!(
        ComputeOptions::default()
            .validate_for(TaskKind::Train, &frame)
            .is_ok()
    );
    frame.capabilities.training = false;
    assert!(
        ComputeOptions::default()
            .validate_for(TaskKind::Train, &frame)
            .is_err()
    );
}

#[test]
fn cuda_flags_and_environment_preserve_the_requested_device() {
    let options =
        ComputeOptions::from_environment_values(Some(" cuda "), Some(" cuda-a ")).unwrap();
    assert_eq!(options.env_overrides()[0].1, "cuda");
    assert_eq!(options.adapter.as_deref(), Some("cuda-a"));
    assert_eq!(
        ComputeOptions::from_flags(None, Some("cuda-a"))
            .unwrap()
            .unwrap(),
        options
    );
    assert!(ComputeOptions::from_environment_values(Some("cuda"), Some("cpu-0")).is_err());
}

#[test]
fn automatic_selection_prefers_rocm_before_wgpu() {
    let mut frame = ready();
    frame.backends.push(Backend::Rocm);
    frame.adapters.push(adapter(
        "rocm-z",
        Backend::Rocm,
        AdapterKind::Discrete,
        true,
    ));
    frame.adapters.push(adapter(
        "rocm-a",
        Backend::Rocm,
        AdapterKind::Integrated,
        true,
    ));
    frame.adapters.push(adapter(
        "rocm-experimental",
        Backend::Rocm,
        AdapterKind::Discrete,
        false,
    ));

    assert_eq!(
        ComputeOptions::default()
            .resolve_adapter(&frame)
            .unwrap()
            .id,
        "rocm-a"
    );
    frame.adapters.reverse();
    assert_eq!(
        ComputeOptions::default()
            .resolve_adapter(&frame)
            .unwrap()
            .id,
        "rocm-a"
    );
    assert_eq!(
        ComputeOptions::new(Backend::Wgpu, None)
            .unwrap()
            .resolve_adapter(&frame)
            .unwrap()
            .id,
        "wgpu-first"
    );
}

#[test]
fn rocm_flags_and_environment_preserve_the_requested_device() {
    let options =
        ComputeOptions::from_environment_values(Some(" rocm "), Some(" rocm-a ")).unwrap();
    assert_eq!(options.backend, Backend::Rocm);
    assert_eq!(options.env_overrides()[0].1, "rocm");
    assert_eq!(options.adapter.as_deref(), Some("rocm-a"));
    assert_eq!(
        ComputeOptions::from_flags(None, Some("rocm-a"))
            .unwrap()
            .unwrap(),
        options
    );
    assert!(ComputeOptions::from_environment_values(Some("rocm"), Some("cpu-0")).is_err());
}

#[test]
fn absent_cli_options_emit_no_override_and_explicit_backend_clears_the_adapter() {
    assert!(ComputeOptions::from_flags(None, None).unwrap().is_none());
    let options = ComputeOptions::from_flags(Some(Backend::Cpu), None)
        .unwrap()
        .unwrap();
    assert_eq!(
        options.env_overrides(),
        vec![
            ("FEATHERTALK_WORKER_BACKEND".into(), "cpu".into()),
            ("FEATHERTALK_WORKER_ADAPTER".into(), "".into()),
        ]
    );
}

#[test]
fn cli_adapter_inference_does_not_change_environment_default_semantics() {
    for (id, backend) in [(" cpu-0 ", Backend::Cpu), ("wgpu-first", Backend::Wgpu)] {
        let options = ComputeOptions::from_flags(None, Some(id)).unwrap().unwrap();
        assert_eq!(options.backend, backend);
        assert_eq!(options.adapter.as_deref(), Some(id.trim()));
    }
    assert!(ComputeOptions::from_flags(Some(Backend::Cpu), Some("wgpu-first")).is_err());
    assert!(ComputeOptions::from_flags(Some(Backend::Wgpu), Some("cpu-0")).is_err());
}
