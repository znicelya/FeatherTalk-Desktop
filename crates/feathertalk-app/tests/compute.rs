use std::path::PathBuf;

use feathertalk_app::compute::{ComputeState, spawn_discovery};
use feathertalk_client::{ComputeOptions, SessionOptions, WorkerLocator};
use feathertalk_domain::{
    AdapterInfo, AdapterKind, Backend, Capabilities, PROTOCOL_VERSION, ReadyFrame, TaskKind,
};

fn ready() -> ReadyFrame {
    let device = |id: &str, backend, kind, certified, vram_bytes| AdapterInfo {
        id: id.into(),
        name: format!("{id} (Vulkan)"),
        backend,
        kind,
        certified,
        vram_bytes,
    };
    ReadyFrame {
        protocol_version: PROTOCOL_VERSION,
        worker_version: "test".into(),
        backends: vec![Backend::Cpu, Backend::Wgpu],
        adapters: vec![
            device(
                "wgpu-first",
                Backend::Wgpu,
                AdapterKind::Discrete,
                true,
                Some(8 * 1024 * 1024 * 1024),
            ),
            device("cpu-0", Backend::Cpu, AdapterKind::Cpu, true, None),
            device(
                "wgpu-second",
                Backend::Wgpu,
                AdapterKind::Integrated,
                true,
                None,
            ),
            device(
                "experimental",
                Backend::Wgpu,
                AdapterKind::Discrete,
                false,
                None,
            ),
            device("software", Backend::Wgpu, AdapterKind::Cpu, true, None),
        ],
        supported_commands: vec![
            TaskKind::Train,
            TaskKind::Render,
            TaskKind::ExtractFrames,
            TaskKind::ExtractFeatures,
        ],
        capabilities: Capabilities {
            training: true,
            wgpu_training: true,
            onnx_validation: false,
            ffmpeg: true,
        },
    }
}

fn state(backend: Option<&str>, adapter: Option<&str>) -> ComputeState {
    ComputeState::new(ComputeOptions::from_environment_values(backend, adapter))
}

#[test]
fn command_admission_waits_for_discovery_and_checks_advertised_support() {
    let mut state = state(None, None);
    assert_eq!(
        state.command_blocked_key(TaskKind::InspectModel),
        Some("compute.unavailable")
    );
    state.begin_discovery();
    assert_eq!(
        state.command_blocked_key(TaskKind::InspectModel),
        Some("compute.discovering")
    );
    state.finish_discovery(Ok(ready()));
    assert_eq!(state.command_blocked_key(TaskKind::Train), None);
    assert_eq!(
        state.command_blocked_key(TaskKind::InspectModel),
        Some("compute.unsupported")
    );
}

#[test]
fn model_tools_remain_available_with_an_invalid_compute_selection() {
    let mut state = state(Some("unsupported-backend"), None);
    let mut ready = ready();
    ready.supported_commands.push(TaskKind::InspectModel);
    state.finish_discovery(Ok(ready));
    assert_eq!(state.command_blocked_key(TaskKind::InspectModel), None);
    assert_eq!(
        state.command_blocked_key(TaskKind::Train),
        Some("compute.blocked")
    );
}

#[test]
fn explicit_cpu_selection_is_independent_of_adapter_order() {
    let mut state = state(Some("cpu"), None);
    assert!(state.environment_for(TaskKind::Render).is_err());
    state.finish_discovery(Ok(ready()));
    assert_eq!(state.selected_adapter().unwrap().id, "cpu-0");
    assert_eq!(
        state.environment_for(TaskKind::Render).unwrap(),
        vec![
            ("FEATHERTALK_WORKER_BACKEND".into(), "cpu".into()),
            ("FEATHERTALK_WORKER_ADAPTER".into(), "cpu-0".into()),
        ]
    );
}

#[test]
fn the_chosen_gpu_persists_across_submissions_and_refreshes() {
    let mut state = state(None, None);
    state.finish_discovery(Ok(ready()));
    state.select("wgpu-second").unwrap();
    for kind in [
        TaskKind::Train,
        TaskKind::Render,
        TaskKind::ExtractFrames,
        TaskKind::ExtractFeatures,
    ] {
        assert_eq!(
            state.environment_for(kind).unwrap(),
            vec![
                ("FEATHERTALK_WORKER_BACKEND".into(), "wgpu".into()),
                ("FEATHERTALK_WORKER_ADAPTER".into(), "wgpu-second".into()),
            ]
        );
    }
    state.begin_discovery();
    assert!(state.environment_for(TaskKind::Render).is_err());
    state.finish_discovery(Ok(ready()));
    assert_eq!(state.selected_adapter().unwrap().id, "wgpu-second");
}

#[test]
fn advertised_arc_and_radeon_devices_are_selectable_for_every_compute_command() {
    for (id, name, kind) in [
        (
            "wgpu-dx12-8086-e20b-luid-00000000-00000001",
            "Intel(R) Arc(TM) B580 Graphics (DX12)",
            AdapterKind::Discrete,
        ),
        (
            "wgpu-vulkan-8086-56a0-uuid-00000000000000000000000000000001",
            "Intel Arc A770 Graphics (Vulkan)",
            AdapterKind::Discrete,
        ),
        (
            "wgpu-dx12-8086-64a0-luid-00000000-00000002",
            "Intel(R) Arc(TM) 140V GPU (DX12)",
            AdapterKind::Integrated,
        ),
        (
            "wgpu-dx12-1002-7550-luid-00000000-00000003",
            "AMD Radeon RX 9070 XT (DX12)",
            AdapterKind::Discrete,
        ),
        (
            "wgpu-vulkan-1002-150e-uuid-00000000000000000000000000000002",
            "AMD Radeon(TM) 890M Graphics (Vulkan)",
            AdapterKind::Integrated,
        ),
    ] {
        let mut frame = ready();
        frame
            .adapters
            .retain(|adapter| adapter.backend == Backend::Cpu);
        frame.adapters.push(AdapterInfo {
            id: id.into(),
            name: name.into(),
            backend: Backend::Wgpu,
            kind,
            certified: true,
            vram_bytes: None,
        });
        frame.validate().unwrap();
        let mut state = state(None, None);
        state.finish_discovery(Ok(frame.clone()));
        assert_eq!(state.selected_adapter().unwrap().id, id);
        assert!(state.can_select(id), "cannot select {name}");
        state.select(id).unwrap();
        for command in [
            TaskKind::Train,
            TaskKind::Render,
            TaskKind::ExtractFrames,
            TaskKind::ExtractFeatures,
        ] {
            assert_eq!(state.command_blocked_key(command), None);
            assert_eq!(
                state.environment_for(command).unwrap(),
                vec![
                    ("FEATHERTALK_WORKER_BACKEND".into(), "wgpu".into()),
                    ("FEATHERTALK_WORKER_ADAPTER".into(), id.into()),
                ]
            );
        }
        state.begin_discovery();
        state.finish_discovery(Ok(frame));
        assert_eq!(state.selected_adapter().unwrap().id, id);
    }
}

#[test]
fn same_named_physical_gpus_keep_separate_selections() {
    let mut frame = ready();
    frame.adapters[0].name = "Intel Arc B580 Graphics (DX12)".into();
    frame.adapters[2].name = frame.adapters[0].name.clone();
    frame.adapters[2].kind = AdapterKind::Discrete;
    let mut state = state(None, None);
    state.finish_discovery(Ok(frame.clone()));
    for id in ["wgpu-first", "wgpu-second"] {
        state.select(id).unwrap();
        frame.adapters.reverse();
        state.begin_discovery();
        state.finish_discovery(Ok(frame.clone()));
        assert_eq!(state.selected_adapter().unwrap().id, id);
        assert_eq!(
            state.environment_for(TaskKind::Train).unwrap()[1],
            ("FEATHERTALK_WORKER_ADAPTER".into(), id.into())
        );
    }
}

#[test]
fn implicit_wgpu_selection_becomes_a_stable_id_and_does_not_switch_after_device_loss() {
    let mut state = state(Some("wgpu"), None);
    state.finish_discovery(Ok(ready()));
    assert_eq!(state.selected_adapter().unwrap().id, "wgpu-first");
    state.begin_discovery();
    let mut changed = ready();
    changed.adapters.retain(|device| device.id != "wgpu-first");
    state.finish_discovery(Ok(changed));
    assert!(state.error().unwrap().contains("wgpu-first"));
    assert!(state.environment_for(TaskKind::Train).is_err());
    state.select("wgpu-second").unwrap();
    assert_eq!(state.selected_adapter().unwrap().id, "wgpu-second");
}

#[test]
fn invalid_environment_is_visible_until_the_user_selects_a_valid_device() {
    for (backend, adapter) in [
        ("invalid", None),
        ("wgpu", Some("missing-gpu")),
        ("cpu", Some("wgpu-first")),
    ] {
        let mut state = state(Some(backend), adapter);
        state.finish_discovery(Ok(ready()));
        assert!(state.error().is_some());
        assert!(state.environment_for(TaskKind::Render).is_err());
        state.select("cpu-0").unwrap();
        assert!(state.error().is_none());
        assert_eq!(state.selected_adapter().unwrap().id, "cpu-0");
    }
}

#[test]
fn experimental_and_software_devices_are_visible_but_cannot_be_selected() {
    let mut state = state(Some("cpu"), None);
    state.finish_discovery(Ok(ready()));
    for id in ["experimental", "software"] {
        assert!(state.adapters().iter().any(|device| device.id == id));
        assert!(!state.can_select(id));
        assert!(state.select(id).is_err());
        assert_eq!(state.selected_adapter().unwrap().id, "cpu-0");
    }
    assert_eq!(state.adapters()[0].vram_bytes, Some(8 * 1024 * 1024 * 1024));
    assert_eq!(state.adapters()[2].vram_bytes, None);
}

#[test]
fn per_task_eligibility_uses_the_current_handshake() {
    let mut state = state(Some("wgpu"), None);
    let mut frame = ready();
    frame.capabilities.wgpu_training = false;
    frame
        .supported_commands
        .retain(|kind| *kind != TaskKind::ExtractFrames);
    state.finish_discovery(Ok(frame));
    assert!(state.blocked_key(TaskKind::Train).is_some());
    assert!(state.blocked_key(TaskKind::ExtractFrames).is_some());
    assert!(state.blocked_key(TaskKind::Render).is_none());
}

#[test]
fn failed_discovery_allows_a_later_refresh_without_resetting_selection() {
    let mut state = state(Some("wgpu"), Some("wgpu-second"));
    state.begin_discovery();
    state.finish_discovery(Err("worker is unavailable".into()));
    assert!(state.error().unwrap().contains("worker is unavailable"));
    assert!(state.environment_for(TaskKind::Train).is_err());
    state.begin_discovery();
    state.finish_discovery(Ok(ready()));
    assert_eq!(state.selected_adapter().unwrap().id, "wgpu-second");
}

#[test]
fn other_operations_use_cpu_even_with_an_invalid_gpu_configuration() {
    let state = state(Some("wgpu"), Some("missing"));
    for kind in [TaskKind::ValidateProject, TaskKind::LockAssetPackage] {
        assert!(state.blocked_key(kind).is_none());
        assert_eq!(
            state.environment_for(kind).unwrap(),
            vec![
                ("FEATHERTALK_WORKER_BACKEND".into(), "cpu".into()),
                ("FEATHERTALK_WORKER_ADAPTER".into(), "".into()),
            ]
        );
    }
}

#[test]
fn automatic_cuda_choice_can_fall_back_on_refresh_without_becoming_pinned() {
    let mut state = state(None, None);
    let mut frame = ready();
    frame.backends.push(Backend::Cuda);
    frame.adapters.push(AdapterInfo {
        id: "cuda-test-0".into(),
        name: "NVIDIA GPU (CUDA)".into(),
        backend: Backend::Cuda,
        kind: AdapterKind::Discrete,
        certified: true,
        vram_bytes: Some(6 * 1024 * 1024 * 1024),
    });
    state.finish_discovery(Ok(frame));
    assert_eq!(state.selected_adapter().unwrap().id, "cuda-test-0");
    assert_eq!(state.requested().unwrap().backend, Backend::Auto);
    state.select("cpu-0").unwrap();
    assert_eq!(state.selected_adapter().unwrap().backend, Backend::Cpu);
    state.select_automatic();
    assert_eq!(state.selected_adapter().unwrap().backend, Backend::Cuda);
    assert!(state.requested().unwrap().adapter.is_none());
    assert!(state.requested().unwrap().adapter.is_none());
    state.begin_discovery();
    state.finish_discovery(Ok(ready()));
    assert_eq!(state.selected_adapter().unwrap().id, "wgpu-first");
    let mut cpu_only = ready();
    cpu_only.backends = vec![Backend::Cpu];
    cpu_only
        .adapters
        .retain(|adapter| adapter.backend == Backend::Cpu);
    state.begin_discovery();
    state.finish_discovery(Ok(cpu_only));
    assert_eq!(state.selected_adapter().unwrap().id, "cpu-0");
    assert!(state.environment_for(TaskKind::Render).is_ok());
}

#[test]
fn background_discovery_overrides_compute_for_the_probe_and_shuts_it_down_cleanly() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("probe-worker.exe");
    std::fs::copy(env!("CARGO_BIN_EXE_feathertalk-app-fake-worker"), &path).unwrap();
    let receiver = spawn_discovery(
        WorkerLocator::from_parts(Some(path), None, None),
        SessionOptions::default(),
    )
    .unwrap();
    let discovery = receiver.recv_blocking().expect("the probe returns");
    assert!(discovery.worker.is_ready());
    let ready = discovery.ready.unwrap();
    assert_eq!(ready.adapters[0].id, "cpu-0");
    assert_eq!(ready.adapters[1].vram_bytes, Some(8 * 1024 * 1024 * 1024));
    assert_eq!(ready.adapters[2].vram_bytes, None);
    assert!(!ready.adapters[2].certified);
    assert_eq!(
        std::fs::read_to_string(directory.path().join("shutdown.txt")).unwrap(),
        "shutdown"
    );
    let received: serde_json::Value =
        serde_json::from_slice(&std::fs::read(directory.path().join("probe-env.json")).unwrap())
            .unwrap();
    assert_eq!(
        received,
        serde_json::json!({"backend": "cpu", "adapter": ""})
    );
}

#[test]
fn a_missing_worker_preserves_the_locator_diagnostics() {
    let directory = tempfile::tempdir().unwrap();
    let missing = directory.path().join("missing-worker.exe");
    let result = spawn_discovery(
        WorkerLocator::from_parts(
            Some(missing.clone()),
            None,
            Some(PathBuf::from("missing-app.exe")),
        ),
        SessionOptions::default(),
    )
    .unwrap()
    .recv_blocking()
    .unwrap();
    assert!(!result.worker.is_ready());
    assert_eq!(result.worker.probed()[0].path.as_ref(), Some(&missing));
    assert!(result.ready.is_err());
}
