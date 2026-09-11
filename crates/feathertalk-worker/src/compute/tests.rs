use super::*;
use feathertalk_domain::Recovery;

fn nvidia(api: wgpu::Backend) -> wgpu::AdapterInfo {
    wgpu::AdapterInfo {
        name: "NVIDIA GeForce RTX 4090".into(),
        vendor: 0x10de,
        device: 0x2684,
        device_type: wgpu::DeviceType::DiscreteGpu,
        device_pci_bus_id: "0000:01:00.0".into(),
        driver: "NVIDIA".into(),
        driver_info: "test driver".into(),
        backend: api,
        subgroup_min_size: 32,
        subgroup_max_size: 32,
        transient_saves_memory: false,
    }
}

fn gpu(id: &str, certified: bool, kind: AdapterKind) -> AdapterInfo {
    AdapterInfo {
        id: id.into(),
        name: "GPU (Vulkan)".into(),
        backend: Backend::Wgpu,
        kind,
        certified,
        vram_bytes: None,
    }
}

fn pc_gpu(
    api: wgpu::Backend,
    vendor: u32,
    name: &str,
    kind: wgpu::DeviceType,
) -> wgpu::AdapterInfo {
    wgpu::AdapterInfo {
        name: name.into(),
        vendor,
        device_type: kind,
        driver: "hardware driver".into(),
        ..nvidia(api)
    }
}

fn arc_and_radeon_cases() -> [(u32, &'static str, wgpu::DeviceType); 8] {
    use wgpu::DeviceType::{DiscreteGpu, IntegratedGpu};
    [
        (0x8086, "Intel(R) Arc(TM) B580 Graphics", DiscreteGpu),
        (0x8086, "Intel Arc A770 Graphics", DiscreteGpu),
        (0x8086, "Intel(R) Arc(TM) Pro B50 Graphics", DiscreteGpu),
        (0x8086, "Intel(R) Arc(TM) 140V GPU", IntegratedGpu),
        (0x8086, "INTEL ARC GRAPHICS", IntegratedGpu),
        (0x1002, "AMD Radeon RX 9070 XT", DiscreteGpu),
        (0x1002, "AMD Radeon(TM) 890M Graphics", IntegratedGpu),
        (0x1002, "AMD Radeon Graphics", IntegratedGpu),
    ]
}

#[test]
fn cpu_resolution_keeps_cpu_zero_and_rejects_mismatched_ids() {
    let registry = ComputeRegistry::cpu_only();
    assert_eq!(registry.resolve(Backend::Cpu, None).unwrap().id, "cpu-0");
    assert_eq!(
        registry.resolve(Backend::Cpu, Some("cpu-0")).unwrap().id,
        "cpu-0"
    );
    for id in ["cpu-1", "", "CPU-0", " cpu-0", "wgpu-dx12-10de"] {
        assert!(
            registry.resolve(Backend::Cpu, Some(id)).is_err(),
            "accepted {id}"
        );
    }
    assert_eq!(registry.adapters().len(), 1);
    assert_eq!(registry.adapters()[0].vram_bytes, None);
}

#[test]
fn explicit_wgpu_requires_a_certified_hardware_id_without_fallback() {
    let adapters = vec![
        crate::handshake::cpu_adapter(),
        gpu("gpu-certified", true, AdapterKind::Discrete),
        gpu("gpu-experimental", false, AdapterKind::Discrete),
        gpu("software", true, AdapterKind::Cpu),
        gpu("virtual", true, AdapterKind::Other),
    ];
    assert_eq!(
        resolve_adapter(&adapters, Backend::Wgpu, Some("gpu-certified"))
            .unwrap()
            .id,
        "gpu-certified"
    );
    for id in [
        "cpu-0",
        "gpu-experimental",
        "software",
        "virtual",
        "unknown",
        "",
        "GPU-CERTIFIED",
    ] {
        assert!(
            resolve_adapter(&adapters, Backend::Wgpu, Some(id)).is_err(),
            "accepted {id}"
        );
    }
    assert!(resolve_adapter(&adapters, Backend::Cpu, Some("gpu-certified")).is_err());
}

#[test]
fn automatic_wgpu_choice_is_independent_of_enumeration_order() {
    let mut adapters = vec![
        gpu("a-experimental", false, AdapterKind::Discrete),
        gpu("z-certified", true, AdapterKind::Discrete),
        gpu("b-certified", true, AdapterKind::Integrated),
    ];
    assert_eq!(
        resolve_adapter(&adapters, Backend::Wgpu, None).unwrap().id,
        "b-certified"
    );
    adapters.reverse();
    assert_eq!(
        resolve_adapter(&adapters, Backend::Wgpu, None).unwrap().id,
        "b-certified"
    );
}

#[test]
fn absent_certified_gpu_is_an_error_even_when_cpu_is_available() {
    let registry = ComputeRegistry::cpu_only();
    assert!(registry.resolve(Backend::Wgpu, None).is_err());
    assert!(registry.resolve(Backend::Wgpu, Some("cpu-0")).is_err());
    assert!(matches!(
        registry.open_wgpu("cpu-0"),
        Err(GpuFailure::Unavailable(_))
    ));
    assert!(matches!(
        registry.open_wgpu("missing"),
        Err(GpuFailure::Unavailable(_))
    ));
}

#[test]
fn certification_requires_the_native_api_vendor_and_hardware_kind() {
    let dx12 = nvidia(wgpu::Backend::Dx12);
    let vulkan = nvidia(wgpu::Backend::Vulkan);
    assert!(certified(Platform::Windows, &vulkan));
    assert!(certified(Platform::Linux, &vulkan));
    assert!(!certified(Platform::Windows, &dx12));
    assert!(!certified(Platform::Linux, &dx12));
    assert!(!certified(Platform::Other, &vulkan));
    for vendor in [0, 0x8086, 0x1414, 0xffff] {
        let mut info = vulkan.clone();
        info.vendor = vendor;
        assert!(
            !certified(Platform::Windows, &info),
            "certified vendor {vendor}"
        );
    }
    for kind in [
        wgpu::DeviceType::Cpu,
        wgpu::DeviceType::VirtualGpu,
        wgpu::DeviceType::Other,
        wgpu::DeviceType::IntegratedGpu,
    ] {
        let mut info = vulkan.clone();
        info.device_type = kind;
        assert!(!certified(Platform::Linux, &info), "certified {kind:?}");
    }
}

#[test]
fn arc_and_radeon_are_supported_on_each_native_pc_api() {
    for (platform, api) in [
        (Platform::Windows, wgpu::Backend::Vulkan),
        (Platform::Linux, wgpu::Backend::Vulkan),
    ] {
        for (vendor, name, kind) in arc_and_radeon_cases() {
            let info = pc_gpu(api, vendor, name, kind);
            assert!(
                certified(platform, &info),
                "rejected {name} on {platform:?}"
            );
        }
    }
}

#[test]
fn arc_and_radeon_still_require_the_native_api_and_physical_gpu_kind() {
    for (platform, native_api) in [
        (Platform::Windows, wgpu::Backend::Vulkan),
        (Platform::Linux, wgpu::Backend::Vulkan),
    ] {
        for (vendor, name, kind) in arc_and_radeon_cases() {
            for api in [
                wgpu::Backend::Dx12,
                wgpu::Backend::Vulkan,
                wgpu::Backend::Metal,
                wgpu::Backend::Gl,
                wgpu::Backend::BrowserWebGpu,
                wgpu::Backend::Noop,
            ] {
                if api != native_api {
                    let info = pc_gpu(api, vendor, name, kind);
                    assert!(!certified(platform, &info), "accepted {name} on {api:?}");
                }
            }
            for kind in [
                wgpu::DeviceType::Cpu,
                wgpu::DeviceType::VirtualGpu,
                wgpu::DeviceType::Other,
            ] {
                let info = pc_gpu(native_api, vendor, name, kind);
                assert!(!certified(platform, &info), "accepted {name} as {kind:?}");
            }
            let metal = pc_gpu(wgpu::Backend::Metal, vendor, name, kind);
            assert!(!certified(Platform::IntelMac, &metal));
            assert!(!certified(Platform::AppleSilicon, &metal));
            assert!(!certified(Platform::Other, &metal));
        }
    }
}

#[test]
fn intel_support_requires_the_arc_product_family_and_vendor() {
    for (platform, api) in [
        (Platform::Windows, wgpu::Backend::Vulkan),
        (Platform::Linux, wgpu::Backend::Vulkan),
    ] {
        for name in [
            "Intel(R) UHD Graphics 770",
            "Intel Iris Xe Graphics",
            "Intel HD Graphics",
            "Intel Arcane Graphics",
        ] {
            let info = pc_gpu(api, 0x8086, name, wgpu::DeviceType::IntegratedGpu);
            assert!(!certified(platform, &info), "accepted {name}");
        }
        for vendor in [0, 0x1414, 0xffff] {
            let info = pc_gpu(
                api,
                vendor,
                "Intel Arc B580 Graphics",
                wgpu::DeviceType::DiscreteGpu,
            );
            assert!(!certified(platform, &info), "accepted vendor {vendor:x}");
        }
    }
}

#[test]
fn arc_and_radeon_selection_reaches_the_worker_and_training_handshake() {
    use feathertalk_domain::TaskKind;

    let directory = tempfile::tempdir().unwrap();
    for (platform, api) in [
        (Platform::Windows, wgpu::Backend::Vulkan),
        (Platform::Linux, wgpu::Backend::Vulkan),
    ] {
        for (vendor, name, kind) in arc_and_radeon_cases() {
            let info = pc_gpu(api, vendor, name, kind);
            let id = adapter_id(&info, Some("physical-gpu"));
            let (mut adapters, _) = physical_adapters(
                platform,
                [(
                    info,
                    native::NativeMetadata {
                        identity: Some("physical-gpu".into()),
                        ..native::NativeMetadata::default()
                    },
                    (),
                )],
            );
            assert_eq!(adapters.len(), 1, "hid {name}");
            let adapter = adapters[0].clone();
            assert_eq!(adapter.id, id);
            adapters.insert(0, crate::handshake::cpu_adapter());
            let registry = ComputeRegistry {
                adapters,
                native: BTreeMap::new(),
                #[cfg(any(target_os = "windows", target_os = "linux"))]
                cuda: BTreeMap::new(),
            };
            assert_eq!(registry.resolve(Backend::Wgpu, None).unwrap(), adapter);
            let config = crate::WorkerConfig::from_values_with_training(
                None,
                None,
                None,
                None,
                None,
                None,
                Some(directory.path().to_str().unwrap().to_owned()),
            )
            .with_compute_registry(registry);
            assert_eq!(config.compute_adapter().unwrap(), adapter);
            let config = config.with_compute_selection(Some("wgpu"), Some(&id));
            for task in [
                TaskKind::Train,
                TaskKind::Render,
                TaskKind::ExtractFrames,
                TaskKind::ExtractFeatures,
                TaskKind::NormalizeMedia,
            ] {
                assert_eq!(config.adapter_for(task).unwrap(), adapter);
            }
            let ready = crate::handshake::ready_frame(&config);
            assert!(ready.backends.contains(&Backend::Wgpu));
            assert!(
                ready.capabilities.wgpu_training,
                "disabled training on {name}"
            );
            assert!(ready.supported_commands.contains(&TaskKind::Train));
            assert_eq!(ready.adapters[1], adapter);
            ready.validate().unwrap();
        }
    }
}

#[test]
fn automatic_compute_uses_cuda_then_existing_gpu_then_cpu() {
    let mut cuda = gpu("cuda-uuid-b", true, AdapterKind::Discrete);
    cuda.backend = Backend::Cuda;
    let mut adapters = vec![
        crate::handshake::cpu_adapter(),
        gpu("wgpu-a", true, AdapterKind::Discrete),
        cuda,
    ];
    assert_eq!(
        resolve_adapter(&adapters, Backend::Auto, None).unwrap().id,
        "cuda-uuid-b"
    );
    adapters.reverse();
    assert_eq!(
        resolve_adapter(&adapters, Backend::Auto, None).unwrap().id,
        "cuda-uuid-b"
    );
    assert_eq!(
        resolve_adapter(&adapters, Backend::Wgpu, None).unwrap().id,
        "wgpu-a"
    );
    assert_eq!(
        resolve_adapter(&adapters, Backend::Cpu, None).unwrap().id,
        "cpu-0"
    );
    assert!(resolve_adapter(&adapters, Backend::Cuda, Some("wgpu-a")).is_err());
    adapters.retain(|adapter| adapter.backend != Backend::Cuda);
    assert_eq!(
        resolve_adapter(&adapters, Backend::Auto, None).unwrap().id,
        "wgpu-a"
    );
    assert!(resolve_adapter(&adapters, Backend::Cuda, None).is_err());
    adapters.retain(|adapter| adapter.backend == Backend::Cpu);
    assert_eq!(
        resolve_adapter(&adapters, Backend::Auto, None).unwrap().id,
        "cpu-0"
    );
}

#[test]
fn duplicate_cuda_identities_preserve_a_valid_handshake_and_automatic_fallback() {
    let mut cuda = gpu("cuda-uuid-shared", true, AdapterKind::Discrete);
    cuda.backend = Backend::Cuda;
    let mut registry = ComputeRegistry::cpu_only();
    registry.adapters.extend([
        gpu("wgpu-fallback", true, AdapterKind::Discrete),
        cuda.clone(),
        cuda,
    ]);
    registry.finish_discovery();
    assert_eq!(
        registry.resolve(Backend::Auto, None).unwrap().id,
        "wgpu-fallback"
    );
    assert!(
        registry
            .resolve(Backend::Cuda, Some("cuda-uuid-shared"))
            .is_err()
    );
    assert!(registry.resolve(Backend::Cuda, None).is_err());
    let config = crate::WorkerConfig::from_values(None, None, None).with_compute_registry(registry);
    crate::handshake::ready_frame(&config).validate().unwrap();
}

#[test]
fn software_adapters_cannot_be_certified_by_a_misleading_device_type() {
    for name in [
        "llvmpipe",
        "lavapipe",
        "SwiftShader Device",
        "Microsoft Basic Render Driver",
        "Software Adapter",
        "WARP",
    ] {
        for (platform, api) in [
            (Platform::Windows, wgpu::Backend::Vulkan),
            (Platform::Linux, wgpu::Backend::Vulkan),
        ] {
            for vendor in [0x10de, 0x8086, 0x1002] {
                let info = pc_gpu(api, vendor, name, wgpu::DeviceType::DiscreteGpu);
                assert!(!certified(platform, &info), "certified {name}");
                let mut disguised = pc_gpu(
                    api,
                    vendor,
                    "Intel Arc B580 Graphics",
                    wgpu::DeviceType::DiscreteGpu,
                );
                disguised.driver_info = name.into();
                assert!(!certified(platform, &disguised), "certified driver {name}");
            }
        }
    }
}

#[test]
fn apple_gpu_is_certified_only_on_apple_silicon_metal() {
    let mut apple = nvidia(wgpu::Backend::Metal);
    apple.name = "Apple M4 Max".into();
    apple.vendor = 0; // WGPU 29 Metal does not expose PCI vendor or device IDs.
    apple.device = 0;
    apple.device_type = wgpu::DeviceType::IntegratedGpu;
    assert!(certified(Platform::AppleSilicon, &apple));
    assert!(!certified(Platform::Other, &apple));
    assert!(!certified(Platform::IntelMac, &apple));
    apple.backend = wgpu::Backend::Vulkan;
    assert!(!certified(Platform::AppleSilicon, &apple));
    apple.backend = wgpu::Backend::Metal;
    apple.name = "AMD Radeon Pro".into();
    assert!(!certified(Platform::AppleSilicon, &apple));
}

#[test]
fn filtering_display_aliases_keeps_linked_physical_gpus_and_their_handles() {
    let info = nvidia(wgpu::Backend::Vulkan);
    let physical_b = "wgpu-vulkan-10de-2684-luid-00000000-00000001-node-00000002";
    let physical_a = "wgpu-vulkan-10de-2684-luid-00000000-00000001-node-00000001";
    let discovered = [
        (
            "luid-00000000-00000001-node-00000002",
            false,
            "physical handle B",
        ),
        (
            "luid-00000000-00000003-node-00000001",
            true,
            "indirect display handle",
        ),
        (
            "luid-00000000-00000001-node-00000001",
            false,
            "physical handle A",
        ),
        (
            "luid-00000000-00000004-node-00000001",
            true,
            "second display handle",
        ),
    ]
    .map(|(identity, software_or_indirect, handle)| {
        (
            info.clone(),
            native::NativeMetadata {
                identity: Some(identity.into()),
                vram_bytes: Some(8 * 1024 * 1024 * 1024),
                software_or_indirect,
            },
            handle,
        )
    });
    let (mut adapters, handles) = physical_adapters(Platform::Windows, discovered);
    assert_eq!(adapters.len(), 2, "display aliases must not appear as GPUs");
    assert_eq!(handles, ["physical handle B", "physical handle A"]);
    assert_eq!(adapters[0].name, adapters[1].name);
    assert_eq!(adapters[0].vram_bytes, adapters[1].vram_bytes);
    disambiguate_ids(&mut adapters);
    let bindings: BTreeMap<_, _> = adapters
        .iter()
        .zip(handles)
        .map(|(adapter, handle)| (adapter.id.clone(), handle))
        .collect();
    adapters.sort_by(|left, right| left.id.cmp(&right.id));
    adapters.insert(0, crate::handshake::cpu_adapter());
    assert_eq!(adapters.len(), 3, "CPU plus two physical GPUs");
    assert_eq!(bindings[physical_a], "physical handle A");
    assert_eq!(bindings[physical_b], "physical handle B");
    for id in [physical_a, physical_b] {
        assert_eq!(
            resolve_adapter(&adapters, Backend::Wgpu, Some(id))
                .unwrap()
                .id,
            id
        );
    }
    for id in [
        "wgpu-vulkan-10de-2684-luid-00000000-00000003-node-00000001",
        "wgpu-vulkan-10de-2684-luid-00000000-00000004-node-00000001",
    ] {
        assert!(resolve_adapter(&adapters, Backend::Wgpu, Some(id)).is_err());
    }
}

#[test]
fn discovery_omits_software_and_virtual_devices_even_with_missing_native_metadata() {
    for (platform, api) in [
        (Platform::Windows, wgpu::Backend::Vulkan),
        (Platform::Linux, wgpu::Backend::Vulkan),
    ] {
        let mut hidden = Vec::new();
        for kind in [
            wgpu::DeviceType::Cpu,
            wgpu::DeviceType::VirtualGpu,
            wgpu::DeviceType::Other,
        ] {
            hidden.push(pc_gpu(api, 0x1002, "AMD Radeon RX 9070 XT", kind));
        }
        for name in ["Microsoft Basic Render Driver", "llvmpipe", "SwiftShader"] {
            hidden.push(pc_gpu(api, 0x1002, name, wgpu::DeviceType::DiscreteGpu));
        }
        let (adapters, handles) = physical_adapters(
            platform,
            hidden
                .into_iter()
                .map(|info| (info, native::NativeMetadata::default(), ())),
        );
        assert!(
            adapters.is_empty(),
            "advertised software devices: {adapters:?}"
        );
        assert!(handles.is_empty(), "retained software handles");
    }
}

#[test]
fn missing_optional_native_metadata_does_not_hide_a_physical_gpu() {
    let info = nvidia(wgpu::Backend::Vulkan);
    let id = adapter_id(&info, None);
    let (adapters, handles) = physical_adapters(
        Platform::Windows,
        [(info, native::NativeMetadata::default(), "physical handle")],
    );
    assert_eq!(handles, ["physical handle"]);
    assert_eq!(adapters.len(), 1);
    assert_eq!(adapters[0].id, id);
    assert!(adapters[0].certified);
    assert_eq!(adapters[0].vram_bytes, None);
}

#[test]
fn ids_preserve_native_identity_and_separate_graphics_apis() {
    let mut info = nvidia(wgpu::Backend::Dx12);
    let first = adapter_id(&info, Some("luid-00112233-44556677"));
    assert_eq!(first, "wgpu-dx12-10de-2684-luid-00112233-44556677");
    assert_ne!(first, adapter_id(&info, Some("luid-00112233-44556678")));
    info.backend = wgpu::Backend::Vulkan;
    assert_ne!(first, adapter_id(&info, Some("luid-00112233-44556677")));
}

#[test]
fn fallback_ids_survive_driver_updates_and_distinguish_models() {
    let mut info = nvidia(wgpu::Backend::Metal);
    info.device_pci_bus_id.clear();
    let first = adapter_id(&info, None);
    assert!(first.starts_with("wgpu-metal-10de-2684-model-"));
    info.driver_info = "updated driver".into();
    info.driver = "new driver label".into();
    assert_eq!(first, adapter_id(&info, None));
    info.name = "Another hardware model".into();
    assert_ne!(first, adapter_id(&info, None));
}

#[test]
fn indistinguishable_adapter_ids_cannot_select_an_arbitrary_physical_device() {
    let mut adapters = vec![
        gpu("same", true, AdapterKind::Discrete),
        gpu("unique", true, AdapterKind::Discrete),
        gpu("same", true, AdapterKind::Discrete),
    ];
    disambiguate_ids(&mut adapters);
    assert_ne!(adapters[0].id, adapters[2].id);
    assert!(!adapters[0].certified);
    assert!(!adapters[2].certified);
    assert!(resolve_adapter(&adapters, Backend::Wgpu, Some(&adapters[0].id)).is_err());
    assert_eq!(
        resolve_adapter(&adapters, Backend::Wgpu, None).unwrap().id,
        "unique"
    );
}

#[test]
fn typed_gpu_failures_keep_their_code_recovery_and_original_stage() {
    let stage = TaskStage::Training {
        epoch: 2,
        step: 34,
        loss: 0.5,
    };
    for (failure, code, recovery) in [
        (
            GpuFailure::OutOfMemory("allocation failed".into()),
            ErrorCode::GpuOutOfMemory,
            Recovery::SelectDifferentAdapter,
        ),
        (
            GpuFailure::DeviceLost("device removed".into()),
            ErrorCode::GpuDeviceLost,
            Recovery::ResumeFromCheckpoint,
        ),
        (
            GpuFailure::Unavailable("adapter missing".into()),
            ErrorCode::WorkerCrashed,
            Recovery::SelectDifferentAdapter,
        ),
        (
            GpuFailure::Other("invalid kernel".into()),
            ErrorCode::WorkerCrashed,
            Recovery::ResumeFromCheckpoint,
        ),
    ] {
        let error = failure.task_error(stage.clone());
        assert_eq!(error.code, code);
        assert_eq!(error.recovery, recovery);
        assert_eq!(error.stage, stage);
        assert_eq!(error.detail, failure.to_string());
        error.validate().unwrap();
    }
}

#[test]
fn gpu_error_details_remain_valid_protocol_messages() {
    let error = GpuFailure::Other("错".repeat(5000)).task_error(TaskStage::Preparing);
    error.validate().unwrap();
    assert_eq!(error.detail.chars().count(), 4000);
}

#[test]
fn uncaptured_oom_is_typed_and_retained_after_follow_on_errors() {
    let faults = FaultState::default();
    faults.record_wgpu(wgpu::Error::OutOfMemory {
        source: Box::new(std::io::Error::other("allocation failed")),
    });
    faults.record(GpuFailure::DeviceLost("device lost after OOM".into()));
    faults.record(GpuFailure::Other("subsequent validation error".into()));
    assert!(matches!(faults.check(), Err(GpuFailure::OutOfMemory(_))));
    assert!(matches!(faults.check(), Err(GpuFailure::OutOfMemory(_))));
}

#[test]
fn typed_fault_replaces_an_earlier_generic_diagnostic() {
    let faults = FaultState::default();
    faults.record(GpuFailure::Other("invalid buffer".into()));
    faults.record(GpuFailure::DeviceLost("driver reset".into()));
    assert!(matches!(faults.check(), Err(GpuFailure::DeviceLost(_))));
}

#[test]
fn telemetry_does_not_open_an_implicit_or_software_device() {
    for device in [
        WgpuDevice::DefaultDevice,
        WgpuDevice::Cpu,
        WgpuDevice::DiscreteGpu(0),
    ] {
        assert_eq!(wgpu_memory_usage_bytes(&device), None);
    }
}

#[test]
fn nested_runtime_and_profiling_errors_preserve_typed_allocation_failure() {
    use burn::cubecl::server::ProfileError;

    let allocation = || LaunchError::OutOfMemory {
        reason: "physical allocation failed".into(),
        backtrace: Default::default(),
    };
    let errors = [
        ServerError::Launch(allocation()),
        ServerError::ServerUnhealthy {
            errors: vec![ServerError::Launch(allocation())],
            backtrace: Default::default(),
        },
        ServerError::Io(IoError::Execution(Box::new(ServerError::Launch(
            allocation(),
        )))),
        ServerError::Profile(ProfileError::Launch(allocation())),
        ServerError::Profile(ProfileError::Server(Box::new(ServerError::Launch(
            allocation(),
        )))),
    ];
    for error in errors {
        assert!(matches!(server_failure(error), GpuFailure::OutOfMemory(_)));
    }
}

#[test]
fn diagnostic_text_does_not_turn_unrelated_errors_into_out_of_memory() {
    let error = ServerError::Generic {
        reason: "out of memory appeared in a shader comment".into(),
        backtrace: Default::default(),
    };
    assert!(matches!(server_failure(error), GpuFailure::Other(_)));
    let error = ServerError::Io(IoError::BufferTooBig {
        size: u64::MAX,
        backtrace: Default::default(),
    });
    assert!(matches!(server_failure(error), GpuFailure::Other(_)));
}

#[test]
#[ignore = "requires a certified native GPU; explicitly enabling this test never falls back or skips"]
fn native_gpu_tensor_smoke() {
    use burn::{
        backend::{Wgpu, wgpu::WgpuRuntime},
        cubecl::Runtime,
        tensor::Tensor,
    };

    let registry = ComputeRegistry::discover();
    let selected = registry
        .resolve(Backend::Wgpu, None)
        .expect("a certified native GPU is required");
    let entry = registry
        .native
        .get(&selected.id)
        .expect("the selected ID must retain its native adapter");
    assert!(
        entry.opened.get().is_none(),
        "discovery must not open devices eagerly"
    );
    let context = registry
        .open_wgpu(&selected.id)
        .expect("the selected GPU must open");
    assert!(
        matches!(context.device, WgpuDevice::Existing(_)),
        "Burn must execute the registered setup"
    );
    assert_eq!(
        entry.adapter, context.setup.adapter,
        "execution must use the exact enumerated handle"
    );
    let actual = context.setup.adapter.get_info();
    assert!(selected.name.contains(&actual.name));
    assert!(matches!(
        actual.device_type,
        wgpu::DeviceType::DiscreteGpu | wgpu::DeviceType::IntegratedGpu
    ));
    assert!(certified(Platform::current(), &actual));
    assert!(!native::metadata(&entry.adapter, &actual).software_or_indirect);
    #[cfg(target_os = "windows")]
    {
        assert_eq!(actual.backend, wgpu::Backend::Vulkan);
        assert_eq!(context.graphics_api(), "vulkan");
        assert!(matches!(actual.vendor, 0x10de | 0x8086 | 0x1002));
    }
    #[cfg(target_os = "linux")]
    {
        assert_eq!(actual.backend, wgpu::Backend::Vulkan);
        assert_eq!(context.graphics_api(), "vulkan");
        assert!(matches!(actual.vendor, 0x10de | 0x8086 | 0x1002));
    }
    #[cfg(target_os = "macos")]
    {
        assert!(cfg!(target_arch = "aarch64"));
        assert_eq!(actual.backend, wgpu::Backend::Metal);
        assert_eq!(context.graphics_api(), "metal");
        assert!(actual.name.starts_with("Apple "));
    }
    assert_eq!(*WgpuRuntime::client(&context.device).info(), actual.backend);
    if actual.backend == wgpu::Backend::Vulkan {
        assert_eq!(
            WgpuRuntime::name(&WgpuRuntime::client(&context.device)),
            "wgpu<spirv>",
            "Vulkan computation must use the native SPIR-V compiler"
        );
    }
    let reopened = registry
        .clone()
        .open_wgpu(&selected.id)
        .expect("reopening must reuse the registered device");
    assert_eq!(context.device, reopened.device);
    assert_eq!(context.setup.device, reopened.setup.device);

    let left = Tensor::<Wgpu, 2>::from_floats([[1.0, 2.0], [3.0, 4.0]], &context.device);
    let right = Tensor::<Wgpu, 2>::from_floats([[5.0, 6.0], [7.0, 8.0]], &context.device);
    let product = left.matmul(right);
    context
        .check()
        .expect("checking must drain pending fused tensor operations");
    let values = product.clone().into_data().to_vec::<f32>().unwrap();
    assert_eq!(values, vec![19.0, 22.0, 43.0, 50.0]);
    context
        .check()
        .expect("the GPU must remain healthy after tensor work");
    let allocator_bytes = context.memory_usage_bytes();
    assert!(allocator_bytes.is_some_and(|bytes| bytes >= 16));

    // A final model upload can contain no compute commands. Checking the
    // context must finish these queue writes too, before a consumer thread
    // receives the model. Mapping below deliberately makes no submission.
    let expected = [0x12, 0x34, 0x56, 0x78, 0x9a, 0xbc, 0xde, 0xf0];
    let upload = context.setup.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("pending upload synchronization regression"),
        size: expected.len() as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    context.setup.queue.write_buffer(&upload, 0, &expected);
    context
        .check()
        .expect("checking must complete a pending upload without compute work");
    let (mapped, receiver) = std::sync::mpsc::channel();
    upload.map_async(wgpu::MapMode::Read, .., move |result| {
        let _ = mapped.send(result);
    });
    context
        .setup
        .device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(Duration::from_secs(5)),
        })
        .expect("mapping the completed upload must finish");
    receiver
        .recv_timeout(Duration::from_secs(5))
        .expect("mapping must report completion without another submission")
        .expect("the completed upload must be readable");
    assert_eq!(
        upload.get_mapped_range(..).as_ref(),
        expected.as_slice(),
        "check returned before the final queued upload completed"
    );
    upload.unmap();

    // Destroy only this logical device to verify the installed native callback
    // and ensure the registry refuses to reuse a lost context.
    context.setup.device.destroy();
    assert!(matches!(context.check(), Err(GpuFailure::DeviceLost(_))));
    assert!(matches!(
        registry.open_wgpu(&selected.id),
        Err(GpuFailure::DeviceLost(_))
    ));
    eprintln!(
        "Native GPU smoke passed: {} [{}], physical VRAM {:?}, stream allocator bytes {:?}",
        selected.name, selected.id, selected.vram_bytes, allocator_bytes
    );
}
