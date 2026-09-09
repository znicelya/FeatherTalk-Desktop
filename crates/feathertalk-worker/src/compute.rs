//! Native compute discovery and exact adapter registration for Burn.
//!
//! Discovery retains adapters without opening logical compute devices. IDs use
//! native hardware identities where WGPU exposes them, with a stable model hash
//! as a fallback. An ambiguous identity is never eligible for computation.

use std::{
    any::Any,
    collections::BTreeMap,
    fmt,
    io::Write,
    panic::{AssertUnwindSafe, catch_unwind},
    sync::{Arc, Mutex, OnceLock},
    time::Duration,
};

use burn::{
    backend::wgpu::{CubeBackend, RuntimeOptions, WgpuDevice, WgpuRuntime, WgpuSetup, init_device},
    cubecl::{
        Runtime,
        future::block_on,
        server::{IoError, LaunchError, ProfileError, ServerError},
    },
};
use feathertalk_domain::{
    AdapterInfo, AdapterKind, Backend, ErrorCode, MAX_DETAIL_CHARS, Recovery, TaskError, TaskStage,
};
use sha2::{Digest, Sha256};

mod native;

/// Adapter metadata and the native handles that metadata identifies.
///
/// Clones share initialization state, so opening the same adapter from multiple
/// worker components never registers the WGPU setup twice with Burn.
#[derive(Clone, Debug)]
pub struct ComputeRegistry {
    adapters: Vec<AdapterInfo>,
    native: BTreeMap<String, Arc<NativeAdapter>>,
}

impl ComputeRegistry {
    pub fn cpu_only() -> Self {
        Self {
            adapters: vec![crate::handshake::cpu_adapter()],
            native: BTreeMap::new(),
        }
    }

    /// Enumerate Vulkan on Windows/Linux or Metal on macOS, once per GPU.
    /// Driver/enumeration failures preserve the CPU adapter and go to stderr.
    pub fn discover() -> Self {
        match catch_unwind(AssertUnwindSafe(Self::discover_native)) {
            Ok(Ok(registry)) => registry,
            Ok(Err(reason)) => {
                diagnostic(format_args!("discovery: {reason}"));
                Self::cpu_only()
            }
            Err(payload) => {
                diagnostic(format_args!("discovery failed: {}", panic_detail(payload)));
                Self::cpu_only()
            }
        }
    }

    fn discover_native() -> Result<Self, String> {
        let platform = Platform::current();
        let backend = platform
            .backend()
            .ok_or_else(|| "this platform has no supported native graphics API".to_owned())?;
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: backend.into(),
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let handles = block_on(instance.enumerate_adapters(backend.into()));
        if handles.is_empty() {
            return Err(format!("no {} adapters were found", api_name(backend)));
        }

        let mut discovered = Vec::with_capacity(handles.len());
        for adapter in handles {
            let info = adapter.get_info();
            if info.backend != backend {
                return Err("enumeration returned an adapter from a different graphics API".into());
            }
            let metadata = native::metadata(&adapter, &info);
            discovered.push((info, metadata, adapter));
        }
        let (mut adapters, handles) = physical_adapters(platform, discovered);
        disambiguate_ids(&mut adapters);

        // Bind IDs before sorting; sorting the metadata independently of its
        // handles would silently select another GPU on multi-adapter machines.
        let native = adapters
            .iter()
            .zip(handles)
            .map(|(info, adapter)| {
                (
                    info.id.clone(),
                    Arc::new(NativeAdapter {
                        instance: instance.clone(),
                        adapter,
                        opened: OnceLock::new(),
                    }),
                )
            })
            .collect();
        adapters.sort_by(|left, right| left.id.cmp(&right.id));
        adapters.insert(0, crate::handshake::cpu_adapter());
        Ok(Self { adapters, native })
    }

    pub fn adapters(&self) -> &[AdapterInfo] {
        &self.adapters
    }

    /// Match an explicit ID exactly. Without a WGPU ID, select the certified
    /// hardware adapter with the lexicographically smallest stable ID.
    pub fn resolve(&self, backend: Backend, id: Option<&str>) -> Result<AdapterInfo, String> {
        resolve_adapter(&self.adapters, backend, id)
    }

    /// Open and register the retained native handle once, without enumerating
    /// again or delegating device selection to Burn's implicit device variants.
    pub fn open_wgpu(&self, id: &str) -> Result<WgpuContext, GpuFailure> {
        self.resolve(Backend::Wgpu, Some(id))
            .map_err(GpuFailure::Unavailable)?;
        let entry = self.native.get(id).ok_or_else(|| {
            GpuFailure::Unavailable(format!("WGPU adapter {id} has no retained native handle"))
        })?;
        let result = entry.opened.get_or_init(|| {
            let faults = Arc::new(FaultState::default());
            match catch_unwind(AssertUnwindSafe(|| entry.open(faults.clone()))) {
                Ok(result) => result,
                Err(payload) => Err(faults.check().err().unwrap_or_else(|| {
                    GpuFailure::Unavailable(format!(
                        "Could not initialize WGPU adapter {id}: {}",
                        panic_detail(payload)
                    ))
                })),
            }
        });
        let context = result.as_ref().map_err(Clone::clone)?;
        context.check()?;
        Ok(context.clone())
    }
}

/// Filter descriptions and original handles together, before assigning IDs.
/// Matching model names must not merge distinct physical GPUs.
fn physical_adapters<T>(
    platform: Platform,
    discovered: impl IntoIterator<Item = (wgpu::AdapterInfo, native::NativeMetadata, T)>,
) -> (Vec<AdapterInfo>, Vec<T>) {
    discovered
        .into_iter()
        .filter_map(|(info, metadata, handle)| {
            if metadata.software_or_indirect
                || !matches!(
                    info.device_type,
                    wgpu::DeviceType::DiscreteGpu | wgpu::DeviceType::IntegratedGpu
                )
                || is_software_adapter(&info)
            {
                return None;
            }
            Some((
                AdapterInfo {
                    id: adapter_id(&info, metadata.identity.as_deref()),
                    name: format!("{} ({})", info.name, api_name(info.backend)),
                    backend: Backend::Wgpu,
                    kind: adapter_kind(info.device_type),
                    certified: certified(platform, &info),
                    vram_bytes: metadata.vram_bytes,
                },
                handle,
            ))
        })
        .unzip()
}

#[derive(Debug)]
struct NativeAdapter {
    instance: wgpu::Instance,
    adapter: wgpu::Adapter,
    opened: OnceLock<Result<WgpuContext, GpuFailure>>,
}

impl NativeAdapter {
    fn open(&self, faults: Arc<FaultState>) -> Result<WgpuContext, GpuFailure> {
        // The WGSL path is still used on Metal. Vulkan must use CubeCL's
        // matching device factory: ordinary request_device does not enable
        // the native SPIR-V/compute extensions that its compiler advertises.
        let descriptor = wgpu::DeviceDescriptor {
            label: Some("FeatherTalk compute"),
            required_features: self
                .adapter
                .features()
                .difference(wgpu::Features::MAPPABLE_PRIMARY_BUFFERS),
            required_limits: self.adapter.limits(),
            memory_hints: wgpu::MemoryHints::MemoryUsage,
            trace: wgpu::Trace::Off,
            // SAFETY: This is the feature configuration used by the pinned
            // CubeCL WGSL backend, which owns shader generation and validation.
            experimental_features: unsafe { wgpu::ExperimentalFeatures::enabled() },
        };
        let (device, queue) = if self.adapter.get_info().backend == wgpu::Backend::Vulkan {
            block_on(burn::cubecl::wgpu::vulkan::request_vulkan_device(
                &self.adapter,
            ))
            .ok_or_else(|| {
                GpuFailure::Unavailable(format!(
                    "{} does not support the Vulkan features required by the SPIR-V compiler",
                    self.adapter.get_info().name
                ))
            })?
        } else {
            block_on(self.adapter.request_device(&descriptor)).map_err(|error| {
                GpuFailure::Unavailable(format!(
                    "Could not open {}: {error}",
                    self.adapter.get_info().name
                ))
            })?
        };

        // Install callbacks before init_device can create runtime resources.
        // The callbacks retain only the fault state, avoiding a device cycle.
        let uncaptured = faults.clone();
        device.on_uncaptured_error(Arc::new(move |error| uncaptured.record_wgpu(error)));
        let lost = faults.clone();
        device.set_device_lost_callback(move |reason, detail| {
            lost.record(GpuFailure::DeviceLost(format!("{reason:?}: {detail}")));
        });
        let setup = WgpuSetup {
            instance: self.instance.clone(),
            adapter: self.adapter.clone(),
            device,
            queue,
            backend: self.adapter.get_info().backend,
        };
        let device = init_device(setup.clone(), RuntimeOptions::default());
        Ok(WgpuContext {
            device,
            setup,
            faults,
        })
    }
}

/// Registered Burn device, its exact native setup, and persistent fault state.
#[derive(Clone, Debug)]
pub struct WgpuContext {
    pub device: WgpuDevice,
    setup: WgpuSetup,
    faults: Arc<FaultState>,
}

impl WgpuContext {
    /// Native API of the registered setup that actually executes this task.
    pub fn graphics_api(&self) -> &'static str {
        api_slug(self.setup.backend)
    }

    /// Bytes occupied by active CubeCL tensor allocations on the calling
    /// compute stream. This excludes allocator reserves and is not VRAM size.
    pub fn memory_usage_bytes(&self) -> Option<u64> {
        self.faults.check().ok()?;
        wgpu_memory_usage_bytes(&self.device)
    }

    /// Flush the calling compute stream and pending uploads, wait for native
    /// work, and return any recorded fault. Failed contexts remain unusable
    /// until the worker restarts.
    pub fn check(&self) -> Result<(), GpuFailure> {
        let result = catch_unwind(AssertUnwindSafe(|| {
            self.setup
                .device
                .poll(wgpu::PollType::Poll)
                .map_err(|error| GpuFailure::Other(error.to_string()))?;
            self.faults.check()?;
            // Fusion can retain the last optimizer or upload operations above
            // CubeCL's queue. Drain it without Backend::sync's unbounded wait;
            // the native poll below owns the timeout and typed fault boundary.
            burn_fusion::get_client::<CubeBackend<WgpuRuntime, f32, i32, u32>>(&self.device)
                .sync(|| ());
            WgpuRuntime::client(&self.device)
                .flush()
                .map_err(server_failure)?;
            // CubeCL can leave a final upload batch without compute commands
            // unsubmitted. Flush those queue writes and wait for this boundary.
            let submission = self.setup.queue.submit([]);
            self.setup
                .device
                .poll(wgpu::PollType::Wait {
                    submission_index: Some(submission),
                    timeout: Some(Duration::from_secs(30)),
                })
                .map_err(|error| {
                    GpuFailure::Other(format!("GPU synchronization failed: {error}"))
                })?;
            Ok(())
        }));
        let failure = match result {
            Ok(Ok(())) => None,
            Ok(Err(failure)) => Some(failure),
            Err(payload) => Some(GpuFailure::Other(format!(
                "GPU runtime failed: {}",
                panic_detail(payload)
            ))),
        };
        if let Some(failure) = failure {
            self.faults.record(failure);
        }
        self.faults.check()
    }
}

/// Real allocator usage for generic worker execution paths. The telemetry
/// helper must never initialize an implicit/default or software device.
pub(crate) fn wgpu_memory_usage_bytes(device: &WgpuDevice) -> Option<u64> {
    if !matches!(device, WgpuDevice::Existing(_)) {
        return None;
    }
    catch_unwind(AssertUnwindSafe(|| {
        WgpuRuntime::client(device)
            .memory_usage()
            .ok()
            .map(|usage| usage.bytes_in_use)
    }))
    .ok()
    .flatten()
}

#[derive(Clone, Debug, thiserror::Error)]
pub enum GpuFailure {
    #[error("{0}")]
    OutOfMemory(String),
    #[error("{0}")]
    DeviceLost(String),
    #[error("{0}")]
    Unavailable(String),
    #[error("{0}")]
    Other(String),
}

impl GpuFailure {
    pub fn task_error(&self, stage: TaskStage) -> TaskError {
        let (code, summary, recovery) = match self {
            Self::OutOfMemory(_) => (
                ErrorCode::GpuOutOfMemory,
                "GPU 显存不足，请选择其他设备",
                Recovery::SelectDifferentAdapter,
            ),
            Self::DeviceLost(_) => (
                ErrorCode::GpuDeviceLost,
                "GPU 设备已断开，请从检查点恢复",
                Recovery::ResumeFromCheckpoint,
            ),
            Self::Unavailable(_) => (
                ErrorCode::WorkerCrashed,
                "所选计算设备不可用，请重新选择",
                Recovery::SelectDifferentAdapter,
            ),
            Self::Other(_) => (
                ErrorCode::WorkerCrashed,
                "GPU 计算失败，请重试或从检查点恢复",
                Recovery::ResumeFromCheckpoint,
            ),
        };
        let detail: String = self.to_string().chars().take(MAX_DETAIL_CHARS).collect();
        let mut error = TaskError::new(code, summary, &detail, stage);
        error.recovery = recovery;
        error
    }

    fn is_typed_hardware_fault(&self) -> bool {
        matches!(self, Self::OutOfMemory(_) | Self::DeviceLost(_))
    }
}

#[derive(Clone, Copy, Debug)]
enum Platform {
    Windows,
    Linux,
    AppleSilicon,
    IntelMac,
    Other,
}

impl Platform {
    fn current() -> Self {
        if cfg!(target_os = "windows") {
            Self::Windows
        } else if cfg!(target_os = "linux") {
            Self::Linux
        } else if cfg!(all(target_os = "macos", target_arch = "aarch64")) {
            Self::AppleSilicon
        } else if cfg!(target_os = "macos") {
            Self::IntelMac
        } else {
            Self::Other
        }
    }

    fn backend(self) -> Option<wgpu::Backend> {
        match self {
            Self::Windows | Self::Linux => Some(wgpu::Backend::Vulkan),
            Self::AppleSilicon | Self::IntelMac => Some(wgpu::Backend::Metal),
            Self::Other => None,
        }
    }
}

// The protocol's `certified` flag expresses device admission, not a claim that
// every supported model/driver combination has undergone a hardware test.
fn certified(platform: Platform, info: &wgpu::AdapterInfo) -> bool {
    if !matches!(
        info.device_type,
        wgpu::DeviceType::DiscreteGpu | wgpu::DeviceType::IntegratedGpu
    ) || is_software_adapter(info)
    {
        return false;
    }
    match platform {
        Platform::Windows | Platform::Linux => {
            Some(info.backend) == platform.backend()
                && match info.vendor {
                    0x10de => info.device_type == wgpu::DeviceType::DiscreteGpu,
                    0x1002 => true,
                    0x8086 => info
                        .name
                        .split(|c: char| !c.is_ascii_alphanumeric())
                        .any(|token| token.eq_ignore_ascii_case("arc")),
                    _ => false,
                }
        }
        Platform::AppleSilicon => {
            info.backend == wgpu::Backend::Metal
                && matches!(info.vendor, 0 | 0x106b)
                && info.name.starts_with("Apple ")
                && info.device_type == wgpu::DeviceType::IntegratedGpu
        }
        Platform::IntelMac | Platform::Other => false,
    }
}

fn is_software_adapter(info: &wgpu::AdapterInfo) -> bool {
    let description =
        format!("{} {} {}", info.name, info.driver, info.driver_info).to_ascii_lowercase();
    [
        "llvmpipe",
        "lavapipe",
        "swiftshader",
        "basic render driver",
        "software",
        "warp",
    ]
    .iter()
    .any(|software| description.contains(software))
}

fn adapter_kind(kind: wgpu::DeviceType) -> AdapterKind {
    match kind {
        wgpu::DeviceType::DiscreteGpu => AdapterKind::Discrete,
        wgpu::DeviceType::IntegratedGpu => AdapterKind::Integrated,
        wgpu::DeviceType::Cpu => AdapterKind::Cpu,
        wgpu::DeviceType::Other | wgpu::DeviceType::VirtualGpu => AdapterKind::Other,
    }
}

fn api_name(api: wgpu::Backend) -> &'static str {
    match api {
        wgpu::Backend::Dx12 => "DX12",
        wgpu::Backend::Metal => "Metal",
        wgpu::Backend::Vulkan => "Vulkan",
        wgpu::Backend::Gl => "OpenGL",
        wgpu::Backend::BrowserWebGpu => "WebGPU",
        wgpu::Backend::Noop => "Noop",
    }
}

fn api_slug(api: wgpu::Backend) -> &'static str {
    match api {
        wgpu::Backend::Dx12 => "dx12",
        wgpu::Backend::Metal => "metal",
        wgpu::Backend::Vulkan => "vulkan",
        wgpu::Backend::Gl => "opengl",
        wgpu::Backend::BrowserWebGpu => "webgpu",
        wgpu::Backend::Noop => "noop",
    }
}

fn adapter_id(info: &wgpu::AdapterInfo, native_id: Option<&str>) -> String {
    let identity = native_id.map(str::to_owned).unwrap_or_else(|| {
        if !info.device_pci_bus_id.is_empty() {
            return format!("pci-{}", info.device_pci_bus_id);
        }
        // Do not hash driver/version strings: installing a new driver must not
        // invalidate a persisted device choice. Never hash Rust's random state.
        let mut hash = Sha256::new();
        hash.update(info.name.as_bytes());
        hash.update([0]);
        hash.update(format!("{:?}", info.device_type).as_bytes());
        let digest = hash.finalize();
        format!("model-{}", hex::encode(&digest[..16]))
    });
    format!(
        "wgpu-{}-{:04x}-{:04x}-{identity}",
        api_slug(info.backend),
        info.vendor,
        info.device
    )
}

fn resolve_adapter(
    adapters: &[AdapterInfo],
    backend: Backend,
    id: Option<&str>,
) -> Result<AdapterInfo, String> {
    if backend == Backend::Cpu {
        let id = id.unwrap_or(crate::handshake::CPU_ADAPTER_ID);
        if id != crate::handshake::CPU_ADAPTER_ID {
            return Err(format!("CPU backend requires adapter cpu-0, got {id}"));
        }
        return adapters
            .iter()
            .find(|adapter| adapter.id == id && adapter.backend == Backend::Cpu)
            .cloned()
            .ok_or_else(|| "CPU adapter cpu-0 is unavailable".into());
    }
    let selectable = |adapter: &&AdapterInfo| {
        adapter.backend == Backend::Wgpu
            && adapter.certified
            && matches!(
                adapter.kind,
                AdapterKind::Discrete | AdapterKind::Integrated
            )
    };
    match id {
        Some(id) => {
            let adapter = adapters
                .iter()
                .find(|adapter| adapter.id == id)
                .ok_or_else(|| format!("Unknown WGPU adapter {id}"))?;
            if !selectable(&adapter) {
                return Err(format!(
                    "Adapter {id} is not a certified native WGPU hardware device"
                ));
            }
            Ok(adapter.clone())
        }
        None => adapters
            .iter()
            .filter(selectable)
            .min_by(|left, right| left.id.cmp(&right.id))
            .cloned()
            .ok_or_else(|| "No certified native WGPU hardware adapter is available".into()),
    }
}

fn disambiguate_ids(adapters: &mut [AdapterInfo]) {
    let mut counts = BTreeMap::new();
    for adapter in adapters.iter() {
        *counts.entry(adapter.id.clone()).or_insert(0_u32) += 1;
    }
    let mut ordinals = BTreeMap::new();
    for adapter in adapters {
        if counts[&adapter.id] > 1 {
            let ordinal = ordinals.entry(adapter.id.clone()).or_insert(0_u32);
            *ordinal += 1;
            adapter.id = format!("{}-ambiguous-{ordinal}", adapter.id);
            adapter.certified = false;
            diagnostic(format_args!(
                "discovery: {} has an ambiguous hardware identity and cannot be selected",
                adapter.name
            ));
        }
    }
}

#[derive(Debug, Default)]
struct FaultState {
    failure: Mutex<Option<GpuFailure>>,
}

impl FaultState {
    fn record(&self, failure: GpuFailure) {
        let mut state = self
            .failure
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        // Preserve the first typed fault: OOM often causes a later device-loss
        // or validation callback. A typed fault can replace an earlier generic
        // diagnostic that merely reported an invalid resource.
        if state
            .as_ref()
            .is_none_or(|old| !old.is_typed_hardware_fault() && failure.is_typed_hardware_fault())
        {
            diagnostic(format_args!("{failure}"));
            *state = Some(failure);
        }
    }

    fn record_wgpu(&self, error: wgpu::Error) {
        let failure = match error {
            wgpu::Error::OutOfMemory { source } => {
                GpuFailure::OutOfMemory(format!("WGPU allocation failed: {source}"))
            }
            wgpu::Error::Validation { description, .. }
            | wgpu::Error::Internal { description, .. } => GpuFailure::Other(description),
        };
        self.record(failure);
    }

    fn check(&self) -> Result<(), GpuFailure> {
        match self
            .failure
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .as_ref()
        {
            Some(failure) => Err(failure.clone()),
            None => Ok(()),
        }
    }
}

fn server_failure(error: ServerError) -> GpuFailure {
    fn allocation_failure(error: &ServerError) -> bool {
        match error {
            ServerError::Launch(error) | ServerError::Profile(ProfileError::Launch(error)) => {
                match error {
                    LaunchError::OutOfMemory { .. } => true,
                    LaunchError::IoError(IoError::Execution(error)) => allocation_failure(error),
                    _ => false,
                }
            }
            ServerError::ServerUnhealthy { errors, .. } => errors.iter().any(allocation_failure),
            ServerError::Io(IoError::Execution(error))
            | ServerError::Profile(ProfileError::Server(error)) => allocation_failure(error),
            _ => false,
        }
    }
    if allocation_failure(&error) {
        GpuFailure::OutOfMemory(error.to_string())
    } else {
        GpuFailure::Other(error.to_string())
    }
}

fn diagnostic(message: fmt::Arguments<'_>) {
    // In particular, a callback must not panic if stderr has been closed.
    let _ = writeln!(std::io::stderr().lock(), "FeatherTalk GPU: {message}");
}

fn panic_detail(payload: Box<dyn Any + Send>) -> String {
    if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else if let Some(message) = payload.downcast_ref::<&str>() {
        (*message).to_owned()
    } else {
        "native graphics runtime panicked without a string diagnostic".into()
    }
}

#[cfg(test)]
mod tests;
