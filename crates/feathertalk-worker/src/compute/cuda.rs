//! CUDA discovery is independent of Vulkan and never makes CUDA a startup dependency.

use std::{
    panic::{AssertUnwindSafe, catch_unwind},
    path::Path,
    sync::Arc};

use burn::{
    backend::{DispatchDevice, wgpu::CubeBackend},
    cubecl::{
        Runtime,
        cuda::{CudaDevice, CudaRuntime},
        future::block_on},
    tensor::{Device, Tensor}};
use cudarc::driver::CudaContext as DriverContext;
use feathertalk_domain::{AdapterInfo, AdapterKind, Backend};

use super::{FaultState, GpuFailure, diagnostic, panic_detail, server_failure};

/// The UUID-selected device and its retained primary driver context.
#[derive(Clone, Debug)]
pub struct CudaContext {
    pub device: Device,
    _driver: Arc<DriverContext>,
    faults: Arc<FaultState>}

impl CudaContext {
    pub fn check(&self) -> Result<(), GpuFailure> {
        self.faults.check()?;
        let result = catch_unwind(AssertUnwindSafe(|| {
            let native = cuda_device(&self.device).ok_or_else(|| {
                GpuFailure::Other("CudaContext holds a non-CUDA device".into())
            })?;
            // Fusion and CubeCL each buffer operations on the calling thread's
            // stream. Flush both before publishing or leaving a loader thread.
            burn_fusion::get_client::<CubeBackend<CudaRuntime>>(native)
                .sync(|| ());
            let client = CudaRuntime::client(native);
            client.flush().map_err(server_failure)?;
            block_on(client.sync()).map_err(server_failure)
        }));
        match result {
            Ok(Ok(())) => {}
            Ok(Err(error)) => self.faults.record(error),
            Err(payload) => self.faults.record(GpuFailure::Other(format!(
                "CUDA synchronization failed: {}",
                panic_detail(payload)
            )))}
        self.faults.check()
    }
}

/// Extract the native CUDA device from the unified [`Device`], or `None` when
/// the device belongs to another backend.
fn cuda_device(device: &Device) -> Option<&CudaDevice> {
    let mut dispatch = device.as_dispatch();
    while let DispatchDevice::Autodiff(inner) = dispatch {
        dispatch = inner;
    }
    match dispatch {
        DispatchDevice::Cuda(native) => Some(native),
        _ => None}
}

pub(crate) fn memory_usage_bytes(device: &Device) -> Option<u64> {
    let device = cuda_device(device)?;
    catch_unwind(AssertUnwindSafe(|| {
        CudaRuntime::client(device)
            .memory_usage()
            .ok()
            .map(|usage| usage.bytes_in_use)
    }))
    .ok()
    .flatten()
}

pub(super) fn discover() -> Vec<(AdapterInfo, CudaContext)> {
    match catch_unwind(AssertUnwindSafe(discover_checked)) {
        Ok(Ok(adapters)) => adapters,
        Ok(Err(reason)) => {
            diagnostic(format_args!(
                "CUDA unavailable; using existing backends: {reason}"
            ));
            Vec::new()
        }
        Err(payload) => {
            diagnostic(format_args!(
                "CUDA unavailable; using existing backends: {}",
                panic_detail(payload)
            ));
            Vec::new()
        }
    }
}

fn validate_headers(include: &Path) -> Result<(), String> {
    for name in ["cuda.h", "cuda_runtime.h", "cuda_fp16.h", "mma.h"] {
        if !include.join(name).is_file() {
            return Err(format!(
                "CUDA toolkit header {} is missing; install CUDA 12.x or newer and set CUDA_PATH",
                include.join(name).display()
            ));
        }
    }
    Ok(())
}

fn discover_checked() -> Result<Vec<(AdapterInfo, CudaContext)>, String> {
    let root = burn::cubecl::cuda::install::cuda_path()
        .ok_or("CUDA toolkit not found; install CUDA 12.x or newer and set CUDA_PATH")?;
    validate_headers(&root.join("include"))?;
    // SAFETY: These upstream helpers only try loading the platform CUDA
    // libraries. No function pointers or device resources escape the probe.
    if !unsafe { cudarc::driver::sys::is_culib_present() } {
        return Err("NVIDIA CUDA driver library is unavailable".into());
    }
    if !unsafe { cudarc::nvrtc::sys::is_culib_present() } {
        return Err(
            "CUDA 12.x+ NVRTC library is unavailable; add the toolkit bin directory to PATH".into(),
        );
    }
    let mut major = 0;
    let mut minor = 0;
    // SAFETY: NVRTC is present and both output pointers refer to live integers.
    unsafe { cudarc::nvrtc::sys::nvrtcVersion(&mut major, &mut minor) }
        .result()
        .map_err(|error| format!("NVRTC version query failed: {error}"))?;
    if major < 12 {
        return Err(format!(
            "NVRTC {major}.{minor} is unsupported; Burn requires CUDA 12.x or newer"
        ));
    }
    let count = DriverContext::device_count().map_err(|error| error.to_string())?;
    let mut adapters = Vec::new();
    for index in 0..count as usize {
        let opened = catch_unwind(AssertUnwindSafe(|| open_and_probe(index)));
        match opened {
            Ok(Ok(adapter)) => adapters.push(adapter),
            Ok(Err(reason)) => {
                diagnostic(format_args!("CUDA device {index} unavailable: {reason}"))
            }
            Err(payload) => diagnostic(format_args!(
                "CUDA device {index} failed its runtime probe: {}",
                panic_detail(payload)
            ))}
    }
    Ok(adapters)
}

fn open_and_probe(index: usize) -> Result<(AdapterInfo, CudaContext), String> {
    let driver = DriverContext::new(index).map_err(|error| error.to_string())?;
    let mut uuid = cudarc::driver::sys::CUuuid { bytes: [0; 16] };
    // SAFETY: The retained driver context owns this valid device handle and
    // uuid is a live output buffer. The v2 API (CUDA 11.4+) distinguishes MIG
    // instances; cudarc's uuid() uses the legacy API with our 12.8 bindings.
    unsafe { cudarc::driver::sys::cuDeviceGetUuid_v2(&mut uuid, driver.cu_device()) }
        .result()
        .map_err(|error| format!("CUDA device UUID query failed: {error}"))?;
    let uuid = uuid.bytes.map(|byte| byte as u8);
    if uuid.iter().all(|byte| *byte == 0) {
        return Err("driver returned an empty device UUID".into());
    }
    let name = driver.name().map_err(|error| error.to_string())?;
    let vram_bytes = driver.total_mem().ok().map(|bytes| bytes as u64);
    let integrated = driver
        .attribute(cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_INTEGRATED)
        .map_err(|error| error.to_string())?
        != 0;
    let context = CudaContext {
        device: Device::new(CudaDevice::new(index)),
        _driver: driver,
        faults: Arc::new(FaultState::default())};
    // A driver or nvcc version alone does not prove that NVRTC, its builtins,
    // headers and the installed driver can compile and run CubeCL kernels.
    let values = (Tensor::<1>::from_floats([1.0, 2.0, 3.0], &context.device) * 2.0)
        .into_data()
        .try_to_vec::<f32>()
        .map_err(|error| format!("CUDA readback failed: {error:?}"))?;
    context.check().map_err(|error| error.to_string())?;
    if values != [2.0, 4.0, 6.0] {
        return Err(format!("CUDA kernel returned incorrect values: {values:?}"));
    }
    Ok((
        AdapterInfo {
            id: format!("cuda-uuid-{}", hex::encode(uuid)),
            name: format!("{name} (CUDA)"),
            backend: Backend::Cuda,
            kind: if integrated {
                AdapterKind::Integrated
            } else {
                AdapterKind::Discrete
            },
            certified: true,
            vram_bytes},
        context,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn incomplete_toolkit_is_rejected_before_any_cuda_library_is_loaded() {
        let directory = tempfile::tempdir().unwrap();
        std::fs::write(directory.path().join("cuda.h"), b"header").unwrap();
        assert!(validate_headers(directory.path()).is_err());
        for name in ["cuda_runtime.h", "cuda_fp16.h", "mma.h"] {
            std::fs::write(directory.path().join(name), b"header").unwrap();
        }
        assert!(validate_headers(directory.path()).is_ok());
    }
}
