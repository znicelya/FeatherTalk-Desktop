//! ROCm discovery is independent of Vulkan and never requires a wgpu adapter.

use std::{
    ffi::CStr,
    panic::{AssertUnwindSafe, catch_unwind},
    sync::Arc};

use burn::{
    backend::{DispatchDevice, wgpu::CubeBackend},
    cubecl::{
        Runtime,
        future::block_on,
        hip::{AmdDevice, HipRuntime}},
    tensor::{Device, Tensor}};
use cubecl_hip_sys::{
    HIP_SUCCESS, hipDeviceProp_tR0600, hipGetDeviceCount, hipGetDevicePropertiesR0600};
use feathertalk_domain::{AdapterInfo, AdapterKind, Backend};

use super::{FaultState, GpuFailure, diagnostic, panic_detail, server_failure};

/// The UUID-selected device and its persistent fault state.
#[derive(Clone, Debug)]
pub struct RocmContext {
    pub device: Device,
    faults: Arc<FaultState>}

impl RocmContext {
    pub fn check(&self) -> Result<(), GpuFailure> {
        self.faults.check()?;
        let result = catch_unwind(AssertUnwindSafe(|| {
            let native = rocm_device(&self.device).ok_or_else(|| {
                GpuFailure::Other("RocmContext holds a non-ROCm device".into())
            })?;
            burn_fusion::get_client::<CubeBackend<HipRuntime>>(native)
                .sync(|| ());
            let client = HipRuntime::client(native);
            client.flush().map_err(server_failure)?;
            block_on(client.sync()).map_err(server_failure)
        }));
        match result {
            Ok(Ok(())) => {}
            Ok(Err(error)) => self.faults.record(error),
            Err(payload) => self.faults.record(GpuFailure::Other(format!(
                "ROCm synchronization failed: {}",
                panic_detail(payload)
            )))}
        self.faults.check()
    }
}

/// Extract the native ROCm device from the unified [`Device`], or `None` when
/// the device belongs to another backend.
fn rocm_device(device: &Device) -> Option<&AmdDevice> {
    let mut dispatch = device.as_dispatch();
    while let DispatchDevice::Autodiff(inner) = dispatch {
        dispatch = inner;
    }
    match dispatch {
        DispatchDevice::Rocm(native) => Some(native),
        _ => None}
}

pub(crate) fn memory_usage_bytes(device: &Device) -> Option<u64> {
    let device = rocm_device(device)?;
    catch_unwind(AssertUnwindSafe(|| {
        HipRuntime::client(device)
            .memory_usage()
            .ok()
            .map(|usage| usage.bytes_in_use)
    }))
    .ok()
    .flatten()
}

pub(super) fn discover() -> Vec<(AdapterInfo, RocmContext)> {
    match catch_unwind(AssertUnwindSafe(discover_checked)) {
        Ok(Ok(adapters)) => adapters,
        Ok(Err(reason)) => {
            diagnostic(format_args!(
                "ROCm unavailable; using existing backends: {reason}"
            ));
            Vec::new()
        }
        Err(payload) => {
            diagnostic(format_args!(
                "ROCm unavailable; using existing backends: {}",
                panic_detail(payload)
            ));
            Vec::new()
        }
    }
}

fn discover_checked() -> Result<Vec<(AdapterInfo, RocmContext)>, String> {
    let mut count = 0_i32;
    // SAFETY: `count` is a live output integer and this probe owns the call.
    let status = unsafe { hipGetDeviceCount(&mut count) };
    if status != HIP_SUCCESS {
        return Err(format!(
            "HIP device count query failed with status {status}"
        ));
    }
    if count <= 0 {
        return Err("no HIP devices were found".into());
    }

    let mut adapters = Vec::new();
    for index in 0..count as usize {
        let opened = catch_unwind(AssertUnwindSafe(|| open_and_probe(index)));
        match opened {
            Ok(Ok(adapter)) => adapters.push(adapter),
            Ok(Err(reason)) => {
                diagnostic(format_args!("ROCm device {index} unavailable: {reason}"))
            }
            Err(payload) => diagnostic(format_args!(
                "ROCm device {index} failed its runtime probe: {}",
                panic_detail(payload)
            ))}
    }
    Ok(adapters)
}

fn open_and_probe(index: usize) -> Result<(AdapterInfo, RocmContext), String> {
    // SAFETY: The struct is plain HIP FFI output storage, and HIP initializes
    // every field on success before it is read.
    let mut properties = unsafe { std::mem::zeroed::<hipDeviceProp_tR0600>() };
    // SAFETY: `properties` is a live output object and the index is bounded by
    // the successful device-count query.
    let status = unsafe { hipGetDevicePropertiesR0600(&mut properties, index as i32) };
    if status != HIP_SUCCESS {
        return Err(format!(
            "HIP property query failed for device {index} with status {status}"
        ));
    }

    // SAFETY: HIP writes a NUL-terminated device name into this fixed array.
    let name = unsafe { CStr::from_ptr(properties.name.as_ptr()) }
        .to_string_lossy()
        .into_owned();
    let uuid = properties.uuid.bytes.map(|byte| byte as u8);
    if uuid.iter().all(|byte| *byte == 0) {
        return Err("driver returned an empty device UUID".into());
    }

    let context = RocmContext {
        device: Device::new(AmdDevice::new(index)),
        faults: Arc::new(FaultState::default())};
    let values = (Tensor::<1>::from_floats([1.0, 2.0, 3.0], &context.device) * 2.0)
        .into_data()
        .try_to_vec::<f32>()
        .map_err(|error| format!("ROCm readback failed: {error:?}"))?;
    context.check().map_err(|error| error.to_string())?;
    if values != [2.0, 4.0, 6.0] {
        return Err(format!("ROCm kernel returned incorrect values: {values:?}"));
    }

    Ok((
        AdapterInfo {
            id: format!("rocm-uuid-{}", hex::encode(uuid)),
            name: format!("{name} (ROCm)"),
            backend: Backend::Rocm,
            kind: if properties.integrated != 0 {
                AdapterKind::Integrated
            } else {
                AdapterKind::Discrete
            },
            certified: true,
            vram_bytes: Some(properties.totalGlobalMem as u64)},
        context,
    ))
}
