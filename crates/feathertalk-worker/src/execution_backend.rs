//! Runtime execution-backend reporting shared by the worker's tensor commands.
//!
//! Burn 0.22 selects the executing backend from the unified [`Device`] at
//! runtime rather than from a compile-time backend type, so the execution name
//! and allocator statistics are derived from the device that a command was
//! handed instead of from a `B: Backend` parameter.

use burn::{backend::DispatchDevice, tensor::Device};

/// Peels the autodiff wrapper, if any, so backend identification always sees
/// the concrete hardware device underneath a training device.
fn hardware_dispatch(device: &Device) -> &DispatchDevice {
    let mut dispatch = device.as_dispatch();
    while let DispatchDevice::Autodiff(inner) = dispatch {
        dispatch = inner;
    }
    dispatch
}

/// The stable execution-name string recorded in task results for `device`.
///
/// The names match the backends the worker advertises in its handshake, and an
/// unrecognized device is reported verbatim as its dispatch backend so a result
/// is never silently mislabeled.
pub fn execution_name(device: &Device) -> &'static str {
    match hardware_dispatch(device) {
        DispatchDevice::Flex(_) => "flex-cpu",
        DispatchDevice::Vulkan(_) | DispatchDevice::Wgpu(_) => "wgpu",
        #[cfg(any(target_os = "windows", target_os = "linux"))]
        DispatchDevice::Cuda(_) => "cuda",
        #[cfg(target_os = "linux")]
        DispatchDevice::Rocm(_) => "rocm",
        _ => "unknown"}
}

/// Bytes occupied by active CubeCL tensor allocations on `device`, or `None`
/// for a backend whose allocator this helper cannot inspect (e.g. the CPU
/// backend, which has no CubeCL client).
pub fn gpu_memory_bytes(device: &Device) -> Option<u64> {
    match hardware_dispatch(device) {
        DispatchDevice::Vulkan(_) | DispatchDevice::Wgpu(_) => {
            crate::compute::wgpu_memory_usage_bytes(device)
        }
        #[cfg(any(target_os = "windows", target_os = "linux"))]
        DispatchDevice::Cuda(_) => crate::compute::cuda::memory_usage_bytes(device),
        #[cfg(target_os = "linux")]
        DispatchDevice::Rocm(_) => crate::compute::rocm::memory_usage_bytes(device),
        _ => None}
}