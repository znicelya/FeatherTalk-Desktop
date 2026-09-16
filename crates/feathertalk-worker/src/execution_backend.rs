//! Reporting shared by the worker's supported tensor backends.

use burn::{
    backend::Autodiff,
    tensor::{Device, backend::Backend},
};
use feathertalk_models::backend::{CpuBackend, GpuBackend};

/// Derives result metadata and allocator statistics from the backend that
/// actually executes the model, including its autodiff wrapper.
pub trait WorkerBackend: Backend {
    const EXECUTION_NAME: &'static str;

    fn gpu_memory_bytes(_device: &Device<Self>) -> Option<u64> {
        None
    }
}

impl WorkerBackend for CpuBackend {
    const EXECUTION_NAME: &'static str = "ndarray-cpu";
}

impl WorkerBackend for GpuBackend {
    const EXECUTION_NAME: &'static str = "wgpu";

    fn gpu_memory_bytes(device: &Device<Self>) -> Option<u64> {
        crate::compute::wgpu_memory_usage_bytes(device)
    }
}

#[cfg(any(target_os = "windows", target_os = "linux"))]
impl WorkerBackend for feathertalk_models::backend::CudaBackend {
    const EXECUTION_NAME: &'static str = "cuda";

    fn gpu_memory_bytes(device: &Device<Self>) -> Option<u64> {
        crate::compute::cuda::memory_usage_bytes(device)
    }
}

#[cfg(target_os = "linux")]
impl WorkerBackend for feathertalk_models::backend::RocmBackend {
    const EXECUTION_NAME: &'static str = "rocm";

    fn gpu_memory_bytes(device: &Device<Self>) -> Option<u64> {
        crate::compute::rocm::memory_usage_bytes(device)
    }
}

impl<B: WorkerBackend> WorkerBackend for Autodiff<B> {
    const EXECUTION_NAME: &'static str = B::EXECUTION_NAME;

    fn gpu_memory_bytes(device: &Device<Self>) -> Option<u64> {
        B::gpu_memory_bytes(device)
    }
}
