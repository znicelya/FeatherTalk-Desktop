#![cfg(any(target_os = "windows", target_os = "linux"))]

mod support;

use burn::tensor::Device;
use feathertalk_domain::Backend;
use feathertalk_models::backend::{
    CudaAutodiffBackend as GpuAutodiffBackend, CudaBackend as GpuBackend,
};
use feathertalk_worker::{ComputeRegistry, GpuContext};

const GPU_BACKEND: Backend = Backend::Cuda;
const GPU_NAME: &str = "cuda";

fn open_gpu(registry: &ComputeRegistry, id: &str) -> (Device<GpuBackend>, GpuContext) {
    let context = registry
        .open_cuda(id)
        .expect("the selected CUDA device opens");
    (context.device.clone(), GpuContext::Cuda(context))
}

#[path = "support/gpu_execution.rs"]
mod gpu_execution;
