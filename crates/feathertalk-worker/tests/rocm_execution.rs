#![cfg(target_os = "linux")]

mod support;

use burn::tensor::Device;
use feathertalk_domain::Backend;
use feathertalk_models::backend::{
    RocmAutodiffBackend as GpuAutodiffBackend, RocmBackend as GpuBackend,
};
use feathertalk_worker::{ComputeRegistry, GpuContext};

const GPU_BACKEND: Backend = Backend::Rocm;
const GPU_NAME: &str = "rocm";

fn open_gpu(registry: &ComputeRegistry, id: &str) -> (Device<GpuBackend>, GpuContext) {
    let context = registry
        .open_rocm(id)
        .expect("the selected ROCm device opens");
    (context.device.clone(), GpuContext::Rocm(context))
}

#[path = "support/gpu_execution.rs"]
mod gpu_execution;
