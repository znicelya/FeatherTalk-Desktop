mod support;

use burn::tensor::Device;
use feathertalk_domain::Backend;
use feathertalk_worker::{ComputeRegistry, GpuContext};

const GPU_BACKEND: Backend = Backend::Wgpu;
const GPU_NAME: &str = "wgpu";

fn open_gpu(registry: &ComputeRegistry, id: &str) -> (Device, GpuContext) {
    let context = registry
        .open_wgpu(id)
        .expect("the selected wgpu device opens");
    (context.device.clone(), GpuContext::Wgpu(context))
}

// Run identical training, checkpoint, render, feature and face-model checks on
// both backends so CUDA cannot bypass the established GPU contracts.
#[path = "support/gpu_execution.rs"]
mod gpu_execution;
