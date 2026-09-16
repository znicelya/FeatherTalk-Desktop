use burn::backend::{Autodiff, NdArray, Wgpu};

pub type CpuBackend = NdArray<f32>;
pub type CpuAutodiffBackend = Autodiff<CpuBackend>;
pub type GpuBackend = Wgpu<f32, i32, u32>;
pub type GpuAutodiffBackend = Autodiff<GpuBackend>;

#[cfg(any(target_os = "windows", target_os = "linux"))]
pub type CudaBackend = burn_cuda::Cuda<f32, i32>;
#[cfg(any(target_os = "windows", target_os = "linux"))]
pub type CudaAutodiffBackend = Autodiff<CudaBackend>;

#[cfg(target_os = "linux")]
pub type RocmBackend = burn_rocm::Rocm<f32, i32>;
#[cfg(target_os = "linux")]
pub type RocmAutodiffBackend = Autodiff<RocmBackend>;
