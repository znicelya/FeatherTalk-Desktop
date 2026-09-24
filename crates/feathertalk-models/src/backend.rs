use burn::backend::{Autodiff, Flex, Wgpu};

pub type CpuBackend = Flex;
pub type CpuAutodiffBackend = Autodiff<CpuBackend>;
pub type GpuBackend = Wgpu;
pub type GpuAutodiffBackend = Autodiff<GpuBackend>;

#[cfg(any(target_os = "windows", target_os = "linux"))]
pub type CudaBackend = burn_cuda::Cuda;
#[cfg(any(target_os = "windows", target_os = "linux"))]
pub type CudaAutodiffBackend = Autodiff<CudaBackend>;

#[cfg(target_os = "linux")]
pub type RocmBackend = burn_rocm::Rocm;
#[cfg(target_os = "linux")]
pub type RocmAutodiffBackend = Autodiff<RocmBackend>;