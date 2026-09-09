use burn::backend::{Autodiff, NdArray, Wgpu};

pub type CpuBackend = NdArray<f32>;
pub type CpuAutodiffBackend = Autodiff<CpuBackend>;
pub type GpuBackend = Wgpu<f32, i32, u32>;
pub type GpuAutodiffBackend = Autodiff<GpuBackend>;
