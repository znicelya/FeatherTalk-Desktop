use clap::ValueEnum;
use serde::Serialize;

use crate::metrics::ParityError;

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum GraphicsSelection {
    Auto,
    Dx12,
    Metal,
    Vulkan}

impl GraphicsSelection {
    pub const fn resolved(self) -> Self {
        match self {
            Self::Auto => {
                #[cfg(target_os = "windows")]
                {
                    Self::Vulkan
                }
                #[cfg(target_os = "macos")]
                {
                    Self::Metal
                }
                #[cfg(target_os = "linux")]
                {
                    Self::Vulkan
                }
                #[cfg(not(any(target_os = "windows", target_os = "macos", target_os = "linux")))]
                {
                    Self::Auto
                }
            }
            selected => selected}
    }

    pub fn validate_for_target(self) -> Result<(), ParityError> {
        let supported = match self {
            Self::Auto => true,
            Self::Dx12 => cfg!(target_os = "windows"),
            Self::Metal => cfg!(target_os = "macos"),
            Self::Vulkan => cfg!(any(target_os = "windows", target_os = "linux"))};
        if supported {
            Ok(())
        } else {
            Err(ParityError::Backend(format!(
                "graphics API {} is unavailable on this target",
                self.as_str()
            )))
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Dx12 => "dx12",
            Self::Metal => "metal",
            Self::Vulkan => "vulkan"}
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ExecutionEvidence {
    pub backend: String,
    pub graphics: String,
    pub device: String,
    pub used_cpu_fallback: bool}

pub fn run_wgpu_probe(graphics: GraphicsSelection) -> Result<ExecutionEvidence, ParityError> {
    let graphics = graphics.resolved();
    graphics.validate_for_target()?;
    match graphics {
        GraphicsSelection::Auto => super::fixture::probe_wgpu_with::<
            burn::backend::wgpu::graphics::AutoGraphicsApi,
        >("auto"),
        GraphicsSelection::Dx12 => {
            super::fixture::probe_wgpu_with::<burn::backend::wgpu::graphics::Dx12>("dx12")
        }
        GraphicsSelection::Metal => {
            super::fixture::probe_wgpu_with::<burn::backend::wgpu::graphics::Metal>("metal")
        }
        GraphicsSelection::Vulkan => {
            super::fixture::probe_wgpu_with::<burn::backend::wgpu::graphics::Vulkan>("vulkan")
        }
    }
}
