use burn::tensor::{Tensor, backend::Backend};

use super::FeatherHubertEncoder;

pub const SAMPLE_RATE: usize = 16_000;
pub const HUBERT_KERNEL: usize = 400;
pub const HUBERT_STRIDE: usize = 320;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FeatherHubertConfig {
    pub channels: usize,
    pub expansion: usize,
    pub num_blocks: usize,
    pub output_dim: usize,
    pub dropout: f64,
}

impl Default for FeatherHubertConfig {
    fn default() -> Self {
        Self {
            channels: 512,
            expansion: 2,
            num_blocks: 12,
            output_dim: 1024,
            dropout: 0.05,
        }
    }
}

impl FeatherHubertConfig {
    pub const fn parity_micro() -> Self {
        Self {
            channels: 32,
            expansion: 2,
            num_blocks: 2,
            output_dim: 64,
            dropout: 0.0,
        }
    }

    pub fn init<B: Backend>(&self, device: &B::Device) -> FeatherHubertEncoder<B> {
        FeatherHubertEncoder::new(self.clone(), device)
    }
}

pub fn expected_hubert_frames(samples: usize) -> usize {
    if samples < HUBERT_KERNEL {
        0
    } else {
        (samples - (HUBERT_KERNEL - HUBERT_STRIDE)) / HUBERT_STRIDE
    }
}

pub fn normalize_waveform<B: Backend>(waveform: Tensor<B, 2>) -> Tensor<B, 2> {
    let mean = waveform.clone().mean_dim(1);
    let centered = waveform - mean;
    let variance = centered.clone().square().mean_dim(1);
    centered / (variance + 1e-7).sqrt()
}

pub fn make_even_tokens<B: Backend>(tokens: Tensor<B, 3>) -> Tensor<B, 3> {
    let [batch, token_count, features] = tokens.dims();
    if token_count % 2 == 0 {
        tokens
    } else {
        tokens.slice([0..batch, 0..token_count - 1, 0..features])
    }
}
