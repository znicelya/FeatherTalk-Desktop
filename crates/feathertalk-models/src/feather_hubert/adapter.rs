use burn::tensor::{Tensor, TensorData, backend::Backend};
use feathertalk_audio::{AudioError, ChunkEncoder};

use super::{FeatherHubertConfig, FeatherHubertEncoder};

/// Burn-backed adapter for the pure long-audio extraction seam.
#[derive(Debug)]
pub struct BurnFeatherHubertEncoder<B: Backend> {
    model: FeatherHubertEncoder<B>,
    device: B::Device,
    output_dim: usize,
}

impl<B: Backend> BurnFeatherHubertEncoder<B> {
    pub fn from_config(config: FeatherHubertConfig, device: &B::Device) -> Self {
        Self::from_model(config.init(device), device)
    }

    pub fn from_model(model: FeatherHubertEncoder<B>, device: &B::Device) -> Self {
        let output_dim = model.config.output_dim;
        Self {
            model,
            device: device.clone(),
            output_dim,
        }
    }

    pub fn model(&self) -> &FeatherHubertEncoder<B> {
        &self.model
    }
}

impl<B: Backend> ChunkEncoder for BurnFeatherHubertEncoder<B> {
    fn output_dim(&self) -> usize {
        self.output_dim
    }

    fn encode(&mut self, _chunk_index: usize, samples: &[f32]) -> Result<Vec<f32>, AudioError> {
        if samples.iter().any(|sample| !sample.is_finite()) {
            let index = samples
                .iter()
                .position(|sample| !sample.is_finite())
                .unwrap_or(0);
            return Err(AudioError::NonFiniteWaveform { index });
        }
        if samples.len() < super::HUBERT_KERNEL {
            return Ok(Vec::new());
        }
        let tensor = Tensor::<B, 2>::from_data(
            TensorData::new(samples.to_vec(), [1, samples.len()]),
            &self.device,
        );
        let output = self.model.forward(tensor);
        let [batch, tokens, dims] = output.dims();
        if batch != 1 || dims != self.output_dim {
            return Err(AudioError::FeatureLengthMismatch {
                actual: tokens.saturating_mul(dims),
                dimension: self.output_dim,
            });
        }
        let values =
            output
                .into_data()
                .to_vec::<f32>()
                .map_err(|error| AudioError::CommitFailed {
                    operation: "burn_tensor_data",
                    message: error.to_string(),
                })?;
        if let Some(index) = values.iter().position(|value| !value.is_finite()) {
            return Err(AudioError::NonFiniteFeature { index });
        }
        Ok(values)
    }
}
