use burn::tensor::{Tensor, TensorData};
use burn::tensor::Device;
use feathertalk_audio::{AudioError, ChunkEncoder};

use super::{FeatherHubertConfig, FeatherHubertEncoder};

/// Burn-backed adapter for the pure long-audio extraction seam.
#[derive(Debug)]
pub struct BurnFeatherHubertEncoder {
    model: FeatherHubertEncoder,
    device: Device,
    output_dim: usize}

impl BurnFeatherHubertEncoder {
    pub fn from_config(config: FeatherHubertConfig, device: &Device) -> Self {
        Self::from_model(config.init(device), device)
    }

    pub fn from_model(model: FeatherHubertEncoder, device: &Device) -> Self {
        let output_dim = model.config.output_dim;
        Self {
            model,
            device: device.clone(),
            output_dim}
    }

    pub fn model(&self) -> &FeatherHubertEncoder {
        &self.model
    }
}

impl ChunkEncoder for BurnFeatherHubertEncoder {
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
        let tensor = Tensor::<2>::from_data(
            TensorData::new(samples.to_vec(), [1, samples.len()]),
            &self.device,
        );
        let output = self.model.forward(tensor);
        let [batch, tokens, dims] = output.dims();
        if batch != 1 || dims != self.output_dim {
            return Err(AudioError::FeatureLengthMismatch {
                actual: tokens.saturating_mul(dims),
                dimension: self.output_dim});
        }
        let values =
            output
                .into_data()
                .try_to_vec::<f32>()
                .map_err(|error| AudioError::CommitFailed {
                    operation: "burn_tensor_data",
                    message: error.to_string()})?;
        if let Some(index) = values.iter().position(|value| !value.is_finite()) {
            return Err(AudioError::NonFiniteFeature { index });
        }
        Ok(values)
    }
}
