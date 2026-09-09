use std::path::Path;

use burn::tensor::{Tensor, backend::Backend};
use burn_store::ModuleSnapshot;
use feathertalk_models::PFLD_GhostOne;

use crate::{
    PFLD_INPUT_SHAPE, PFLD_OUTPUT_SHAPE, PfldRuntimeError, PfldRuntimeManifest,
    artifact::load_artifact,
};

pub struct PfldRuntime<B: Backend> {
    model: PFLD_GhostOne<B>,
    manifest: PfldRuntimeManifest,
}

impl<B: Backend> PfldRuntime<B> {
    pub fn load(directory: &Path, device: &B::Device) -> Result<Self, PfldRuntimeError> {
        let (model, manifest) = load_artifact::<B>(directory, device)?;
        Ok(Self { model, manifest })
    }

    pub fn forward(&self, input: Tensor<B, 4>) -> Result<Tensor<B, 2>, PfldRuntimeError> {
        let actual = input.dims();
        if actual != PFLD_INPUT_SHAPE {
            return Err(PfldRuntimeError::InvalidInputShape { actual });
        }
        let values = input
            .clone()
            .into_data()
            .to_vec::<f32>()
            .map_err(|error| PfldRuntimeError::Store(error.to_string()))?;
        if values.iter().any(|value| !value.is_finite()) {
            return Err(PfldRuntimeError::NonFiniteInput);
        }
        let output = self.model.forward(input);
        let output_shape = output.dims();
        if output_shape != PFLD_OUTPUT_SHAPE {
            return Err(PfldRuntimeError::InvalidOutputShape {
                actual: output_shape,
            });
        }
        let output_values = output
            .clone()
            .into_data()
            .to_vec::<f32>()
            .map_err(|error| PfldRuntimeError::Store(error.to_string()))?;
        if output_values.iter().any(|value| !value.is_finite()) {
            return Err(PfldRuntimeError::NonFiniteOutput);
        }
        Ok(output)
    }

    pub fn manifest(&self) -> &PfldRuntimeManifest {
        &self.manifest
    }

    pub fn tensor_count(&self) -> usize
    where
        PFLD_GhostOne<B>: ModuleSnapshot<B>,
    {
        self.model.collect(None, None, false).len()
    }
}
