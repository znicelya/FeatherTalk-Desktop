use burn::tensor::Tensor;

use crate::{
    ScrfdArtifactManifest, ScrfdArtifactPaths, ScrfdError,
    artifact::load_model,
    generated::scrfd_2_5g,
    output::{GeneratedOutput, ScrfdRawOutput, assemble},
};

pub struct ScrfdModel {
    pub(crate) model: scrfd_2_5g::Model,
    pub(crate) manifest: ScrfdArtifactManifest,
}

impl ScrfdModel {
    pub fn load(paths: &ScrfdArtifactPaths, device: &burn::tensor::Device) -> Result<Self, ScrfdError> {
        let (model, manifest) = load_model(paths, device)?;
        Ok(Self { model, manifest })
    }

    pub fn forward(&self, input: Tensor<4>) -> Result<ScrfdRawOutput, ScrfdError> {
        let actual = input.dims();
        if actual != crate::SCRFD_INPUT_SHAPE {
            return Err(ScrfdError::InvalidInputShape { actual });
        }
        let outputs: GeneratedOutput = self.model.forward(input);
        assemble(outputs)
    }

    pub fn manifest(&self) -> &ScrfdArtifactManifest {
        &self.manifest
    }
}
