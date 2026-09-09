use burn::{tensor::Tensor, tensor::backend::Backend};

use crate::{
    ScrfdArtifactManifest, ScrfdArtifactPaths, ScrfdError,
    artifact::load_model,
    generated::scrfd_2_5g,
    output::{GeneratedOutput, ScrfdRawOutput, assemble},
};

pub struct ScrfdModel<B: Backend> {
    pub(crate) model: scrfd_2_5g::Model<B>,
    pub(crate) manifest: ScrfdArtifactManifest,
}

impl<B: Backend> ScrfdModel<B> {
    pub fn load(paths: &ScrfdArtifactPaths, device: &B::Device) -> Result<Self, ScrfdError> {
        let (model, manifest) = load_model(paths, device)?;
        Ok(Self { model, manifest })
    }

    pub fn forward(&self, input: Tensor<B, 4>) -> Result<ScrfdRawOutput<B>, ScrfdError> {
        let actual = input.dims();
        if actual != crate::SCRFD_INPUT_SHAPE {
            return Err(ScrfdError::InvalidInputShape { actual });
        }
        let outputs: GeneratedOutput<B> = self.model.forward(input);
        assemble(outputs)
    }

    pub fn manifest(&self) -> &ScrfdArtifactManifest {
        &self.manifest
    }
}
