use burn::{
    module::Module,
    tensor::Tensor,
};

use super::{MobileOneUnet, OriginalUnet};

/// Common training boundary for talking-head UNet graphs.
///
/// The reparameterized MobileOne inference graph must not cross this boundary:
///
/// ```compile_fail
/// use feathertalk_models::{
///     backend::CpuBackend,
///     unet::{MobileOneUnetConfig, TrainableTalkingHead},
/// };
///
/// fn require_training_graph<M: TrainableTalkingHead<CpuBackend>>(_model: &M) {}
///
/// let device = Default::default();
/// let inference_graph = MobileOneUnetConfig::parity_micro()
///     .init(&device)
///     .reparameterize();
/// require_training_graph(&inference_graph);
/// ```
pub trait TrainableTalkingHead {
    fn forward_training(&self, image: Tensor<4>, audio: Tensor<4>) -> Tensor<4>;

    fn freeze_audio(self) -> Self;
}

impl TrainableTalkingHead for OriginalUnet {
    fn forward_training(&self, image: Tensor<4>, audio: Tensor<4>) -> Tensor<4> {
        self.forward(image, audio)
    }

    fn freeze_audio(mut self) -> Self {
        self.audio_model = self.audio_model.no_grad();
        self
    }
}

impl TrainableTalkingHead for MobileOneUnet {
    fn forward_training(&self, image: Tensor<4>, audio: Tensor<4>) -> Tensor<4> {
        self.forward(image, audio)
    }

    fn freeze_audio(mut self) -> Self {
        self.audio_model = self.audio_model.no_grad();
        self
    }
}
