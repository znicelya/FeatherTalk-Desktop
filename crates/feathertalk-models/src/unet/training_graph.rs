use burn::tensor::{Tensor, backend::Backend};

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
///     .init::<CpuBackend>(&device)
///     .reparameterize();
/// require_training_graph(&inference_graph);
/// ```
pub trait TrainableTalkingHead<B: Backend> {
    fn forward_training(&self, image: Tensor<B, 4>, audio: Tensor<B, 4>) -> Tensor<B, 4>;
}

impl<B: Backend> TrainableTalkingHead<B> for OriginalUnet<B> {
    fn forward_training(&self, image: Tensor<B, 4>, audio: Tensor<B, 4>) -> Tensor<B, 4> {
        self.forward(image, audio)
    }
}

impl<B: Backend> TrainableTalkingHead<B> for MobileOneUnet<B> {
    fn forward_training(&self, image: Tensor<B, 4>, audio: Tensor<B, 4>) -> Tensor<B, 4> {
        self.forward(image, audio)
    }
}
