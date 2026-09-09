use burn::tensor::{Tensor, backend::Backend};

use super::{MobileOneUnetInference, OriginalUnet};

/// Common inference boundary for product talking-head UNet graphs.
///
/// The MobileOne training graph must be reparameterized before it can cross this boundary:
///
/// ```compile_fail
/// use feathertalk_models::{
///     backend::CpuBackend,
///     unet::{MobileOneUnetConfig, TalkingHeadModel},
/// };
///
/// fn require_inference_graph<M: TalkingHeadModel<CpuBackend>>(_model: &M) {}
///
/// let device = Default::default();
/// let training_graph = MobileOneUnetConfig::parity_micro().init::<CpuBackend>(&device);
/// require_inference_graph(&training_graph);
/// ```
pub trait TalkingHeadModel<B: Backend> {
    fn forward_talking_head(&self, image: Tensor<B, 4>, audio: Tensor<B, 4>) -> Tensor<B, 4>;
}

impl<B: Backend> TalkingHeadModel<B> for OriginalUnet<B> {
    fn forward_talking_head(&self, image: Tensor<B, 4>, audio: Tensor<B, 4>) -> Tensor<B, 4> {
        self.forward(image, audio)
    }
}

impl<B: Backend> TalkingHeadModel<B> for MobileOneUnetInference<B> {
    fn forward_talking_head(&self, image: Tensor<B, 4>, audio: Tensor<B, 4>) -> Tensor<B, 4> {
        self.forward(image, audio)
    }
}
