use burn::tensor::Tensor;

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
/// let training_graph = MobileOneUnetConfig::parity_micro().init(&device);
/// require_inference_graph(&training_graph);
/// ```
pub trait TalkingHeadModel {
    fn forward_talking_head(&self, image: Tensor<4>, audio: Tensor<4>) -> Tensor<4>;
}

impl TalkingHeadModel for OriginalUnet {
    fn forward_talking_head(&self, image: Tensor<4>, audio: Tensor<4>) -> Tensor<4> {
        self.forward(image, audio)
    }
}

impl TalkingHeadModel for MobileOneUnetInference {
    fn forward_talking_head(&self, image: Tensor<4>, audio: Tensor<4>) -> Tensor<4> {
        self.forward(image, audio)
    }
}
