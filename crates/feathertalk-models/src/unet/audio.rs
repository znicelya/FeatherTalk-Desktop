use super::{InvertedResidual, InvertedResidualConfig};
use crate::unet::config::{batch_norm, conv2d};
use burn::nn::{BatchNorm, Relu, conv::Conv2d};
use burn::tensor::Tensor;
use burn::tensor::Device;

#[derive(burn::module::Module, Debug)]
pub struct AudioConvHubert {
    pub conv1: InvertedResidual,
    pub conv2: InvertedResidual,
    pub conv3: Conv2d,
    pub bn3: BatchNorm,
    pub conv4: InvertedResidual,
    pub conv5: Conv2d,
    pub bn5: BatchNorm,
    pub relu: Relu,
    pub conv6: InvertedResidual,
    pub conv7: InvertedResidual}

impl AudioConvHubert {
    pub(crate) fn new(channels: [usize; 5], device: &Device) -> Self {
        Self {
            conv1: InvertedResidualConfig::new(16, channels[1])
                .with_expansion(2)
                .init(device),
            conv2: InvertedResidualConfig::new(channels[1], channels[2])
                .with_expansion(2)
                .init(device),
            conv3: conv2d(
                [channels[2], channels[3]],
                [3, 3],
                [2, 2],
                burn::nn::PaddingConfig2d::Explicit(1, 1, 1, 1),
                true,
                device,
            ),
            bn3: batch_norm(channels[3], device),
            conv4: InvertedResidualConfig::new(channels[3], channels[3])
                .with_expansion(2)
                .init(device),
            conv5: conv2d(
                [channels[3], channels[4]],
                [3, 3],
                [2, 2],
                burn::nn::PaddingConfig2d::Explicit(3, 3, 3, 3),
                true,
                device,
            ),
            bn5: batch_norm(channels[4], device),
            relu: Relu,
            conv6: InvertedResidualConfig::new(channels[4], channels[4])
                .with_expansion(2)
                .init(device),
            conv7: InvertedResidualConfig::new(channels[4], channels[4])
                .with_expansion(2)
                .init(device)}
    }

    pub fn forward(&self, input: Tensor<4>) -> Tensor<4> {
        let output = self.conv1.forward(input);
        let output = self.conv2.forward(output);
        let output = self
            .relu
            .forward(self.bn3.forward(self.conv3.forward(output)));
        let output = self.conv4.forward(output);
        let output = self
            .relu
            .forward(self.bn5.forward(self.conv5.forward(output)));
        let output = self.conv6.forward(output);
        self.conv7.forward(output)
    }
}
