use super::{InvertedResidual, InvertedResidualConfig};
use crate::unet::config::{batch_norm, conv2d};
use burn::nn::{BatchNorm, Relu, conv::Conv2d};
use burn::tensor::{Tensor, backend::Backend};

#[derive(burn::module::Module, Debug)]
pub struct AudioConvHubert<B: Backend> {
    pub conv1: InvertedResidual<B>,
    pub conv2: InvertedResidual<B>,
    pub conv3: Conv2d<B>,
    pub bn3: BatchNorm<B>,
    pub conv4: InvertedResidual<B>,
    pub conv5: Conv2d<B>,
    pub bn5: BatchNorm<B>,
    pub relu: Relu,
    pub conv6: InvertedResidual<B>,
    pub conv7: InvertedResidual<B>,
}

impl<B: Backend> AudioConvHubert<B> {
    pub(crate) fn new(channels: [usize; 5], device: &B::Device) -> Self {
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
                .init(device),
        }
    }

    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
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
