use crate::{MobileOneBlock, ReparameterizedMobileOneBlock};
use burn::module::{Param, RunningState};
use burn::nn::{
    BatchNorm, Relu,
    conv::Conv2d,
    interpolate::{Interpolate2d, Interpolate2dConfig, InterpolateMode},
};
use burn::tensor::Tensor;
use burn::tensor::Device;

use super::{
    blocks::upsample_and_concat,
    config::{batch_norm, conv2d},
};

#[derive(burn::module::Module, Debug)]
pub struct MobileOneSeparableBlock {
    pub depthwise: MobileOneBlock,
    pub pointwise: MobileOneBlock,
    #[module(skip)]
    use_residual: bool}

impl MobileOneSeparableBlock {
    pub(crate) fn new(
        in_channels: usize,
        out_channels: usize,
        stride: usize,
        num_conv_branches: usize,
        use_residual: bool,
        device: &Device,
    ) -> Self {
        Self {
            depthwise: MobileOneBlock::new(
                in_channels,
                in_channels,
                3,
                stride,
                1,
                in_channels,
                num_conv_branches,
                false,
                device,
            ),
            pointwise: MobileOneBlock::new(
                in_channels,
                out_channels,
                1,
                1,
                0,
                1,
                num_conv_branches,
                false,
                device,
            ),
            use_residual: use_residual && stride == 1 && in_channels == out_channels}
    }

    pub fn forward(&self, input: Tensor<4>) -> Tensor<4> {
        let output = self
            .pointwise
            .forward(self.depthwise.forward(input.clone()));
        if self.use_residual {
            input + output
        } else {
            output
        }
    }

    pub(crate) fn reparameterize(&self) -> ReparameterizedMobileOneSeparableBlock {
        ReparameterizedMobileOneSeparableBlock {
            depthwise: self.depthwise.reparameterize(),
            pointwise: self.pointwise.reparameterize(),
            use_residual: self.use_residual}
    }
}

#[derive(burn::module::Module, Debug)]
pub struct ReparameterizedMobileOneSeparableBlock {
    pub depthwise: ReparameterizedMobileOneBlock,
    pub pointwise: ReparameterizedMobileOneBlock,
    #[module(skip)]
    use_residual: bool}

impl ReparameterizedMobileOneSeparableBlock {
    pub fn forward(&self, input: Tensor<4>) -> Tensor<4> {
        let output = self
            .pointwise
            .forward(self.depthwise.forward(input.clone()));
        if self.use_residual {
            input + output
        } else {
            output
        }
    }
}

#[derive(burn::module::Module, Debug)]
pub struct MobileOneDoubleConv {
    pub first: MobileOneSeparableBlock,
    pub second: MobileOneSeparableBlock}

impl MobileOneDoubleConv {
    pub(crate) fn new(
        in_channels: usize,
        out_channels: usize,
        stride: usize,
        num_conv_branches: usize,
        device: &Device,
    ) -> Self {
        Self {
            first: MobileOneSeparableBlock::new(
                in_channels,
                out_channels,
                stride,
                num_conv_branches,
                false,
                device,
            ),
            second: MobileOneSeparableBlock::new(
                out_channels,
                out_channels,
                1,
                num_conv_branches,
                true,
                device,
            )}
    }

    pub fn forward(&self, input: Tensor<4>) -> Tensor<4> {
        self.second.forward(self.first.forward(input))
    }

    pub(crate) fn reparameterize(&self) -> ReparameterizedMobileOneDoubleConv {
        ReparameterizedMobileOneDoubleConv {
            first: self.first.reparameterize(),
            second: self.second.reparameterize()}
    }
}

#[derive(burn::module::Module, Debug)]
pub struct ReparameterizedMobileOneDoubleConv {
    pub first: ReparameterizedMobileOneSeparableBlock,
    pub second: ReparameterizedMobileOneSeparableBlock}

impl ReparameterizedMobileOneDoubleConv {
    pub fn forward(&self, input: Tensor<4>) -> Tensor<4> {
        self.second.forward(self.first.forward(input))
    }
}

#[derive(burn::module::Module, Debug)]
pub struct MobileOneDown {
    pub maxpool_conv: MobileOneDoubleConv}

impl MobileOneDown {
    pub(crate) fn new(
        in_channels: usize,
        out_channels: usize,
        num_conv_branches: usize,
        device: &Device,
    ) -> Self {
        Self {
            maxpool_conv: MobileOneDoubleConv::new(
                in_channels,
                out_channels,
                2,
                num_conv_branches,
                device,
            )}
    }

    pub fn forward(&self, input: Tensor<4>) -> Tensor<4> {
        self.maxpool_conv.forward(input)
    }

    pub(crate) fn reparameterize(&self) -> ReparameterizedMobileOneDown {
        ReparameterizedMobileOneDown {
            maxpool_conv: self.maxpool_conv.reparameterize()}
    }
}

#[derive(burn::module::Module, Debug)]
pub struct ReparameterizedMobileOneDown {
    pub maxpool_conv: ReparameterizedMobileOneDoubleConv}

impl ReparameterizedMobileOneDown {
    pub fn forward(&self, input: Tensor<4>) -> Tensor<4> {
        self.maxpool_conv.forward(input)
    }
}

#[derive(burn::module::Module, Debug)]
pub struct MobileOneUp {
    pub up: Interpolate2d,
    pub conv: MobileOneDoubleConv}

impl MobileOneUp {
    pub(crate) fn new(
        in_channels: usize,
        out_channels: usize,
        num_conv_branches: usize,
        device: &Device,
    ) -> Self {
        Self {
            up: Interpolate2dConfig::new()
                .with_mode(InterpolateMode::Linear)
                .with_scale_factor(Some([2.0, 2.0]))
                .with_align_corners(true)
                .init(),
            conv: MobileOneDoubleConv::new(in_channels, out_channels, 1, num_conv_branches, device)}
    }

    pub fn forward(&self, input: Tensor<4>, skip: Tensor<4>) -> Tensor<4> {
        self.conv
            .forward(upsample_and_concat(&self.up, input, skip))
    }

    pub(crate) fn reparameterize(&self) -> ReparameterizedMobileOneUp {
        ReparameterizedMobileOneUp {
            up: self.up.clone(),
            conv: self.conv.reparameterize()}
    }
}

#[derive(burn::module::Module, Debug)]
pub struct ReparameterizedMobileOneUp {
    pub up: Interpolate2d,
    pub conv: ReparameterizedMobileOneDoubleConv}

impl ReparameterizedMobileOneUp {
    pub fn forward(&self, input: Tensor<4>, skip: Tensor<4>) -> Tensor<4> {
        self.conv
            .forward(upsample_and_concat(&self.up, input, skip))
    }
}

#[derive(burn::module::Module, Debug)]
pub struct ConvBnAct {
    pub conv: Conv2d,
    pub batch_norm: BatchNorm,
    pub activation: Relu}

impl ConvBnAct {
    pub(crate) fn new(
        in_channels: usize,
        out_channels: usize,
        stride: [usize; 2],
        padding: usize,
        device: &Device,
    ) -> Self {
        Self {
            conv: conv2d(
                [in_channels, out_channels],
                [3, 3],
                stride,
                burn::nn::PaddingConfig2d::Explicit(padding, padding, padding, padding),
                false,
                device,
            ),
            batch_norm: batch_norm(out_channels, device),
            activation: Relu}
    }

    pub fn forward(&self, input: Tensor<4>) -> Tensor<4> {
        self.activation
            .forward(self.batch_norm.forward(self.conv.forward(input)))
    }

    pub(crate) fn detached_clone(&self) -> Self {
        Self {
            conv: detached_conv2d(&self.conv),
            batch_norm: detached_batch_norm(&self.batch_norm),
            activation: Relu}
    }
}

#[derive(burn::module::Module, Debug)]
pub struct MobileOneAudioConvHubert {
    pub conv1: MobileOneSeparableBlock,
    pub conv2: MobileOneSeparableBlock,
    pub conv3: MobileOneBlock,
    pub conv4: MobileOneSeparableBlock,
    pub conv5: ConvBnAct,
    pub conv6: MobileOneSeparableBlock,
    pub conv7: MobileOneSeparableBlock}

impl MobileOneAudioConvHubert {
    pub(crate) fn new(channels: [usize; 5], num_conv_branches: usize, device: &Device) -> Self {
        Self {
            conv1: MobileOneSeparableBlock::new(
                16,
                channels[1],
                1,
                num_conv_branches,
                false,
                device,
            ),
            conv2: MobileOneSeparableBlock::new(
                channels[1],
                channels[2],
                1,
                num_conv_branches,
                false,
                device,
            ),
            conv3: MobileOneBlock::new_with_stride(
                channels[2],
                channels[3],
                3,
                [2, 2],
                1,
                1,
                num_conv_branches,
                false,
                device,
            ),
            conv4: MobileOneSeparableBlock::new(
                channels[3],
                channels[3],
                1,
                num_conv_branches,
                true,
                device,
            ),
            conv5: ConvBnAct::new(channels[3], channels[4], [2, 2], 3, device),
            conv6: MobileOneSeparableBlock::new(
                channels[4],
                channels[4],
                1,
                num_conv_branches,
                true,
                device,
            ),
            conv7: MobileOneSeparableBlock::new(
                channels[4],
                channels[4],
                1,
                num_conv_branches,
                true,
                device,
            )}
    }

    pub fn forward(&self, input: Tensor<4>) -> Tensor<4> {
        let output = self.conv1.forward(input);
        let output = self.conv2.forward(output);
        let output = self.conv3.forward(output);
        let output = self.conv4.forward(output);
        let output = self.conv5.forward(output);
        let output = self.conv6.forward(output);
        self.conv7.forward(output)
    }

    pub(crate) fn reparameterize(&self) -> ReparameterizedMobileOneAudioConvHubert {
        ReparameterizedMobileOneAudioConvHubert {
            conv1: self.conv1.reparameterize(),
            conv2: self.conv2.reparameterize(),
            conv3: self.conv3.reparameterize(),
            conv4: self.conv4.reparameterize(),
            conv5: self.conv5.detached_clone(),
            conv6: self.conv6.reparameterize(),
            conv7: self.conv7.reparameterize()}
    }
}

#[derive(burn::module::Module, Debug)]
pub struct ReparameterizedMobileOneAudioConvHubert {
    pub conv1: ReparameterizedMobileOneSeparableBlock,
    pub conv2: ReparameterizedMobileOneSeparableBlock,
    pub conv3: ReparameterizedMobileOneBlock,
    pub conv4: ReparameterizedMobileOneSeparableBlock,
    pub conv5: ConvBnAct,
    pub conv6: ReparameterizedMobileOneSeparableBlock,
    pub conv7: ReparameterizedMobileOneSeparableBlock}

impl ReparameterizedMobileOneAudioConvHubert {
    pub fn forward(&self, input: Tensor<4>) -> Tensor<4> {
        let output = self.conv1.forward(input);
        let output = self.conv2.forward(output);
        let output = self.conv3.forward(output);
        let output = self.conv4.forward(output);
        let output = self.conv5.forward(output);
        let output = self.conv6.forward(output);
        self.conv7.forward(output)
    }
}

pub(crate) fn detached_conv2d(source: &Conv2d) -> Conv2d {
    Conv2d {
        weight: Param::from_tensor(source.weight.val().detach()),
        bias: source
            .bias
            .as_ref()
            .map(|bias| Param::from_tensor(bias.val().detach())),
        stride: source.stride,
        kernel_size: source.kernel_size,
        dilation: source.dilation,
        groups: source.groups,
        padding: source.padding.clone()}
}

fn detached_batch_norm(source: &BatchNorm) -> BatchNorm {
    BatchNorm {
        gamma: Param::from_tensor(source.gamma.val().detach()),
        beta: Param::from_tensor(source.beta.val().detach()),
        training: source.training.clone(),
        running_mean: RunningState::new(source.running_mean.value().detach()),
        running_var: RunningState::new(source.running_var.value().detach()),
        momentum: source.momentum,
        epsilon: source.epsilon}
}
