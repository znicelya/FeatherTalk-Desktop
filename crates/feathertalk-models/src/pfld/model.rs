use burn::nn::{
    Relu,
    conv::Conv2d,
    pool::{AvgPool2d, AvgPool2dConfig},
};
use burn::tensor::{Tensor, backend::Backend};

use super::config::{PFLD_INPUT_CHANNELS, PFLD_OUTPUT_VALUES, PfldConfig};
use super::ghost::GhostOneBottleneck;
use super::mobileone::MobileOneBlock;

#[derive(burn::module::Module, Debug)]
pub struct PfldGhostOne<B: Backend> {
    #[module(skip)]
    config: PfldConfig,
    conv1: MobileOneBlock<B>,
    conv2: MobileOneBlock<B>,
    conv3_1: GhostOneBottleneck<B>,
    conv3_2: GhostOneBottleneck<B>,
    conv3_3: GhostOneBottleneck<B>,
    conv4_1: GhostOneBottleneck<B>,
    conv4_2: GhostOneBottleneck<B>,
    conv4_3: GhostOneBottleneck<B>,
    conv5_1: GhostOneBottleneck<B>,
    conv5_2: GhostOneBottleneck<B>,
    conv5_3: GhostOneBottleneck<B>,
    conv5_4: GhostOneBottleneck<B>,
    conv6: GhostOneBottleneck<B>,
    conv7: MobileOneBlock<B>,
    conv8: Conv2d<B>,
    conv8_activation: Relu,
    pool1: AvgPool2d,
    pool2: AvgPool2d,
    pool3: AvgPool2d,
    pool4: AvgPool2d,
    head: Conv2d<B>,
}

impl<B: Backend> PfldGhostOne<B> {
    pub fn new(config: PfldConfig, device: &B::Device) -> Self {
        assert_eq!(config.width_factor, 0.5);
        assert_eq!(config.input_size, 192);
        assert_eq!(config.landmark_count * 2, PFLD_OUTPUT_VALUES);
        assert_eq!(PFLD_INPUT_CHANNELS, 3);
        let branches = config.num_conv_branches;
        let conv = |channels: [usize; 2],
                    kernel: usize,
                    stride: usize,
                    padding: usize,
                    groups: usize,
                    linear: bool| {
            MobileOneBlock::new(
                channels[0],
                channels[1],
                kernel,
                stride,
                padding,
                groups,
                branches,
                linear,
                device,
            )
        };
        let ghost = |input, hidden, output, stride| {
            GhostOneBottleneck::new(input, hidden, output, stride, branches, device)
        };
        let pool = |size| AvgPool2dConfig::new([size, size]).init();
        let conv8 = burn::nn::conv::Conv2dConfig::new(
            [16, 64],
            [config.input_size / 16, config.input_size / 16],
        )
        .with_stride([1, 1])
        .with_padding(burn::nn::PaddingConfig2d::Valid)
        .with_bias(false)
        .init(device);
        let head = burn::nn::conv::Conv2dConfig::new(
            [config.pooled_channels(), PFLD_OUTPUT_VALUES],
            [1, 1],
        )
        .with_stride([1, 1])
        .with_padding(burn::nn::PaddingConfig2d::Valid)
        .with_bias(true)
        .init(device);
        Self {
            config,
            conv1: conv([3, 32], 3, 2, 1, 1, false),
            conv2: conv([32, 32], 3, 1, 1, 32, false),
            conv3_1: ghost(32, 48, 40, 2),
            conv3_2: ghost(40, 60, 40, 1),
            conv3_3: ghost(40, 60, 40, 1),
            conv4_1: ghost(40, 100, 48, 2),
            conv4_2: ghost(48, 120, 48, 1),
            conv4_3: ghost(48, 120, 48, 1),
            conv5_1: ghost(48, 168, 72, 2),
            conv5_2: ghost(72, 252, 72, 1),
            conv5_3: ghost(72, 252, 72, 1),
            conv5_4: ghost(72, 252, 72, 1),
            conv6: ghost(72, 108, 8, 1),
            conv7: conv([8, 16], 3, 1, 1, 1, false),
            conv8,
            conv8_activation: Relu,
            pool1: pool(96),
            pool2: pool(48),
            pool3: pool(24),
            pool4: pool(12),
            head,
        }
    }

    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 2> {
        let [batch, channels, height, width] = input.dims();
        assert_eq!(channels, PFLD_INPUT_CHANNELS);
        assert_eq!(height, self.config.input_size);
        assert_eq!(width, self.config.input_size);
        let mut x = self.conv1.forward(input);
        x = self.conv2.forward(x);
        let x1 = self.pool1.forward(x.clone());

        x = self.conv3_1.forward(x);
        x = self.conv3_2.forward(x);
        x = self.conv3_3.forward(x);
        let x2 = self.pool2.forward(x.clone());

        x = self.conv4_1.forward(x);
        x = self.conv4_2.forward(x);
        x = self.conv4_3.forward(x);
        let x3 = self.pool3.forward(x.clone());

        x = self.conv5_1.forward(x);
        x = self.conv5_2.forward(x);
        x = self.conv5_3.forward(x);
        x = self.conv5_4.forward(x);
        let x4 = self.pool4.forward(x.clone());

        x = self.conv6.forward(x);
        x = self.conv7.forward(x);
        let x5 = self.conv8_activation.forward(self.conv8.forward(x));

        let features = Tensor::cat(vec![x1, x2, x3, x4, x5], 1);
        let output = self.head.forward(features);
        let [output_batch, channels, height, width] = output.dims();
        assert_eq!(output_batch, batch);
        assert_eq!(channels, self.config.output_values());
        assert_eq!([height, width], [1, 1]);
        output.reshape([batch, self.config.output_values()])
    }
}

#[allow(non_camel_case_types)]
pub type PFLD_GhostOne<B> = PfldGhostOne<B>;
