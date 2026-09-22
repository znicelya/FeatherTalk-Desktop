use burn::tensor::Tensor;
use burn::tensor::Device;

use super::mobileone::MobileOneBlock;

#[derive(burn::module::Module, Debug)]
pub struct GhostOneModule {
    primary: MobileOneBlock,
    cheap: MobileOneBlock,
    #[module(skip)]
    out_channels: usize}

impl GhostOneModule {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        is_linear: bool,
        num_conv_branches: usize,
        device: &Device,
    ) -> Self {
        let half = out_channels.div_ceil(2);
        Self {
            primary: MobileOneBlock::new(
                in_channels,
                half,
                1,
                1,
                0,
                1,
                num_conv_branches,
                is_linear,
                device,
            ),
            cheap: MobileOneBlock::new(
                half,
                half,
                3,
                1,
                1,
                half,
                num_conv_branches,
                is_linear,
                device,
            ),
            out_channels}
    }

    pub fn forward(&self, input: Tensor<4>) -> Tensor<4> {
        let primary = self.primary.forward(input);
        let cheap = self.cheap.forward(primary.clone());
        let output = Tensor::cat(vec![primary, cheap], 1);
        let [batch, _, height, width] = output.dims();
        output.slice([0..batch, 0..self.out_channels, 0..height, 0..width])
    }
}

#[derive(burn::module::Module, Debug)]
pub struct GhostOneBottleneck {
    ghost: GhostOneModule,
    depthwise: Option<MobileOneBlock>,
    linear: GhostOneModule}

impl GhostOneBottleneck {
    pub fn new(
        in_channels: usize,
        hidden_channels: usize,
        out_channels: usize,
        stride: usize,
        num_conv_branches: usize,
        device: &Device,
    ) -> Self {
        assert!(matches!(stride, 1 | 2));
        Self {
            ghost: GhostOneModule::new(
                in_channels,
                hidden_channels,
                false,
                num_conv_branches,
                device,
            ),
            depthwise: (stride == 2).then(|| {
                MobileOneBlock::new(
                    hidden_channels,
                    hidden_channels,
                    3,
                    stride,
                    1,
                    hidden_channels,
                    num_conv_branches,
                    true,
                    device,
                )
            }),
            linear: GhostOneModule::new(
                hidden_channels,
                out_channels,
                true,
                num_conv_branches,
                device,
            ),
        }
    }

    pub fn forward(&self, input: Tensor<4>) -> Tensor<4> {
        let output = self.ghost.forward(input);
        let output = match &self.depthwise {
            Some(depthwise) => depthwise.forward(output),
            None => output};
        self.linear.forward(output)
    }
}
