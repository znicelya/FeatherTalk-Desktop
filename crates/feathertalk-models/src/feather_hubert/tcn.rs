use burn::nn::{
    Dropout, DropoutConfig, Gelu, GroupNorm, GroupNormConfig, PaddingConfig1d,
    conv::{Conv1d, Conv1dConfig},
};
use burn::tensor::{Tensor, backend::Backend};

use super::frontend::pick_group_count;

#[derive(burn::module::Module, Debug)]
pub struct DepthwiseTcnBlock<B: Backend> {
    pub norm: GroupNorm<B>,
    pub pw_expand: Conv1d<B>,
    pub dw_conv: Conv1d<B>,
    pub act: Gelu,
    pub pw_project: Conv1d<B>,
    pub dropout: Dropout,
}

impl<B: Backend> DepthwiseTcnBlock<B> {
    pub(crate) fn new(
        channels: usize,
        expansion: usize,
        dilation: usize,
        dropout: f64,
        device: &B::Device,
    ) -> Self {
        let hidden_channels = channels * expansion;
        let norm = GroupNormConfig::new(pick_group_count(channels), channels).init(device);
        let pw_expand = Conv1dConfig::new(channels, hidden_channels, 1)
            .with_bias(false)
            .init(device);
        let dw_conv = Conv1dConfig::new(hidden_channels, hidden_channels, 5)
            .with_dilation(dilation)
            .with_groups(hidden_channels)
            .with_padding(PaddingConfig1d::Explicit(2 * dilation, 2 * dilation))
            .with_bias(false)
            .init(device);
        let pw_project = Conv1dConfig::new(hidden_channels, channels, 1)
            .with_bias(false)
            .init(device);

        Self {
            norm,
            pw_expand,
            dw_conv,
            act: Gelu::new(),
            pw_project,
            dropout: DropoutConfig::new(dropout).init(),
        }
    }

    pub(crate) fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let residual = input.clone();
        let output = self.norm.forward(input);
        let output = self.pw_expand.forward(output);
        let output = self.act.forward(self.dw_conv.forward(output));
        let output = self.pw_project.forward(output);
        residual + self.dropout.forward(output)
    }
}
