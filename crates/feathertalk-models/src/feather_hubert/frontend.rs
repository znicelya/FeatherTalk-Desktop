use burn::nn::{
    Gelu, GroupNorm, GroupNormConfig, PaddingConfig1d,
    conv::{Conv1d, Conv1dConfig},
};
use burn::tensor::{Tensor, backend::Backend};

const KERNELS: [usize; 7] = [10, 3, 3, 3, 3, 2, 2];
const STRIDES: [usize; 7] = [5, 2, 2, 2, 2, 2, 2];

#[derive(burn::module::Module, Debug)]
pub struct ConvNormAct1d<B: Backend> {
    pub conv: Conv1d<B>,
    pub norm: GroupNorm<B>,
    pub act: Gelu,
}

impl<B: Backend> ConvNormAct1d<B> {
    fn new(
        channels_in: usize,
        channels_out: usize,
        kernel_size: usize,
        stride: usize,
        device: &B::Device,
    ) -> Self {
        let conv = Conv1dConfig::new(channels_in, channels_out, kernel_size)
            .with_stride(stride)
            .with_padding(PaddingConfig1d::Valid)
            .with_bias(false)
            .init(device);
        let norm = GroupNormConfig::new(pick_group_count(channels_out), channels_out).init(device);
        Self {
            conv,
            norm,
            act: Gelu::new(),
        }
    }

    fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        self.act
            .forward(self.norm.forward(self.conv.forward(input)))
    }
}

#[derive(burn::module::Module, Debug)]
pub struct HubertStrideFrontend<B: Backend> {
    pub layers: Vec<ConvNormAct1d<B>>,
}

impl<B: Backend> HubertStrideFrontend<B> {
    pub(crate) fn new(channels: [usize; 7], device: &B::Device) -> Self {
        let mut layers = Vec::with_capacity(channels.len());
        let mut channels_in = 1;
        for ((channels_out, kernel_size), stride) in channels.into_iter().zip(KERNELS).zip(STRIDES)
        {
            layers.push(ConvNormAct1d::new(
                channels_in,
                channels_out,
                kernel_size,
                stride,
                device,
            ));
            channels_in = channels_out;
        }
        Self { layers }
    }

    pub(crate) fn forward(&self, waveform: Tensor<B, 2>) -> Tensor<B, 3> {
        let [batch, samples] = waveform.dims();
        let mut output = waveform.reshape([batch, 1, samples]);
        for layer in &self.layers {
            output = layer.forward(output);
        }
        output
    }
}

pub(crate) fn pick_group_count(channels: usize) -> usize {
    for groups in [32, 16, 8, 4, 2] {
        if channels.is_multiple_of(groups) {
            return groups;
        }
    }
    1
}
