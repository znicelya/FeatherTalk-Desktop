use burn::nn::{
    Gelu, GroupNorm, GroupNormConfig, PaddingConfig1d,
    conv::{Conv1d, Conv1dConfig},
};
use burn::tensor::Tensor;
use burn::tensor::Device;

const KERNELS: [usize; 7] = [10, 3, 3, 3, 3, 2, 2];
const STRIDES: [usize; 7] = [5, 2, 2, 2, 2, 2, 2];

#[derive(burn::module::Module, Debug)]
pub struct ConvNormAct1d {
    pub conv: Conv1d,
    pub norm: GroupNorm,
    pub act: Gelu}

impl ConvNormAct1d {
    fn new(
        channels_in: usize,
        channels_out: usize,
        kernel_size: usize,
        stride: usize,
        device: &Device,
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
            act: Gelu::new()}
    }

    fn forward(&self, input: Tensor<3>) -> Tensor<3> {
        self.act
            .forward(self.norm.forward(self.conv.forward(input)))
    }
}

#[derive(burn::module::Module, Debug)]
pub struct HubertStrideFrontend {
    pub layers: Vec<ConvNormAct1d>}

impl HubertStrideFrontend {
    pub(crate) fn new(channels: [usize; 7], device: &Device) -> Self {
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

    pub(crate) fn forward(&self, waveform: Tensor<2>) -> Tensor<3> {
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
