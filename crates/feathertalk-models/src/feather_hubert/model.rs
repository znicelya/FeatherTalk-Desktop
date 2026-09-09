use burn::nn::{
    Gelu, GroupNorm, GroupNormConfig, PaddingConfig1d,
    conv::{Conv1d, Conv1dConfig},
};
use burn::tensor::{Tensor, backend::Backend, ops::PadMode};

use super::{
    config::{FeatherHubertConfig, expected_hubert_frames},
    frontend::HubertStrideFrontend,
    frontend::pick_group_count,
    tcn::DepthwiseTcnBlock,
};

#[derive(burn::module::Module, Debug)]
pub struct FeatherHubertEncoder<B: Backend> {
    pub frontend: HubertStrideFrontend<B>,
    pub encoder: Vec<DepthwiseTcnBlock<B>>,
    pub final_norm: GroupNorm<B>,
    pub proj: Conv1d<B>,
    #[module(skip)]
    pub config: FeatherHubertConfig,
}

impl<B: Backend> FeatherHubertEncoder<B> {
    pub(crate) fn new(config: FeatherHubertConfig, device: &B::Device) -> Self {
        let frontend = HubertStrideFrontend::new(
            [
                64,
                128,
                256,
                384,
                config.channels,
                config.channels,
                config.channels,
            ],
            device,
        );
        let encoder = (0..config.num_blocks)
            .map(|index| {
                DepthwiseTcnBlock::new(
                    config.channels,
                    config.expansion,
                    [1, 2, 4, 8][index % 4],
                    config.dropout,
                    device,
                )
            })
            .collect();
        let final_norm =
            GroupNormConfig::new(pick_group_count(config.channels), config.channels).init(device);
        let proj = Conv1dConfig::new(config.channels, config.output_dim, 1)
            .with_padding(PaddingConfig1d::Valid)
            .init(device);

        Self {
            frontend,
            encoder,
            final_norm,
            proj,
            config,
        }
    }

    pub fn forward(&self, waveform: Tensor<B, 2>) -> Tensor<B, 3> {
        let expected_frames = expected_hubert_frames(waveform.dims()[1]);
        assert!(
            expected_frames > 0,
            "Waveform is too short for HuBERT-compatible output: {} samples",
            waveform.dims()[1]
        );

        let mut output = self.frontend.forward(waveform);
        for block in &self.encoder {
            output = block.forward(output);
        }
        output = self
            .proj
            .forward(Gelu::new().forward(self.final_norm.forward(output)));
        output = output.transpose();

        let [batch, actual_frames, output_dim] = output.dims();
        if actual_frames < expected_frames {
            output = output.pad(
                [(0, 0), (0, expected_frames - actual_frames), (0, 0)],
                PadMode::Constant(0.0),
            );
        } else if actual_frames > expected_frames {
            output = output.slice([0..batch, 0..expected_frames, 0..output_dim]);
        }
        output
    }
}
