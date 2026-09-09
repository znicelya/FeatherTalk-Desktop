use super::{
    AudioConvHubert,
    blocks::{DoubleConvDw, Down, InConvDw, OutConv, Up},
    config::{AudioConvHubertConfig, DownConfig},
};
use burn::tensor::{Tensor, backend::Backend};

#[derive(burn::module::Module, Debug)]
pub struct OriginalUnet<B: Backend> {
    pub audio_model: AudioConvHubert<B>,
    pub fuse_first: DoubleConvDw<B>,
    pub fuse_second: DoubleConvDw<B>,
    pub inc: InConvDw<B>,
    pub down1: Down<B>,
    pub down2: Down<B>,
    pub down3: Down<B>,
    pub down4: Down<B>,
    pub up1: Up<B>,
    pub up2: Up<B>,
    pub up3: Up<B>,
    pub up4: Up<B>,
    pub outc: OutConv<B>,
}

impl<B: Backend> OriginalUnet<B> {
    pub(crate) fn new(channels: [usize; 5], device: &B::Device) -> Self {
        Self {
            audio_model: AudioConvHubertConfig::new(channels).init(device),
            fuse_first: DoubleConvDw::new(channels[4] * 2, channels[4], 1, device),
            fuse_second: DoubleConvDw::new(channels[4], channels[3], 1, device),
            inc: InConvDw::new(6, channels[0], device),
            down1: DownConfig::new(channels[0], channels[1]).init(device),
            down2: DownConfig::new(channels[1], channels[2]).init(device),
            down3: DownConfig::new(channels[2], channels[3]).init(device),
            down4: DownConfig::new(channels[3], channels[4]).init(device),
            up1: Up::new(channels[4], channels[3] / 2, device),
            up2: Up::new(channels[3] / 2 + channels[2], channels[2] / 2, device),
            up3: Up::new(channels[2] / 2 + channels[1], channels[1] / 2, device),
            up4: Up::new(channels[1] / 2 + channels[0], channels[0], device),
            outc: OutConv::new(channels[0], device),
        }
    }

    pub fn forward(&self, image: Tensor<B, 4>, audio: Tensor<B, 4>) -> Tensor<B, 4> {
        let x1 = self.inc.forward(image);
        let x2 = self.down1.forward(x1.clone());
        let x3 = self.down2.forward(x2.clone());
        let x4 = self.down3.forward(x3.clone());
        let x5 = self.down4.forward(x4.clone());

        let audio = self.audio_model.forward(audio);
        let x5 = Tensor::cat(vec![x5, audio], 1);
        let x5 = self.fuse_second.forward(self.fuse_first.forward(x5));

        let output = self.up1.forward(x5, x4);
        let output = self.up2.forward(output, x3);
        let output = self.up3.forward(output, x2);
        let output = self.up4.forward(output, x1);
        burn::tensor::activation::sigmoid(self.outc.forward(output))
    }
}
