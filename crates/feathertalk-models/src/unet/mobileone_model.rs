use burn::tensor::{Tensor, backend::Backend};

use super::{
    blocks::OutConv,
    mobileone_blocks::{
        MobileOneAudioConvHubert, MobileOneDoubleConv, MobileOneDown, MobileOneSeparableBlock,
        MobileOneUp, ReparameterizedMobileOneAudioConvHubert, ReparameterizedMobileOneDoubleConv,
        ReparameterizedMobileOneDown, ReparameterizedMobileOneSeparableBlock,
        ReparameterizedMobileOneUp, detached_conv2d,
    },
};

#[derive(burn::module::Module, Debug)]
pub struct MobileOneUnet<B: Backend> {
    pub audio_model: MobileOneAudioConvHubert<B>,
    pub fuse_first: MobileOneDoubleConv<B>,
    pub fuse_second: MobileOneDoubleConv<B>,
    pub inc: MobileOneSeparableBlock<B>,
    pub down1: MobileOneDown<B>,
    pub down2: MobileOneDown<B>,
    pub down3: MobileOneDown<B>,
    pub down4: MobileOneDown<B>,
    pub up1: MobileOneUp<B>,
    pub up2: MobileOneUp<B>,
    pub up3: MobileOneUp<B>,
    pub up4: MobileOneUp<B>,
    pub outc: OutConv<B>,
}

impl<B: Backend> MobileOneUnet<B> {
    pub(crate) fn new(channels: [usize; 5], num_conv_branches: usize, device: &B::Device) -> Self {
        assert!(num_conv_branches > 0);
        assert!(channels.into_iter().all(|channel| channel > 0));
        assert!(channels[1].is_multiple_of(2));
        assert!(channels[2].is_multiple_of(2));
        assert!(channels[3].is_multiple_of(2));
        Self {
            audio_model: MobileOneAudioConvHubert::new(channels, num_conv_branches, device),
            fuse_first: MobileOneDoubleConv::new(
                channels[4] * 2,
                channels[4],
                1,
                num_conv_branches,
                device,
            ),
            fuse_second: MobileOneDoubleConv::new(
                channels[4],
                channels[3],
                1,
                num_conv_branches,
                device,
            ),
            inc: MobileOneSeparableBlock::new(6, channels[0], 1, num_conv_branches, false, device),
            down1: MobileOneDown::new(channels[0], channels[1], num_conv_branches, device),
            down2: MobileOneDown::new(channels[1], channels[2], num_conv_branches, device),
            down3: MobileOneDown::new(channels[2], channels[3], num_conv_branches, device),
            down4: MobileOneDown::new(channels[3], channels[4], num_conv_branches, device),
            up1: MobileOneUp::new(channels[4], channels[3] / 2, num_conv_branches, device),
            up2: MobileOneUp::new(channels[3], channels[2] / 2, num_conv_branches, device),
            up3: MobileOneUp::new(channels[2], channels[1] / 2, num_conv_branches, device),
            up4: MobileOneUp::new(channels[1], channels[0], num_conv_branches, device),
            outc: OutConv::new(channels[0], device),
        }
    }

    pub fn forward(&self, image: Tensor<B, 4>, audio: Tensor<B, 4>) -> Tensor<B, 4> {
        validate_inputs(&image, &audio);
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

    pub fn reparameterize(&self) -> MobileOneUnetInference<B> {
        MobileOneUnetInference {
            audio_model: self.audio_model.reparameterize(),
            fuse_first: self.fuse_first.reparameterize(),
            fuse_second: self.fuse_second.reparameterize(),
            inc: self.inc.reparameterize(),
            down1: self.down1.reparameterize(),
            down2: self.down2.reparameterize(),
            down3: self.down3.reparameterize(),
            down4: self.down4.reparameterize(),
            up1: self.up1.reparameterize(),
            up2: self.up2.reparameterize(),
            up3: self.up3.reparameterize(),
            up4: self.up4.reparameterize(),
            outc: OutConv {
                conv: detached_conv2d(&self.outc.conv),
            },
        }
    }
}

#[derive(burn::module::Module, Debug)]
pub struct MobileOneUnetInference<B: Backend> {
    pub audio_model: ReparameterizedMobileOneAudioConvHubert<B>,
    pub fuse_first: ReparameterizedMobileOneDoubleConv<B>,
    pub fuse_second: ReparameterizedMobileOneDoubleConv<B>,
    pub inc: ReparameterizedMobileOneSeparableBlock<B>,
    pub down1: ReparameterizedMobileOneDown<B>,
    pub down2: ReparameterizedMobileOneDown<B>,
    pub down3: ReparameterizedMobileOneDown<B>,
    pub down4: ReparameterizedMobileOneDown<B>,
    pub up1: ReparameterizedMobileOneUp<B>,
    pub up2: ReparameterizedMobileOneUp<B>,
    pub up3: ReparameterizedMobileOneUp<B>,
    pub up4: ReparameterizedMobileOneUp<B>,
    pub outc: OutConv<B>,
}

impl<B: Backend> MobileOneUnetInference<B> {
    pub fn forward(&self, image: Tensor<B, 4>, audio: Tensor<B, 4>) -> Tensor<B, 4> {
        validate_inputs(&image, &audio);
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

fn validate_inputs<B: Backend>(image: &Tensor<B, 4>, audio: &Tensor<B, 4>) {
    let [image_batch, image_channels, image_height, image_width] = image.dims();
    let [audio_batch, audio_channels, audio_height, audio_width] = audio.dims();
    assert_eq!(
        [image_channels, image_height, image_width],
        [6, 160, 160],
        "MobileOne UNet image input must be [B,6,160,160]"
    );
    assert_eq!(
        [audio_channels, audio_height, audio_width],
        [16, 32, 32],
        "MobileOne UNet audio input must be [B,16,32,32]"
    );
    assert_eq!(
        image_batch, audio_batch,
        "MobileOne UNet image and audio batch sizes must match"
    );
}
