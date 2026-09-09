use burn::{
    nn::{PaddingConfig2d, conv::Conv2d, conv::Conv2dConfig},
    tensor::{Tensor, activation::relu, backend::Backend, module::max_pool2d},
};

#[derive(burn::module::Module, Debug)]
pub struct Vgg19Conv3_3<B: Backend> {
    pub conv1_1: Conv2d<B>,
    pub conv1_2: Conv2d<B>,
    pub conv2_1: Conv2d<B>,
    pub conv2_2: Conv2d<B>,
    pub conv3_1: Conv2d<B>,
    pub conv3_2: Conv2d<B>,
    pub conv3_3: Conv2d<B>,
}

impl<B: Backend> Vgg19Conv3_3<B> {
    pub fn new_for_import(device: &B::Device) -> Self {
        Self {
            conv1_1: conv([3, 64], device),
            conv1_2: conv([64, 64], device),
            conv2_1: conv([64, 128], device),
            conv2_2: conv([128, 128], device),
            conv3_1: conv([128, 256], device),
            conv3_2: conv([256, 256], device),
            conv3_3: conv([256, 256], device),
        }
    }

    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let [batch, channels, height, width] = input.dims();
        assert!(batch > 0, "VGG19 input batch must be non-zero");
        assert_eq!(channels, 3, "VGG19 input must have exactly 3 channels");
        assert!(
            height >= 4 && width >= 4,
            "VGG19 input spatial dimensions must both be at least 4"
        );

        let output = relu(self.conv1_1.forward(input));
        let output = relu(self.conv1_2.forward(output));
        let output = max_pool2d(output, [2, 2], [2, 2], [0, 0], [1, 1], false);
        let output = relu(self.conv2_1.forward(output));
        let output = relu(self.conv2_2.forward(output));
        let output = max_pool2d(output, [2, 2], [2, 2], [0, 0], [1, 1], false);
        let output = relu(self.conv3_1.forward(output));
        let output = relu(self.conv3_2.forward(output));
        self.conv3_3.forward(output)
    }
}

fn conv<B: Backend>(channels: [usize; 2], device: &B::Device) -> Conv2d<B> {
    Conv2dConfig::new(channels, [3, 3])
        .with_stride([1, 1])
        .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
        .with_dilation([1, 1])
        .with_groups(1)
        .with_bias(true)
        .init(device)
}
