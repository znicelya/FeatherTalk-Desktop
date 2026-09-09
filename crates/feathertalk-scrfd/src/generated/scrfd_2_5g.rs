// Generated from ONNX "data_utils/scrfd_2.5g_kps.onnx" by burn-onnx
use burn::prelude::*;
use burn::nn::PaddingConfig2d;
use burn::nn::conv::Conv2d;
use burn::nn::conv::Conv2dConfig;
use burn::nn::pool::AvgPool2d;
use burn::nn::pool::AvgPool2dConfig;
use burn::nn::pool::MaxPool2d;
use burn::nn::pool::MaxPool2dConfig;


#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    constant39: burn::module::Param<Tensor<B, 1>>,
    constant40: burn::module::Param<Tensor<B, 1>>,
    constant41: burn::module::Param<Tensor<B, 1>>,
    conv2d1: Conv2d<B>,
    conv2d2: Conv2d<B>,
    conv2d3: Conv2d<B>,
    maxpool2d1: MaxPool2d,
    conv2d4: Conv2d<B>,
    conv2d5: Conv2d<B>,
    conv2d6: Conv2d<B>,
    conv2d7: Conv2d<B>,
    conv2d8: Conv2d<B>,
    conv2d9: Conv2d<B>,
    conv2d10: Conv2d<B>,
    conv2d11: Conv2d<B>,
    averagepool2d1: AvgPool2d,
    conv2d12: Conv2d<B>,
    conv2d13: Conv2d<B>,
    conv2d14: Conv2d<B>,
    conv2d15: Conv2d<B>,
    conv2d16: Conv2d<B>,
    conv2d17: Conv2d<B>,
    conv2d18: Conv2d<B>,
    conv2d19: Conv2d<B>,
    conv2d20: Conv2d<B>,
    conv2d21: Conv2d<B>,
    conv2d22: Conv2d<B>,
    averagepool2d2: AvgPool2d,
    conv2d23: Conv2d<B>,
    conv2d24: Conv2d<B>,
    conv2d25: Conv2d<B>,
    conv2d26: Conv2d<B>,
    conv2d27: Conv2d<B>,
    conv2d28: Conv2d<B>,
    conv2d29: Conv2d<B>,
    averagepool2d3: AvgPool2d,
    conv2d30: Conv2d<B>,
    conv2d31: Conv2d<B>,
    conv2d32: Conv2d<B>,
    conv2d33: Conv2d<B>,
    conv2d34: Conv2d<B>,
    conv2d35: Conv2d<B>,
    conv2d36: Conv2d<B>,
    conv2d37: Conv2d<B>,
    conv2d38: Conv2d<B>,
    conv2d39: Conv2d<B>,
    conv2d40: Conv2d<B>,
    conv2d41: Conv2d<B>,
    conv2d42: Conv2d<B>,
    conv2d43: Conv2d<B>,
    conv2d44: Conv2d<B>,
    conv2d45: Conv2d<B>,
    conv2d46: Conv2d<B>,
    conv2d47: Conv2d<B>,
    conv2d48: Conv2d<B>,
    conv2d49: Conv2d<B>,
    conv2d50: Conv2d<B>,
    conv2d51: Conv2d<B>,
    conv2d52: Conv2d<B>,
    conv2d53: Conv2d<B>,
    conv2d54: Conv2d<B>,
    conv2d55: Conv2d<B>,
    conv2d56: Conv2d<B>,
    conv2d57: Conv2d<B>,
    phantom: core::marker::PhantomData<B>,
    #[module(skip)]
    device: B::Device,
}

impl<B: Backend> Model<B> {
    #[allow(unused_variables)]
    pub fn new(device: &B::Device) -> Self {
        let constant39: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<
                B,
                1,
            >::from_data(
                burn::tensor::TensorData::from([0.880725085735321f64]),
                (device, burn::tensor::DType::F32),
            ),
            device.clone(),
            false,
            [1].into(),
        );
        let constant40: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<
                B,
                1,
            >::from_data(
                burn::tensor::TensorData::from([0.9315465092658997f64]),
                (device, burn::tensor::DType::F32),
            ),
            device.clone(),
            false,
            [1].into(),
        );
        let constant41: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<
                B,
                1,
            >::from_data(
                burn::tensor::TensorData::from([1.1423660516738892f64]),
                (device, burn::tensor::DType::F32),
            ),
            device.clone(),
            false,
            [1].into(),
        );
        let conv2d1 = Conv2dConfig::new([3, 12], [3, 3])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d2 = Conv2dConfig::new([12, 12], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d3 = Conv2dConfig::new([12, 24], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let maxpool2d1 = MaxPool2dConfig::new([2, 2])
            .with_strides([2, 2])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_ceil_mode(false)
            .init();
        let conv2d4 = Conv2dConfig::new([24, 24], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d5 = Conv2dConfig::new([24, 24], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d6 = Conv2dConfig::new([24, 24], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d7 = Conv2dConfig::new([24, 24], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d8 = Conv2dConfig::new([24, 24], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d9 = Conv2dConfig::new([24, 24], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d10 = Conv2dConfig::new([24, 48], [3, 3])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d11 = Conv2dConfig::new([48, 48], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let averagepool2d1 = AvgPool2dConfig::new([2, 2])
            .with_strides([2, 2])
            .with_padding(PaddingConfig2d::Valid)
            .with_count_include_pad(false)
            .with_ceil_mode(true)
            .init();
        let conv2d12 = Conv2dConfig::new([24, 48], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d13 = Conv2dConfig::new([48, 48], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d14 = Conv2dConfig::new([48, 48], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d15 = Conv2dConfig::new([48, 48], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d16 = Conv2dConfig::new([48, 48], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d17 = Conv2dConfig::new([48, 48], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d18 = Conv2dConfig::new([48, 48], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d19 = Conv2dConfig::new([48, 48], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d20 = Conv2dConfig::new([48, 48], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d21 = Conv2dConfig::new([48, 48], [3, 3])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d22 = Conv2dConfig::new([48, 48], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let averagepool2d2 = AvgPool2dConfig::new([2, 2])
            .with_strides([2, 2])
            .with_padding(PaddingConfig2d::Valid)
            .with_count_include_pad(false)
            .with_ceil_mode(true)
            .init();
        let conv2d23 = Conv2dConfig::new([48, 48], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d24 = Conv2dConfig::new([48, 48], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d25 = Conv2dConfig::new([48, 48], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d26 = Conv2dConfig::new([48, 48], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d27 = Conv2dConfig::new([48, 48], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d28 = Conv2dConfig::new([48, 80], [3, 3])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d29 = Conv2dConfig::new([80, 80], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let averagepool2d3 = AvgPool2dConfig::new([2, 2])
            .with_strides([2, 2])
            .with_padding(PaddingConfig2d::Valid)
            .with_count_include_pad(false)
            .with_ceil_mode(true)
            .init();
        let conv2d30 = Conv2dConfig::new([48, 80], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d31 = Conv2dConfig::new([80, 80], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d32 = Conv2dConfig::new([80, 80], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d33 = Conv2dConfig::new([48, 24], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d34 = Conv2dConfig::new([48, 24], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d35 = Conv2dConfig::new([80, 24], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d36 = Conv2dConfig::new([24, 24], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d37 = Conv2dConfig::new([24, 24], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d38 = Conv2dConfig::new([24, 24], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d39 = Conv2dConfig::new([24, 24], [3, 3])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d40 = Conv2dConfig::new([24, 24], [3, 3])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d41 = Conv2dConfig::new([24, 24], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d42 = Conv2dConfig::new([24, 24], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d43 = Conv2dConfig::new([24, 64], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d44 = Conv2dConfig::new([64, 64], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d45 = Conv2dConfig::new([64, 2], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d46 = Conv2dConfig::new([64, 8], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d47 = Conv2dConfig::new([64, 20], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d48 = Conv2dConfig::new([24, 64], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d49 = Conv2dConfig::new([64, 64], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d50 = Conv2dConfig::new([64, 2], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d51 = Conv2dConfig::new([64, 8], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d52 = Conv2dConfig::new([64, 20], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d53 = Conv2dConfig::new([24, 64], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d54 = Conv2dConfig::new([64, 64], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d55 = Conv2dConfig::new([64, 2], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d56 = Conv2dConfig::new([64, 8], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d57 = Conv2dConfig::new([64, 20], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        Self {
            constant39,
            constant40,
            constant41,
            conv2d1,
            conv2d2,
            conv2d3,
            maxpool2d1,
            conv2d4,
            conv2d5,
            conv2d6,
            conv2d7,
            conv2d8,
            conv2d9,
            conv2d10,
            conv2d11,
            averagepool2d1,
            conv2d12,
            conv2d13,
            conv2d14,
            conv2d15,
            conv2d16,
            conv2d17,
            conv2d18,
            conv2d19,
            conv2d20,
            conv2d21,
            conv2d22,
            averagepool2d2,
            conv2d23,
            conv2d24,
            conv2d25,
            conv2d26,
            conv2d27,
            conv2d28,
            conv2d29,
            averagepool2d3,
            conv2d30,
            conv2d31,
            conv2d32,
            conv2d33,
            conv2d34,
            conv2d35,
            conv2d36,
            conv2d37,
            conv2d38,
            conv2d39,
            conv2d40,
            conv2d41,
            conv2d42,
            conv2d43,
            conv2d44,
            conv2d45,
            conv2d46,
            conv2d47,
            conv2d48,
            conv2d49,
            conv2d50,
            conv2d51,
            conv2d52,
            conv2d53,
            conv2d54,
            conv2d55,
            conv2d56,
            conv2d57,
            phantom: core::marker::PhantomData,
            device: device.clone(),
        }
    }

    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(
        &self,
        images: Tensor<B, 4>,
    ) -> (
        Tensor<B, 3>,
        Tensor<B, 3>,
        Tensor<B, 3>,
        Tensor<B, 3>,
        Tensor<B, 3>,
        Tensor<B, 3>,
        Tensor<B, 3>,
        Tensor<B, 3>,
        Tensor<B, 3>,
    ) {
        let constant39_out1 = self.constant39.val();
        let constant40_out1 = self.constant40.val();
        let constant41_out1 = self.constant41.val();
        let constant118_out1: [i64; 1] = [-1i64];
        let constant119_out1: [i64; 1] = [1i64];
        let constant120_out1: [i64; 1] = [-1i64];
        let constant121_out1: [i64; 1] = [4i64];
        let constant122_out1: [i64; 1] = [-1i64];
        let constant123_out1: [i64; 1] = [10i64];
        let constant124_out1: [i64; 1] = [-1i64];
        let constant125_out1: [i64; 1] = [1i64];
        let constant126_out1: [i64; 1] = [-1i64];
        let constant127_out1: [i64; 1] = [4i64];
        let constant128_out1: [i64; 1] = [-1i64];
        let constant129_out1: [i64; 1] = [10i64];
        let constant130_out1: [i64; 1] = [-1i64];
        let constant131_out1: [i64; 1] = [1i64];
        let constant132_out1: [i64; 1] = [-1i64];
        let constant133_out1: [i64; 1] = [4i64];
        let constant134_out1: [i64; 1] = [-1i64];
        let constant135_out1: [i64; 1] = [10i64];
        let conv2d1_out1 = self.conv2d1.forward(images);
        let relu1_out1 = burn::tensor::activation::relu(conv2d1_out1);
        let conv2d2_out1 = self.conv2d2.forward(relu1_out1);
        let relu2_out1 = burn::tensor::activation::relu(conv2d2_out1);
        let conv2d3_out1 = self.conv2d3.forward(relu2_out1);
        let relu3_out1 = burn::tensor::activation::relu(conv2d3_out1);
        let maxpool2d1_out1 = self.maxpool2d1.forward(relu3_out1);
        let conv2d4_out1 = self.conv2d4.forward(maxpool2d1_out1.clone());
        let relu4_out1 = burn::tensor::activation::relu(conv2d4_out1);
        let conv2d5_out1 = self.conv2d5.forward(relu4_out1);
        let add1_out1 = conv2d5_out1.add(maxpool2d1_out1);
        let relu5_out1 = burn::tensor::activation::relu(add1_out1);
        let conv2d6_out1 = self.conv2d6.forward(relu5_out1.clone());
        let relu6_out1 = burn::tensor::activation::relu(conv2d6_out1);
        let conv2d7_out1 = self.conv2d7.forward(relu6_out1);
        let add2_out1 = conv2d7_out1.add(relu5_out1);
        let relu7_out1 = burn::tensor::activation::relu(add2_out1);
        let conv2d8_out1 = self.conv2d8.forward(relu7_out1.clone());
        let relu8_out1 = burn::tensor::activation::relu(conv2d8_out1);
        let conv2d9_out1 = self.conv2d9.forward(relu8_out1);
        let add3_out1 = conv2d9_out1.add(relu7_out1);
        let relu9_out1 = burn::tensor::activation::relu(add3_out1);
        let conv2d10_out1 = self.conv2d10.forward(relu9_out1.clone());
        let relu10_out1 = burn::tensor::activation::relu(conv2d10_out1);
        let conv2d11_out1 = self.conv2d11.forward(relu10_out1);
        let averagepool2d1_out1 = self.averagepool2d1.forward(relu9_out1);
        let conv2d12_out1 = self.conv2d12.forward(averagepool2d1_out1);
        let add4_out1 = conv2d11_out1.add(conv2d12_out1);
        let relu11_out1 = burn::tensor::activation::relu(add4_out1);
        let conv2d13_out1 = self.conv2d13.forward(relu11_out1.clone());
        let relu12_out1 = burn::tensor::activation::relu(conv2d13_out1);
        let conv2d14_out1 = self.conv2d14.forward(relu12_out1);
        let add5_out1 = conv2d14_out1.add(relu11_out1);
        let relu13_out1 = burn::tensor::activation::relu(add5_out1);
        let conv2d15_out1 = self.conv2d15.forward(relu13_out1.clone());
        let relu14_out1 = burn::tensor::activation::relu(conv2d15_out1);
        let conv2d16_out1 = self.conv2d16.forward(relu14_out1);
        let add6_out1 = conv2d16_out1.add(relu13_out1);
        let relu15_out1 = burn::tensor::activation::relu(add6_out1);
        let conv2d17_out1 = self.conv2d17.forward(relu15_out1.clone());
        let relu16_out1 = burn::tensor::activation::relu(conv2d17_out1);
        let conv2d18_out1 = self.conv2d18.forward(relu16_out1);
        let add7_out1 = conv2d18_out1.add(relu15_out1);
        let relu17_out1 = burn::tensor::activation::relu(add7_out1);
        let conv2d19_out1 = self.conv2d19.forward(relu17_out1.clone());
        let relu18_out1 = burn::tensor::activation::relu(conv2d19_out1);
        let conv2d20_out1 = self.conv2d20.forward(relu18_out1);
        let add8_out1 = conv2d20_out1.add(relu17_out1);
        let relu19_out1 = burn::tensor::activation::relu(add8_out1);
        let conv2d21_out1 = self.conv2d21.forward(relu19_out1.clone());
        let relu20_out1 = burn::tensor::activation::relu(conv2d21_out1);
        let conv2d22_out1 = self.conv2d22.forward(relu20_out1);
        let averagepool2d2_out1 = self.averagepool2d2.forward(relu19_out1.clone());
        let conv2d23_out1 = self.conv2d23.forward(averagepool2d2_out1);
        let add9_out1 = conv2d22_out1.add(conv2d23_out1);
        let relu21_out1 = burn::tensor::activation::relu(add9_out1);
        let conv2d24_out1 = self.conv2d24.forward(relu21_out1.clone());
        let relu22_out1 = burn::tensor::activation::relu(conv2d24_out1);
        let conv2d25_out1 = self.conv2d25.forward(relu22_out1);
        let add10_out1 = conv2d25_out1.add(relu21_out1);
        let relu23_out1 = burn::tensor::activation::relu(add10_out1);
        let conv2d26_out1 = self.conv2d26.forward(relu23_out1.clone());
        let relu24_out1 = burn::tensor::activation::relu(conv2d26_out1);
        let conv2d27_out1 = self.conv2d27.forward(relu24_out1);
        let add11_out1 = conv2d27_out1.add(relu23_out1);
        let relu25_out1 = burn::tensor::activation::relu(add11_out1);
        let conv2d28_out1 = self.conv2d28.forward(relu25_out1.clone());
        let relu26_out1 = burn::tensor::activation::relu(conv2d28_out1);
        let conv2d29_out1 = self.conv2d29.forward(relu26_out1);
        let averagepool2d3_out1 = self.averagepool2d3.forward(relu25_out1.clone());
        let conv2d30_out1 = self.conv2d30.forward(averagepool2d3_out1);
        let add12_out1 = conv2d29_out1.add(conv2d30_out1);
        let relu27_out1 = burn::tensor::activation::relu(add12_out1);
        let conv2d31_out1 = self.conv2d31.forward(relu27_out1.clone());
        let relu28_out1 = burn::tensor::activation::relu(conv2d31_out1);
        let conv2d32_out1 = self.conv2d32.forward(relu28_out1);
        let add13_out1 = conv2d32_out1.add(relu27_out1);
        let relu29_out1 = burn::tensor::activation::relu(add13_out1);
        let conv2d33_out1 = self.conv2d33.forward(relu19_out1);
        let conv2d34_out1 = self.conv2d34.forward(relu25_out1);
        let conv2d35_out1 = self.conv2d35.forward(relu29_out1);
        let shape1_out1: [i64; 4] = {
            let axes = &conv2d34_out1.clone().dims()[0..4];
            let mut output = [0i64; 4];
            for i in 0..4 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let gather1_out1 = shape1_out1[2] as i64;
        let gather2_out1 = shape1_out1[3] as i64;
        let unsqueeze1_out1 = [gather1_out1 as i64];
        let unsqueeze2_out1 = [gather2_out1 as i64];
        let concat1_out1: [i64; 2usize] = [&unsqueeze1_out1[..], &unsqueeze2_out1[..]]
            .concat()
            .try_into()
            .unwrap();
        let shape3_out1: [i64; 4] = {
            let axes = &conv2d35_out1.clone().dims()[0..4];
            let mut output = [0i64; 4];
            for i in 0..4 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let slice1_out1: [i64; 2] = shape3_out1[0..2].try_into().unwrap();
        let concat2_out1: [i64; 4usize] = [&slice1_out1[..], &concat1_out1[..]]
            .concat()
            .try_into()
            .unwrap();
        let resize1_out1 = {
            let target_height = concat2_out1[2] as usize;
            let target_width = concat2_out1[3] as usize;
            burn::tensor::module::interpolate(
                conv2d35_out1.clone(),
                [target_height, target_width],
                burn::tensor::ops::InterpolateOptions::new(
                        burn::tensor::ops::InterpolateMode::Nearest,
                    )
                    .with_align_corners(false),
            )
        };
        let add14_out1 = conv2d34_out1.add(resize1_out1);
        let shape4_out1: [i64; 4] = {
            let axes = &conv2d33_out1.clone().dims()[0..4];
            let mut output = [0i64; 4];
            for i in 0..4 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let gather3_out1 = shape4_out1[2] as i64;
        let gather4_out1 = shape4_out1[3] as i64;
        let unsqueeze3_out1 = [gather3_out1 as i64];
        let unsqueeze4_out1 = [gather4_out1 as i64];
        let concat3_out1: [i64; 2usize] = [&unsqueeze3_out1[..], &unsqueeze4_out1[..]]
            .concat()
            .try_into()
            .unwrap();
        let shape6_out1: [i64; 4] = {
            let axes = &add14_out1.clone().dims()[0..4];
            let mut output = [0i64; 4];
            for i in 0..4 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let slice2_out1: [i64; 2] = shape6_out1[0..2].try_into().unwrap();
        let concat4_out1: [i64; 4usize] = [&slice2_out1[..], &concat3_out1[..]]
            .concat()
            .try_into()
            .unwrap();
        let resize2_out1 = {
            let target_height = concat4_out1[2] as usize;
            let target_width = concat4_out1[3] as usize;
            burn::tensor::module::interpolate(
                add14_out1.clone(),
                [target_height, target_width],
                burn::tensor::ops::InterpolateOptions::new(
                        burn::tensor::ops::InterpolateMode::Nearest,
                    )
                    .with_align_corners(false),
            )
        };
        let add15_out1 = conv2d33_out1.add(resize2_out1);
        let conv2d36_out1 = self.conv2d36.forward(add15_out1);
        let conv2d37_out1 = self.conv2d37.forward(add14_out1);
        let conv2d38_out1 = self.conv2d38.forward(conv2d35_out1);
        let conv2d39_out1 = self.conv2d39.forward(conv2d36_out1.clone());
        let add16_out1 = conv2d37_out1.add(conv2d39_out1);
        let conv2d40_out1 = self.conv2d40.forward(add16_out1.clone());
        let add17_out1 = conv2d38_out1.add(conv2d40_out1);
        let conv2d41_out1 = self.conv2d41.forward(add16_out1);
        let conv2d42_out1 = self.conv2d42.forward(add17_out1);
        let conv2d43_out1 = self.conv2d43.forward(conv2d36_out1);
        let relu30_out1 = burn::tensor::activation::relu(conv2d43_out1);
        let conv2d44_out1 = self.conv2d44.forward(relu30_out1);
        let relu31_out1 = burn::tensor::activation::relu(conv2d44_out1);
        let conv2d45_out1 = self.conv2d45.forward(relu31_out1.clone());
        let conv2d46_out1 = self.conv2d46.forward(relu31_out1.clone());
        let mul1_out1 = conv2d46_out1
            .mul((constant39_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let conv2d47_out1 = self.conv2d47.forward(relu31_out1);
        let shape7_out1: [i64; 4] = {
            let axes = &conv2d45_out1.clone().dims()[0..4];
            let mut output = [0i64; 4];
            for i in 0..4 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let gather5_out1 = shape7_out1[0] as i64;
        let transpose1_out1 = conv2d45_out1.permute([0, 2, 3, 1]);
        let unsqueeze5_out1 = [gather5_out1 as i64];
        let concat5_out1: [i64; 3usize] = [
            &unsqueeze5_out1[..],
            &constant118_out1[..],
            &constant119_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let reshape1_out1 = transpose1_out1.reshape(concat5_out1);
        let sigmoid1_out1 = burn::tensor::activation::sigmoid(reshape1_out1);
        let transpose2_out1 = mul1_out1.permute([0, 2, 3, 1]);
        let concat6_out1: [i64; 3usize] = [
            &unsqueeze5_out1[..],
            &constant120_out1[..],
            &constant121_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let reshape2_out1 = transpose2_out1.reshape(concat6_out1);
        let transpose3_out1 = conv2d47_out1.permute([0, 2, 3, 1]);
        let concat7_out1: [i64; 3usize] = [
            &unsqueeze5_out1[..],
            &constant122_out1[..],
            &constant123_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let reshape3_out1 = transpose3_out1.reshape(concat7_out1);
        let conv2d48_out1 = self.conv2d48.forward(conv2d41_out1);
        let relu32_out1 = burn::tensor::activation::relu(conv2d48_out1);
        let conv2d49_out1 = self.conv2d49.forward(relu32_out1);
        let relu33_out1 = burn::tensor::activation::relu(conv2d49_out1);
        let conv2d50_out1 = self.conv2d50.forward(relu33_out1.clone());
        let conv2d51_out1 = self.conv2d51.forward(relu33_out1.clone());
        let mul2_out1 = conv2d51_out1
            .mul((constant40_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let conv2d52_out1 = self.conv2d52.forward(relu33_out1);
        let shape8_out1: [i64; 4] = {
            let axes = &conv2d50_out1.clone().dims()[0..4];
            let mut output = [0i64; 4];
            for i in 0..4 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let gather6_out1 = shape8_out1[0] as i64;
        let transpose4_out1 = conv2d50_out1.permute([0, 2, 3, 1]);
        let unsqueeze8_out1 = [gather6_out1 as i64];
        let concat8_out1: [i64; 3usize] = [
            &unsqueeze8_out1[..],
            &constant124_out1[..],
            &constant125_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let reshape4_out1 = transpose4_out1.reshape(concat8_out1);
        let sigmoid2_out1 = burn::tensor::activation::sigmoid(reshape4_out1);
        let transpose5_out1 = mul2_out1.permute([0, 2, 3, 1]);
        let concat9_out1: [i64; 3usize] = [
            &unsqueeze8_out1[..],
            &constant126_out1[..],
            &constant127_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let reshape5_out1 = transpose5_out1.reshape(concat9_out1);
        let transpose6_out1 = conv2d52_out1.permute([0, 2, 3, 1]);
        let concat10_out1: [i64; 3usize] = [
            &unsqueeze8_out1[..],
            &constant128_out1[..],
            &constant129_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let reshape6_out1 = transpose6_out1.reshape(concat10_out1);
        let conv2d53_out1 = self.conv2d53.forward(conv2d42_out1);
        let relu34_out1 = burn::tensor::activation::relu(conv2d53_out1);
        let conv2d54_out1 = self.conv2d54.forward(relu34_out1);
        let relu35_out1 = burn::tensor::activation::relu(conv2d54_out1);
        let conv2d55_out1 = self.conv2d55.forward(relu35_out1.clone());
        let conv2d56_out1 = self.conv2d56.forward(relu35_out1.clone());
        let mul3_out1 = conv2d56_out1
            .mul((constant41_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let conv2d57_out1 = self.conv2d57.forward(relu35_out1);
        let shape9_out1: [i64; 4] = {
            let axes = &conv2d55_out1.clone().dims()[0..4];
            let mut output = [0i64; 4];
            for i in 0..4 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let gather7_out1 = shape9_out1[0] as i64;
        let transpose7_out1 = conv2d55_out1.permute([0, 2, 3, 1]);
        let unsqueeze11_out1 = [gather7_out1 as i64];
        let concat11_out1: [i64; 3usize] = [
            &unsqueeze11_out1[..],
            &constant130_out1[..],
            &constant131_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let reshape7_out1 = transpose7_out1.reshape(concat11_out1);
        let sigmoid3_out1 = burn::tensor::activation::sigmoid(reshape7_out1);
        let transpose8_out1 = mul3_out1.permute([0, 2, 3, 1]);
        let concat12_out1: [i64; 3usize] = [
            &unsqueeze11_out1[..],
            &constant132_out1[..],
            &constant133_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let reshape8_out1 = transpose8_out1.reshape(concat12_out1);
        let transpose9_out1 = conv2d57_out1.permute([0, 2, 3, 1]);
        let concat13_out1: [i64; 3usize] = [
            &unsqueeze11_out1[..],
            &constant134_out1[..],
            &constant135_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let reshape9_out1 = transpose9_out1.reshape(concat13_out1);
        (
            sigmoid1_out1,
            sigmoid2_out1,
            sigmoid3_out1,
            reshape2_out1,
            reshape5_out1,
            reshape8_out1,
            reshape3_out1,
            reshape6_out1,
            reshape9_out1,
        )
    }
}
