use super::config::{InvertedResidualConfig, batch_norm, conv2d};
use burn::nn::{
    BatchNorm, Relu,
    conv::{Conv2d, Conv2dConfig},
    interpolate::{Interpolate2d, Interpolate2dConfig, InterpolateMode},
};
use burn::tensor::Device;
use burn::tensor::{Tensor, TensorData, ops::PadMode};

#[derive(burn::module::Module, Debug)]
pub struct InvertedResidual {
    pub expand_conv: Conv2d,
    pub expand_bn: BatchNorm,
    pub depthwise_conv: Conv2d,
    pub depthwise_bn: BatchNorm,
    pub project_conv: Conv2d,
    pub project_bn: BatchNorm,
    #[module(skip)]
    pub use_residual: bool,
}

impl InvertedResidual {
    pub(crate) fn new(config: &InvertedResidualConfig, device: &Device) -> Self {
        assert!(matches!(config.stride, 1 | 2));
        let hidden = config.inp * config.expansion;
        Self {
            expand_conv: conv2d(
                [config.inp, hidden],
                [1, 1],
                [1, 1],
                burn::nn::PaddingConfig2d::Valid,
                false,
                device,
            ),
            expand_bn: batch_norm(hidden, device),
            depthwise_conv: Conv2dConfig::new([hidden, hidden], [3, 3])
                .with_stride([config.stride, config.stride])
                .with_padding(burn::nn::PaddingConfig2d::Explicit(1, 1, 1, 1))
                .with_groups(hidden)
                .with_bias(false)
                .init(device),
            depthwise_bn: batch_norm(hidden, device),
            project_conv: conv2d(
                [hidden, config.oup],
                [1, 1],
                [1, 1],
                burn::nn::PaddingConfig2d::Valid,
                false,
                device,
            ),
            project_bn: batch_norm(config.oup, device),
            use_residual: config.stride == 1 && config.inp == config.oup,
        }
    }

    pub fn forward(&self, input: Tensor<4>) -> Tensor<4> {
        let relu = Relu;
        let output = relu.forward(
            self.expand_bn
                .forward(self.expand_conv.forward(input.clone())),
        );
        let output = relu.forward(
            self.depthwise_bn
                .forward(self.depthwise_conv.forward(output)),
        );
        let output = self.project_bn.forward(self.project_conv.forward(output));
        if self.use_residual {
            input + output
        } else {
            output
        }
    }
}

#[derive(burn::module::Module, Debug)]
pub struct DoubleConvDw {
    pub first: InvertedResidual,
    pub second: InvertedResidual,
}

impl DoubleConvDw {
    pub(crate) fn new(inp: usize, oup: usize, stride: usize, device: &Device) -> Self {
        Self {
            first: InvertedResidualConfig::new(inp, oup)
                .with_expansion(2)
                .with_stride(stride)
                .init(device),
            second: InvertedResidualConfig::new(oup, oup)
                .with_expansion(2)
                .with_stride(1)
                .init(device),
        }
    }

    pub fn forward(&self, input: Tensor<4>) -> Tensor<4> {
        self.second.forward(self.first.forward(input))
    }
}

#[derive(burn::module::Module, Debug)]
pub struct InConvDw {
    pub inconv: InvertedResidual,
}

impl InConvDw {
    pub(crate) fn new(inp: usize, oup: usize, device: &Device) -> Self {
        Self {
            inconv: InvertedResidualConfig::new(inp, oup)
                .with_expansion(2)
                .init(device),
        }
    }

    pub fn forward(&self, input: Tensor<4>) -> Tensor<4> {
        self.inconv.forward(input)
    }
}

#[derive(burn::module::Module, Debug)]
pub struct Down {
    pub maxpool_conv: DoubleConvDw,
}

impl Down {
    pub(crate) fn new(inp: usize, oup: usize, device: &Device) -> Self {
        Self {
            maxpool_conv: DoubleConvDw::new(inp, oup, 2, device),
        }
    }

    pub fn forward(&self, input: Tensor<4>) -> Tensor<4> {
        self.maxpool_conv.forward(input)
    }
}

#[derive(burn::module::Module, Debug)]
pub struct Up {
    pub up: Interpolate2d,
    pub conv: DoubleConvDw,
}

impl Up {
    pub(crate) fn new(inp: usize, oup: usize, device: &Device) -> Self {
        let up = Interpolate2dConfig::new()
            .with_mode(InterpolateMode::Linear)
            .with_scale_factor(Some([2.0, 2.0]))
            .with_align_corners(true)
            .init();
        Self {
            up,
            conv: DoubleConvDw::new(inp, oup, 1, device),
        }
    }

    pub fn forward(&self, input: Tensor<4>, skip: Tensor<4>) -> Tensor<4> {
        self.conv
            .forward(upsample_and_concat(&self.up, input, skip))
    }
}

pub(crate) fn upsample_and_concat(
    up: &Interpolate2d,
    input: Tensor<4>,
    skip: Tensor<4>,
) -> Tensor<4> {
    // Native cuDNN-style bilinear interpolate on every device. The original
    // FeatherTalk code took a matmul-based resampling path under autodiff to
    // dodge a slow scatter-add interpolate backward on CUDA; on this
    // burn/cubecl build the native interpolate (fwd+bwd) benchmarks 6-10x
    // faster per up-stage, so we always use it.
    //
    // let input = if input.device().is_autodiff() {
    //     bilinear_upsample_2x_align_corners(input)
    // } else {
    //     up.forward(input)
    // };
    let input = up.forward(input);
    let [_, _, input_h, input_w] = input.dims();
    let [_, _, skip_h, skip_w] = skip.dims();
    assert!(
        skip_h >= input_h,
        "skip height is smaller than upsampled input"
    );
    assert!(
        skip_w >= input_w,
        "skip width is smaller than upsampled input"
    );
    let diff_h = skip_h - input_h;
    let diff_w = skip_w - input_w;
    let input = input.pad(
        [
            (0, 0),
            (0, 0),
            (diff_h / 2, diff_h - diff_h / 2),
            (diff_w / 2, diff_w - diff_w / 2),
        ],
        PadMode::Constant(0.0),
    );
    Tensor::cat(vec![input, skip], 1)
}

fn bilinear_upsample_2x_align_corners(input: Tensor<4>) -> Tensor<4> {
    let [_, _, height, width] = input.dims();
    let input = interpolate_axis_align_corners(input, 2, height, height * 2);
    interpolate_axis_align_corners(input, 3, width, width * 2)
}

fn interpolate_axis_align_corners(
    input: Tensor<4>,
    axis: usize,
    input_size: usize,
    output_size: usize,
) -> Tensor<4> {
    let device = input.device();
    let scale = (input_size - 1) as f64 / (output_size - 1) as f64;
    // A dense [output_size, input_size] resampling matrix carries the same
    // align-corners bilinear weights the gather form used, but its backward is a
    // matmul rather than a scatter-add. Scatter-add on CUDA serialises through
    // f32 atomics and made backward ~3.5x slower than the wgpu/Vulkan backend;
    // the GEMM form runs the fast path on every backend.
    let mut matrix = vec![0.0f32; output_size * input_size];
    for output_index in 0..output_size {
        let source = output_index as f64 * scale;
        let lower_index = source.floor() as usize;
        let upper_index = (lower_index + 1).min(input_size - 1);
        let weight = (source - lower_index as f64) as f32;
        let row = output_index * input_size;
        matrix[row + lower_index] += 1.0 - weight;
        matrix[row + upper_index] += weight;
    }

    // resample[o, i] contracts against the interpolated axis. For height this is
    // resample @ input; for width it is input @ resample^T. matmul broadcasts the
    // leading batch/channel dimensions, so the output keeps the target rank.
    let resample =
        Tensor::<2>::from_data(TensorData::new(matrix, [output_size, input_size]), &device);
    match axis {
        2 => resample
            .reshape([1, 1, output_size, input_size])
            .matmul(input),
        3 => input.matmul(
            resample
                .transpose()
                .reshape([1, 1, input_size, output_size]),
        ),
        _ => unreachable!("2D interpolation axis must be height or width"),
    }
}

#[derive(burn::module::Module, Debug)]
pub struct OutConv {
    pub conv: Conv2d,
}

impl OutConv {
    pub(crate) fn new(inp: usize, device: &Device) -> Self {
        Self {
            conv: conv2d(
                [inp, 3],
                [1, 1],
                [1, 1],
                burn::nn::PaddingConfig2d::Valid,
                true,
                device,
            ),
        }
    }

    pub fn forward(&self, input: Tensor<4>) -> Tensor<4> {
        self.conv.forward(input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::TensorData;

    fn assert_close(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len());
        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() < 1e-5,
                "mismatch at {index}: {actual} vs {expected}"
            );
        }
    }

    #[test]
    fn bilinear_2x_align_corners_matches_separable_interpolation() {
        let device = Default::default();
        let input =
            Tensor::<4>::from_data(TensorData::from([[[[1.0_f32, -2.0], [0.5, 4.0]]]]), &device);
        let actual = bilinear_upsample_2x_align_corners(input)
            .into_data()
            .try_to_vec::<f32>()
            .unwrap();
        assert_close(
            &actual,
            &[
                1.0,
                0.0,
                -1.0,
                -2.0,
                5.0 / 6.0,
                5.0 / 9.0,
                5.0 / 18.0,
                0.0,
                2.0 / 3.0,
                10.0 / 9.0,
                14.0 / 9.0,
                2.0,
                0.5,
                5.0 / 3.0,
                8.5 / 3.0,
                4.0,
            ],
        );
    }

    #[test]
    fn bilinear_2x_align_corners_backward_distributes_ones_by_sample_weight() {
        let device = burn::tensor::Device::default().autodiff();
        let input = Tensor::<4>::from_data(
            TensorData::from([[[[1.0_f32, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]]),
            &device,
        )
        .require_grad();
        let loss = bilinear_upsample_2x_align_corners(input.clone()).sum();
        let grads = loss.backward();
        let actual = input
            .grad(&grads)
            .expect("input gradient")
            .into_data()
            .try_to_vec::<f32>()
            .unwrap();
        assert_close(
            &actual,
            &[3.24, 4.32, 3.24, 4.32, 5.76, 4.32, 3.24, 4.32, 3.24],
        );
    }
}
