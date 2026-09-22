use burn::module::Param;
use burn::nn::{BatchNorm, BatchNormConfig, Relu, conv::Conv2d};
use burn::tensor::Tensor;
use burn::tensor::Device;

#[derive(burn::module::Module, Debug)]
pub struct MobileOneBlock {
    branches: Vec<(Conv2d, BatchNorm)>,
    scale: Option<(Conv2d, BatchNorm)>,
    skip: Option<BatchNorm>,
    #[module(skip)]
    activation: bool}

#[derive(burn::module::Module, Debug)]
pub struct ReparameterizedMobileOneBlock {
    pub conv: Conv2d,
    #[module(skip)]
    activation: bool}

impl MobileOneBlock {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        groups: usize,
        num_conv_branches: usize,
        is_linear: bool,
        device: &Device,
    ) -> Self {
        Self::new_with_stride(
            in_channels,
            out_channels,
            kernel_size,
            [stride, stride],
            padding,
            groups,
            num_conv_branches,
            is_linear,
            device,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new_with_stride(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: [usize; 2],
        padding: usize,
        groups: usize,
        num_conv_branches: usize,
        is_linear: bool,
        device: &Device,
    ) -> Self {
        assert!(in_channels > 0);
        assert!(out_channels > 0);
        assert!(kernel_size > 0 && kernel_size % 2 == 1);
        if kernel_size > 1 {
            assert_eq!(padding, kernel_size / 2);
        }
        assert!(stride.into_iter().all(|value| matches!(value, 1 | 2)));
        assert!(groups > 0);
        assert!(num_conv_branches > 0);
        assert!(in_channels.is_multiple_of(groups));
        assert!(out_channels.is_multiple_of(groups));

        let branches = (0..num_conv_branches)
            .map(|_| {
                let conv = burn::nn::conv::Conv2dConfig::new(
                    [in_channels, out_channels],
                    [kernel_size, kernel_size],
                )
                .with_stride(stride)
                .with_padding(burn::nn::PaddingConfig2d::Explicit(
                    padding, padding, padding, padding,
                ))
                .with_groups(groups)
                .with_bias(false)
                .init(device);
                let bn = BatchNormConfig::new(out_channels).init(device);
                (conv, bn)
            })
            .collect();
        let scale = if kernel_size > 1 {
            let conv = burn::nn::conv::Conv2dConfig::new([in_channels, out_channels], [1, 1])
                .with_stride(stride)
                .with_padding(burn::nn::PaddingConfig2d::Valid)
                .with_groups(groups)
                .with_bias(false)
                .init(device);
            Some((conv, BatchNormConfig::new(out_channels).init(device)))
        } else {
            None
        };
        let skip = if stride == [1, 1] && in_channels == out_channels {
            Some(BatchNormConfig::new(in_channels).init(device))
        } else {
            None
        };
        Self {
            branches,
            scale,
            skip,
            activation: !is_linear}
    }

    pub fn forward(&self, input: Tensor<4>) -> Tensor<4> {
        let mut output: Option<Tensor<4>> = None;
        for (conv, bn) in &self.branches {
            let branch = bn.forward(conv.forward(input.clone()));
            output = Some(match output {
                Some(current) => current + branch,
                None => branch});
        }
        if let Some((conv, bn)) = &self.scale {
            let branch = bn.forward(conv.forward(input.clone()));
            output = Some(match output {
                Some(current) => current + branch,
                None => branch});
        }
        if let Some(skip) = &self.skip {
            let branch = skip.forward(input);
            output = Some(match output {
                Some(current) => current + branch,
                None => branch});
        }
        let output = output.expect("MobileOneBlock always has a convolution branch");
        if self.activation {
            Relu.forward(output)
        } else {
            output
        }
    }

    pub fn reparameterize(&self) -> ReparameterizedMobileOneBlock {
        let (first_conv, _) = self
            .branches
            .first()
            .expect("MobileOneBlock always has a convolution branch");
        let [out_channels, input_per_group, kernel_height, kernel_width] =
            first_conv.weight.val().dims();
        assert_eq!(kernel_height, kernel_width);
        assert!(kernel_height % 2 == 1);
        let device = first_conv.weight.val().device();

        let mut fused_kernel = Tensor::<4>::zeros(
            [out_channels, input_per_group, kernel_height, kernel_width],
            &device,
        );
        let mut fused_bias = Tensor::<1>::zeros([out_channels], &device);

        for (conv, batch_norm) in &self.branches {
            let (kernel, bias) = fuse_conv_batch_norm(conv, batch_norm);
            fused_kernel = fused_kernel + kernel;
            fused_bias = fused_bias + bias;
        }

        if let Some((conv, batch_norm)) = &self.scale {
            let (kernel, bias) = fuse_conv_batch_norm(conv, batch_norm);
            fused_kernel = fused_kernel + center_pad_kernel(kernel, [kernel_height, kernel_width]);
            fused_bias = fused_bias + bias;
        }

        if let Some(batch_norm) = &self.skip {
            let (kernel, bias) = fuse_identity_batch_norm(
                batch_norm,
                out_channels,
                first_conv.groups,
                [kernel_height, kernel_width],
            );
            fused_kernel = fused_kernel + kernel;
            fused_bias = fused_bias + bias;
        }

        let conv = Conv2d {
            weight: Param::from_tensor(fused_kernel.detach()),
            bias: Some(Param::from_tensor(fused_bias.detach())),
            stride: first_conv.stride,
            kernel_size: first_conv.kernel_size,
            dilation: first_conv.dilation,
            groups: first_conv.groups,
            padding: first_conv.padding.clone()};
        ReparameterizedMobileOneBlock {
            conv,
            activation: self.activation}
    }
}

impl ReparameterizedMobileOneBlock {
    pub fn forward(&self, input: Tensor<4>) -> Tensor<4> {
        let output = self.conv.forward(input);
        if self.activation {
            Relu.forward(output)
        } else {
            output
        }
    }
}

fn fuse_conv_batch_norm(
    conv: &Conv2d,
    batch_norm: &BatchNorm,
) -> (Tensor<4>, Tensor<1>) {
    let weight = conv.weight.val().detach();
    let running_mean = batch_norm.running_mean.value().detach();
    let running_var = batch_norm.running_var.value().detach();
    let gamma = batch_norm.gamma.val().detach();
    let beta = batch_norm.beta.val().detach();
    let scale = gamma.div(running_var.add_scalar(batch_norm.epsilon).sqrt());
    let [out_channels] = scale.dims();
    let kernel = weight.mul(scale.clone().reshape([out_channels, 1, 1, 1]));
    let bias = beta - running_mean.mul(scale);
    (kernel, bias)
}

fn fuse_identity_batch_norm(
    batch_norm: &BatchNorm,
    channels: usize,
    groups: usize,
    kernel_size: [usize; 2],
) -> (Tensor<4>, Tensor<1>) {
    assert!(channels.is_multiple_of(groups));
    let input_per_group = channels / groups;
    let [kernel_height, kernel_width] = kernel_size;
    let device = batch_norm.gamma.val().device();
    let mut identity = vec![0.0_f32; channels * input_per_group * kernel_height * kernel_width];
    let center = (kernel_height / 2) * kernel_width + kernel_width / 2;
    for output_channel in 0..channels {
        let input_channel = output_channel % input_per_group;
        let index =
            (output_channel * input_per_group + input_channel) * kernel_height * kernel_width
                + center;
        identity[index] = 1.0;
    }
    let identity = Tensor::<1>::from_data(identity.as_slice(), &device).reshape([
        channels,
        input_per_group,
        kernel_height,
        kernel_width,
    ]);
    let running_mean = batch_norm.running_mean.value().detach();
    let running_var = batch_norm.running_var.value().detach();
    let gamma = batch_norm.gamma.val().detach();
    let beta = batch_norm.beta.val().detach();
    let scale = gamma.div(running_var.add_scalar(batch_norm.epsilon).sqrt());
    let kernel = identity.mul(scale.clone().reshape([channels, 1, 1, 1]));
    let bias = beta - running_mean.mul(scale);
    (kernel, bias)
}

fn center_pad_kernel(kernel: Tensor<4>, target_size: [usize; 2]) -> Tensor<4> {
    let [out_channels, input_per_group, source_height, source_width] = kernel.dims();
    assert_eq!([source_height, source_width], [1, 1]);
    let [target_height, target_width] = target_size;
    let device = kernel.device();
    let top = target_height / 2;
    let left = target_width / 2;
    Tensor::<4>::zeros(
        [out_channels, input_per_group, target_height, target_width],
        &device,
    )
    .slice_assign(
        [
            0..out_channels,
            0..input_per_group,
            top..top + 1,
            left..left + 1,
        ],
        kernel,
    )
}
