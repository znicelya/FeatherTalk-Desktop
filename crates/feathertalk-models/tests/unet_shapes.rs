use burn::tensor::Tensor;
use feathertalk_models::{
    backend::CpuBackend,
    unet::{AudioConvHubertConfig, DownConfig, InvertedResidualConfig, OriginalUnetConfig},
};

#[test]
fn inverted_residual_preserves_shape_when_residual_is_enabled() {
    let device = Default::default();
    let block = InvertedResidualConfig::new(8, 8)
        .with_expansion(2)
        .with_stride(1)
        .init::<CpuBackend>(&device);
    let input = Tensor::<CpuBackend, 4>::ones([1, 8, 20, 20], &device);
    assert_eq!(block.forward(input).dims(), [1, 8, 20, 20]);
}

#[test]
fn inverted_residual_uses_depthwise_groups() {
    let device = Default::default();
    let block = InvertedResidualConfig::new(8, 8)
        .with_expansion(2)
        .init::<CpuBackend>(&device);
    assert_eq!(block.depthwise_conv.groups, 16);
}

#[test]
fn down_blocks_produce_80_40_20_10_spatial_sizes() {
    let device = Default::default();
    let input = Tensor::<CpuBackend, 4>::ones([1, 32, 160, 160], &device);
    let down1 = DownConfig::new(32, 64).init::<CpuBackend>(&device);
    let down2 = DownConfig::new(64, 128).init::<CpuBackend>(&device);
    let down3 = DownConfig::new(128, 256).init::<CpuBackend>(&device);
    let down4 = DownConfig::new(256, 512).init::<CpuBackend>(&device);

    let x1 = down1.forward(input);
    assert_eq!(x1.dims(), [1, 64, 80, 80]);
    let x2 = down2.forward(x1);
    assert_eq!(x2.dims(), [1, 128, 40, 40]);
    let x3 = down3.forward(x2);
    assert_eq!(x3.dims(), [1, 256, 20, 20]);
    assert_eq!(down4.forward(x3).dims(), [1, 512, 10, 10]);
}

#[test]
fn hubert_audio_branch_matches_image_bottleneck_shape() {
    let device = Default::default();
    let branch = AudioConvHubertConfig::new([32, 64, 128, 256, 512]).init::<CpuBackend>(&device);
    let audio = Tensor::<CpuBackend, 4>::ones([1, 16, 32, 32], &device);
    assert_eq!(branch.forward(audio).dims(), [1, 512, 10, 10]);
}

#[test]
fn production_unet_returns_three_by_160_by_160() {
    let device = Default::default();
    let model = OriginalUnetConfig::production().init::<CpuBackend>(&device);
    let image = Tensor::<CpuBackend, 4>::zeros([1, 6, 160, 160], &device);
    let audio = Tensor::<CpuBackend, 4>::zeros([1, 16, 32, 32], &device);
    assert_eq!(model.forward(image, audio).dims(), [1, 3, 160, 160]);
}

#[test]
fn output_is_bounded_by_sigmoid() {
    let device = Default::default();
    let model = OriginalUnetConfig::parity_micro().init::<CpuBackend>(&device);
    let image = Tensor::<CpuBackend, 4>::ones([1, 6, 160, 160], &device);
    let audio = Tensor::<CpuBackend, 4>::ones([1, 16, 32, 32], &device);
    let values = model
        .forward(image, audio)
        .into_data()
        .to_vec::<f32>()
        .unwrap();
    assert!(values.iter().all(|value| value.is_finite()));
    assert!(values.iter().all(|value| (0.0..=1.0).contains(value)));
}
