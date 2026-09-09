use burn::{module::Module, tensor::Tensor};
use feathertalk_models::{
    backend::{CpuAutodiffBackend, CpuBackend},
    train_step::l1_loss,
    unet::{
        MobileOneAudioConvHubertConfig, MobileOneDownConfig, MobileOneUnetConfig,
        MobileOneUnetInference, MobileOneUpConfig,
    },
};

fn assert_module<M: Module<CpuBackend>>() {}

fn assert_max_abs(left: Tensor<CpuBackend, 4>, right: Tensor<CpuBackend, 4>, tolerance: f32) {
    let left = left.into_data().to_vec::<f32>().unwrap();
    let right = right.into_data().to_vec::<f32>().unwrap();
    let max_abs = left
        .iter()
        .zip(right.iter())
        .map(|(left, right)| (left - right).abs())
        .fold(0.0_f32, f32::max);
    assert!(max_abs <= tolerance, "max_abs={max_abs}");
}

#[test]
fn production_mobileone_config_is_fixed() {
    let config = MobileOneUnetConfig::production();
    assert_eq!(config.channels, [32, 64, 128, 256, 512]);
    assert_eq!(config.num_conv_branches, 2);
}

#[test]
fn mobileone_down_halves_the_spatial_size() {
    let device = Default::default();
    let down = MobileOneDownConfig::new(2, 4, 2).init::<CpuBackend>(&device);
    let input = Tensor::zeros([1, 2, 160, 160], &device);
    assert_eq!(down.forward(input).dims(), [1, 4, 80, 80]);
}

#[test]
fn mobileone_up_restores_the_skip_spatial_size() {
    let device = Default::default();
    let up = MobileOneUpConfig::new(16, 4, 2).init::<CpuBackend>(&device);
    let input = Tensor::zeros([1, 8, 20, 20], &device);
    let skip = Tensor::zeros([1, 8, 40, 40], &device);
    assert_eq!(up.forward(input, skip).dims(), [1, 4, 40, 40]);
}

#[test]
fn mobileone_hubert_audio_branch_matches_the_micro_bottleneck() {
    let device = Default::default();
    let branch =
        MobileOneAudioConvHubertConfig::new([2, 4, 8, 16, 32], 2).init::<CpuBackend>(&device);
    let audio = Tensor::zeros([1, 16, 32, 32], &device);
    assert_eq!(branch.forward(audio).dims(), [1, 32, 10, 10]);
}

#[test]
fn mobileone_training_and_inference_graphs_are_burn_modules() {
    assert_module::<feathertalk_models::unet::MobileOneUnet<CpuBackend>>();
    assert_module::<MobileOneUnetInference<CpuBackend>>();
}

#[test]
fn micro_mobileone_unet_returns_fixed_bounded_output() {
    let device = Default::default();
    let model = MobileOneUnetConfig::parity_micro().init::<CpuBackend>(&device);
    let image = Tensor::ones([1, 6, 160, 160], &device);
    let audio = Tensor::ones([1, 16, 32, 32], &device);
    let output = model.forward(image, audio);
    assert_eq!(output.dims(), [1, 3, 160, 160]);
    let values = output.into_data().to_vec::<f32>().unwrap();
    assert!(values.iter().all(|value| value.is_finite()));
    assert!(values.iter().all(|value| (0.0..=1.0).contains(value)));
}

#[test]
fn production_mobileone_unet_returns_fixed_shape() {
    let device = Default::default();
    let model = MobileOneUnetConfig::production().init::<CpuBackend>(&device);
    let image = Tensor::zeros([1, 6, 160, 160], &device);
    let audio = Tensor::zeros([1, 16, 32, 32], &device);
    assert_eq!(model.forward(image, audio).dims(), [1, 3, 160, 160]);
}

#[test]
fn micro_training_and_inference_graphs_are_equivalent() {
    let device = Default::default();
    let model = MobileOneUnetConfig::parity_micro().init::<CpuBackend>(&device);
    let image = Tensor::ones([1, 6, 160, 160], &device);
    let audio = Tensor::ones([1, 16, 32, 32], &device);
    let expected = model.forward(image.clone(), audio.clone());
    let inference = model.reparameterize();
    let actual = inference.forward(image, audio);
    assert_max_abs(expected, actual, 1.0e-4);
}

#[test]
fn reparameterization_does_not_mutate_the_training_graph() {
    let device = Default::default();
    let model = MobileOneUnetConfig::parity_micro().init::<CpuBackend>(&device);
    let image = Tensor::ones([1, 6, 160, 160], &device);
    let audio = Tensor::ones([1, 16, 32, 32], &device);
    let before = model.forward(image.clone(), audio.clone());
    let _inference = model.reparameterize();
    let after = model.forward(image, audio);
    assert_max_abs(before, after, 0.0);
}

#[test]
#[should_panic(expected = "MobileOne UNet image input must be [B,6,160,160]")]
fn mobileone_unet_rejects_wrong_image_shape_before_forward() {
    let device = Default::default();
    let model = MobileOneUnetConfig::parity_micro().init::<CpuBackend>(&device);
    let image = Tensor::zeros([1, 3, 160, 160], &device);
    let audio = Tensor::zeros([1, 16, 32, 32], &device);
    let _ = model.forward(image, audio);
}

#[test]
fn mobileone_output_weight_receives_gradient() {
    let device = Default::default();
    let model = MobileOneUnetConfig::parity_micro().init::<CpuAutodiffBackend>(&device);
    let image = Tensor::ones([1, 6, 160, 160], &device);
    let audio = Tensor::ones([1, 16, 32, 32], &device);
    let target = Tensor::zeros([1, 3, 160, 160], &device);
    let gradients = l1_loss(model.forward(image, audio), target).backward();
    assert!(model.outc.conv.weight.grad(&gradients).is_some());
}
