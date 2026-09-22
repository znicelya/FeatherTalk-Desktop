use burn::tensor::Tensor;
use feathertalk_models::{
        unet::{
        MobileOneUnet, MobileOneUnetConfig, OriginalUnet, OriginalUnetConfig, TrainableTalkingHead},
};

type CpuDevice = burn::tensor::Device;

fn assert_trainable_talking_head<M: TrainableTalkingHead>() {}

fn image(device: &CpuDevice) -> Tensor<4> {
    Tensor::zeros([1, 6, 160, 160], device)
}

fn audio(device: &CpuDevice) -> Tensor<4> {
    Tensor::zeros([1, 16, 32, 32], device)
}

#[test]
fn both_training_graphs_implement_the_public_training_trait() {
    assert_trainable_talking_head::<OriginalUnet>();
    assert_trainable_talking_head::<MobileOneUnet>();
}

#[test]
fn training_trait_forward_preserves_the_fixed_unet_contract() {
    let device = CpuDevice::default();

    let original = OriginalUnetConfig::parity_micro().init(&device);
    let original_output = original.forward_training(image(&device), audio(&device));
    assert_eq!(original_output.dims(), [1, 3, 160, 160]);

    let mobile = MobileOneUnetConfig::parity_micro().init(&device);
    let mobile_output = mobile.forward_training(image(&device), audio(&device));
    assert_eq!(mobile_output.dims(), [1, 3, 160, 160]);
}
