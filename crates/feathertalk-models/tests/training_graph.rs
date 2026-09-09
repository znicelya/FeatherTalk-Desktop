use burn::tensor::Tensor;
use feathertalk_models::{
    backend::CpuBackend,
    unet::{
        MobileOneUnet, MobileOneUnetConfig, OriginalUnet, OriginalUnetConfig, TrainableTalkingHead,
    },
};

type CpuDevice = burn::backend::ndarray::NdArrayDevice;

fn assert_trainable_talking_head<M: TrainableTalkingHead<CpuBackend>>() {}

fn image(device: &CpuDevice) -> Tensor<CpuBackend, 4> {
    Tensor::zeros([1, 6, 160, 160], device)
}

fn audio(device: &CpuDevice) -> Tensor<CpuBackend, 4> {
    Tensor::zeros([1, 16, 32, 32], device)
}

#[test]
fn both_training_graphs_implement_the_public_training_trait() {
    assert_trainable_talking_head::<OriginalUnet<CpuBackend>>();
    assert_trainable_talking_head::<MobileOneUnet<CpuBackend>>();
}

#[test]
fn training_trait_forward_preserves_the_fixed_unet_contract() {
    let device = CpuDevice::default();

    let original = OriginalUnetConfig::parity_micro().init::<CpuBackend>(&device);
    let original_output = original.forward_training(image(&device), audio(&device));
    assert_eq!(original_output.dims(), [1, 3, 160, 160]);

    let mobile = MobileOneUnetConfig::parity_micro().init::<CpuBackend>(&device);
    let mobile_output = mobile.forward_training(image(&device), audio(&device));
    assert_eq!(mobile_output.dims(), [1, 3, 160, 160]);
}
