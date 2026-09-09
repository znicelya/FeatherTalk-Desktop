use burn::{
    optim::AdamConfig,
    tensor::{Tensor, TensorData},
};
use feathertalk_models::{
    backend::{CpuAutodiffBackend, CpuBackend},
    train_step::{adam_train_step, l1_loss},
    unet::OriginalUnetConfig,
};

fn micro_batch(
    device: &burn::tensor::Device<CpuBackend>,
) -> (
    Tensor<CpuAutodiffBackend, 4>,
    Tensor<CpuAutodiffBackend, 4>,
    Tensor<CpuAutodiffBackend, 4>,
) {
    (
        Tensor::ones([1, 6, 160, 160], device),
        Tensor::ones([1, 16, 32, 32], device),
        Tensor::zeros([1, 3, 160, 160], device),
    )
}

#[test]
fn l1_loss_matches_hand_computed_value() {
    let device = Default::default();
    let prediction =
        Tensor::<CpuBackend, 2>::from_data(TensorData::from([[1.0_f32, -2.0, 3.0]]), &device);
    let target = Tensor::<CpuBackend, 2>::zeros([1, 3], &device);
    let actual = l1_loss(prediction, target).into_scalar();
    assert!((actual - 2.0).abs() <= f32::EPSILON);
}

#[test]
fn backward_registers_output_weight_gradient() {
    let device = Default::default();
    let model = OriginalUnetConfig::parity_micro().init::<CpuAutodiffBackend>(&device);
    let (image, audio, target) = micro_batch(&device);
    let loss = l1_loss(model.forward(image, audio), target);
    let gradients = loss.backward();
    assert!(model.outc.conv.weight.grad(&gradients).is_some());
}

#[test]
fn zero_learning_rate_leaves_output_weight_unchanged() {
    let device = Default::default();
    let model = OriginalUnetConfig::parity_micro().init::<CpuAutodiffBackend>(&device);
    let before = model.outc.conv.weight.val().into_data();
    let (image, audio, target) = micro_batch(&device);
    let mut optimizer = AdamConfig::new().init();
    let (model, loss) = adam_train_step(model, &mut optimizer, image, audio, target, 0.0);
    let after = model.outc.conv.weight.val().into_data();
    assert!(loss.is_finite());
    assert_eq!(before, after);
}
