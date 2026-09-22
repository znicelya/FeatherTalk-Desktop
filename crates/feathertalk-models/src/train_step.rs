//! Shared training-step primitives.

use burn::{
    optim::{GradientsParams, ModuleOptimizer},
    tensor::{ElementConversion, Tensor},
};

use crate::unet::OriginalUnet;

pub fn l1_loss<const D: usize>(prediction: Tensor<D>, target: Tensor<D>) -> Tensor<1> {
    (prediction - target).abs().mean()
}

pub fn adam_train_step(
    model: OriginalUnet,
    optimizer: &mut ModuleOptimizer,
    image: Tensor<4>,
    audio: Tensor<4>,
    target: Tensor<4>,
    learning_rate: f64,
) -> (OriginalUnet, f32) {
    let prediction = model.forward(image, audio);
    let loss = l1_loss(prediction, target);
    let loss_value = loss.clone().into_scalar::<f32>().elem::<f32>();
    let gradients = loss.backward();
    let gradients = GradientsParams::from_grads(gradients, &model);
    let model = optimizer.step(learning_rate, model, gradients);
    (model, loss_value)
}