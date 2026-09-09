//! Shared training-step primitives.

use burn::{
    optim::{GradientsParams, Optimizer},
    tensor::{
        ElementConversion, Tensor,
        backend::{AutodiffBackend, Backend},
    },
};

use crate::unet::OriginalUnet;

pub fn l1_loss<B: Backend, const D: usize>(
    prediction: Tensor<B, D>,
    target: Tensor<B, D>,
) -> Tensor<B, 1> {
    (prediction - target).abs().mean()
}

pub fn adam_train_step<B>(
    model: OriginalUnet<B>,
    optimizer: &mut impl Optimizer<OriginalUnet<B>, B>,
    image: Tensor<B, 4>,
    audio: Tensor<B, 4>,
    target: Tensor<B, 4>,
    learning_rate: f64,
) -> (OriginalUnet<B>, f32)
where
    B: AutodiffBackend,
{
    let prediction = model.forward(image, audio);
    let loss = l1_loss(prediction, target);
    let loss_value = loss.clone().into_scalar().elem::<f32>();
    let gradients = loss.backward();
    let gradients = GradientsParams::from_grads(gradients, &model);
    let model = optimizer.step(learning_rate, model, gradients);
    (model, loss_value)
}
