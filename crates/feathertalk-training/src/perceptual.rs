use burn::tensor::{Tensor};

use crate::{TrainingError, Vgg19Conv3_3};

pub trait PerceptualFeatureExtractor {
    fn forward(&self, image: Tensor<4>) -> Tensor<4>;
}

impl PerceptualFeatureExtractor for Vgg19Conv3_3 {
    fn forward(&self, image: Tensor<4>) -> Tensor<4> {
        Vgg19Conv3_3::forward(self, image)
    }
}

pub fn perceptual_mse(
    extractor: &impl PerceptualFeatureExtractor,
    prediction: Tensor<4>,
    target: Tensor<4>,
) -> Result<Tensor<1>, TrainingError> {
    validate_image_pair(&prediction, &target)?;
    let predicted = extractor.forward(prediction);
    let expected = extractor.forward(target.detach());
    Ok((predicted - expected).square().mean())
}

pub(crate) fn validate_image_pair(
    prediction: &Tensor<4>,
    target: &Tensor<4>,
) -> Result<(), TrainingError> {
    let prediction_shape = prediction.dims();
    let target_shape = target.dims();
    if prediction_shape != target_shape {
        return Err(TrainingError::InvalidInput(format!(
            "perceptual prediction/target shape mismatch: {prediction_shape:?} != {target_shape:?}"
        )));
    }
    let [batch, channels, height, width] = prediction_shape;
    if batch == 0 {
        return Err(TrainingError::InvalidInput(
            "perceptual input batch must be non-zero".to_owned(),
        ));
    }
    if channels != 3 {
        return Err(TrainingError::InvalidInput(format!(
            "perceptual input must have exactly 3 channels, got {channels}"
        )));
    }
    if height < 4 || width < 4 {
        return Err(TrainingError::InvalidInput(format!(
            "perceptual input spatial dimensions must both be at least 4, got {height}x{width}"
        )));
    }
    Ok(())
}
