use burn::tensor::{Tensor, backend::Backend};
use serde::{Deserialize, Serialize};

use crate::{
    PerceptualFeatureExtractor, TrainingError, perceptual::validate_image_pair, perceptual_mse,
};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct BaselineLossConfig {
    pub perceptual_weight: f64,
}

impl Default for BaselineLossConfig {
    fn default() -> Self {
        Self {
            perceptual_weight: 0.01,
        }
    }
}

impl BaselineLossConfig {
    pub fn validate(&self) -> Result<(), TrainingError> {
        validate_weight("perceptual_weight", self.perceptual_weight)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MouthRoiLossConfig {
    pub mouth_weight: f64,
    pub perceptual_weight: f64,
}

impl Default for MouthRoiLossConfig {
    fn default() -> Self {
        Self {
            mouth_weight: 4.0,
            perceptual_weight: 0.01,
        }
    }
}

impl MouthRoiLossConfig {
    pub fn validate(&self) -> Result<(), TrainingError> {
        validate_weight("mouth_weight", self.mouth_weight)?;
        validate_weight("perceptual_weight", self.perceptual_weight)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TemporalLossConfig {
    pub mouth_weight: f64,
    pub temporal_weight: f64,
    pub temporal_mouth_weight: f64,
    pub perceptual_weight: f64,
}

impl Default for TemporalLossConfig {
    fn default() -> Self {
        Self {
            mouth_weight: 4.0,
            temporal_weight: 0.5,
            temporal_mouth_weight: 4.0,
            perceptual_weight: 0.01,
        }
    }
}

impl TemporalLossConfig {
    pub fn validate(&self) -> Result<(), TrainingError> {
        validate_weight("mouth_weight", self.mouth_weight)?;
        validate_weight("temporal_weight", self.temporal_weight)?;
        validate_weight("temporal_mouth_weight", self.temporal_mouth_weight)?;
        validate_weight("perceptual_weight", self.perceptual_weight)
    }
}

#[derive(Debug)]
pub struct LossBreakdown<B: Backend> {
    pub total: Tensor<B, 1>,
    pub full: Tensor<B, 1>,
    pub perceptual: Tensor<B, 1>,
    pub mouth: Option<Tensor<B, 1>>,
    pub temporal: Option<Tensor<B, 1>>,
    pub temporal_mouth: Option<Tensor<B, 1>>,
}

pub fn mouth_l1_loss<B: Backend>(
    prediction: Tensor<B, 4>,
    target: Tensor<B, 4>,
    mask: Tensor<B, 4>,
) -> Result<Tensor<B, 1>, TrainingError> {
    validate_image_pair(&prediction, &target)?;
    validate_mask(&mask, prediction.dims())?;

    let channels = prediction.dims()[1] as f64;
    let denominator = mask.clone().sum().clamp_min(1.0).mul_scalar(channels);
    Ok(((prediction - target).abs() * mask).sum() / denominator)
}

pub fn baseline_loss<B: Backend, E: PerceptualFeatureExtractor<B>>(
    extractor: &E,
    prediction: Tensor<B, 4>,
    target: Tensor<B, 4>,
    config: &BaselineLossConfig,
) -> Result<LossBreakdown<B>, TrainingError> {
    config.validate()?;
    validate_image_pair(&prediction, &target)?;

    let full = (prediction.clone() - target.clone()).abs().mean();
    let perceptual = perceptual_mse(extractor, prediction, target)?;
    let total = full.clone() + perceptual.clone().mul_scalar(config.perceptual_weight);

    Ok(LossBreakdown {
        total,
        full,
        perceptual,
        mouth: None,
        temporal: None,
        temporal_mouth: None,
    })
}

pub fn mouth_roi_loss<B: Backend, E: PerceptualFeatureExtractor<B>>(
    extractor: &E,
    prediction: Tensor<B, 4>,
    target: Tensor<B, 4>,
    mask: Tensor<B, 4>,
    config: &MouthRoiLossConfig,
) -> Result<LossBreakdown<B>, TrainingError> {
    config.validate()?;
    validate_image_pair(&prediction, &target)?;
    validate_mask(&mask, prediction.dims())?;

    let full = (prediction.clone() - target.clone()).abs().mean();
    let mouth = mouth_l1_loss(prediction.clone(), target.clone(), mask)?;
    let perceptual = perceptual_mse(extractor, prediction, target)?;
    let total = full.clone()
        + mouth.clone().mul_scalar(config.mouth_weight)
        + perceptual.clone().mul_scalar(config.perceptual_weight);

    Ok(LossBreakdown {
        total,
        full,
        perceptual,
        mouth: Some(mouth),
        temporal: None,
        temporal_mouth: None,
    })
}

pub fn temporal_loss<B: Backend, E: PerceptualFeatureExtractor<B>>(
    extractor: &E,
    prediction: Tensor<B, 5>,
    target: Tensor<B, 5>,
    mask: Tensor<B, 5>,
    config: &TemporalLossConfig,
) -> Result<LossBreakdown<B>, TrainingError> {
    config.validate()?;
    let [batch, pair_len, channels, height, width] = prediction.dims();
    if target.dims() != [batch, pair_len, channels, height, width] {
        return Err(TrainingError::InvalidInput(format!(
            "temporal prediction/target shape mismatch: {:?} != {:?}",
            prediction.dims(),
            target.dims()
        )));
    }
    if pair_len != 2 {
        return Err(TrainingError::InvalidInput(format!(
            "temporal input pair length must be exactly 2, got {pair_len}"
        )));
    }
    if batch == 0 {
        return Err(TrainingError::InvalidInput(
            "temporal input batch must be non-zero".to_owned(),
        ));
    }
    if channels != 3 {
        return Err(TrainingError::InvalidInput(format!(
            "temporal input must have exactly 3 image channels, got {channels}"
        )));
    }
    if height < 4 || width < 4 {
        return Err(TrainingError::InvalidInput(format!(
            "temporal input spatial dimensions must both be at least 4, got {height}x{width}"
        )));
    }
    let expected_mask = [batch, pair_len, 1, height, width];
    if mask.dims() != expected_mask {
        return Err(TrainingError::InvalidInput(format!(
            "temporal mask must have shape {expected_mask:?}, got {:?}",
            mask.dims()
        )));
    }

    let flat_batch = batch * pair_len;
    let flat_prediction = prediction
        .clone()
        .reshape([flat_batch, channels, height, width]);
    let flat_target = target
        .clone()
        .reshape([flat_batch, channels, height, width]);
    let flat_mask = mask.clone().reshape([flat_batch, 1, height, width]);

    let full = (flat_prediction.clone() - flat_target.clone()).abs().mean();
    let mouth = mouth_l1_loss(flat_prediction.clone(), flat_target.clone(), flat_mask)?;
    let perceptual = perceptual_mse(extractor, flat_prediction, flat_target)?;

    let first_prediction = prediction
        .clone()
        .slice([0..batch, 0..1, 0..channels, 0..height, 0..width])
        .squeeze_dim::<4>(1);
    let second_prediction = prediction
        .clone()
        .slice([0..batch, 1..2, 0..channels, 0..height, 0..width])
        .squeeze_dim::<4>(1);
    let first_target = target
        .clone()
        .slice([0..batch, 0..1, 0..channels, 0..height, 0..width])
        .squeeze_dim::<4>(1);
    let second_target = target
        .clone()
        .slice([0..batch, 1..2, 0..channels, 0..height, 0..width])
        .squeeze_dim::<4>(1);
    let first_mask = mask
        .clone()
        .slice([0..batch, 0..1, 0..1, 0..height, 0..width])
        .squeeze_dim::<4>(1);
    let second_mask = mask
        .slice([0..batch, 1..2, 0..1, 0..height, 0..width])
        .squeeze_dim::<4>(1);

    let prediction_delta = second_prediction - first_prediction;
    let target_delta = second_target - first_target;
    let union_mask = first_mask.max_pair(second_mask);
    let temporal = (prediction_delta.clone() - target_delta.clone())
        .abs()
        .mean();
    let temporal_mouth = mouth_l1_loss(prediction_delta, target_delta, union_mask)?;

    let total = full.clone()
        + mouth.clone().mul_scalar(config.mouth_weight)
        + temporal.clone().mul_scalar(config.temporal_weight)
        + temporal_mouth
            .clone()
            .mul_scalar(config.temporal_mouth_weight)
        + perceptual.clone().mul_scalar(config.perceptual_weight);

    Ok(LossBreakdown {
        total,
        full,
        perceptual,
        mouth: Some(mouth),
        temporal: Some(temporal),
        temporal_mouth: Some(temporal_mouth),
    })
}

fn validate_weight(name: &str, value: f64) -> Result<(), TrainingError> {
    if !value.is_finite() || value < 0.0 {
        return Err(TrainingError::InvalidConfig(format!(
            "{name} must be finite and non-negative, got {value}"
        )));
    }
    Ok(())
}

fn validate_mask<B: Backend>(
    mask: &Tensor<B, 4>,
    image_shape: [usize; 4],
) -> Result<(), TrainingError> {
    let expected = [image_shape[0], 1, image_shape[2], image_shape[3]];
    if mask.dims() != expected {
        return Err(TrainingError::InvalidInput(format!(
            "mouth mask must have shape {expected:?}, got {:?}",
            mask.dims()
        )));
    }
    Ok(())
}
