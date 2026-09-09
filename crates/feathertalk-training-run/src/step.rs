use burn::{
    module::AutodiffModule,
    optim::{GradientsParams, Optimizer},
    tensor::backend::AutodiffBackend,
};
use feathertalk_models::unet::TrainableTalkingHead;
use feathertalk_training::{
    BaselineLossConfig, DataLoaderConfig, LossBreakdown, MouthRoiLossConfig,
    PerceptualFeatureExtractor, TemporalLossConfig, TrainingConfig, TrainingError, TrainingMode,
    baseline_loss, mouth_roi_loss, temporal_loss,
};
use feathertalk_training_data::{SingleFrameBatch, TemporalBatch};

use crate::LossValues;

/// Derives the data-loader config a training mode needs.
pub fn data_loader_config_for(
    config: &TrainingConfig,
    seed: u64,
) -> Result<DataLoaderConfig, TrainingError> {
    config.validate()?;
    Ok(match config.mode {
        TrainingMode::Baseline | TrainingMode::MouthRoi => {
            DataLoaderConfig::single_frame(config.batch_size, seed)
        }
        TrainingMode::MouthRoiTemporal => {
            DataLoaderConfig::temporal_pair(config.batch_size, seed, config.temporal_stride)
        }
    })
}

fn commit_gradients<B, M, O>(
    model: M,
    optimizer: &mut O,
    breakdown: LossBreakdown<B>,
    learning_rate: f64,
) -> Result<(M, LossValues), TrainingError>
where
    B: AutodiffBackend,
    M: AutodiffModule<B>,
    O: Optimizer<M, B>,
{
    let values = LossValues::from_breakdown(&breakdown);
    values.require_finite()?;
    let gradients = GradientsParams::from_grads(breakdown.total.backward(), &model);
    Ok((optimizer.step(learning_rate, model, gradients), values))
}

/// Runs one optimizer step over a single-frame batch.
pub fn train_single_frame_step<B, M, O, E>(
    model: M,
    optimizer: &mut O,
    extractor: &E,
    batch: SingleFrameBatch<B>,
    config: &TrainingConfig,
) -> Result<(M, LossValues), TrainingError>
where
    B: AutodiffBackend,
    M: TrainableTalkingHead<B> + AutodiffModule<B>,
    O: Optimizer<M, B>,
    E: PerceptualFeatureExtractor<B>,
{
    let prediction = model.forward_training(batch.image, batch.audio);
    let breakdown = match config.mode {
        TrainingMode::Baseline => {
            let loss_config = BaselineLossConfig {
                perceptual_weight: config.perceptual_weight,
            };
            baseline_loss(extractor, prediction, batch.target, &loss_config)?
        }
        TrainingMode::MouthRoi => {
            let loss_config = MouthRoiLossConfig {
                mouth_weight: config.mouth_weight,
                perceptual_weight: config.perceptual_weight,
            };
            mouth_roi_loss(
                extractor,
                prediction,
                batch.target,
                batch.mouth_mask,
                &loss_config,
            )?
        }
        TrainingMode::MouthRoiTemporal => {
            return Err(TrainingError::InvalidConfig(
                "the temporal mode needs train_temporal_step".to_owned(),
            ));
        }
    };
    commit_gradients(model, optimizer, breakdown, config.learning_rate)
}

/// Runs one optimizer step over a temporal pair batch.
pub fn train_temporal_step<B, M, O, E>(
    model: M,
    optimizer: &mut O,
    extractor: &E,
    batch: TemporalBatch<B>,
    config: &TrainingConfig,
) -> Result<(M, LossValues), TrainingError>
where
    B: AutodiffBackend,
    M: TrainableTalkingHead<B> + AutodiffModule<B>,
    O: Optimizer<M, B>,
    E: PerceptualFeatureExtractor<B>,
{
    if config.mode != TrainingMode::MouthRoiTemporal {
        return Err(TrainingError::InvalidConfig(
            "the non-temporal modes need train_single_frame_step".to_owned(),
        ));
    }
    let [pairs, pair_len, ..] = batch.target.dims();
    let flat = model.forward_training(batch.image, batch.audio);
    let [rows, channels, height, width] = flat.dims();
    if rows != pairs.saturating_mul(pair_len) {
        return Err(TrainingError::InvalidInput(format!(
            "temporal rows {rows} do not match {pairs}x{pair_len}"
        )));
    }
    let prediction = flat.reshape([pairs, pair_len, channels, height, width]);
    let loss_config = TemporalLossConfig {
        mouth_weight: config.mouth_weight,
        temporal_weight: config.temporal_weight,
        temporal_mouth_weight: config.temporal_mouth_weight,
        perceptual_weight: config.perceptual_weight,
    };
    let breakdown = temporal_loss(
        extractor,
        prediction,
        batch.target,
        batch.mouth_mask,
        &loss_config,
    )?;
    commit_gradients(model, optimizer, breakdown, config.learning_rate)
}
