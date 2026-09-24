use burn::{
    module::Module,
    optim::GradientsParams,
};
use feathertalk_models::unet::TrainableTalkingHead;
use feathertalk_training::CheckpointableOptimizer;
use feathertalk_training::{
    BaselineLossConfig, DataLoaderConfig, LossBreakdown, MouthRoiLossConfig,
    PerceptualFeatureExtractor, TemporalLossConfig, TrainingConfig, TrainingError, TrainingMode,
    baseline_loss, mouth_roi_loss, temporal_loss};
use feathertalk_training_data::{SingleFrameBatch, TemporalBatch};
use std::time::Instant;

use crate::LossValues;

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct StepComputeTimings {
    pub forward_secs: f64,
    pub backward_secs: f64,
    pub optim_secs: f64}

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

fn commit_gradients<M, O>(
    model: M,
    optimizer: &mut O,
    breakdown: LossBreakdown,
    learning_rate: f64,
    forward_started: Instant,
) -> Result<(M, LossValues, StepComputeTimings), TrainingError>
where
    M: Module,
    O: CheckpointableOptimizer,
{
    let values = LossValues::from_breakdown(&breakdown);
    values.require_finite()?;
    let forward_secs = forward_started.elapsed().as_secs_f64();
    let device = breakdown.total.device();
    // Full-device syncs isolate the backward and optimizer phases so their
    // timings are attributable, but each one is a pipeline stall. Pay for them
    // only while profiling; otherwise the per-step loss readback in
    // `LossValues::from_breakdown` already flushes the in-flight queue to one
    // step, so the extra barriers are pure overhead.
    let profile = profiling_enabled();
    let backward_started = Instant::now();
    let gradients = GradientsParams::from_grads(breakdown.total.backward(), &model);
    if profile {
        sync_backend(&device)?;
    }
    let backward_secs = backward_started.elapsed().as_secs_f64();
    let optim_started = Instant::now();
    let model = optimizer.optimizer_step(learning_rate, model, gradients);
    if profile {
        sync_backend(&device)?;
    }
    Ok((
        model,
        values,
        StepComputeTimings {
            forward_secs,
            backward_secs,
            optim_secs: optim_started.elapsed().as_secs_f64()},
    ))
}

fn sync_backend(device: &burn::tensor::Device) -> Result<(), TrainingError> {
    device.sync().map_err(|error| {
        TrainingError::InvalidInput(format!("training backend sync failed: {error}"))
    })
}

/// Whether step profiling is active, read from the environment once and cached.
///
/// Mirrors the `FEATHERTALK_STEP_PROFILE` gate the runner and worker use, so
/// the per-phase backend syncs that make `backward_secs`/`optim_secs`
/// attributable are only paid when those numbers are actually recorded.
fn profiling_enabled() -> bool {
    use std::sync::OnceLock;
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("FEATHERTALK_STEP_PROFILE").is_some())
}

/// Runs one optimizer step over a single-frame batch.
pub fn train_single_frame_step<M, O, E>(
    model: M,
    optimizer: &mut O,
    extractor: &E,
    batch: SingleFrameBatch,
    config: &TrainingConfig,
) -> Result<(M, LossValues), TrainingError>
where
    M: TrainableTalkingHead + Module,
    O: CheckpointableOptimizer,
    E: PerceptualFeatureExtractor,
{
    let (model, values, _) =
        train_single_frame_step_profiled(model, optimizer, extractor, batch, config)?;
    Ok((model, values))
}

pub(crate) fn train_single_frame_step_profiled<M, O, E>(
    model: M,
    optimizer: &mut O,
    extractor: &E,
    batch: SingleFrameBatch,
    config: &TrainingConfig,
) -> Result<(M, LossValues, StepComputeTimings), TrainingError>
where
    M: TrainableTalkingHead + Module,
    O: CheckpointableOptimizer,
    E: PerceptualFeatureExtractor,
{
    let forward_started = Instant::now();
    let prediction = model.forward_training(batch.image, batch.audio);
    let breakdown = match config.mode {
        TrainingMode::Baseline => {
            let loss_config = BaselineLossConfig {
                perceptual_weight: config.perceptual_weight};
            baseline_loss(extractor, prediction, batch.target, &loss_config)?
        }
        TrainingMode::MouthRoi => {
            let loss_config = MouthRoiLossConfig {
                mouth_weight: config.mouth_weight,
                perceptual_weight: config.perceptual_weight};
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
    commit_gradients(
        model,
        optimizer,
        breakdown,
        config.learning_rate,
        forward_started,
    )
}

/// Runs one optimizer step over a temporal pair batch.
pub fn train_temporal_step<M, O, E>(
    model: M,
    optimizer: &mut O,
    extractor: &E,
    batch: TemporalBatch,
    config: &TrainingConfig,
) -> Result<(M, LossValues), TrainingError>
where
    M: TrainableTalkingHead + Module,
    O: CheckpointableOptimizer,
    E: PerceptualFeatureExtractor,
{
    let (model, values, _) =
        train_temporal_step_profiled(model, optimizer, extractor, batch, config)?;
    Ok((model, values))
}

pub(crate) fn train_temporal_step_profiled<M, O, E>(
    model: M,
    optimizer: &mut O,
    extractor: &E,
    batch: TemporalBatch,
    config: &TrainingConfig,
) -> Result<(M, LossValues, StepComputeTimings), TrainingError>
where
    M: TrainableTalkingHead + Module,
    O: CheckpointableOptimizer,
    E: PerceptualFeatureExtractor,
{
    if config.mode != TrainingMode::MouthRoiTemporal {
        return Err(TrainingError::InvalidConfig(
            "the non-temporal modes need train_single_frame_step".to_owned(),
        ));
    }
    let [pairs, pair_len, ..] = batch.target.dims();
    let forward_started = Instant::now();
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
        perceptual_weight: config.perceptual_weight};
    let breakdown = temporal_loss(
        extractor,
        prediction,
        batch.target,
        batch.mouth_mask,
        &loss_config,
    )?;
    commit_gradients(
        model,
        optimizer,
        breakdown,
        config.learning_rate,
        forward_started,
    )
}
