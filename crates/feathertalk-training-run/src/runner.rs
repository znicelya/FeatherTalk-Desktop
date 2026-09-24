use std::{collections::BTreeMap, path::Path, thread::JoinHandle, time::Duration};

use burn::module::Module;
use feathertalk_models::unet::TrainableTalkingHead;
use feathertalk_training::PreparedBatchWithTiming;
use feathertalk_training::{
    CheckpointDescriptor, PerceptualFeatureExtractor, Provenance, RestoredTrainingState,
    TRAINING_STATE_SCHEMA_VERSION, TrainingCheckpointManifest, TrainingCheckpointState,
    TrainingConfig, TrainingDataLoader, TrainingDataset, TrainingError, TrainingMetrics,
    TrainingMode, save_training_checkpoint};
use feathertalk_training_data::{TrainingItem, stack_single_frame_batch, stack_temporal_batch};

use crate::step::{train_single_frame_step_profiled, train_temporal_step_profiled};
use crate::{LossValues, data_loader_config_for};

const POISONED: &str = "training runner was poisoned by a failed step";

fn poisoned() -> TrainingError {
    TrainingError::InvalidInput(POISONED.to_owned())
}

fn overflow(operation: &'static str) -> TrainingError {
    TrainingError::DataLoaderOverflow { operation }
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct StepTimings {
    pub load_secs: f64,
    pub prefetch_wait_secs: f64,
    pub forward_secs: f64,
    pub backward_secs: f64,
    pub optim_secs: f64,
    pub total_secs: f64}

/// What one committed optimizer step did.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StepReport {
    pub epoch: u64,
    pub global_step: u64,
    pub samples_in_batch: u64,
    pub losses: LossValues,
    pub timings: StepTimings}

/// Owns a training run: the loader, the model, the optimizer, and the progress counters.
pub struct TrainingRunner<M, O, D>
where
    D: TrainingDataset<Item = TrainingItem> + Send + Sync + 'static,
{
    model: Option<M>,
    optimizer: O,
    loader: TrainingDataLoader<D>,
    config: TrainingConfig,
    device: burn::tensor::Device,
    global_step: u64,
    samples_seen: u64,
    prefetch: Option<JoinHandle<Result<PreparedBatchWithTiming<D::Item>, TrainingError>>>,
    prefetch_enabled: bool}

impl<M, O, D> TrainingRunner<M, O, D>
where
    M: TrainableTalkingHead + Module + Clone,
    O: feathertalk_training::CheckpointableOptimizer + Clone,
    D: TrainingDataset<Item = TrainingItem> + Send + Sync + 'static,
{
    pub fn new(
        dataset: D,
        model: M,
        optimizer: O,
        config: TrainingConfig,
        seed: u64,
        device: burn::tensor::Device,
    ) -> Result<Self, TrainingError> {
        let loader_config = data_loader_config_for(&config, seed)?;
        let loader = TrainingDataLoader::new(dataset, loader_config)?;
        Ok(Self {
            model: Some(model),
            optimizer,
            loader,
            config,
            device,
            global_step: 0,
            samples_seen: 0,
            prefetch: None,
            prefetch_enabled: prefetch_enabled()})
    }

    fn run_step<E>(
        &mut self,
        model: M,
        items: &[TrainingItem],
        extractor: &E,
    ) -> Result<(M, LossValues, crate::StepComputeTimings), TrainingError>
    where
        E: PerceptualFeatureExtractor,
    {
        let config = &self.config;
        match config.mode {
            TrainingMode::Baseline | TrainingMode::MouthRoi => {
                let batch = stack_single_frame_batch(items, &self.device)?;
                train_single_frame_step_profiled(
                    model,
                    &mut self.optimizer,
                    extractor,
                    batch,
                    config,
                )
            }
            TrainingMode::MouthRoiTemporal => {
                let batch = stack_temporal_batch(items, &self.device)?;
                train_temporal_step_profiled(model, &mut self.optimizer, extractor, batch, config)
            }
        }
    }

    /// Prepares one batch, trains on it, and commits the loader position.
    pub fn step<E>(&mut self, extractor: &E) -> Result<StepReport, TrainingError>
    where
        E: PerceptualFeatureExtractor,
    {
        let step_started = std::time::Instant::now();
        let (prepared, prefetch_wait_secs) = self.take_prepared()?;
        let load_secs = prepared.load_secs;
        let batch = prepared.batch;
        let epoch = batch.epoch();
        let samples_in_batch =
            u64::try_from(batch.items().len()).map_err(|_| overflow("counting batch items"))?;
        let model = self.model.take().ok_or_else(poisoned)?;
        let (model, losses, compute) = self.run_step(model, batch.items(), extractor)?;
        self.loader.commit_batch(batch)?;
        self.model = Some(model);
        self.global_step = self
            .global_step
            .checked_add(1)
            .ok_or_else(|| overflow("counting training steps"))?;
        self.samples_seen = self
            .samples_seen
            .checked_add(samples_in_batch)
            .ok_or_else(|| overflow("counting seen samples"))?;
        if self.prefetch_enabled && !self.is_finished() {
            self.prefetch = Some(self.loader.spawn_prefetch_batch()?);
        }
        let total_secs = step_started.elapsed().as_secs_f64();
        let report = StepReport {
            epoch,
            global_step: self.global_step,
            samples_in_batch,
            losses,
            timings: StepTimings {
                load_secs,
                prefetch_wait_secs,
                forward_secs: compute.forward_secs,
                backward_secs: compute.backward_secs,
                optim_secs: compute.optim_secs,
                total_secs}};
        log_step_profile(&report);
        Ok(report)
    }

    pub fn epoch(&self) -> u64 {
        self.loader.state().epoch
    }

    pub fn global_step(&self) -> u64 {
        self.global_step
    }

    pub fn samples_seen(&self) -> u64 {
        self.samples_seen
    }

    pub fn is_finished(&self) -> bool {
        self.epoch() >= self.config.total_epochs
    }

    pub fn training_config(&self) -> &TrainingConfig {
        &self.config
    }

    pub fn dataset(&self) -> &D {
        self.loader.dataset()
    }

    pub fn model(&self) -> Result<&M, TrainingError> {
        self.model.as_ref().ok_or_else(poisoned)
    }

    /// Turns one `StepReport` plus wall-clock elapsed time into wire metrics.
    pub fn metrics(
        &self,
        report: &StepReport,
        elapsed: Duration,
        gpu_memory_bytes: Option<u64>,
        worker_state: &str,
    ) -> Result<TrainingMetrics, TrainingError> {
        let state = self.loader.state();
        let sample_count = state.config.sample_count(state.frame_count)?;
        let total = self.config.total_epochs.saturating_mul(sample_count);
        let done = state
            .epoch
            .saturating_mul(sample_count)
            .saturating_add(state.next_position);
        let remaining = total.saturating_sub(done);
        let seconds = elapsed.as_secs_f64();
        let samples_per_second = if seconds > 0.0 {
            self.samples_seen as f64 / seconds
        } else {
            0.0
        };
        let estimated_remaining_seconds = if samples_per_second > 0.0 {
            remaining as f64 / samples_per_second
        } else {
            0.0
        };
        TrainingMetrics::new(
            self.config.mode,
            report.epoch,
            report.global_step,
            report.losses.total,
            report.losses.full,
            report.losses.perceptual,
            report.losses.mouth,
            report.losses.temporal,
            report.losses.temporal_mouth,
            self.samples_seen,
            samples_per_second,
            estimated_remaining_seconds,
            gpu_memory_bytes,
            worker_state,
        )
    }

    /// Snapshots everything a checkpoint needs besides the weights themselves.
    pub fn checkpoint_state(&self) -> TrainingCheckpointState {
        let state = self.loader.state();
        TrainingCheckpointState {
            schema_version: TRAINING_STATE_SCHEMA_VERSION,
            epoch: state.epoch,
            global_step: self.global_step,
            random_seed: state.config.seed,
            data_loader: state.clone(),
            training_config: self.config.clone(),
            asset_provenance: Provenance {
                entries: BTreeMap::new()},
            model_provenance: Provenance {
                entries: BTreeMap::new()}}
    }

    /// Writes a complete checkpoint directory: weights, optimizer, and state.
    pub fn save_checkpoint(
        &self,
        destination: impl AsRef<Path>,
        descriptor: CheckpointDescriptor,
    ) -> Result<TrainingCheckpointManifest, TrainingError> {
        let model = self.model()?;
        save_training_checkpoint::<M, O>(
            destination,
            model,
            &self.optimizer,
            descriptor,
            self.checkpoint_state(),
        )
    }

    /// Rebuilds a runner from a loaded checkpoint and a freshly opened dataset.
    pub fn restore(
        dataset: D,
        restored: RestoredTrainingState<M, O>,
        device: burn::tensor::Device,
    ) -> Result<Self, TrainingError> {
        restored.state.validate()?;
        let global_step = restored.state.global_step;
        let loader = TrainingDataLoader::restore(dataset, restored.state.data_loader)?;
        Ok(Self {
            model: Some(restored.model),
            optimizer: restored.optimizer,
            loader,
            config: restored.state.training_config,
            device,
            global_step,
            samples_seen: 0,
            prefetch: None,
            prefetch_enabled: prefetch_enabled()})
    }

    fn take_prepared(&mut self) -> Result<(PreparedBatchWithTiming<D::Item>, f64), TrainingError> {
        if self.prefetch_enabled {
            if let Some(handle) = self.prefetch.take() {
                let started = std::time::Instant::now();
                let prepared = handle.join().unwrap_or_else(|_| {
                    Err(TrainingError::InvalidInput("prefetch worker panic".into()))
                })?;
                return Ok((prepared, started.elapsed().as_secs_f64()));
            }
            return Ok((self.loader.prepare_next_batch_parallel()?, 0.0));
        }
        Ok((self.loader.prepare_next_batch_with_timing()?, 0.0))
    }
}

fn prefetch_enabled() -> bool {
    match std::env::var("FEATHERTALK_TRAIN_PREFETCH") {
        Ok(value) => !matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "0" | "false" | "off" | "no"
        ),
        Err(_) => true}
}

fn log_step_profile(report: &StepReport) {
    if std::env::var_os("FEATHERTALK_STEP_PROFILE").is_none() {
        return;
    }
    let t = report.timings;
    eprintln!(
        "step={} load={:.4} wait={:.4} fwd={:.4} bwd={:.4} optim={:.4} total={:.4}",
        report.global_step,
        t.load_secs,
        t.prefetch_wait_secs,
        t.forward_secs,
        t.backward_secs,
        t.optim_secs,
        t.total_secs,
    );
}
