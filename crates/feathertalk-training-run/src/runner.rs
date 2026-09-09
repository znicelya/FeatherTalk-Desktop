use std::{collections::BTreeMap, path::Path, time::Duration};

use burn::{module::AutodiffModule, optim::Optimizer, tensor::backend::AutodiffBackend};
use feathertalk_models::unet::TrainableTalkingHead;
use feathertalk_training::{
    CheckpointDescriptor, PerceptualFeatureExtractor, Provenance, RestoredTrainingState,
    TRAINING_STATE_SCHEMA_VERSION, TrainingCheckpointManifest, TrainingCheckpointState,
    TrainingConfig, TrainingDataLoader, TrainingDataset, TrainingError, TrainingMetrics,
    TrainingMode, save_training_checkpoint,
};
use feathertalk_training_data::{TrainingItem, stack_single_frame_batch, stack_temporal_batch};

use crate::{LossValues, data_loader_config_for, train_single_frame_step, train_temporal_step};

const POISONED: &str = "training runner was poisoned by a failed step";

fn poisoned() -> TrainingError {
    TrainingError::InvalidInput(POISONED.to_owned())
}

fn overflow(operation: &'static str) -> TrainingError {
    TrainingError::DataLoaderOverflow { operation }
}

/// What one committed optimizer step did.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StepReport {
    pub epoch: u64,
    pub global_step: u64,
    pub samples_in_batch: u64,
    pub losses: LossValues,
}

/// Owns a training run: the loader, the model, the optimizer, and the progress counters.
pub struct TrainingRunner<B, M, O, D>
where
    B: AutodiffBackend,
    D: TrainingDataset<Item = TrainingItem>,
{
    model: Option<M>,
    optimizer: O,
    loader: TrainingDataLoader<D>,
    config: TrainingConfig,
    device: B::Device,
    global_step: u64,
    samples_seen: u64,
}

impl<B, M, O, D> TrainingRunner<B, M, O, D>
where
    B: AutodiffBackend,
    M: TrainableTalkingHead<B> + AutodiffModule<B> + Clone,
    O: Optimizer<M, B> + Clone,
    D: TrainingDataset<Item = TrainingItem>,
{
    pub fn new(
        dataset: D,
        model: M,
        optimizer: O,
        config: TrainingConfig,
        seed: u64,
        device: B::Device,
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
        })
    }

    fn run_step<E>(
        &mut self,
        model: M,
        items: &[TrainingItem],
        extractor: &E,
    ) -> Result<(M, LossValues), TrainingError>
    where
        E: PerceptualFeatureExtractor<B>,
    {
        let config = &self.config;
        match config.mode {
            TrainingMode::Baseline | TrainingMode::MouthRoi => {
                let batch = stack_single_frame_batch::<B>(items, &self.device)?;
                train_single_frame_step(model, &mut self.optimizer, extractor, batch, config)
            }
            TrainingMode::MouthRoiTemporal => {
                let batch = stack_temporal_batch::<B>(items, &self.device)?;
                train_temporal_step(model, &mut self.optimizer, extractor, batch, config)
            }
        }
    }

    /// Prepares one batch, trains on it, and commits the loader position.
    pub fn step<E>(&mut self, extractor: &E) -> Result<StepReport, TrainingError>
    where
        E: PerceptualFeatureExtractor<B>,
    {
        let prepared = self.loader.prepare_next_batch()?;
        let epoch = prepared.epoch();
        let samples_in_batch =
            u64::try_from(prepared.items().len()).map_err(|_| overflow("counting batch items"))?;
        let model = self.model.take().ok_or_else(poisoned)?;
        let (model, losses) = self.run_step(model, prepared.items(), extractor)?;
        self.loader.commit_batch(prepared)?;
        self.model = Some(model);
        self.global_step = self
            .global_step
            .checked_add(1)
            .ok_or_else(|| overflow("counting training steps"))?;
        self.samples_seen = self
            .samples_seen
            .checked_add(samples_in_batch)
            .ok_or_else(|| overflow("counting seen samples"))?;
        Ok(StepReport {
            epoch,
            global_step: self.global_step,
            samples_in_batch,
            losses,
        })
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
                entries: BTreeMap::new(),
            },
            model_provenance: Provenance {
                entries: BTreeMap::new(),
            },
        }
    }

    /// Writes a complete checkpoint directory: weights, optimizer, and state.
    pub fn save_checkpoint(
        &self,
        destination: impl AsRef<Path>,
        descriptor: CheckpointDescriptor,
    ) -> Result<TrainingCheckpointManifest, TrainingError> {
        let model = self.model()?;
        save_training_checkpoint::<B, M, O>(
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
        device: B::Device,
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
        })
    }
}
