use std::sync::atomic::{AtomicU64, Ordering};

use serde::{Deserialize, Serialize};

use crate::{
    TrainingError,
    random::{epoch_permutation, reference_index as choose_reference_index},
};

pub const DATA_LOADER_STATE_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RandomAlgorithm {
    Splitmix64FisherYatesV1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SamplingKind {
    SingleFrame,
    TemporalPair,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SamplingConfig {
    pub kind: SamplingKind,
    pub temporal_stride: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DataLoaderConfig {
    pub batch_size: u64,
    pub seed: u64,
    pub sampling: SamplingConfig,
}

impl DataLoaderConfig {
    pub const fn single_frame(batch_size: u64, seed: u64) -> Self {
        Self {
            batch_size,
            seed,
            sampling: SamplingConfig {
                kind: SamplingKind::SingleFrame,
                temporal_stride: 0,
            },
        }
    }

    pub const fn temporal_pair(batch_size: u64, seed: u64, temporal_stride: u64) -> Self {
        Self {
            batch_size,
            seed,
            sampling: SamplingConfig {
                kind: SamplingKind::TemporalPair,
                temporal_stride,
            },
        }
    }

    pub fn validate(&self, frame_count: u64) -> Result<(), TrainingError> {
        self.sample_count(frame_count).map(|_| ())
    }

    /// Samples per epoch: `frame_count` for single frames, `frame_count - stride` for pairs.
    pub fn sample_count(&self, frame_count: u64) -> Result<u64, TrainingError> {
        if self.batch_size == 0 {
            return Err(TrainingError::InvalidDataLoaderConfig(
                "batch_size must be greater than zero".into(),
            ));
        }
        usize::try_from(self.batch_size).map_err(|_| TrainingError::DataLoaderOverflow {
            operation: "converting batch size",
        })?;
        if frame_count == 0 {
            return Err(TrainingError::InvalidDataLoaderConfig(
                "frame_count must be greater than zero".into(),
            ));
        }
        usize::try_from(frame_count).map_err(|_| TrainingError::DataLoaderOverflow {
            operation: "converting frame count",
        })?;

        let sample_count = match self.sampling.kind {
            SamplingKind::SingleFrame => {
                if self.sampling.temporal_stride != 0 {
                    return Err(TrainingError::InvalidDataLoaderConfig(
                        "single_frame requires temporal_stride zero".into(),
                    ));
                }
                frame_count
            }
            SamplingKind::TemporalPair => {
                let stride = self.sampling.temporal_stride;
                if stride == 0 || stride >= frame_count {
                    return Err(TrainingError::InvalidDataLoaderConfig(
                        "temporal_pair requires 1 <= temporal_stride < frame_count".into(),
                    ));
                }
                frame_count
                    .checked_sub(stride)
                    .ok_or(TrainingError::DataLoaderOverflow {
                        operation: "computing temporal sample count",
                    })?
            }
        };
        usize::try_from(sample_count).map_err(|_| TrainingError::DataLoaderOverflow {
            operation: "converting sample count",
        })?;
        Ok(sample_count)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DataLoaderState {
    pub schema_version: u32,
    pub random_algorithm: RandomAlgorithm,
    pub config: DataLoaderConfig,
    pub frame_count: u64,
    pub epoch: u64,
    pub next_position: u64,
}

impl DataLoaderState {
    pub fn validate(&self, dataset_frame_count: u64) -> Result<(), TrainingError> {
        if self.schema_version != DATA_LOADER_STATE_SCHEMA_VERSION {
            return Err(TrainingError::InvalidDataLoaderState(
                "unsupported schema_version".into(),
            ));
        }
        if self.random_algorithm != RandomAlgorithm::Splitmix64FisherYatesV1 {
            return Err(TrainingError::InvalidDataLoaderState(
                "unsupported random_algorithm".into(),
            ));
        }
        if self.frame_count != dataset_frame_count {
            return Err(TrainingError::InvalidDataLoaderState(
                "dataset frame_count does not match saved state".into(),
            ));
        }
        let sample_count = self.config.sample_count(self.frame_count)?;
        if self.next_position >= sample_count {
            return Err(TrainingError::InvalidDataLoaderState(
                "next_position must be inside the current epoch".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TrainingSample {
    SingleFrame {
        target_index: u64,
        reference_index: u64,
    },
    TemporalPair {
        first_target_index: u64,
        second_target_index: u64,
        reference_index: u64,
    },
}

pub trait TrainingDataset {
    type Item;

    fn frame_count(&self) -> u64;

    fn load_sample(&self, sample: &TrainingSample) -> Result<Self::Item, TrainingError>;
}

pub struct PreparedBatch<T> {
    loader_id: u64,
    epoch: u64,
    start_position: u64,
    end_position: u64,
    samples: Vec<TrainingSample>,
    items: Vec<T>,
    next_epoch: Option<u64>,
    next_permutation: Option<Vec<u64>>,
}

impl<T> PreparedBatch<T> {
    pub fn epoch(&self) -> u64 {
        self.epoch
    }

    pub fn start_position(&self) -> u64 {
        self.start_position
    }

    pub fn samples(&self) -> &[TrainingSample] {
        &self.samples
    }

    pub fn items(&self) -> &[T] {
        &self.items
    }
}

pub struct TrainingDataLoader<D: TrainingDataset> {
    dataset: D,
    state: DataLoaderState,
    sample_count: u64,
    permutation: Vec<u64>,
    loader_id: u64,
}

static NEXT_LOADER_ID: AtomicU64 = AtomicU64::new(1);

fn allocate_loader_id() -> Result<u64, TrainingError> {
    NEXT_LOADER_ID
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
            current.checked_add(1)
        })
        .map_err(|_| TrainingError::DataLoaderOverflow {
            operation: "allocating loader identifier",
        })
}

impl<D: TrainingDataset> TrainingDataLoader<D> {
    pub fn new(dataset: D, config: DataLoaderConfig) -> Result<Self, TrainingError> {
        let frame_count = dataset.frame_count();
        let sample_count = config.sample_count(frame_count)?;
        let state = DataLoaderState {
            schema_version: DATA_LOADER_STATE_SCHEMA_VERSION,
            random_algorithm: RandomAlgorithm::Splitmix64FisherYatesV1,
            config,
            frame_count,
            epoch: 0,
            next_position: 0,
        };
        let permutation = epoch_permutation(sample_count, config.seed, 0)?;
        let loader_id = allocate_loader_id()?;
        Ok(Self {
            dataset,
            state,
            sample_count,
            permutation,
            loader_id,
        })
    }

    pub fn restore(dataset: D, state: DataLoaderState) -> Result<Self, TrainingError> {
        let dataset_frame_count = dataset.frame_count();
        state.validate(dataset_frame_count)?;
        let sample_count = state.config.sample_count(state.frame_count)?;
        let permutation = epoch_permutation(sample_count, state.config.seed, state.epoch)?;
        let loader_id = allocate_loader_id()?;
        Ok(Self {
            dataset,
            state,
            sample_count,
            permutation,
            loader_id,
        })
    }

    pub fn state(&self) -> &DataLoaderState {
        &self.state
    }

    /// Lends out the dataset this loader samples from.
    pub fn dataset(&self) -> &D {
        &self.dataset
    }

    pub fn prepare_next_batch(&self) -> Result<PreparedBatch<D::Item>, TrainingError> {
        let start_position = self.state.next_position;
        let remaining = self
            .sample_count
            .checked_sub(start_position)
            .ok_or_else(|| {
                TrainingError::InvalidDataLoaderState("next_position exceeds sample_count".into())
            })?;
        let batch_items = self.state.config.batch_size.min(remaining);
        let end_position =
            start_position
                .checked_add(batch_items)
                .ok_or(TrainingError::DataLoaderOverflow {
                    operation: "computing batch end",
                })?;
        let batch_length =
            usize::try_from(batch_items).map_err(|_| TrainingError::DataLoaderOverflow {
                operation: "converting batch length",
            })?;

        let mut samples = Vec::new();
        samples.try_reserve_exact(batch_length).map_err(|source| {
            TrainingError::BatchAllocation {
                items: batch_items,
                source,
            }
        })?;
        for position in start_position..end_position {
            samples.push(self.sample_at_position(position)?);
        }

        let (next_epoch, next_permutation) = if end_position == self.sample_count {
            let next_epoch =
                self.state
                    .epoch
                    .checked_add(1)
                    .ok_or(TrainingError::DataLoaderOverflow {
                        operation: "advancing epoch",
                    })?;
            let next_permutation =
                epoch_permutation(self.sample_count, self.state.config.seed, next_epoch)?;
            (Some(next_epoch), Some(next_permutation))
        } else {
            (None, None)
        };

        let mut items = Vec::new();
        items
            .try_reserve_exact(batch_length)
            .map_err(|source| TrainingError::BatchAllocation {
                items: batch_items,
                source,
            })?;
        for sample in &samples {
            items.push(self.dataset.load_sample(sample)?);
        }

        Ok(PreparedBatch {
            loader_id: self.loader_id,
            epoch: self.state.epoch,
            start_position,
            end_position,
            samples,
            items,
            next_epoch,
            next_permutation,
        })
    }

    pub fn commit_batch(&mut self, prepared: PreparedBatch<D::Item>) -> Result<(), TrainingError> {
        let remaining = self
            .sample_count
            .checked_sub(self.state.next_position)
            .ok_or(TrainingError::StalePreparedBatch)?;
        let expected_items = self.state.config.batch_size.min(remaining);
        let expected_end = self
            .state
            .next_position
            .checked_add(expected_items)
            .ok_or(TrainingError::StalePreparedBatch)?;
        let expected_length =
            usize::try_from(expected_items).map_err(|_| TrainingError::StalePreparedBatch)?;
        let is_final = expected_end == self.sample_count;

        if prepared.loader_id != self.loader_id
            || prepared.epoch != self.state.epoch
            || prepared.start_position != self.state.next_position
            || prepared.end_position != expected_end
            || prepared.samples.len() != expected_length
            || prepared.items.len() != expected_length
        {
            return Err(TrainingError::StalePreparedBatch);
        }

        if is_final {
            let expected_next_epoch = self
                .state
                .epoch
                .checked_add(1)
                .ok_or(TrainingError::StalePreparedBatch)?;
            let Some(next_epoch) = prepared.next_epoch else {
                return Err(TrainingError::StalePreparedBatch);
            };
            let Some(next_permutation) = prepared.next_permutation else {
                return Err(TrainingError::StalePreparedBatch);
            };
            if next_epoch != expected_next_epoch || next_permutation.len() != self.permutation.len()
            {
                return Err(TrainingError::StalePreparedBatch);
            }
            self.state.epoch = next_epoch;
            self.state.next_position = 0;
            self.permutation = next_permutation;
        } else {
            if prepared.next_epoch.is_some() || prepared.next_permutation.is_some() {
                return Err(TrainingError::StalePreparedBatch);
            }
            self.state.next_position = expected_end;
        }
        Ok(())
    }

    fn sample_at_position(&self, position: u64) -> Result<TrainingSample, TrainingError> {
        let index = usize::try_from(position).map_err(|_| TrainingError::DataLoaderOverflow {
            operation: "converting sample position",
        })?;
        let target_index = *self.permutation.get(index).ok_or_else(|| {
            TrainingError::InvalidDataLoaderState(
                "sample position is outside the epoch permutation".into(),
            )
        })?;
        let reference_index = choose_reference_index(
            self.state.frame_count,
            self.state.config.seed,
            self.state.epoch,
            position,
        )?;
        match self.state.config.sampling.kind {
            SamplingKind::SingleFrame => Ok(TrainingSample::SingleFrame {
                target_index,
                reference_index,
            }),
            SamplingKind::TemporalPair => {
                let second_target_index = target_index
                    .checked_add(self.state.config.sampling.temporal_stride)
                    .ok_or(TrainingError::DataLoaderOverflow {
                        operation: "computing temporal second target",
                    })?;
                if second_target_index >= self.state.frame_count {
                    return Err(TrainingError::InvalidDataLoaderState(
                        "temporal target is outside frame_count".into(),
                    ));
                }
                Ok(TrainingSample::TemporalPair {
                    first_target_index: target_index,
                    second_target_index,
                    reference_index,
                })
            }
        }
    }
}
