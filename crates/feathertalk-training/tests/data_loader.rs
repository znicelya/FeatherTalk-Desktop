use feathertalk_training::{
    DATA_LOADER_STATE_SCHEMA_VERSION, DataLoaderConfig, DataLoaderState, PreparedBatch,
    RandomAlgorithm, SamplingConfig, SamplingKind, TrainingDataLoader, TrainingDataset,
    TrainingError, TrainingSample,
};
use std::{cell::Cell, rc::Rc};

#[derive(Debug)]
struct PlanDataset {
    frames: u64,
}

impl TrainingDataset for PlanDataset {
    type Item = TrainingSample;

    fn frame_count(&self) -> u64 {
        self.frames
    }

    fn load_sample(&self, sample: &TrainingSample) -> Result<Self::Item, TrainingError> {
        Ok(sample.clone())
    }
}

fn collect_committed(
    loader: &mut TrainingDataLoader<PlanDataset>,
    batch_count: usize,
) -> Vec<TrainingSample> {
    let mut samples = Vec::new();
    for _ in 0..batch_count {
        let prepared = loader.prepare_next_batch().unwrap();
        samples.extend_from_slice(prepared.samples());
        loader.commit_batch(prepared).unwrap();
    }
    samples
}

fn assert_prepared_batch_api(batch: &PreparedBatch<TrainingSample>) {
    assert_eq!(batch.samples(), batch.items());
}

#[test]
fn single_and_temporal_configs_expose_the_fixed_contract() {
    assert_eq!(DATA_LOADER_STATE_SCHEMA_VERSION, 1);
    assert_eq!(
        DataLoaderConfig::single_frame(4, 7),
        DataLoaderConfig {
            batch_size: 4,
            seed: 7,
            sampling: SamplingConfig {
                kind: SamplingKind::SingleFrame,
                temporal_stride: 0,
            },
        }
    );
    assert_eq!(
        DataLoaderConfig::temporal_pair(3, 42, 2),
        DataLoaderConfig {
            batch_size: 3,
            seed: 42,
            sampling: SamplingConfig {
                kind: SamplingKind::TemporalPair,
                temporal_stride: 2,
            },
        }
    );
}

#[test]
fn invalid_config_and_state_are_rejected_before_loading() {
    let invalid = [
        DataLoaderConfig::single_frame(0, 7),
        DataLoaderConfig {
            batch_size: 2,
            seed: 7,
            sampling: SamplingConfig {
                kind: SamplingKind::SingleFrame,
                temporal_stride: 1,
            },
        },
        DataLoaderConfig::temporal_pair(2, 7, 0),
    ];
    for config in invalid {
        assert!(matches!(
            config.validate(5),
            Err(TrainingError::InvalidDataLoaderConfig(_))
        ));
    }

    assert!(matches!(
        DataLoaderConfig::single_frame(2, 7).validate(0),
        Err(TrainingError::InvalidDataLoaderConfig(_))
    ));
    assert!(matches!(
        DataLoaderConfig::temporal_pair(2, 7, 5).validate(5),
        Err(TrainingError::InvalidDataLoaderConfig(_))
    ));

    let unsupported_schema = DataLoaderState {
        schema_version: 99,
        random_algorithm: RandomAlgorithm::Splitmix64FisherYatesV1,
        config: DataLoaderConfig::single_frame(2, 7),
        frame_count: 5,
        epoch: 0,
        next_position: 0,
    };
    assert!(matches!(
        unsupported_schema.validate(5),
        Err(TrainingError::InvalidDataLoaderState(_))
    ));

    let cursor_at_end = DataLoaderState {
        schema_version: DATA_LOADER_STATE_SCHEMA_VERSION,
        random_algorithm: RandomAlgorithm::Splitmix64FisherYatesV1,
        config: DataLoaderConfig::single_frame(2, 7),
        frame_count: 5,
        epoch: 0,
        next_position: 5,
    };
    assert!(matches!(
        cursor_at_end.validate(5),
        Err(TrainingError::InvalidDataLoaderState(_))
    ));

    let invalid_temporal_state = DataLoaderState {
        schema_version: DATA_LOADER_STATE_SCHEMA_VERSION,
        random_algorithm: RandomAlgorithm::Splitmix64FisherYatesV1,
        config: DataLoaderConfig::temporal_pair(2, 7, 5),
        frame_count: 5,
        epoch: 0,
        next_position: 0,
    };
    assert!(matches!(
        invalid_temporal_state.validate(5),
        Err(TrainingError::InvalidDataLoaderConfig(_))
    ));
}

#[test]
fn fixed_single_frame_and_temporal_sample_plans_match_version_one() {
    let mut single = TrainingDataLoader::new(
        PlanDataset { frames: 5 },
        DataLoaderConfig::single_frame(2, 7),
    )
    .unwrap();
    assert_eq!(
        collect_committed(&mut single, 3),
        vec![
            TrainingSample::SingleFrame {
                target_index: 0,
                reference_index: 1,
            },
            TrainingSample::SingleFrame {
                target_index: 4,
                reference_index: 2,
            },
            TrainingSample::SingleFrame {
                target_index: 2,
                reference_index: 4,
            },
            TrainingSample::SingleFrame {
                target_index: 1,
                reference_index: 2,
            },
            TrainingSample::SingleFrame {
                target_index: 3,
                reference_index: 2,
            },
        ]
    );
    assert_eq!(
        collect_committed(&mut single, 3),
        vec![
            TrainingSample::SingleFrame {
                target_index: 0,
                reference_index: 0,
            },
            TrainingSample::SingleFrame {
                target_index: 3,
                reference_index: 4,
            },
            TrainingSample::SingleFrame {
                target_index: 1,
                reference_index: 1,
            },
            TrainingSample::SingleFrame {
                target_index: 4,
                reference_index: 0,
            },
            TrainingSample::SingleFrame {
                target_index: 2,
                reference_index: 0,
            },
        ]
    );

    let mut temporal = TrainingDataLoader::new(
        PlanDataset { frames: 8 },
        DataLoaderConfig::temporal_pair(4, 42, 2),
    )
    .unwrap();
    assert_eq!(
        collect_committed(&mut temporal, 2),
        vec![
            TrainingSample::TemporalPair {
                first_target_index: 2,
                second_target_index: 4,
                reference_index: 4,
            },
            TrainingSample::TemporalPair {
                first_target_index: 0,
                second_target_index: 2,
                reference_index: 1,
            },
            TrainingSample::TemporalPair {
                first_target_index: 4,
                second_target_index: 6,
                reference_index: 6,
            },
            TrainingSample::TemporalPair {
                first_target_index: 1,
                second_target_index: 3,
                reference_index: 6,
            },
            TrainingSample::TemporalPair {
                first_target_index: 3,
                second_target_index: 5,
                reference_index: 4,
            },
            TrainingSample::TemporalPair {
                first_target_index: 5,
                second_target_index: 7,
                reference_index: 5,
            },
        ]
    );

    let self_reference = TrainingDataLoader::new(
        PlanDataset { frames: 5 },
        DataLoaderConfig::single_frame(5, 0),
    )
    .unwrap()
    .prepare_next_batch()
    .unwrap();
    assert_eq!(
        self_reference.samples()[1],
        TrainingSample::SingleFrame {
            target_index: 0,
            reference_index: 0,
        }
    );
}

#[test]
fn commit_advances_only_after_success_and_preserves_partial_batches() {
    let mut loader = TrainingDataLoader::new(
        PlanDataset { frames: 5 },
        DataLoaderConfig::single_frame(2, 7),
    )
    .unwrap();

    let first = loader.prepare_next_batch().unwrap();
    assert_prepared_batch_api(&first);
    assert_eq!(first.epoch(), 0);
    assert_eq!(first.start_position(), 0);
    assert_eq!(first.samples().len(), 2);
    assert_eq!(loader.state().next_position, 0);
    loader.commit_batch(first).unwrap();
    assert_eq!(loader.state().next_position, 2);

    let second = loader.prepare_next_batch().unwrap();
    loader.commit_batch(second).unwrap();
    let tail = loader.prepare_next_batch().unwrap();
    assert_eq!(tail.start_position(), 4);
    assert_eq!(tail.samples().len(), 1);
    loader.commit_batch(tail).unwrap();
    assert_eq!((loader.state().epoch, loader.state().next_position), (1, 0));

    let mut oversized = TrainingDataLoader::new(
        PlanDataset { frames: 5 },
        DataLoaderConfig::single_frame(8, 7),
    )
    .unwrap();
    let only_batch = oversized.prepare_next_batch().unwrap();
    assert_eq!(only_batch.samples().len(), 5);
    oversized.commit_batch(only_batch).unwrap();
    assert_eq!(
        (oversized.state().epoch, oversized.state().next_position),
        (1, 0)
    );
}

#[test]
fn repeated_prepare_stale_and_foreign_commits_do_not_change_state() {
    let mut loader = TrainingDataLoader::new(
        PlanDataset { frames: 5 },
        DataLoaderConfig::single_frame(2, 7),
    )
    .unwrap();
    let before = loader.state().clone();
    let first = loader.prepare_next_batch().unwrap();
    let duplicate = loader.prepare_next_batch().unwrap();
    assert_eq!(first.samples(), duplicate.samples());
    assert_eq!(loader.state(), &before);

    loader.commit_batch(first).unwrap();
    let after_first = loader.state().clone();
    assert!(matches!(
        loader.commit_batch(duplicate),
        Err(TrainingError::StalePreparedBatch)
    ));
    assert_eq!(loader.state(), &after_first);

    let mut target = TrainingDataLoader::new(
        PlanDataset { frames: 5 },
        DataLoaderConfig::single_frame(2, 7),
    )
    .unwrap();
    let source = TrainingDataLoader::new(
        PlanDataset { frames: 5 },
        DataLoaderConfig::single_frame(2, 7),
    )
    .unwrap();
    let foreign = source.prepare_next_batch().unwrap();
    let target_before = target.state().clone();
    assert!(matches!(
        target.commit_batch(foreign),
        Err(TrainingError::StalePreparedBatch)
    ));
    assert_eq!(target.state(), &target_before);
}

struct FailingDataset {
    frames: u64,
    calls: Rc<Cell<usize>>,
    fail_at: usize,
}

impl TrainingDataset for FailingDataset {
    type Item = TrainingSample;

    fn frame_count(&self) -> u64 {
        self.frames
    }

    fn load_sample(&self, sample: &TrainingSample) -> Result<Self::Item, TrainingError> {
        let call = self.calls.get() + 1;
        self.calls.set(call);
        if call == self.fail_at {
            return Err(TrainingError::InvalidInput(
                "injected dataset failure".into(),
            ));
        }
        Ok(sample.clone())
    }
}

#[test]
fn dataset_failure_during_prepare_leaves_state_unchanged() {
    let calls = Rc::new(Cell::new(0));
    let loader = TrainingDataLoader::new(
        FailingDataset {
            frames: 5,
            calls: calls.clone(),
            fail_at: 2,
        },
        DataLoaderConfig::single_frame(3, 7),
    )
    .unwrap();
    let before = loader.state().clone();

    assert!(matches!(
        loader.prepare_next_batch(),
        Err(TrainingError::InvalidInput(message)) if message == "injected dataset failure"
    ));
    assert_eq!(calls.get(), 2);
    assert_eq!(loader.state(), &before);
}

#[test]
fn sample_count_is_public_for_each_sampling_kind() {
    assert_eq!(
        DataLoaderConfig::single_frame(4, 7)
            .sample_count(10)
            .unwrap(),
        10
    );
    assert_eq!(
        DataLoaderConfig::temporal_pair(4, 7, 3)
            .sample_count(10)
            .unwrap(),
        7
    );
}

#[test]
fn the_loader_lends_out_its_dataset() {
    let loader = TrainingDataLoader::new(
        PlanDataset { frames: 6 },
        DataLoaderConfig::single_frame(2, 7),
    )
    .unwrap();
    assert_eq!(loader.dataset().frame_count(), 6);
}
