use feathertalk_training::{
    DATA_LOADER_STATE_SCHEMA_VERSION, DataLoaderConfig, DataLoaderState, RandomAlgorithm,
    SamplingConfig, SamplingKind, TrainingDataLoader, TrainingDataset, TrainingError,
    TrainingSample,
};
use std::{cell::Cell, rc::Rc};

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

fn collect_committed<D>(
    loader: &mut TrainingDataLoader<D>,
    batch_count: usize,
) -> Vec<Vec<TrainingSample>>
where
    D: TrainingDataset,
{
    let mut batches = Vec::with_capacity(batch_count);
    for _ in 0..batch_count {
        let prepared = loader.prepare_next_batch().unwrap();
        batches.push(prepared.samples().to_vec());
        loader.commit_batch(prepared).unwrap();
    }
    batches
}

fn state(
    config: DataLoaderConfig,
    frame_count: u64,
    epoch: u64,
    next_position: u64,
) -> DataLoaderState {
    DataLoaderState {
        schema_version: DATA_LOADER_STATE_SCHEMA_VERSION,
        random_algorithm: RandomAlgorithm::Splitmix64FisherYatesV1,
        config,
        frame_count,
        epoch,
        next_position,
    }
}

#[test]
fn state_json_is_exact_strict_and_contains_no_permutation() {
    let state = state(DataLoaderConfig::single_frame(2, 7), 5, 3, 2);
    let json = serde_json::to_string(&state).unwrap();
    assert_eq!(
        json,
        r#"{"schema_version":1,"random_algorithm":"splitmix64_fisher_yates_v1","config":{"batch_size":2,"seed":7,"sampling":{"kind":"single_frame","temporal_stride":0}},"frame_count":5,"epoch":3,"next_position":2}"#
    );
    assert!(!json.contains("permutation"));
    assert_eq!(
        serde_json::from_str::<DataLoaderState>(&json).unwrap(),
        state
    );

    let unknown_root = format!("{},\"extra\":true}}", json.strip_suffix('}').unwrap());
    assert!(serde_json::from_str::<DataLoaderState>(&unknown_root).is_err());
    assert!(
        serde_json::from_str::<DataLoaderState>(&json.replacen(
            "\"batch_size\":2",
            "\"batch_size\":2,\"extra\":true",
            1,
        ))
        .is_err()
    );
    assert!(
        serde_json::from_str::<DataLoaderState>(&json.replacen(
            "\"temporal_stride\":0",
            "\"temporal_stride\":0,\"extra\":true",
            1,
        ))
        .is_err()
    );
    assert!(
        serde_json::from_str::<DataLoaderState>(&json.replacen(
            "splitmix64_fisher_yates_v1",
            "future_rng",
            1,
        ))
        .is_err()
    );
}

#[test]
fn single_frame_resume_matches_uninterrupted_batches_across_epochs() {
    let config = DataLoaderConfig::single_frame(2, 7);
    let mut uninterrupted = TrainingDataLoader::new(PlanDataset { frames: 5 }, config).unwrap();
    let expected = collect_committed(&mut uninterrupted, 8);

    let mut interrupted = TrainingDataLoader::new(PlanDataset { frames: 5 }, config).unwrap();
    let mut actual = collect_committed(&mut interrupted, 2);
    let json = serde_json::to_string(interrupted.state()).unwrap();
    let restored_state = serde_json::from_str::<DataLoaderState>(&json).unwrap();
    let mut resumed =
        TrainingDataLoader::restore(PlanDataset { frames: 5 }, restored_state).unwrap();
    actual.extend(collect_committed(&mut resumed, 6));

    assert_eq!(actual, expected);
}

#[test]
fn temporal_resume_matches_uninterrupted_tail_and_later_epochs() {
    let config = DataLoaderConfig::temporal_pair(4, 42, 2);
    let mut uninterrupted = TrainingDataLoader::new(PlanDataset { frames: 8 }, config).unwrap();
    let expected = collect_committed(&mut uninterrupted, 6);

    let mut interrupted = TrainingDataLoader::new(PlanDataset { frames: 8 }, config).unwrap();
    let mut actual = collect_committed(&mut interrupted, 1);
    let json = serde_json::to_string(interrupted.state()).unwrap();
    let restored_state = serde_json::from_str::<DataLoaderState>(&json).unwrap();
    let mut resumed =
        TrainingDataLoader::restore(PlanDataset { frames: 8 }, restored_state).unwrap();
    actual.extend(collect_committed(&mut resumed, 5));

    assert_eq!(actual, expected);
}

struct CountingDataset {
    frames: u64,
    loads: Rc<Cell<usize>>,
}

impl TrainingDataset for CountingDataset {
    type Item = TrainingSample;

    fn frame_count(&self) -> u64 {
        self.frames
    }

    fn load_sample(&self, sample: &TrainingSample) -> Result<Self::Item, TrainingError> {
        self.loads.set(self.loads.get() + 1);
        Ok(sample.clone())
    }
}

fn assert_restore_fails_without_loading(saved: DataLoaderState, dataset_frames: u64) {
    let loads = Rc::new(Cell::new(0));
    assert!(
        TrainingDataLoader::restore(
            CountingDataset {
                frames: dataset_frames,
                loads: loads.clone(),
            },
            saved,
        )
        .is_err()
    );
    assert_eq!(loads.get(), 0);
}

#[test]
fn strict_restore_rejects_incompatible_state_before_loading() {
    assert_restore_fails_without_loading(state(DataLoaderConfig::single_frame(2, 7), 5, 0, 0), 6);

    let mut unsupported_schema = state(DataLoaderConfig::single_frame(2, 7), 5, 0, 0);
    unsupported_schema.schema_version = 2;
    assert_restore_fails_without_loading(unsupported_schema, 5);

    assert_restore_fails_without_loading(state(DataLoaderConfig::single_frame(0, 7), 5, 0, 0), 5);
    assert_restore_fails_without_loading(
        state(
            DataLoaderConfig {
                batch_size: 2,
                seed: 7,
                sampling: SamplingConfig {
                    kind: SamplingKind::SingleFrame,
                    temporal_stride: 1,
                },
            },
            5,
            0,
            0,
        ),
        5,
    );
    assert_restore_fails_without_loading(
        state(DataLoaderConfig::temporal_pair(2, 7, 0), 5, 0, 0),
        5,
    );
    assert_restore_fails_without_loading(
        state(DataLoaderConfig::temporal_pair(2, 7, 5), 5, 0, 0),
        5,
    );
    assert_restore_fails_without_loading(state(DataLoaderConfig::single_frame(2, 7), 5, 0, 5), 5);
}

#[test]
fn impossible_restore_lengths_fail_before_loading() {
    let loads = Rc::new(Cell::new(0));
    let result = TrainingDataLoader::restore(
        CountingDataset {
            frames: u64::MAX,
            loads: loads.clone(),
        },
        state(DataLoaderConfig::single_frame(2, 7), u64::MAX, 0, 0),
    );
    assert!(matches!(
        result,
        Err(TrainingError::DataLoaderOverflow { .. })
            | Err(TrainingError::PermutationAllocation { .. })
    ));
    assert_eq!(loads.get(), 0);
}

#[test]
fn epoch_overflow_fails_before_loading_and_keeps_state() {
    let loads = Rc::new(Cell::new(0));
    let loader = TrainingDataLoader::restore(
        CountingDataset {
            frames: 5,
            loads: loads.clone(),
        },
        state(DataLoaderConfig::single_frame(2, 7), 5, u64::MAX, 4),
    )
    .unwrap();
    let before = loader.state().clone();

    assert!(matches!(
        loader.prepare_next_batch(),
        Err(TrainingError::DataLoaderOverflow {
            operation: "advancing epoch"
        })
    ));
    assert_eq!(loads.get(), 0);
    assert_eq!(loader.state(), &before);

    let retry = loader.prepare_next_batch();
    assert!(matches!(
        retry,
        Err(TrainingError::DataLoaderOverflow {
            operation: "advancing epoch"
        })
    ));
    assert_eq!(loads.get(), 0);
    assert_eq!(loader.state(), &before);
}
