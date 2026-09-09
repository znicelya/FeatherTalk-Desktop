use std::collections::BTreeMap;

use feathertalk_training::{
    CheckpointCompatibility, CheckpointDescriptor, DATA_LOADER_STATE_SCHEMA_VERSION,
    DataLoaderConfig, DataLoaderState, Provenance, RandomAlgorithm, SamplingConfig, SamplingKind,
    TRAINING_STATE_SCHEMA_VERSION, TrainingCheckpointState, TrainingConfig, TrainingError,
    TrainingMode,
};

fn loader_state() -> DataLoaderState {
    DataLoaderState {
        schema_version: DATA_LOADER_STATE_SCHEMA_VERSION,
        random_algorithm: RandomAlgorithm::Splitmix64FisherYatesV1,
        config: DataLoaderConfig {
            batch_size: 2,
            seed: 17,
            sampling: SamplingConfig {
                kind: SamplingKind::SingleFrame,
                temporal_stride: 0,
            },
        },
        frame_count: 5,
        epoch: 3,
        next_position: 4,
    }
}

fn training_config() -> TrainingConfig {
    TrainingConfig {
        mode: TrainingMode::Baseline,
        batch_size: 2,
        learning_rate: 1e-3,
        total_epochs: 10,
        temporal_stride: 0,
        mouth_weight: 0.0,
        temporal_weight: 0.0,
        temporal_mouth_weight: 0.0,
        perceptual_weight: 0.01,
    }
}

fn state() -> TrainingCheckpointState {
    TrainingCheckpointState {
        schema_version: TRAINING_STATE_SCHEMA_VERSION,
        epoch: 3,
        global_step: 14,
        random_seed: 17,
        data_loader: loader_state(),
        training_config: training_config(),
        asset_provenance: Provenance {
            entries: BTreeMap::from([("assets".into(), "a".repeat(64))]),
        },
        model_provenance: Provenance {
            entries: BTreeMap::from([("vgg19".into(), "b".repeat(64))]),
        },
    }
}

#[test]
fn state_json_is_schema_one_and_round_trips_exactly() {
    let value = state();
    value.validate().unwrap();
    let json = serde_json::to_string(&value).unwrap();
    assert!(json.contains("\"schema_version\":1"));
    assert!(json.contains("\"global_step\":14"));
    assert!(!json.contains("permutation"));
    assert_eq!(
        serde_json::from_str::<TrainingCheckpointState>(&json).unwrap(),
        value
    );
}

#[test]
fn unknown_fields_and_inconsistent_progress_are_rejected() {
    let json = serde_json::to_value(state()).unwrap();
    let mut extra = json.clone();
    extra["unexpected"] = true.into();
    assert!(serde_json::from_value::<TrainingCheckpointState>(extra).is_err());

    let mut mismatch = state();
    mismatch.epoch = 2;
    assert!(matches!(
        mismatch.validate(),
        Err(TrainingError::InvalidCheckpoint(_))
    ));

    let mut bad_hash = state();
    bad_hash
        .asset_provenance
        .entries
        .insert("bad".into(), "ABC".into());
    assert!(matches!(
        bad_hash.validate(),
        Err(TrainingError::InvalidCheckpoint(_))
    ));

    let mut provenance_json = serde_json::to_value(state()).unwrap();
    provenance_json["asset_provenance"]["unexpected"] = true.into();
    assert!(serde_json::from_value::<TrainingCheckpointState>(provenance_json).is_err());
}

#[test]
fn manifest_descriptor_and_compatibility_use_fixed_identifiers() {
    let descriptor = CheckpointDescriptor::new("original-unet", "original-unet-v1", "c".repeat(64));
    descriptor.validate().unwrap();
    assert_eq!(descriptor.optimizer_kind, "adam");
    assert_eq!(descriptor.optimizer_schema_version, 1);
    let compatibility = CheckpointCompatibility::new(descriptor.clone(), training_config(), 5);
    compatibility.validate().unwrap();
}
