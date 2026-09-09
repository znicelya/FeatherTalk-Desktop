mod support;

use feathertalk_training::{DataLoaderConfig, SamplingKind, TrainingError, TrainingMode};
use feathertalk_training_run::data_loader_config_for;
use support::training_config;

fn rejection(mode: TrainingMode, temporal_stride: u64) -> String {
    let config = training_config(mode, 4, 1, temporal_stride);
    let error = data_loader_config_for(&config, 7).unwrap_err();
    let TrainingError::InvalidCheckpoint(message) = error else {
        panic!("expected an invalid-checkpoint rejection, got {error:?}");
    };
    message
}

#[test]
fn baseline_maps_to_single_frame_sampling() {
    let config = training_config(TrainingMode::Baseline, 4, 1, 0);
    let loader_config = data_loader_config_for(&config, 7).unwrap();
    assert_eq!(loader_config, DataLoaderConfig::single_frame(4, 7));
    assert_eq!(loader_config.sampling.kind, SamplingKind::SingleFrame);
    assert_eq!(loader_config.sampling.temporal_stride, 0);
}

#[test]
fn mouth_roi_maps_to_single_frame_sampling() {
    let config = training_config(TrainingMode::MouthRoi, 4, 1, 0);
    let loader_config = data_loader_config_for(&config, 42).unwrap();
    assert_eq!(loader_config, DataLoaderConfig::single_frame(4, 42));
}

#[test]
fn temporal_mode_maps_to_temporal_pair_sampling() {
    let config = training_config(TrainingMode::MouthRoiTemporal, 4, 1, 2);
    let loader_config = data_loader_config_for(&config, 42).unwrap();
    assert_eq!(loader_config, DataLoaderConfig::temporal_pair(4, 42, 2));
    assert_eq!(loader_config.sampling.kind, SamplingKind::TemporalPair);
}

#[test]
fn a_non_temporal_mode_rejects_a_temporal_stride() {
    assert_eq!(
        rejection(TrainingMode::Baseline, 3),
        "training_config.temporal_stride must be zero for non-temporal modes"
    );
}

#[test]
fn the_temporal_mode_rejects_a_zero_stride() {
    assert_eq!(
        rejection(TrainingMode::MouthRoiTemporal, 0),
        "training_config.temporal_stride must be greater than zero for temporal mode"
    );
}
