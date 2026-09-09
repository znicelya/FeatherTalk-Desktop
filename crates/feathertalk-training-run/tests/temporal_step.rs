mod support;

use burn::optim::AdamConfig;
use burn::tensor::Tensor;
use feathertalk_training::{TrainingError, TrainingMode};
use feathertalk_training_data::TemporalBatch;
use feathertalk_training_run::train_temporal_step;
use support::{
    CpuAutodiffBackend, CpuDevice, IdentityExtractor, assert_close, model, on_step_stack,
    training_config,
};

fn stacked(
    values: &[f32],
    channels: usize,
    size: usize,
    device: &CpuDevice,
) -> Tensor<CpuAutodiffBackend, 4> {
    let rows = values
        .iter()
        .map(|value| Tensor::full([1, channels, size, size], *value, device))
        .collect::<Vec<_>>();
    Tensor::cat(rows, 0)
}

fn pairs(values: &[f32], device: &CpuDevice) -> Tensor<CpuAutodiffBackend, 5> {
    stacked(values, 3, 160, device).reshape([2, 2, 3, 160, 160])
}

fn batch(device: &CpuDevice) -> TemporalBatch<CpuAutodiffBackend> {
    TemporalBatch {
        image: stacked(&[0.25, 0.25, 0.75, 0.75], 6, 160, device),
        audio: stacked(&[0.5, 0.5, 0.1, 0.1], 16, 32, device),
        target: pairs(&[0.0, 0.5, 0.25, 1.0], device),
        mouth_mask: Tensor::ones([2, 2, 1, 160, 160], device),
    }
}

#[test]
fn a_temporal_step_reports_every_loss_component() {
    on_step_stack("temporal-step", || {
        let device = CpuDevice::default();
        let mut optimizer = AdamConfig::new().init();
        let (_model, values) = train_temporal_step(
            model(&device),
            &mut optimizer,
            &IdentityExtractor,
            batch(&device),
            &training_config(TrainingMode::MouthRoiTemporal, 2, 1, 1),
        )
        .unwrap();
        let mouth = values.mouth.unwrap();
        let temporal = values.temporal.unwrap();
        let temporal_mouth = values.temporal_mouth.unwrap();
        assert_close(mouth, values.full);
        assert_close(temporal, 0.625);
        assert_close(temporal_mouth, 0.625);
        assert_close(
            values.total,
            values.full
                + 4.0 * mouth
                + 0.5 * temporal
                + 4.0 * temporal_mouth
                + 0.01 * values.perceptual,
        );
    });
}

#[test]
fn a_non_temporal_mode_is_rejected() {
    on_step_stack("non-temporal-rejection", || {
        let device = CpuDevice::default();
        let mut optimizer = AdamConfig::new().init();
        let error = train_temporal_step(
            model(&device),
            &mut optimizer,
            &IdentityExtractor,
            batch(&device),
            &training_config(TrainingMode::Baseline, 2, 1, 0),
        )
        .unwrap_err();
        let TrainingError::InvalidConfig(message) = error else {
            panic!("expected an invalid-config rejection, got {error:?}");
        };
        assert_eq!(
            message,
            "the non-temporal modes need train_single_frame_step"
        );
    });
}

#[test]
fn a_row_count_that_does_not_fill_the_pairs_is_rejected() {
    on_step_stack("temporal-row-guard", || {
        let device = CpuDevice::default();
        let mut optimizer = AdamConfig::new().init();
        let mut wrong = batch(&device);
        wrong.image = stacked(&[0.25, 0.25, 0.75], 6, 160, &device);
        wrong.audio = stacked(&[0.5, 0.5, 0.1], 16, 32, &device);
        let error = train_temporal_step(
            model(&device),
            &mut optimizer,
            &IdentityExtractor,
            wrong,
            &training_config(TrainingMode::MouthRoiTemporal, 2, 1, 1),
        )
        .unwrap_err();
        let TrainingError::InvalidInput(message) = error else {
            panic!("expected an invalid-input rejection, got {error:?}");
        };
        assert_eq!(message, "temporal rows 3 do not match 2x2");
    });
}
