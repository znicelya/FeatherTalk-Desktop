mod support;

use burn::optim::AdamConfig;
use burn::tensor::Tensor;
use feathertalk_training::{TrainingError, TrainingMode};
use feathertalk_training_data::SingleFrameBatch;
use feathertalk_training_run::train_single_frame_step;
use support::{
    CpuAutodiffBackend, CpuDevice, IdentityExtractor, NanExtractor, assert_close, model,
    on_step_stack, training_config,
};

fn batch(device: &CpuDevice) -> SingleFrameBatch<CpuAutodiffBackend> {
    SingleFrameBatch {
        image: Tensor::ones([2, 6, 160, 160], device),
        audio: Tensor::ones([2, 16, 32, 32], device),
        target: Tensor::zeros([2, 3, 160, 160], device),
        mouth_mask: Tensor::ones([2, 1, 160, 160], device),
    }
}

#[test]
fn a_baseline_step_reports_only_the_required_losses() {
    on_step_stack("baseline-step", || {
        let device = CpuDevice::default();
        let mut optimizer = AdamConfig::new().init();
        let (_model, values) = train_single_frame_step(
            model(&device),
            &mut optimizer,
            &IdentityExtractor,
            batch(&device),
            &training_config(TrainingMode::Baseline, 2, 1, 0),
        )
        .unwrap();
        assert_eq!(values.mouth, None);
        assert_eq!(values.temporal, None);
        assert_eq!(values.temporal_mouth, None);
        assert_close(values.total, values.full + 0.01 * values.perceptual);
    });
}

#[test]
fn a_mouth_roi_step_adds_the_weighted_mouth_loss() {
    on_step_stack("mouth-roi-step", || {
        let device = CpuDevice::default();
        let mut optimizer = AdamConfig::new().init();
        let (_model, values) = train_single_frame_step(
            model(&device),
            &mut optimizer,
            &IdentityExtractor,
            batch(&device),
            &training_config(TrainingMode::MouthRoi, 2, 1, 0),
        )
        .unwrap();
        assert_eq!(values.temporal, None);
        assert_eq!(values.temporal_mouth, None);
        let mouth = values.mouth.unwrap();
        assert_close(mouth, values.full);
        assert_close(
            values.total,
            values.full + 4.0 * mouth + 0.01 * values.perceptual,
        );
    });
}

#[test]
fn the_mouth_roi_total_exceeds_the_baseline_total() {
    on_step_stack("mouth-roi-vs-baseline", || {
        let device = CpuDevice::default();
        let start = model(&device);
        let mut baseline_optimizer = AdamConfig::new().init();
        let (_model, baseline) = train_single_frame_step(
            start.clone(),
            &mut baseline_optimizer,
            &IdentityExtractor,
            batch(&device),
            &training_config(TrainingMode::Baseline, 2, 1, 0),
        )
        .unwrap();
        let mut mouth_optimizer = AdamConfig::new().init();
        let (_model, mouth) = train_single_frame_step(
            start,
            &mut mouth_optimizer,
            &IdentityExtractor,
            batch(&device),
            &training_config(TrainingMode::MouthRoi, 2, 1, 0),
        )
        .unwrap();
        assert_close(mouth.full, baseline.full);
        assert!(
            mouth.total > baseline.total,
            "expected {} > {}",
            mouth.total,
            baseline.total
        );
    });
}

#[test]
fn a_zero_learning_rate_leaves_the_weights_untouched() {
    on_step_stack("zero-learning-rate", || {
        let device = CpuDevice::default();
        let mut config = training_config(TrainingMode::Baseline, 2, 1, 0);
        config.learning_rate = 0.0;
        let mut optimizer = AdamConfig::new().init();
        let start = model(&device);
        let before = start.outc.conv.weight.val().into_data();
        let (trained, _values) = train_single_frame_step(
            start,
            &mut optimizer,
            &IdentityExtractor,
            batch(&device),
            &config,
        )
        .unwrap();
        let after = trained.outc.conv.weight.val().into_data();
        assert_eq!(before, after);
    });
}

#[test]
fn the_temporal_mode_is_rejected() {
    on_step_stack("temporal-rejection", || {
        let device = CpuDevice::default();
        let mut optimizer = AdamConfig::new().init();
        let error = train_single_frame_step(
            model(&device),
            &mut optimizer,
            &IdentityExtractor,
            batch(&device),
            &training_config(TrainingMode::MouthRoiTemporal, 2, 1, 1),
        )
        .unwrap_err();
        let TrainingError::InvalidConfig(message) = error else {
            panic!("expected an invalid-config rejection, got {error:?}");
        };
        assert_eq!(message, "the temporal mode needs train_temporal_step");
    });
}

#[test]
fn a_non_finite_loss_is_rejected_and_the_optimizer_survives() {
    on_step_stack("non-finite-loss", || {
        let device = CpuDevice::default();
        let mut optimizer = AdamConfig::new().init();
        let start = model(&device);
        let error = train_single_frame_step(
            start.clone(),
            &mut optimizer,
            &NanExtractor,
            batch(&device),
            &training_config(TrainingMode::Baseline, 2, 1, 0),
        )
        .unwrap_err();
        let TrainingError::InvalidInput(message) = error else {
            panic!("expected an invalid-input rejection, got {error:?}");
        };
        assert!(
            message.contains("total") && message.contains("is not finite"),
            "unexpected message: {message}"
        );

        let (_model, values) = train_single_frame_step(
            start,
            &mut optimizer,
            &IdentityExtractor,
            batch(&device),
            &training_config(TrainingMode::Baseline, 2, 1, 0),
        )
        .unwrap();
        assert!(values.total.is_finite());
    });
}
