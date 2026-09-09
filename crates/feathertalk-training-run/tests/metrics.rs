mod fixture;
mod support;

use std::time::Duration;

use burn::optim::AdamConfig;
use feathertalk_training::TrainingMode;
use feathertalk_training_run::TrainingRunner;
use fixture::{dataset, locked_project};
use support::{
    CpuAutodiffBackend, CpuDevice, IdentityExtractor, assert_close, model, on_step_stack,
    training_config,
};

#[test]
fn zero_elapsed_time_reports_no_rate_and_no_eta() {
    on_step_stack("zero-elapsed", || {
        let device = CpuDevice::default();
        let (_temp, project_dir) = locked_project(4);
        let mut runner = TrainingRunner::<CpuAutodiffBackend, _, _, _>::new(
            dataset(&project_dir),
            model(&device),
            AdamConfig::new().init(),
            training_config(TrainingMode::Baseline, 2, 1, 0),
            7,
            device,
        )
        .unwrap();

        let report = runner.step(&IdentityExtractor).unwrap();
        let metrics = runner
            .metrics(&report, Duration::ZERO, None, "training")
            .unwrap();

        assert_eq!(metrics.mode, TrainingMode::Baseline);
        assert_eq!(metrics.epoch, 0);
        assert_eq!(metrics.global_step, 1);
        assert_eq!(metrics.samples_seen, 2);
        assert_close(metrics.samples_per_second, 0.0);
        assert_close(metrics.estimated_remaining_seconds, 0.0);
        assert_eq!(metrics.gpu_memory_bytes, None);
        assert_eq!(metrics.worker_state, "training");
        assert!(metrics.mouth_loss.is_none());
    });
}

#[test]
fn the_metrics_copy_every_loss_component() {
    on_step_stack("loss-components", || {
        let device = CpuDevice::default();
        let (_temp, project_dir) = locked_project(4);
        let mut runner = TrainingRunner::<CpuAutodiffBackend, _, _, _>::new(
            dataset(&project_dir),
            model(&device),
            AdamConfig::new().init(),
            training_config(TrainingMode::MouthRoi, 2, 1, 0),
            7,
            device,
        )
        .unwrap();

        let report = runner.step(&IdentityExtractor).unwrap();
        let metrics = runner
            .metrics(&report, Duration::from_secs(1), Some(4096), "training")
            .unwrap();

        assert_eq!(metrics.total_loss, report.losses.total);
        assert_eq!(metrics.full_loss, report.losses.full);
        assert_eq!(metrics.perceptual_loss, report.losses.perceptual);
        assert_eq!(metrics.mouth_loss, report.losses.mouth);
        assert!(metrics.mouth_loss.is_some());
        assert_eq!(metrics.temporal_loss, None);
        assert_eq!(metrics.temporal_mouth_loss, None);
        assert_eq!(metrics.gpu_memory_bytes, Some(4096));
    });
}

#[test]
fn the_eta_shrinks_as_the_run_progresses() {
    on_step_stack("eta-shrinks", || {
        let device = CpuDevice::default();
        let (_temp, project_dir) = locked_project(4);
        let mut runner = TrainingRunner::<CpuAutodiffBackend, _, _, _>::new(
            dataset(&project_dir),
            model(&device),
            AdamConfig::new().init(),
            training_config(TrainingMode::Baseline, 2, 2, 0),
            7,
            device,
        )
        .unwrap();

        let first = runner.step(&IdentityExtractor).unwrap();
        let after_one = runner
            .metrics(&first, Duration::from_secs(1), None, "training")
            .unwrap();
        assert_close(after_one.samples_per_second, 2.0);
        assert_close(after_one.estimated_remaining_seconds, 3.0);

        let second = runner.step(&IdentityExtractor).unwrap();
        let after_two = runner
            .metrics(&second, Duration::from_secs(2), None, "training")
            .unwrap();
        assert_close(after_two.samples_per_second, 2.0);
        assert_close(after_two.estimated_remaining_seconds, 2.0);
        assert_eq!(after_two.epoch, 0);
        assert_eq!(after_two.samples_seen, 4);
    });
}
