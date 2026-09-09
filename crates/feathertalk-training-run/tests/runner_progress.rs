mod fixture;
mod support;

use burn::optim::AdamConfig;
use feathertalk_training::{TrainingDataset, TrainingMode};
use feathertalk_training_run::TrainingRunner;
use fixture::{dataset, locked_project};
use support::{
    CpuAutodiffBackend, CpuDevice, IdentityExtractor, model, on_step_stack, training_config,
};

#[test]
fn a_full_batch_advances_the_epoch_without_reporting_it() {
    on_step_stack("full-batch", || {
        let device = CpuDevice::default();
        let (_temp, project_dir) = locked_project(4);
        let mut runner = TrainingRunner::<CpuAutodiffBackend, _, _, _>::new(
            dataset(&project_dir),
            model(&device),
            AdamConfig::new().init(),
            training_config(TrainingMode::Baseline, 4, 2, 0),
            7,
            device,
        )
        .unwrap();

        let report = runner.step(&IdentityExtractor).unwrap();
        assert_eq!(report.epoch, 0);
        assert_eq!(report.global_step, 1);
        assert_eq!(report.samples_in_batch, 4);
        assert!(report.losses.total.is_finite());
        assert_eq!(runner.epoch(), 1);
        assert_eq!(runner.global_step(), 1);
        assert_eq!(runner.samples_seen(), 4);
        assert!(!runner.is_finished());
    });
}

#[test]
fn a_short_final_batch_still_counts_every_sample() {
    on_step_stack("short-batch", || {
        let device = CpuDevice::default();
        let (_temp, project_dir) = locked_project(5);
        let mut runner = TrainingRunner::<CpuAutodiffBackend, _, _, _>::new(
            dataset(&project_dir),
            model(&device),
            AdamConfig::new().init(),
            training_config(TrainingMode::Baseline, 2, 2, 0),
            7,
            device,
        )
        .unwrap();

        let sizes = [
            runner.step(&IdentityExtractor).unwrap().samples_in_batch,
            runner.step(&IdentityExtractor).unwrap().samples_in_batch,
            runner.step(&IdentityExtractor).unwrap().samples_in_batch,
        ];
        assert_eq!(sizes, [2, 2, 1]);
        assert_eq!(runner.samples_seen(), 5);
        assert_eq!(runner.global_step(), 3);
        assert_eq!(runner.epoch(), 1);
        assert_eq!(runner.dataset().frame_count(), 5);
        assert_eq!(runner.training_config().batch_size, 2);
        assert!(runner.model().is_ok());
    });
}

#[test]
fn the_runner_finishes_after_its_last_epoch() {
    on_step_stack("finishes", || {
        let device = CpuDevice::default();
        let (_temp, project_dir) = locked_project(4);
        let mut runner = TrainingRunner::<CpuAutodiffBackend, _, _, _>::new(
            dataset(&project_dir),
            model(&device),
            AdamConfig::new().init(),
            training_config(TrainingMode::Baseline, 4, 2, 0),
            7,
            device,
        )
        .unwrap();

        runner.step(&IdentityExtractor).unwrap();
        let second = runner.step(&IdentityExtractor).unwrap();
        assert_eq!(second.epoch, 1);
        assert_eq!(runner.epoch(), 2);
        assert!(runner.is_finished());
    });
}

#[test]
fn a_temporal_run_steps_through_its_pairs() {
    on_step_stack("temporal-run", || {
        let device = CpuDevice::default();
        let (_temp, project_dir) = locked_project(5);
        let mut runner = TrainingRunner::<CpuAutodiffBackend, _, _, _>::new(
            dataset(&project_dir),
            model(&device),
            AdamConfig::new().init(),
            training_config(TrainingMode::MouthRoiTemporal, 2, 1, 1),
            7,
            device,
        )
        .unwrap();

        let first = runner.step(&IdentityExtractor).unwrap();
        assert!(first.losses.temporal.is_some());
        assert!(first.losses.temporal_mouth.is_some());
        runner.step(&IdentityExtractor).unwrap();
        assert_eq!(runner.samples_seen(), 4);
        assert_eq!(runner.epoch(), 1);
        assert!(runner.is_finished());
    });
}
