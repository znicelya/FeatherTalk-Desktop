mod fixture;
mod support;

use burn::optim::AdamConfig;
use feathertalk_training::{
    CheckpointCompatibility, CheckpointDescriptor, TRAINING_STATE_SCHEMA_VERSION, TrainingMode,
    load_training_checkpoint,
};
use feathertalk_training_run::{TrainingRunner, data_loader_config_for};
use fixture::{dataset, locked_project};
use support::{
    CpuAutodiffBackend, CpuDevice, IdentityExtractor, NanExtractor, assert_close, model,
    on_step_stack, training_config,
};

fn descriptor() -> CheckpointDescriptor {
    CheckpointDescriptor::new("original-unet", "original-unet-v1", "0".repeat(64))
}

#[test]
fn a_restored_runner_reproduces_the_next_steps() {
    on_step_stack("restore-replay", || {
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

        runner.step(&IdentityExtractor).unwrap();
        runner.step(&IdentityExtractor).unwrap();

        let root = tempfile::tempdir().unwrap();
        let checkpoint = root.path().join("checkpoint-000002");
        let manifest = runner.save_checkpoint(&checkpoint, descriptor()).unwrap();
        assert_eq!(manifest.model_kind, "original-unet");

        let third = runner.step(&IdentityExtractor).unwrap();
        let state_after_third = runner.checkpoint_state();
        let fourth = runner.step(&IdentityExtractor).unwrap();

        let (_temp_b, project_b) = locked_project(4);
        let replay = dataset(&project_b);
        let expected = CheckpointCompatibility::new(
            descriptor(),
            training_config(TrainingMode::Baseline, 2, 2, 0),
            4,
        );
        let template_model = model(&device);
        let template_optimizer = AdamConfig::new().init();
        let restored = load_training_checkpoint::<CpuAutodiffBackend, _, _>(
            &checkpoint,
            &template_model,
            &template_optimizer,
            &device,
            &expected,
        )
        .unwrap();
        let mut replayed =
            TrainingRunner::<CpuAutodiffBackend, _, _, _>::restore(replay, restored, device)
                .unwrap();

        let replayed_third = replayed.step(&IdentityExtractor).unwrap();
        assert_eq!(replayed_third.global_step, 3);
        assert_eq!(replayed_third.epoch, third.epoch);
        assert_eq!(replayed_third.samples_in_batch, third.samples_in_batch);
        assert_close(replayed_third.losses.total, third.losses.total);
        assert_close(replayed_third.losses.full, third.losses.full);
        assert_eq!(replayed.checkpoint_state(), state_after_third);

        let replayed_fourth = replayed.step(&IdentityExtractor).unwrap();
        assert_close(replayed_fourth.losses.total, fourth.losses.total);
    });
}

#[test]
fn the_checkpoint_state_matches_the_runner() {
    on_step_stack("checkpoint-state", || {
        let device = CpuDevice::default();
        let (_temp, project_dir) = locked_project(4);
        let config = training_config(TrainingMode::Baseline, 2, 2, 0);
        let mut runner = TrainingRunner::<CpuAutodiffBackend, _, _, _>::new(
            dataset(&project_dir),
            model(&device),
            AdamConfig::new().init(),
            config.clone(),
            7,
            device,
        )
        .unwrap();

        runner.step(&IdentityExtractor).unwrap();
        runner.step(&IdentityExtractor).unwrap();

        let state = runner.checkpoint_state();
        let expected_loader = data_loader_config_for(&config, 7).unwrap();

        assert!(state.validate().is_ok());
        assert_eq!(state.schema_version, TRAINING_STATE_SCHEMA_VERSION);
        assert_eq!(state.epoch, 1);
        assert_eq!(state.epoch, runner.epoch());
        assert_eq!(state.global_step, 2);
        assert_eq!(state.random_seed, 7);
        assert_eq!(state.data_loader.config, expected_loader);
        assert_eq!(state.data_loader.next_position, 0);
        assert_eq!(state.training_config, config);
        assert!(state.asset_provenance.entries.is_empty());
        assert!(state.model_provenance.entries.is_empty());
    });
}

#[test]
fn a_mismatched_dataset_refuses_to_restore() {
    on_step_stack("mismatched-dataset", || {
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

        runner.step(&IdentityExtractor).unwrap();

        let root = tempfile::tempdir().unwrap();
        let checkpoint = root.path().join("checkpoint-000001");
        runner.save_checkpoint(&checkpoint, descriptor()).unwrap();

        let (_temp_b, project_b) = locked_project(6);
        let replay = dataset(&project_b);
        let expected = CheckpointCompatibility::new(
            descriptor(),
            training_config(TrainingMode::Baseline, 2, 2, 0),
            4,
        );
        let template_model = model(&device);
        let template_optimizer = AdamConfig::new().init();
        let restored = load_training_checkpoint::<CpuAutodiffBackend, _, _>(
            &checkpoint,
            &template_model,
            &template_optimizer,
            &device,
            &expected,
        )
        .unwrap();

        let error =
            TrainingRunner::<CpuAutodiffBackend, _, _, _>::restore(replay, restored, device)
                .map(|_| ())
                .unwrap_err();
        assert_eq!(
            error.to_string(),
            "invalid data loader state: dataset frame_count does not match saved state"
        );
    });
}

#[test]
fn a_poisoned_runner_refuses_to_save() {
    on_step_stack("poisoned-save", || {
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

        runner.step(&NanExtractor).unwrap_err();

        let root = tempfile::tempdir().unwrap();
        let checkpoint = root.path().join("checkpoint-000000");
        let error = runner
            .save_checkpoint(&checkpoint, descriptor())
            .unwrap_err();
        assert_eq!(
            error.to_string(),
            "invalid training input: training runner was poisoned by a failed step"
        );
        assert!(!checkpoint.exists());
    });
}
