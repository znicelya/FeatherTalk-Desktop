mod fixture;
mod support;

use burn::optim::AdamConfig;
use feathertalk_training::{TrainingError, TrainingMode};
use feathertalk_training_run::TrainingRunner;
use fixture::{dataset, locked_project};
use support::{
    CpuAutodiffBackend, CpuDevice, IdentityExtractor, NanExtractor, model, on_step_stack,
    training_config,
};

fn message(error: TrainingError) -> String {
    let TrainingError::InvalidInput(message) = error else {
        panic!("expected an invalid-input rejection, got {error:?}");
    };
    message
}

#[test]
fn a_non_finite_loss_poisons_the_runner() {
    on_step_stack("poisoned", || {
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

        let first = message(runner.step(&NanExtractor).unwrap_err());
        assert!(
            first.contains("is not finite"),
            "unexpected message: {first}"
        );

        let second = message(runner.step(&IdentityExtractor).unwrap_err());
        assert_eq!(second, "training runner was poisoned by a failed step");
        let third = message(runner.model().unwrap_err());
        assert_eq!(third, "training runner was poisoned by a failed step");
    });
}
