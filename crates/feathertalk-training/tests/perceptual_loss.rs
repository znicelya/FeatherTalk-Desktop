use burn::{
    module::Module,
    tensor::{Tensor, TensorData, backend::Backend},
};
use feathertalk_training::{
    PerceptualFeatureExtractor, TrainingError, Vgg19Conv3_3, perceptual_mse,
};

type CpuBackend = burn::backend::NdArray<f32>;
type CpuAutodiffBackend = burn::backend::Autodiff<CpuBackend>;

#[derive(Debug, Clone, Copy)]
struct IdentityExtractor;

impl<B: Backend> PerceptualFeatureExtractor<B> for IdentityExtractor {
    fn forward(&self, image: Tensor<B, 4>) -> Tensor<B, 4> {
        image
    }
}

#[test]
fn perceptual_mse_matches_a_hand_computed_mean_square() {
    let device = Default::default();
    let prediction = Tensor::<CpuBackend, 4>::full([1, 3, 4, 4], 1.0, &device);
    let target = Tensor::<CpuBackend, 4>::full([1, 3, 4, 4], 0.5, &device);

    let actual = perceptual_mse(&IdentityExtractor, prediction, target)
        .unwrap()
        .into_scalar();

    assert!((actual - 0.25).abs() <= f32::EPSILON);
}

#[test]
fn identical_perceptual_inputs_have_exactly_zero_loss() {
    let device = Default::default();
    let input = Tensor::<CpuBackend, 4>::from_data(
        TensorData::from([[
            [
                [0.0_f32, 0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6, 0.7],
                [0.8, 0.9, 1.0, 0.2],
                [0.3, 0.4, 0.5, 0.6],
            ],
            [
                [0.7, 0.8, 0.9, 1.0],
                [0.1, 0.2, 0.3, 0.4],
                [0.5, 0.6, 0.7, 0.8],
                [0.9, 1.0, 0.1, 0.2],
            ],
            [
                [0.3, 0.4, 0.5, 0.6],
                [0.7, 0.8, 0.9, 1.0],
                [0.2, 0.3, 0.4, 0.5],
                [0.6, 0.7, 0.8, 0.9],
            ],
        ]]),
        &device,
    );

    let actual = perceptual_mse(&IdentityExtractor, input.clone(), input)
        .unwrap()
        .into_scalar();

    assert_eq!(actual, 0.0);
}

#[test]
fn perceptual_mse_rejects_shape_mismatch_before_extraction() {
    let device = Default::default();
    let prediction = Tensor::<CpuBackend, 4>::zeros([1, 3, 4, 4], &device);
    let target = Tensor::<CpuBackend, 4>::zeros([1, 3, 4, 5], &device);

    let error = perceptual_mse(&IdentityExtractor, prediction, target).unwrap_err();

    assert!(matches!(error, TrainingError::InvalidInput(message) if message.contains("shape")));
}

#[test]
fn perceptual_mse_rejects_non_three_channel_input() {
    let device = Default::default();
    let prediction = Tensor::<CpuBackend, 4>::zeros([1, 1, 4, 4], &device);
    let target = Tensor::<CpuBackend, 4>::zeros([1, 1, 4, 4], &device);

    let error = perceptual_mse(&IdentityExtractor, prediction, target).unwrap_err();

    assert!(
        matches!(error, TrainingError::InvalidInput(message) if message.contains("3 channels"))
    );
}

#[test]
fn perceptual_mse_rejects_empty_batch() {
    let device = Default::default();
    let prediction = Tensor::<CpuBackend, 4>::zeros([0, 3, 4, 4], &device);
    let target = Tensor::<CpuBackend, 4>::zeros([0, 3, 4, 4], &device);

    let error = perceptual_mse(&IdentityExtractor, prediction, target).unwrap_err();

    assert!(matches!(error, TrainingError::InvalidInput(message) if message.contains("batch")));
}

#[test]
fn perceptual_mse_rejects_spatial_dimensions_smaller_than_four() {
    let device = Default::default();
    let prediction = Tensor::<CpuBackend, 4>::zeros([1, 3, 3, 4], &device);
    let target = Tensor::<CpuBackend, 4>::zeros([1, 3, 3, 4], &device);

    let error = perceptual_mse(&IdentityExtractor, prediction, target).unwrap_err();

    assert!(matches!(error, TrainingError::InvalidInput(message) if message.contains("spatial")));
}

#[test]
fn frozen_vgg_keeps_prediction_gradient_and_drops_target_and_vgg_gradients() {
    let device = Default::default();
    let vgg = Vgg19Conv3_3::<CpuAutodiffBackend>::new_for_import(&device).no_grad();
    let prediction = Tensor::<CpuAutodiffBackend, 4>::ones([1, 3, 8, 8], &device).require_grad();
    let target = Tensor::<CpuAutodiffBackend, 4>::zeros([1, 3, 8, 8], &device).require_grad();

    let loss = perceptual_mse(&vgg, prediction.clone(), target.clone()).unwrap();
    let gradients = loss.backward();

    assert!(prediction.grad(&gradients).is_some());
    assert!(target.grad(&gradients).is_none());
    assert!(vgg.conv1_1.weight.val().grad(&gradients).is_none());
}
