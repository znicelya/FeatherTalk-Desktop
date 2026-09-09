use burn::tensor::{Tensor, backend::Backend};
use feathertalk_training::{
    BaselineLossConfig, LossBreakdown, MouthRoiLossConfig, PerceptualFeatureExtractor,
    TemporalLossConfig, TrainingError, baseline_loss, mouth_l1_loss, mouth_roi_loss, temporal_loss,
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

#[derive(Debug, Clone, Copy)]
struct PanicExtractor;

impl<B: Backend> PerceptualFeatureExtractor<B> for PanicExtractor {
    fn forward(&self, _image: Tensor<B, 4>) -> Tensor<B, 4> {
        panic!("extractor must not run for invalid input shapes")
    }
}

#[test]
fn config_defaults_are_exact_and_serde_round_trip() {
    let baseline = BaselineLossConfig::default();
    let mouth = MouthRoiLossConfig::default();
    let temporal = TemporalLossConfig::default();

    assert_eq!(baseline.perceptual_weight, 0.01);
    assert_eq!(mouth.mouth_weight, 4.0);
    assert_eq!(mouth.perceptual_weight, 0.01);
    assert_eq!(temporal.mouth_weight, 4.0);
    assert_eq!(temporal.temporal_weight, 0.5);
    assert_eq!(temporal.temporal_mouth_weight, 4.0);
    assert_eq!(temporal.perceptual_weight, 0.01);
    assert_eq!(
        serde_json::from_str::<TemporalLossConfig>(&serde_json::to_string(&temporal).unwrap())
            .unwrap(),
        temporal
    );
}

#[test]
fn negative_nan_and_infinite_weights_are_rejected() {
    for perceptual_weight in [-1.0, f64::NAN, f64::INFINITY] {
        assert_invalid_config(BaselineLossConfig { perceptual_weight }.validate());
    }

    for config in [
        MouthRoiLossConfig {
            mouth_weight: f64::NAN,
            ..Default::default()
        },
        MouthRoiLossConfig {
            perceptual_weight: f64::INFINITY,
            ..Default::default()
        },
    ] {
        assert_invalid_config(config.validate());
    }

    for config in [
        TemporalLossConfig {
            mouth_weight: -1.0,
            ..Default::default()
        },
        TemporalLossConfig {
            temporal_weight: f64::NAN,
            ..Default::default()
        },
        TemporalLossConfig {
            temporal_mouth_weight: f64::INFINITY,
            ..Default::default()
        },
        TemporalLossConfig {
            perceptual_weight: -1.0,
            ..Default::default()
        },
    ] {
        assert_invalid_config(config.validate());
    }
}

#[test]
fn baseline_components_match_hand_computed_literals() {
    let device = Default::default();
    let prediction = Tensor::<CpuBackend, 4>::ones([1, 3, 4, 4], &device);
    let target = Tensor::<CpuBackend, 4>::zeros([1, 3, 4, 4], &device);

    let result = baseline_loss(
        &IdentityExtractor,
        prediction,
        target,
        &BaselineLossConfig::default(),
    )
    .unwrap();

    assert_breakdown_value(&result, result.full.clone(), 1.0);
    assert_breakdown_value(&result, result.perceptual.clone(), 1.0);
    assert_breakdown_value(&result, result.total.clone(), 1.01);
    assert!(result.mouth.is_none());
    assert!(result.temporal.is_none());
    assert!(result.temporal_mouth.is_none());
}

#[test]
fn mouth_roi_components_use_channel_scaled_mask_denominator() {
    let device = Default::default();
    let prediction = Tensor::<CpuBackend, 4>::ones([1, 3, 4, 4], &device);
    let target = Tensor::<CpuBackend, 4>::zeros([1, 3, 4, 4], &device);
    let mask = Tensor::<CpuBackend, 4>::zeros([1, 1, 4, 4], &device)
        .slice_fill([0..1, 0..1, 0..2, 0..4], 1.0);

    let result = mouth_roi_loss(
        &IdentityExtractor,
        prediction,
        target,
        mask,
        &MouthRoiLossConfig::default(),
    )
    .unwrap();

    assert_breakdown_value(&result, result.full.clone(), 1.0);
    assert_breakdown_value(&result, result.mouth.clone().unwrap(), 1.0);
    assert_breakdown_value(&result, result.perceptual.clone(), 1.0);
    assert_breakdown_value(&result, result.total.clone(), 5.01);
    assert!(result.temporal.is_none());
    assert!(result.temporal_mouth.is_none());
}

#[test]
fn empty_mouth_mask_returns_zero_without_nan() {
    let device = Default::default();
    let prediction = Tensor::<CpuBackend, 4>::ones([1, 3, 4, 4], &device);
    let target = Tensor::<CpuBackend, 4>::zeros([1, 3, 4, 4], &device);
    let mask = Tensor::<CpuBackend, 4>::zeros([1, 1, 4, 4], &device);

    let actual = mouth_l1_loss(prediction, target, mask)
        .unwrap()
        .into_scalar();

    assert_eq!(actual, 0.0);
}

#[test]
fn temporal_components_match_hand_computed_pair_and_union_literals() {
    let device = Default::default();
    let prediction = Tensor::<CpuBackend, 5>::zeros([1, 2, 3, 4, 4], &device)
        .slice_fill([0..1, 1..2, 0..3, 0..2, 0..4], 1.0)
        .slice_fill([0..1, 1..2, 0..3, 2..4, 0..4], 3.0);
    let target = Tensor::<CpuBackend, 5>::zeros([1, 2, 3, 4, 4], &device);
    let mask = Tensor::<CpuBackend, 5>::zeros([1, 2, 1, 4, 4], &device)
        .slice_fill([0..1, 0..1, 0..1, 0..2, 0..4], 1.0)
        .slice_fill([0..1, 1..2, 0..1, 2..4, 0..4], 1.0);

    let result = temporal_loss(
        &IdentityExtractor,
        prediction,
        target,
        mask,
        &TemporalLossConfig::default(),
    )
    .unwrap();

    assert_breakdown_value(&result, result.full.clone(), 1.0);
    assert_breakdown_value(&result, result.mouth.clone().unwrap(), 1.5);
    assert_breakdown_value(&result, result.temporal.clone().unwrap(), 2.0);
    assert_breakdown_value(&result, result.temporal_mouth.clone().unwrap(), 2.0);
    assert_breakdown_value(&result, result.perceptual.clone(), 2.5);
    assert_breakdown_value(&result, result.total.clone(), 16.025);
}

#[test]
fn invalid_masks_and_temporal_pair_lengths_are_rejected() {
    let device = Default::default();
    let image = Tensor::<CpuBackend, 4>::zeros([1, 3, 4, 4], &device);
    let target = Tensor::<CpuBackend, 4>::zeros([1, 3, 4, 4], &device);
    let bad_channels = Tensor::<CpuBackend, 4>::zeros([1, 2, 4, 4], &device);
    assert!(matches!(
        mouth_l1_loss(image.clone(), target.clone(), bad_channels),
        Err(TrainingError::InvalidInput(_))
    ));

    let bad_spatial = Tensor::<CpuBackend, 4>::zeros([1, 1, 3, 4], &device);
    assert!(matches!(
        mouth_l1_loss(image.clone(), target.clone(), bad_spatial),
        Err(TrainingError::InvalidInput(_))
    ));

    let bad_batch = Tensor::<CpuBackend, 4>::zeros([2, 1, 4, 4], &device);
    assert!(matches!(
        mouth_l1_loss(image.clone(), target.clone(), bad_batch),
        Err(TrainingError::InvalidInput(_))
    ));

    let pair_one = Tensor::<CpuBackend, 5>::zeros([1, 1, 3, 4, 4], &device);
    let pair_three = Tensor::<CpuBackend, 5>::zeros([1, 3, 3, 4, 4], &device);
    let mask_one = Tensor::<CpuBackend, 5>::zeros([1, 1, 1, 4, 4], &device);
    let mask_three = Tensor::<CpuBackend, 5>::zeros([1, 3, 1, 4, 4], &device);
    assert!(matches!(
        temporal_loss(
            &IdentityExtractor,
            pair_one.clone(),
            pair_one,
            mask_one,
            &TemporalLossConfig::default()
        ),
        Err(TrainingError::InvalidInput(_))
    ));
    assert!(matches!(
        temporal_loss(
            &IdentityExtractor,
            pair_three.clone(),
            pair_three,
            mask_three,
            &TemporalLossConfig::default()
        ),
        Err(TrainingError::InvalidInput(_))
    ));
}

#[test]
fn target_shape_mismatch_is_rejected_before_extractor_invocation() {
    let device = Default::default();
    let prediction = Tensor::<CpuBackend, 4>::zeros([1, 3, 4, 4], &device);
    let target = Tensor::<CpuBackend, 4>::zeros([1, 3, 5, 4], &device);

    assert!(matches!(
        baseline_loss(
            &PanicExtractor,
            prediction,
            target,
            &BaselineLossConfig::default()
        ),
        Err(TrainingError::InvalidInput(_))
    ));
}

#[test]
fn all_three_loss_totals_propagate_prediction_gradients() {
    let device = Default::default();
    let prediction = Tensor::<CpuAutodiffBackend, 4>::ones([1, 3, 4, 4], &device).require_grad();
    let target = Tensor::<CpuAutodiffBackend, 4>::zeros([1, 3, 4, 4], &device);
    let mask = Tensor::<CpuAutodiffBackend, 4>::ones([1, 1, 4, 4], &device);

    let baseline = baseline_loss(
        &IdentityExtractor,
        prediction.clone(),
        target.clone(),
        &BaselineLossConfig::default(),
    )
    .unwrap();
    assert!(prediction.grad(&baseline.total.backward()).is_some());

    let prediction = Tensor::<CpuAutodiffBackend, 4>::ones([1, 3, 4, 4], &device).require_grad();
    let mouth = mouth_roi_loss(
        &IdentityExtractor,
        prediction.clone(),
        target.clone(),
        mask.clone(),
        &MouthRoiLossConfig::default(),
    )
    .unwrap();
    assert!(prediction.grad(&mouth.total.backward()).is_some());

    let prediction = Tensor::<CpuAutodiffBackend, 5>::ones([1, 2, 3, 4, 4], &device).require_grad();
    let target = Tensor::<CpuAutodiffBackend, 5>::zeros([1, 2, 3, 4, 4], &device);
    let mask = Tensor::<CpuAutodiffBackend, 5>::ones([1, 2, 1, 4, 4], &device);
    let temporal = temporal_loss(
        &IdentityExtractor,
        prediction.clone(),
        target,
        mask,
        &TemporalLossConfig::default(),
    )
    .unwrap();
    assert!(prediction.grad(&temporal.total.backward()).is_some());
}

fn assert_breakdown_value(
    breakdown: &LossBreakdown<CpuBackend>,
    tensor: Tensor<CpuBackend, 1>,
    expected: f32,
) {
    let actual = tensor.into_scalar();
    assert!(
        (actual - expected).abs() <= 1e-5,
        "actual={actual}, expected={expected}"
    );
    assert!(breakdown.total.clone().into_scalar().is_finite());
}

fn assert_invalid_config(result: Result<(), TrainingError>) {
    assert!(matches!(result, Err(TrainingError::InvalidConfig(_))));
}
