use burn::{
    backend::{Autodiff, Flex},
    tensor::Tensor};
use feathertalk_training::LossBreakdown;
use feathertalk_training_run::LossValues;

fn scalar(value: f32) -> Tensor<1> {
    Tensor::from_floats([value], &burn::tensor::Device::default().autodiff())
}

#[test]
fn loss_readback_preserves_optional_fields_and_their_order() {
    for present in 0..8 {
        let breakdown = LossBreakdown {
            total: scalar(1.0),
            full: scalar(2.0),
            perceptual: scalar(3.0),
            mouth: (present & 1 != 0).then(|| scalar(4.0)),
            temporal: (present & 2 != 0).then(|| scalar(5.0)),
            temporal_mouth: (present & 4 != 0).then(|| scalar(6.0))};
        assert_eq!(
            LossValues::from_breakdown(&breakdown),
            LossValues {
                total: 1.0,
                full: 2.0,
                perceptual: 3.0,
                mouth: (present & 1 != 0).then_some(4.0),
                temporal: (present & 2 != 0).then_some(5.0),
                temporal_mouth: (present & 4 != 0).then_some(6.0)},
            "optional loss mask {present}"
        );
    }
}

#[test]
fn loss_readback_leaves_the_graph_available_for_backward() {
    let input = scalar(2.0).require_grad();
    let breakdown = LossBreakdown {
        total: input.clone().mul_scalar(7.0),
        full: input.clone().mul_scalar(2.0),
        perceptual: input.clone().mul_scalar(3.0),
        mouth: None,
        temporal: None,
        temporal_mouth: None};
    let values = LossValues::from_breakdown(&breakdown);
    assert_eq!(values.total, 14.0);
    let gradients = breakdown.total.backward();
    assert_eq!(input.grad(&gradients).unwrap().into_scalar::<f32>(), 7.0);
}

#[test]
fn a_non_finite_optional_loss_is_still_rejected_after_readback() {
    let breakdown = LossBreakdown {
        total: scalar(1.0),
        full: scalar(2.0),
        perceptual: scalar(3.0),
        mouth: None,
        temporal: None,
        temporal_mouth: Some(scalar(f32::NAN))};
    let error = LossValues::from_breakdown(&breakdown)
        .require_finite()
        .unwrap_err();
    assert!(error.to_string().contains("temporal_mouth is not finite"));
}
