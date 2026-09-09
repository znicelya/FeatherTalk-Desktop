#![cfg(feature = "ort-runtime")]

use feathertalk_onnx_validate::compare_output_arrays;
use ndarray::{ArrayD, IxDyn};

#[test]
fn comparison_reports_hand_checked_error_metrics() {
    let actual = ArrayD::from_shape_vec(IxDyn(&[1, 2]), vec![1.0, 3.0]).unwrap();
    let expected = ArrayD::from_shape_vec(IxDyn(&[1, 2]), vec![1.0, 2.0]).unwrap();

    let metrics = compare_output_arrays(&actual, &expected, 1.0).unwrap();

    assert_eq!(metrics.max_absolute_error, 1.0);
    assert_eq!(metrics.mean_absolute_error, 0.5);
    assert!(metrics.passed);
}

#[test]
fn comparison_rejects_shape_mismatch() {
    let actual = ArrayD::zeros(IxDyn(&[1, 2]));
    let expected = ArrayD::zeros(IxDyn(&[2, 1]));

    let error = compare_output_arrays(&actual, &expected, 1.0).unwrap_err();

    assert!(error.to_string().contains("output shape mismatch"));
}
