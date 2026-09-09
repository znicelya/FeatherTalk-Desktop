use ndarray::ArrayViewD;
use serde::Serialize;
use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct ParityMetrics {
    pub max_abs: f32,
    pub mean_abs: f32,
    pub max_relative: f32,
}

#[derive(Debug, Error)]
pub enum ParityError {
    #[error("backend initialization failed: {0}")]
    Backend(String),
    #[error("array shapes differ: actual {actual:?}, expected {expected:?}")]
    ShapeMismatch {
        actual: Vec<usize>,
        expected: Vec<usize>,
    },
    #[error("cannot compare empty arrays")]
    EmptyArray,
    #[error("non-finite value at element {index}: actual {actual}, expected {expected}")]
    NonFinite {
        index: usize,
        actual: f32,
        expected: f32,
    },
    #[error("{metric} cannot be represented as a finite f32: {value}")]
    MetricOverflow { metric: &'static str, value: f64 },
    #[error("archive sidecar verification failed: {0}")]
    ArchiveVerification(crate::archive::FixtureError),
    #[error("{case} fixture contract mismatch for {field}: expected {expected}, actual {actual}")]
    FixtureContract {
        case: &'static str,
        field: &'static str,
        expected: String,
        actual: String,
    },
    #[error("{case} fixture {role} arrays differ: expected {expected:?}, actual {actual:?}")]
    FixtureArraySet {
        case: &'static str,
        role: &'static str,
        expected: Vec<String>,
        actual: Vec<String>,
    },
    #[error("{case} fixture {role} array {name} has shape {actual:?}, expected {expected:?}")]
    FixtureArrayShape {
        case: &'static str,
        role: &'static str,
        name: &'static str,
        expected: Vec<usize>,
        actual: Vec<usize>,
    },
    #[error("tensor {name} has shape {actual:?}, expected {expected:?}")]
    TensorShape {
        name: &'static str,
        expected: Vec<usize>,
        actual: Vec<usize>,
    },
    #[error("fixture error: {0}")]
    Fixture(#[from] crate::archive::FixtureError),
    #[error("weight import error: {0}")]
    WeightImport(#[from] feathertalk_weights::WeightImportError),
    #[error("tensor data error: {0}")]
    TensorData(String),
    #[error("array construction error: {0}")]
    Array(String),
    #[error("fixture array is missing: {0}")]
    MissingArray(String),
}

pub fn compare_f32(
    actual: ArrayViewD<'_, f32>,
    expected: ArrayViewD<'_, f32>,
) -> Result<ParityMetrics, ParityError> {
    if actual.shape() != expected.shape() {
        return Err(ParityError::ShapeMismatch {
            actual: actual.shape().to_vec(),
            expected: expected.shape().to_vec(),
        });
    }
    if actual.is_empty() {
        return Err(ParityError::EmptyArray);
    }

    let mut max_abs = 0.0_f64;
    let mut absolute_sum = 0.0_f64;
    let mut max_relative = 0.0_f64;
    for (index, (&actual, &expected)) in actual.iter().zip(expected.iter()).enumerate() {
        if !actual.is_finite() || !expected.is_finite() {
            return Err(ParityError::NonFinite {
                index,
                actual,
                expected,
            });
        }
        let actual = f64::from(actual);
        let expected = f64::from(expected);
        let absolute = (actual - expected).abs();
        max_abs = max_abs.max(absolute);
        absolute_sum += absolute;
        if !absolute_sum.is_finite() {
            return Err(ParityError::MetricOverflow {
                metric: "mean_abs_sum",
                value: absolute_sum,
            });
        }
        max_relative = max_relative.max(absolute / expected.abs().max(1e-7));
    }

    Ok(ParityMetrics {
        max_abs: narrow_metric("max_abs", max_abs)?,
        mean_abs: narrow_metric("mean_abs", absolute_sum / actual.len() as f64)?,
        max_relative: narrow_metric("max_relative", max_relative)?,
    })
}

fn narrow_metric(metric: &'static str, value: f64) -> Result<f32, ParityError> {
    if !value.is_finite() || value > f64::from(f32::MAX) {
        return Err(ParityError::MetricOverflow { metric, value });
    }
    Ok(value as f32)
}
