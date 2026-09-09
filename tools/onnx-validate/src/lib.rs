#![cfg(feature = "ort-runtime")]

use std::{error::Error, path::Path};

use ndarray::ArrayD;
use ort::{
    session::{Session, SessionInputValue},
    value::Tensor,
};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ComparisonMetrics {
    pub max_absolute_error: f32,
    pub mean_absolute_error: f32,
    pub passed: bool,
}

pub fn compare_output_arrays(
    actual: &ArrayD<f32>,
    expected: &ArrayD<f32>,
    threshold: f32,
) -> Result<ComparisonMetrics, Box<dyn Error>> {
    if !threshold.is_finite() || threshold < 0.0 {
        return Err(invalid_data(
            "threshold must be a finite non-negative number",
        ));
    }
    if actual.shape() != expected.shape() {
        return Err(invalid_data(format!(
            "output shape mismatch: actual {:?}, expected {:?}",
            actual.shape(),
            expected.shape()
        )));
    }
    if actual.is_empty() {
        return Err(invalid_data("output arrays must not be empty"));
    }

    let mut maximum = 0.0_f32;
    let mut total = 0.0_f64;
    for (index, (&actual, &expected)) in actual.iter().zip(expected.iter()).enumerate() {
        if !actual.is_finite() || !expected.is_finite() {
            return Err(invalid_data(format!(
                "non-finite output value at element {index}"
            )));
        }
        let difference = (actual - expected).abs();
        maximum = maximum.max(difference);
        total += f64::from(difference);
    }
    let mean = (total / actual.len() as f64) as f32;
    Ok(ComparisonMetrics {
        max_absolute_error: maximum,
        mean_absolute_error: mean,
        passed: maximum <= threshold,
    })
}

pub fn run_cpu_session(
    model: &Path,
    inputs: Vec<(String, ArrayD<f32>)>,
    output_name: &str,
) -> Result<ArrayD<f32>, Box<dyn Error>> {
    let mut session = Session::builder()?.commit_from_file(model)?;
    let values = inputs
        .into_iter()
        .map(|(name, array)| {
            let tensor = Tensor::from_array(array)?;
            Ok((name, SessionInputValue::from(tensor)))
        })
        .collect::<ort::Result<Vec<_>>>()?;
    let outputs = session.run(values)?;
    let output = outputs
        .get(output_name)
        .ok_or_else(|| invalid_data(format!("ONNX Runtime did not return output {output_name}")))?;
    Ok(output.try_extract_array::<f32>()?.to_owned())
}

fn invalid_data(message: impl Into<String>) -> Box<dyn Error> {
    std::io::Error::new(std::io::ErrorKind::InvalidData, message.into()).into()
}
