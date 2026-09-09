use std::{fs, path::PathBuf};

use clap::{Parser, ValueEnum};
use feathertalk_export::onnx::{OnnxModelKind, validate_model_contract};
use serde::Serialize;
use sha2::{Digest, Sha256};

#[cfg(feature = "ort-runtime")]
use feathertalk_onnx_validate::{compare_output_arrays, run_cpu_session};
#[cfg(feature = "ort-runtime")]
use ndarray::ArrayD;
#[cfg(feature = "ort-runtime")]
use ndarray_npy::ReadNpyExt;

#[derive(Debug, Parser)]
#[command(name = "feathertalk-onnx-validate")]
struct Arguments {
    #[arg(long)]
    model: PathBuf,
    #[arg(long, value_enum)]
    kind: Kind,
    #[arg(long)]
    input: Vec<PathBuf>,
    #[arg(long = "expected-output")]
    expected_output: Option<PathBuf>,
    #[arg(long)]
    structural_only: bool,
    #[arg(long, default_value_t = 1.0e-4)]
    threshold: f32,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum Kind {
    FeatherHubert,
    OriginalUnet,
    MobileoneUnet,
}

impl From<Kind> for OnnxModelKind {
    fn from(value: Kind) -> Self {
        match value {
            Kind::FeatherHubert => Self::FeatherHubert,
            Kind::OriginalUnet => Self::OriginalUnet,
            Kind::MobileoneUnet => Self::MobileOneUnet,
        }
    }
}

#[derive(Debug, Serialize)]
struct ValidationReport {
    provider: &'static str,
    model_bytes: usize,
    model_sha256: String,
    input_metadata: Vec<TensorMetadata>,
    output_metadata: Option<TensorMetadata>,
    max_absolute_error: Option<f32>,
    mean_absolute_error: Option<f32>,
    threshold: Option<f32>,
    passed: bool,
}

#[derive(Debug, Serialize)]
struct TensorMetadata {
    name: String,
    shape: Vec<usize>,
    elements: usize,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let arguments = Arguments::parse();
    let bytes = fs::read(&arguments.model)?;
    let kind: OnnxModelKind = arguments.kind.into();
    validate_model_contract(&bytes, &kind.public_contract())?;
    let has_input = !arguments.input.is_empty();
    let has_expected_output = arguments.expected_output.is_some();
    if arguments.structural_only {
        if has_input || has_expected_output {
            return Err(
                "--structural-only cannot be combined with runtime fixture arguments".into(),
            );
        }
    } else {
        match (has_input, has_expected_output) {
            (false, false) => {
                return Err(
                    "runtime mode requires --input and --expected-output; use --structural-only when ONNX Runtime is unavailable"
                        .into(),
                );
            }
            (true, true) => {}
            _ => {
                return Err("--input and --expected-output must be provided together".into());
            }
        }
    }
    if !arguments.structural_only {
        return run_runtime(arguments, bytes, kind);
    }

    let report = ValidationReport {
        provider: "structural-only",
        model_bytes: bytes.len(),
        model_sha256: hex::encode(Sha256::digest(&bytes)),
        input_metadata: Vec::new(),
        output_metadata: None,
        max_absolute_error: None,
        mean_absolute_error: None,
        threshold: None,
        passed: true,
    };
    println!("{}", serde_json::to_string(&report)?);
    Ok(())
}

#[cfg(not(feature = "ort-runtime"))]
fn run_runtime(
    _arguments: Arguments,
    _bytes: Vec<u8>,
    _kind: OnnxModelKind,
) -> Result<(), Box<dyn std::error::Error>> {
    Err(
        "ONNX Runtime compatibility mode is unavailable in this build; rebuild with --features ort-runtime or use --structural-only"
            .into(),
    )
}

#[cfg(feature = "ort-runtime")]
fn run_runtime(
    arguments: Arguments,
    bytes: Vec<u8>,
    kind: OnnxModelKind,
) -> Result<(), Box<dyn std::error::Error>> {
    let contract = kind.public_contract();
    if arguments.input.len() != contract.inputs.len() {
        return Err(invalid_input(format!(
            "{} requires {} --input values, received {}",
            kind_name(kind),
            contract.inputs.len(),
            arguments.input.len()
        )));
    }
    let expected_path = arguments
        .expected_output
        .as_ref()
        .expect("runtime argument pair was validated");
    let mut input_metadata = Vec::with_capacity(contract.inputs.len());
    let mut runtime_inputs = Vec::with_capacity(contract.inputs.len());
    for (path, tensor) in arguments.input.iter().zip(&contract.inputs) {
        let array = read_npy(path, "input")?;
        validate_shape(array.shape(), &tensor.shape, &tensor.name)?;
        input_metadata.push(metadata(&tensor.name, &array));
        runtime_inputs.push((tensor.name.clone(), array));
    }
    let expected = read_npy(expected_path, "expected-output")?;
    let output_contract = &contract.outputs[0];
    validate_shape(
        expected.shape(),
        &output_contract.shape,
        &output_contract.name,
    )?;

    let actual = run_cpu_session(&arguments.model, runtime_inputs, &output_contract.name)?;
    let metrics = compare_output_arrays(&actual, &expected, arguments.threshold)?;
    let report = ValidationReport {
        provider: "CPUExecutionProvider",
        model_bytes: bytes.len(),
        model_sha256: hex::encode(Sha256::digest(&bytes)),
        input_metadata,
        output_metadata: Some(metadata(&output_contract.name, &actual)),
        max_absolute_error: Some(metrics.max_absolute_error),
        mean_absolute_error: Some(metrics.mean_absolute_error),
        threshold: Some(arguments.threshold),
        passed: metrics.passed,
    };
    println!("{}", serde_json::to_string(&report)?);
    if !metrics.passed {
        return Err(invalid_input(format!(
            "maximum absolute error {} exceeds threshold {}",
            metrics.max_absolute_error, arguments.threshold
        )));
    }
    Ok(())
}

#[cfg(feature = "ort-runtime")]
fn read_npy(path: &std::path::Path, role: &str) -> Result<ArrayD<f32>, Box<dyn std::error::Error>> {
    const MAX_NPY_BYTES: u64 = 512 * 1024 * 1024;

    let metadata = fs::symlink_metadata(path).map_err(|error| {
        invalid_input(format!(
            "failed to inspect {role} NPY {}: {error}",
            path.display()
        ))
    })?;
    if !metadata.file_type().is_file() || metadata.file_type().is_symlink() {
        return Err(invalid_input(format!(
            "{role} NPY must be a regular file: {}",
            path.display()
        )));
    }
    if metadata.len() > MAX_NPY_BYTES {
        return Err(invalid_input(format!(
            "{role} NPY exceeds {MAX_NPY_BYTES} bytes: {}",
            path.display()
        )));
    }
    let file = fs::File::open(path).map_err(|error| {
        invalid_input(format!(
            "failed to open {role} NPY {}: {error}",
            path.display()
        ))
    })?;
    let array = ArrayD::<f32>::read_npy(std::io::BufReader::new(file)).map_err(|error| {
        invalid_input(format!(
            "failed to read {role} NPY {}: {error}",
            path.display()
        ))
    })?;
    if let Some(index) = array.iter().position(|value| !value.is_finite()) {
        return Err(invalid_input(format!(
            "{role} NPY contains a non-finite value at element {index}: {}",
            path.display()
        )));
    }
    Ok(array)
}

#[cfg(feature = "ort-runtime")]
fn validate_shape(
    actual: &[usize],
    expected: &[i64],
    name: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let valid = actual.len() == expected.len()
        && actual.iter().zip(expected).all(|(&actual, &expected)| {
            (expected == -1 && actual > 0)
                || (expected > 0 && usize::try_from(expected) == Ok(actual))
        });
    if !valid {
        return Err(invalid_input(format!(
            "tensor {name} shape mismatch: actual {actual:?}, contract {expected:?}"
        )));
    }
    Ok(())
}

#[cfg(feature = "ort-runtime")]
fn metadata(name: &str, array: &ArrayD<f32>) -> TensorMetadata {
    TensorMetadata {
        name: name.to_owned(),
        shape: array.shape().to_vec(),
        elements: array.len(),
    }
}

#[cfg(feature = "ort-runtime")]
fn kind_name(kind: OnnxModelKind) -> &'static str {
    match kind {
        OnnxModelKind::FeatherHubert => "feather-hubert",
        OnnxModelKind::OriginalUnet => "original-unet",
        OnnxModelKind::MobileOneUnet => "mobileone-unet",
    }
}

#[cfg(feature = "ort-runtime")]
fn invalid_input(message: impl Into<String>) -> Box<dyn std::error::Error> {
    std::io::Error::new(std::io::ErrorKind::InvalidInput, message.into()).into()
}
