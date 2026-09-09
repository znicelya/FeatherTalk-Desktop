#![allow(dead_code)]

use std::{
    collections::BTreeMap,
    fs::File,
    io::{BufReader, Read},
    path::{Path, PathBuf},
};

use burn::{backend::NdArray, tensor::Tensor};
use feathertalk_scrfd::ScrfdArtifactPaths;
use ndarray::ArrayD;
use ndarray_npy::ReadNpyExt;
use serde::Deserialize;
use sha2::{Digest, Sha256};

const SOURCE_SHA256: &str = "32d20c77b9e2dc1d07e94c2ab9d25bdd5cd05eddbe0b46e7b38e7a1eca22e99a";
const FIXTURE_FILES: [(&str, &[usize]); 10] = [
    ("input.npy", &[1, 3, 640, 640]),
    ("out0.npy", &[1, 12_800, 1]),
    ("out1.npy", &[1, 3_200, 1]),
    ("out2.npy", &[1, 800, 1]),
    ("out3.npy", &[1, 12_800, 4]),
    ("out4.npy", &[1, 3_200, 4]),
    ("out5.npy", &[1, 800, 4]),
    ("out6.npy", &[1, 12_800, 10]),
    ("out7.npy", &[1, 3_200, 10]),
    ("out8.npy", &[1, 800, 10]),
];

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FixtureManifest {
    pub schema_version: u32,
    pub case: String,
    pub source: FixtureSource,
    pub generator: FixtureGenerator,
    pub files: BTreeMap<String, FixtureFile>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FixtureSource {
    pub file_name: String,
    pub file_bytes: u64,
    pub sha256: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FixtureGenerator {
    pub python_version: String,
    pub numpy_version: String,
    pub opencv_version: String,
    pub backend: String,
    pub target: String,
    pub threads: u32,
    pub opencl: bool,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FixtureFile {
    pub dtype: String,
    pub shape: Vec<usize>,
    pub bytes: u64,
    pub sha256: String,
}

#[derive(Debug)]
pub struct VerifiedFixture {
    pub root: PathBuf,
    pub manifest: FixtureManifest,
}

pub fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/opencv_cpu_v1")
}

pub fn artifact_paths() -> ScrfdArtifactPaths {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("artifacts/scrfd_2_5g");
    ScrfdArtifactPaths {
        manifest: root.join("manifest.json"),
        weights: root.join("model.safetensors"),
    }
}

pub fn read_array(path: &Path) -> Result<ArrayD<f32>, String> {
    let file = File::open(path).map_err(|error| format!("{}: {error}", path.display()))?;
    ArrayD::<f32>::read_npy(BufReader::new(file))
        .map_err(|error| format!("{}: {error}", path.display()))
}

pub fn sha256_bytes(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

pub fn load_and_verify_fixture() -> Result<VerifiedFixture, String> {
    load_and_verify_fixture_at(&fixture_dir())
}

pub fn load_and_verify_fixture_at(root: &Path) -> Result<VerifiedFixture, String> {
    let manifest_path = root.join("fixture.json");
    let manifest_bytes = std::fs::read(&manifest_path)
        .map_err(|error| format!("{}: {error}", manifest_path.display()))?;
    let manifest: FixtureManifest = serde_json::from_slice(&manifest_bytes)
        .map_err(|error| format!("{}: {error}", manifest_path.display()))?;

    require_eq(&manifest_path, "schema_version", manifest.schema_version, 1)?;
    require_eq(
        &manifest_path,
        "case",
        manifest.case.as_str(),
        "opencv_cpu_v1",
    )?;
    require_eq(
        &manifest_path,
        "source.file_name",
        manifest.source.file_name.as_str(),
        "scrfd_2.5g_kps.onnx",
    )?;
    require_eq(
        &manifest_path,
        "source.file_bytes",
        manifest.source.file_bytes,
        3_291_017,
    )?;
    require_eq(
        &manifest_path,
        "source.sha256",
        manifest.source.sha256.as_str(),
        SOURCE_SHA256,
    )?;
    for (field, actual, expected) in [
        (
            "generator.python_version",
            manifest.generator.python_version.as_str(),
            "3.11",
        ),
        (
            "generator.numpy_version",
            manifest.generator.numpy_version.as_str(),
            "2.2.6",
        ),
        (
            "generator.opencv_version",
            manifest.generator.opencv_version.as_str(),
            "4.12.0",
        ),
        (
            "generator.backend",
            manifest.generator.backend.as_str(),
            "opencv",
        ),
        (
            "generator.target",
            manifest.generator.target.as_str(),
            "cpu",
        ),
    ] {
        require_eq(&manifest_path, field, actual, expected)?;
    }
    require_eq(
        &manifest_path,
        "generator.threads",
        manifest.generator.threads,
        1,
    )?;
    require_eq(
        &manifest_path,
        "generator.opencl",
        manifest.generator.opencl,
        false,
    )?;

    let actual_names = manifest
        .files
        .keys()
        .map(String::as_str)
        .collect::<Vec<_>>();
    let expected_names = FIXTURE_FILES
        .iter()
        .map(|(name, _)| *name)
        .collect::<Vec<_>>();
    if actual_names != expected_names {
        return Err(format!(
            "{}: expected files {expected_names:?}, got {actual_names:?}",
            manifest_path.display()
        ));
    }

    for (name, expected_shape) in FIXTURE_FILES {
        let descriptor = manifest
            .files
            .get(name)
            .expect("the exact key set is checked above");
        let path = root.join(name);
        if descriptor.dtype != "float32" {
            return Err(format!(
                "{}: expected dtype float32, got {}",
                path.display(),
                descriptor.dtype
            ));
        }
        if descriptor.shape.contains(&0) || descriptor.shape.as_slice() != expected_shape {
            return Err(format!(
                "{}: expected shape {expected_shape:?}, got {:?}",
                path.display(),
                descriptor.shape
            ));
        }
        let (actual_bytes, actual_sha256) = stream_hash(&path)?;
        if actual_bytes != descriptor.bytes {
            return Err(format!(
                "{}: expected {} bytes, got {actual_bytes}",
                path.display(),
                descriptor.bytes
            ));
        }
        if actual_sha256 != descriptor.sha256 {
            return Err(format!(
                "{}: expected SHA-256 {}, got {actual_sha256}",
                path.display(),
                descriptor.sha256
            ));
        }
        let array = read_array(&path)?;
        if array.shape() != expected_shape {
            return Err(format!(
                "{}: decoded shape {:?}, expected {expected_shape:?}",
                path.display(),
                array.shape()
            ));
        }
        if let Some((index, value)) = array
            .iter()
            .copied()
            .enumerate()
            .find(|(_, value)| !value.is_finite())
        {
            return Err(format!(
                "{}: non-finite value {value} at flattened index {index}",
                path.display()
            ));
        }
    }

    Ok(VerifiedFixture {
        root: root.to_owned(),
        manifest,
    })
}

fn stream_hash(path: &Path) -> Result<(u64, String), String> {
    let mut file = File::open(path).map_err(|error| format!("{}: {error}", path.display()))?;
    let mut digest = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    let mut bytes = 0_u64;
    loop {
        let read = file
            .read(&mut buffer)
            .map_err(|error| format!("{}: {error}", path.display()))?;
        if read == 0 {
            break;
        }
        bytes += read as u64;
        digest.update(&buffer[..read]);
    }
    Ok((bytes, hex::encode(digest.finalize())))
}

fn require_eq<T: std::fmt::Debug + PartialEq>(
    path: &Path,
    field: &str,
    actual: T,
    expected: T,
) -> Result<(), String> {
    if actual != expected {
        return Err(format!(
            "{}: {field} expected {expected:?}, got {actual:?}",
            path.display()
        ));
    }
    Ok(())
}

#[derive(Debug, Clone, Copy)]
pub struct ParityMetrics {
    pub max_abs: f32,
    pub mean_abs: f32,
    pub max_relative: f32,
}

pub fn compare_f32(actual: &[f32], expected: &[f32]) -> Result<ParityMetrics, String> {
    if actual.len() != expected.len() || actual.is_empty() {
        return Err("arrays must have the same nonzero length".to_owned());
    }
    let mut max_abs = 0.0_f64;
    let mut sum_abs = 0.0_f64;
    let mut max_relative = 0.0_f64;
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        if !actual.is_finite() || !expected.is_finite() {
            return Err(format!("non-finite value at index {index}"));
        }
        let actual = f64::from(actual);
        let expected = f64::from(expected);
        let absolute = (actual - expected).abs();
        max_abs = max_abs.max(absolute);
        sum_abs += absolute;
        max_relative = max_relative.max(absolute / expected.abs().max(1e-7));
    }
    Ok(ParityMetrics {
        max_abs: max_abs as f32,
        mean_abs: (sum_abs / actual.len() as f64) as f32,
        max_relative: max_relative as f32,
    })
}

pub fn assert_cpu_tensor_matches_fixture<const D: usize>(
    tensor: Tensor<NdArray<f32>, D>,
    path: &Path,
) {
    let expected = read_array(path).unwrap();
    let raw_shape = expected.shape().to_vec();
    let public_shape = match (D, raw_shape.as_slice()) {
        (2, [1, anchors, 1]) => vec![1, *anchors],
        (3, [1, anchors, width]) => vec![1, *anchors, *width],
        _ => panic!(
            "{}: fixture shape {raw_shape:?} is incompatible with public rank {D}",
            path.display(),
        ),
    };
    assert_eq!(
        tensor.dims().to_vec(),
        public_shape,
        "{}: public tensor shape mismatch",
        path.display()
    );

    let actual = tensor.into_data().to_vec::<f32>().unwrap();
    let expected = expected.iter().copied().collect::<Vec<_>>();
    let metrics = compare_f32(&actual, &expected).unwrap();
    assert!(metrics.max_abs <= 1e-3, "{}: {metrics:?}", path.display());
    assert!(metrics.mean_abs <= 1e-4, "{}: {metrics:?}", path.display());
    assert!(
        metrics.max_relative.is_finite(),
        "{}: {metrics:?}",
        path.display()
    );
}
