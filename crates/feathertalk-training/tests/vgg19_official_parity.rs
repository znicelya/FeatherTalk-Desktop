use std::{
    io::{Cursor, Read},
    path::{Path, PathBuf},
};

use burn::{
    backend::NdArray,
    tensor::{Tensor, TensorData},
};
use feathertalk_training::{load_vgg19_package, read_vgg19_manifest};
use ndarray::ArrayD;
use ndarray_npy::ReadNpyExt;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use zip::ZipArchive;

type CpuBackend = NdArray<f32>;

const GOLDEN_ARCHIVE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../tests/golden/vgg19-conv3-3-v1.zip"
);

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct GoldenManifest {
    schema_version: u32,
    fixture: String,
    source_sha256: String,
    output_layer: String,
    input_contract: InputContract,
    input: ArrayManifest,
    expected: ArrayManifest,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct InputContract {
    channels: usize,
    color_order: String,
    value_range: String,
    normalization: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ArrayManifest {
    path: String,
    shape: Vec<usize>,
    dtype: String,
    sha256: String,
}

#[test]
#[ignore = "requires an explicitly supplied licensed VGG19 package"]
fn official_python_golden_matches_burn_package() {
    let package = std::env::var("FEATHERTALK_VGG19_PACKAGE")
        .expect("FEATHERTALK_VGG19_PACKAGE must point to an explicitly built VGG19 package");
    let package = PathBuf::from(package);

    let archive_path = Path::new(GOLDEN_ARCHIVE);
    let archive_bytes = std::fs::read(archive_path).expect("read committed VGG19 golden archive");
    let sidecar = std::fs::read_to_string(archive_path.with_extension("sha256"))
        .expect("read VGG19 golden archive sidecar");
    let expected_archive_hash = sidecar
        .split_whitespace()
        .next()
        .expect("golden sidecar must contain a SHA-256");
    let actual_archive_hash = hex::encode(Sha256::digest(&archive_bytes));
    assert_eq!(actual_archive_hash, expected_archive_hash);

    let manifest_bytes = read_zip_entry(&archive_bytes, "manifest.json");
    let manifest: GoldenManifest = serde_json::from_slice(&manifest_bytes)
        .expect("golden manifest must use the locked schema");
    assert_eq!(manifest.schema_version, 1);
    assert_eq!(manifest.fixture, "vgg19-conv3-3-v1");
    assert_eq!(manifest.output_layer, "features.14");
    assert_eq!(manifest.input_contract.channels, 3);
    assert_eq!(manifest.input_contract.color_order, "bgr");
    assert_eq!(manifest.input_contract.value_range, "0..1");
    assert_eq!(manifest.input_contract.normalization, "none");

    let input_bytes = read_zip_entry(&archive_bytes, &manifest.input.path);
    let expected_bytes = read_zip_entry(&archive_bytes, &manifest.expected.path);
    assert_eq!(
        hex::encode(Sha256::digest(&input_bytes)),
        manifest.input.sha256
    );
    assert_eq!(
        hex::encode(Sha256::digest(&expected_bytes)),
        manifest.expected.sha256
    );
    assert_eq!(manifest.input.dtype, "float32");
    assert_eq!(manifest.expected.dtype, "float32");

    let input = ArrayD::<f32>::read_npy(Cursor::new(input_bytes)).expect("read golden input");
    let expected =
        ArrayD::<f32>::read_npy(Cursor::new(expected_bytes)).expect("read golden output");
    assert_eq!(input.shape(), manifest.input.shape.as_slice());
    assert_eq!(expected.shape(), manifest.expected.shape.as_slice());

    let package_manifest = read_vgg19_manifest(&package).expect("read VGG19 package manifest");
    assert_eq!(package_manifest.source.sha256, manifest.source_sha256);

    let device = Default::default();
    let model = load_vgg19_package::<CpuBackend>(&package, &device).expect("load VGG19 package");
    let input_tensor = Tensor::<CpuBackend, 4>::from_data(
        TensorData::new(
            input.iter().copied().collect::<Vec<_>>(),
            input.shape().to_vec(),
        ),
        &device,
    );
    let actual = model
        .forward(input_tensor)
        .into_data()
        .to_vec::<f32>()
        .unwrap();
    let expected = expected.iter().copied().collect::<Vec<_>>();
    assert_eq!(actual.len(), expected.len());

    let (max_abs, sum_abs) = actual
        .iter()
        .zip(expected.iter())
        .map(|(actual, expected)| (actual - expected).abs())
        .fold((0.0_f32, 0.0_f32), |(max_abs, sum_abs), error| {
            (max_abs.max(error), sum_abs + error)
        });
    let mean_abs = sum_abs / actual.len() as f32;
    println!("vgg19_official_parity max_abs={max_abs:.9e} mean_abs={mean_abs:.9e}");
    assert!(max_abs <= 1e-4, "max_abs={max_abs}");
    assert!(mean_abs <= 1e-5, "mean_abs={mean_abs}");
}

fn read_zip_entry(archive_bytes: &[u8], name: &str) -> Vec<u8> {
    let mut archive = ZipArchive::new(Cursor::new(archive_bytes)).expect("open golden ZIP");
    let mut entry = archive.by_name(name).expect("golden entry exists");
    let mut bytes = Vec::new();
    entry.read_to_end(&mut bytes).expect("read golden entry");
    bytes
}
