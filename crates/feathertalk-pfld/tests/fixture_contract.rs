use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
};

use serde::Deserialize;
use sha2::{Digest, Sha256};

const INPUT_SHAPE: [usize; 4] = [1, 3, 192, 192];
const OUTPUT_SHAPE: [usize; 2] = [1, 220];

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct FixtureManifest {
    schema_version: u32,
    case: String,
    model_type: String,
    source: Source,
    artifact: Artifact,
    generator: Generator,
    files: BTreeMap<String, FileDescriptor>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Source {
    file_name: String,
    sha256: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Artifact {
    file_name: String,
    sha256: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Generator {
    python_version: String,
    torch_version: String,
    numpy_version: String,
    platform: String,
    threads: u32,
    input_formula: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct FileDescriptor {
    file_name: String,
    dtype: String,
    shape: Vec<usize>,
    bytes: u64,
    sha256: String,
}

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/pytorch_cpu_v1")
}

fn sha256(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

fn read_f32(path: &Path, descriptor: &FileDescriptor) -> Vec<f32> {
    let bytes = fs::read(path).unwrap();
    assert_eq!(bytes.len() as u64, descriptor.bytes);
    assert_eq!(sha256(&bytes), descriptor.sha256);
    assert_eq!(bytes.len() % 4, 0);
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect()
}

#[test]
fn committed_python_fixture_has_fixed_schema_hashes_and_finite_arrays() {
    let root = fixture_dir();
    let manifest: FixtureManifest =
        serde_json::from_slice(&fs::read(root.join("fixture.json")).unwrap()).unwrap();
    assert_eq!(manifest.schema_version, 1);
    assert_eq!(manifest.case, "pfld_cpu_v1");
    assert_eq!(manifest.model_type, "pfld_ghost_one");
    assert_eq!(manifest.source.file_name, "checkpoint_epoch_335.pth.tar");
    assert_eq!(
        manifest.source.sha256,
        "bada866661ad5fa1080a085f51fe9c016c69958c406951afa4afc7840f856de0"
    );
    assert_eq!(manifest.artifact.file_name, "model.safetensors");
    assert_eq!(
        manifest.artifact.sha256,
        "e131dd764236fde54a27b2f7084906119f06c28b140bf127b459ec967e92915b"
    );
    assert_eq!(manifest.generator.threads, 1);
    assert_eq!(manifest.generator.input_formula, "bgr_u8_channel_affine_v1");
    assert!(!manifest.generator.python_version.is_empty());
    assert!(!manifest.generator.torch_version.is_empty());
    assert!(!manifest.generator.numpy_version.is_empty());
    assert!(!manifest.generator.platform.is_empty());
    let names = manifest
        .files
        .keys()
        .map(String::as_str)
        .collect::<Vec<_>>();
    assert_eq!(names, vec!["input.f32", "output.f32"]);

    let input = &manifest.files["input.f32"];
    assert_eq!(input.file_name, "input.f32");
    assert_eq!(input.dtype, "f32-le");
    assert_eq!(input.shape, INPUT_SHAPE);
    let input_values = read_f32(&root.join(&input.file_name), input);
    assert_eq!(
        input_values.len(),
        INPUT_SHAPE.into_iter().product::<usize>()
    );
    assert!(input_values.iter().all(|value| value.is_finite()));

    let output = &manifest.files["output.f32"];
    assert_eq!(output.file_name, "output.f32");
    assert_eq!(output.dtype, "f32-le");
    assert_eq!(output.shape, OUTPUT_SHAPE);
    let output_values = read_f32(&root.join(&output.file_name), output);
    assert_eq!(
        output_values.len(),
        OUTPUT_SHAPE.into_iter().product::<usize>()
    );
    assert!(output_values.iter().all(|value| value.is_finite()));
}

#[test]
fn fixture_manifest_rejects_unknown_fields() {
    let root = fixture_dir();
    let mut value: serde_json::Value =
        serde_json::from_slice(&fs::read(root.join("fixture.json")).unwrap()).unwrap();
    value["future_field"] = serde_json::json!(true);
    assert!(serde_json::from_value::<FixtureManifest>(value).is_err());
}
