#![allow(dead_code)]

use std::{
    fs::File,
    io::{BufReader, Read},
    path::{Path, PathBuf},
};

use feathertalk_image::BgrImage;
use ndarray::ArrayD;
use ndarray_npy::ReadNpyExt;
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};

pub const CASE: &str = "opencv_cpu_v1";

/// Every committed array, in manifest key order, with its dtype and shape.
pub const FIXTURE_ARRAYS: [(&str, &str, &[usize]); 15] = [
    ("area_int_2x2_dst.npy", "uint8", &[4, 4, 3]),
    ("area_int_2x2_src.npy", "uint8", &[8, 8, 3]),
    ("area_int_4x4_dst.npy", "uint8", &[2, 2, 3]),
    ("area_int_4x4_src.npy", "uint8", &[8, 8, 3]),
    ("area_shrink_dst.npy", "uint8", &[5, 7, 3]),
    ("area_shrink_src.npy", "uint8", &[9, 13, 3]),
    ("area_upscale_dst.npy", "uint8", &[8, 8, 3]),
    ("area_upscale_src.npy", "uint8", &[5, 5, 3]),
    ("gray_dst.npy", "uint8", &[64, 64]),
    ("gray_src.npy", "uint8", &[64, 64, 3]),
    ("laplacian_response.npy", "float64", &[64, 64]),
    ("linear_shrink_dst.npy", "uint8", &[192, 192, 3]),
    ("linear_shrink_src.npy", "uint8", &[200, 200, 3]),
    ("linear_upscale_dst.npy", "uint8", &[192, 192, 3]),
    ("linear_upscale_src.npy", "uint8", &[47, 61, 3]),
];

#[derive(Debug)]
pub struct VerifiedFixture {
    pub root: PathBuf,
    pub manifest: Value,
}

pub fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/opencv_cpu_v1")
}

pub fn load_and_verify_fixture() -> Result<VerifiedFixture, String> {
    load_and_verify_fixture_at(&fixture_dir())
}

/// Validate every manifest field without touching the filesystem.
///
/// The generator writes with `sort_keys=True`, so key iteration is alphabetical
/// whichever map type `serde_json` was compiled with.
pub fn verify_manifest(label: &str, bytes: &[u8]) -> Result<Value, String> {
    let manifest: Value =
        serde_json::from_slice(bytes).map_err(|error| format!("{label}: {error}"))?;
    let root = object(label, "manifest", &manifest)?;
    require_keys(
        label,
        "manifest",
        root,
        &[
            "case",
            "files",
            "generator",
            "scalars",
            "schema_version",
            "source",
        ],
    )?;
    require_eq(
        label,
        "schema_version",
        number(label, "schema_version", &manifest["schema_version"])?,
        1,
    )?;
    require_eq(label, "case", text(label, "case", &manifest["case"])?, CASE)?;

    let source = object(label, "source", &manifest["source"])?;
    require_keys(
        label,
        "source",
        source,
        &["kind", "pattern", "pattern_edge"],
    )?;
    for (field, expected) in [
        ("kind", "synthetic"),
        ("pattern", "bgr_u8_channel_affine_v1"),
    ] {
        let qualified = format!("source.{field}");
        let value = &manifest["source"][field];
        require_eq(label, &qualified, text(label, &qualified, value)?, expected)?;
    }
    require_eq(
        label,
        "source.pattern_edge",
        number(
            label,
            "source.pattern_edge",
            &manifest["source"]["pattern_edge"],
        )?,
        640,
    )?;

    let generator = object(label, "generator", &manifest["generator"])?;
    require_keys(
        label,
        "generator",
        generator,
        &[
            "backend",
            "numpy_version",
            "opencl",
            "opencv_version",
            "python_version",
            "target",
            "threads",
        ],
    )?;
    for (field, expected) in [
        ("backend", "opencv"),
        ("numpy_version", "2.4.6"),
        ("opencv_version", "5.0.0"),
        ("python_version", "3.11"),
        ("target", "cpu"),
    ] {
        let qualified = format!("generator.{field}");
        let value = &manifest["generator"][field];
        require_eq(label, &qualified, text(label, &qualified, value)?, expected)?;
    }
    require_eq(
        label,
        "generator.threads",
        number(
            label,
            "generator.threads",
            &manifest["generator"]["threads"],
        )?,
        1,
    )?;
    require_eq(
        label,
        "generator.opencl",
        flag(label, "generator.opencl", &manifest["generator"]["opencl"])?,
        false,
    )?;

    let scalars = object(label, "scalars", &manifest["scalars"])?;
    require_keys(label, "scalars", scalars, &["laplacian_variance"])?;
    let variance = manifest["scalars"]["laplacian_variance"]
        .as_f64()
        .ok_or_else(|| format!("{label}: scalars.laplacian_variance must be a number"))?;
    if !variance.is_finite() || variance <= 0.0 {
        return Err(format!(
            "{label}: scalars.laplacian_variance must be finite and positive, got {variance}"
        ));
    }

    let files = object(label, "files", &manifest["files"])?;
    let expected_names = FIXTURE_ARRAYS
        .iter()
        .map(|(name, _, _)| *name)
        .collect::<Vec<_>>();
    require_keys(label, "files", files, &expected_names)?;
    for (name, dtype, shape) in FIXTURE_ARRAYS {
        let qualified = format!("files.{name}");
        let descriptor = object(label, &qualified, &manifest["files"][name])?;
        require_keys(
            label,
            &qualified,
            descriptor,
            &["bytes", "dtype", "sha256", "shape"],
        )?;
        require_eq(
            label,
            &format!("{qualified}.dtype"),
            text(label, &qualified, &descriptor["dtype"])?,
            dtype,
        )?;
        let actual_shape = descriptor["shape"]
            .as_array()
            .ok_or_else(|| format!("{label}: {qualified}.shape must be an array"))?
            .iter()
            .map(|value| {
                value
                    .as_u64()
                    .filter(|value| *value > 0)
                    .map(|value| value as usize)
                    .ok_or_else(|| {
                        format!("{label}: {qualified}.shape must hold positive integers")
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        if actual_shape.as_slice() != shape {
            return Err(format!(
                "{label}: {qualified}.shape expected {shape:?}, got {actual_shape:?}"
            ));
        }
        number(label, &format!("{qualified}.bytes"), &descriptor["bytes"])?;
        let sha256 = text(label, &format!("{qualified}.sha256"), &descriptor["sha256"])?;
        if sha256.len() != 64 || !sha256.bytes().all(|byte| byte.is_ascii_hexdigit()) {
            return Err(format!(
                "{label}: {qualified}.sha256 is not a 64 digit hex string"
            ));
        }
    }

    Ok(manifest)
}

pub fn load_and_verify_fixture_at(root: &Path) -> Result<VerifiedFixture, String> {
    let manifest_path = root.join("fixture.json");
    let label = manifest_path.display().to_string();
    let bytes = std::fs::read(&manifest_path).map_err(|error| format!("{label}: {error}"))?;
    let manifest = verify_manifest(&label, &bytes)?;

    let mut actual_names = std::fs::read_dir(root)
        .map_err(|error| format!("{}: {error}", root.display()))?
        .map(|entry| {
            entry
                .map(|entry| entry.file_name().to_string_lossy().into_owned())
                .map_err(|error| format!("{}: {error}", root.display()))
        })
        .collect::<Result<Vec<_>, _>>()?;
    actual_names.sort();
    let mut expected_names = FIXTURE_ARRAYS
        .iter()
        .map(|(name, _, _)| (*name).to_owned())
        .collect::<Vec<_>>();
    expected_names.push("fixture.json".to_owned());
    expected_names.sort();
    if actual_names != expected_names {
        return Err(format!(
            "{}: expected files {expected_names:?}, got {actual_names:?}",
            root.display()
        ));
    }

    for (name, dtype, shape) in FIXTURE_ARRAYS {
        let path = root.join(name);
        let descriptor = &manifest["files"][name];
        let (actual_bytes, actual_sha256) = stream_hash(&path)?;
        let expected_bytes = descriptor["bytes"]
            .as_u64()
            .expect("verify_manifest checked the type");
        if actual_bytes != expected_bytes {
            return Err(format!(
                "{}: expected {expected_bytes} bytes, got {actual_bytes}",
                path.display()
            ));
        }
        let expected_sha256 = descriptor["sha256"]
            .as_str()
            .expect("verify_manifest checked the type");
        if actual_sha256 != expected_sha256 {
            return Err(format!(
                "{}: expected SHA-256 {expected_sha256}, got {actual_sha256}",
                path.display()
            ));
        }
        match dtype {
            "uint8" => require_shape(&path, read_u8_array(&path)?.shape(), shape)?,
            "float64" => {
                let array = read_f64_array(&path)?;
                require_shape(&path, array.shape(), shape)?;
                check_finite(&path.display().to_string(), &array)?;
            }
            other => return Err(format!("{}: unsupported dtype {other}", path.display())),
        }
    }

    Ok(VerifiedFixture {
        root: root.to_owned(),
        manifest,
    })
}

pub fn read_u8_array(path: &Path) -> Result<ArrayD<u8>, String> {
    let file = File::open(path).map_err(|error| format!("{}: {error}", path.display()))?;
    ArrayD::<u8>::read_npy(BufReader::new(file))
        .map_err(|error| format!("{}: {error}", path.display()))
}

pub fn read_f64_array(path: &Path) -> Result<ArrayD<f64>, String> {
    let file = File::open(path).map_err(|error| format!("{}: {error}", path.display()))?;
    ArrayD::<f64>::read_npy(BufReader::new(file))
        .map_err(|error| format!("{}: {error}", path.display()))
}

pub fn check_finite(label: &str, array: &ArrayD<f64>) -> Result<(), String> {
    if let Some((index, value)) = array
        .iter()
        .copied()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(format!(
            "{label}: non-finite value {value} at flattened index {index}"
        ));
    }
    Ok(())
}

/// Row-major bytes of a fixture array, whatever its rank.
pub fn flatten_u8(array: &ArrayD<u8>) -> Vec<u8> {
    array.iter().copied().collect()
}

/// Build a `BgrImage` from an `[h, w, 3]` uint8 fixture array.
pub fn bgr_from_array(array: &ArrayD<u8>) -> BgrImage {
    let shape = array.shape();
    assert_eq!(shape.len(), 3, "expected an [h, w, 3] array, got {shape:?}");
    assert_eq!(shape[2], 3, "expected three channels, got {shape:?}");
    BgrImage::new(shape[1] as u32, shape[0] as u32, flatten_u8(array))
        .expect("fixture geometry is valid")
}

pub fn sha256_bytes(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

pub fn scalar(fixture: &VerifiedFixture, name: &str) -> f64 {
    fixture.manifest["scalars"][name]
        .as_f64()
        .unwrap_or_else(|| panic!("scalars.{name} is missing or not a number"))
}

fn require_shape(path: &Path, actual: &[usize], expected: &[usize]) -> Result<(), String> {
    if actual != expected {
        return Err(format!(
            "{}: decoded shape {actual:?}, expected {expected:?}",
            path.display()
        ));
    }
    Ok(())
}

fn object<'a>(
    label: &str,
    field: &str,
    value: &'a Value,
) -> Result<&'a Map<String, Value>, String> {
    value
        .as_object()
        .ok_or_else(|| format!("{label}: {field} must be a JSON object"))
}

fn text<'a>(label: &str, field: &str, value: &'a Value) -> Result<&'a str, String> {
    value
        .as_str()
        .ok_or_else(|| format!("{label}: {field} must be a string"))
}

fn number(label: &str, field: &str, value: &Value) -> Result<u64, String> {
    value
        .as_u64()
        .ok_or_else(|| format!("{label}: {field} must be an unsigned integer"))
}

fn flag(label: &str, field: &str, value: &Value) -> Result<bool, String> {
    value
        .as_bool()
        .ok_or_else(|| format!("{label}: {field} must be a boolean"))
}

fn require_keys(
    label: &str,
    field: &str,
    map: &Map<String, Value>,
    expected: &[&str],
) -> Result<(), String> {
    let actual = map.keys().map(String::as_str).collect::<Vec<_>>();
    if actual.as_slice() != expected {
        return Err(format!(
            "{label}: {field} expected keys {expected:?}, got {actual:?}"
        ));
    }
    Ok(())
}

fn require_eq<T: std::fmt::Debug + PartialEq>(
    label: &str,
    field: &str,
    actual: T,
    expected: T,
) -> Result<(), String> {
    if actual != expected {
        return Err(format!(
            "{label}: {field} expected {expected:?}, got {actual:?}"
        ));
    }
    Ok(())
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
