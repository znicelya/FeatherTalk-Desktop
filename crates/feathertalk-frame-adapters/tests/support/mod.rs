#![allow(dead_code)]

use std::{
    ffi::OsStr,
    fs::File,
    io::{BufReader, Read},
    path::{Path, PathBuf},
};

use feathertalk_frame_pipeline::CommandSpec;
use feathertalk_image::BgrImage;
use ndarray::ArrayD;
use ndarray_npy::ReadNpyExt;
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};

pub const CASE: &str = "opencv_cpu_v1";

/// The value `blobFromImage` leaves in the letterbox padding: `(0.0 - 127.5) / 128.0`.
pub const PADDED_BLOB_VALUE: f32 = -0.99609375;

/// SHA-256 of the committed `feathertalk-scrfd` blob, which the generator
/// reproduces from the pattern instead of copying.
pub const REFERENCE_INPUT_SHA256: &str =
    "3d1bcdaf3874b28af337d5b596902143b59655a1bf8411c034b1aab1162f04db";

/// Committed `.npy` arrays, in manifest key order, with dtype and shape.
pub const FIXTURE_ARRAYS: [(&str, &str, &[usize]); 3] = [
    ("crop_blob.npy", "float32", &[1, 3, 192, 192]),
    ("crop_blob_padded.npy", "float32", &[1, 3, 192, 192]),
    ("frame_decode.npy", "uint8", &[640, 640, 3]),
];

/// Committed payloads that are not `.npy`, in manifest key order.
pub const FIXTURE_BLOBS: [&str; 3] = ["detections_thr002.json", "frame.jpg", "landmarks.json"];

/// The sibling `feathertalk-scrfd` arrays this case reuses, with their shapes.
pub const REFERENCE_ARRAYS: [(&str, &[usize]); 10] = [
    ("input.npy", &[1, 3, 640, 640]),
    ("out0.npy", &[1, 12800, 1]),
    ("out1.npy", &[1, 3200, 1]),
    ("out2.npy", &[1, 800, 1]),
    ("out3.npy", &[1, 12800, 4]),
    ("out4.npy", &[1, 3200, 4]),
    ("out5.npy", &[1, 800, 4]),
    ("out6.npy", &[1, 12800, 10]),
    ("out7.npy", &[1, 3200, 10]),
    ("out8.npy", &[1, 800, 10]),
];

/// The two crop geometries the fixture pins, in manifest key order.
pub const CROP_CASES: [&str; 2] = ["in_bounds", "padded"];

/// The single hash-pinned array, whose bytes are deliberately not committed.
pub const LETTERBOX_KEY: &str = "letterbox_1280x720";

#[derive(Debug)]
pub struct VerifiedFixture {
    pub root: PathBuf,
    pub manifest: Value,
}

/// Aggregate difference between a computed tensor and its reference.
#[derive(Debug, Clone, Copy)]
pub struct ParityMetrics {
    pub max_abs: f64,
    pub mean_abs: f64,
    pub max_relative: f64,
}

/// One entry of `detections_thr002.json`, in SCRFD's xywh convention.
#[derive(Debug, Clone, Copy)]
pub struct ExpectedDetection {
    pub score: f32,
    pub bbox: [f32; 4],
    pub keypoints: [[f32; 2]; 5],
}

/// The contents of `landmarks.json`.
#[derive(Debug, Clone)]
pub struct ExpectedLandmarks {
    pub bbox: [f32; 4],
    pub size: u32,
    pub origin_x: i32,
    pub origin_y: i32,
    pub points: Vec<[i32; 2]>,
}

/// One `crops` entry: the input bbox and the geometry it must produce.
#[derive(Debug, Clone)]
pub struct CropCase {
    pub bbox: [f32; 4],
    pub size: u32,
    pub origin_x: i32,
    pub origin_y: i32,
    /// Left, top, right, bottom.
    pub padding: [u32; 4],
    /// The clipped source rectangle as x, y, width, height.
    pub source: [i64; 4],
    pub array: String,
}

/// The hash-pinned 1280x720 letterbox blob plus eight spot samples.
#[derive(Debug, Clone)]
pub struct LetterboxPin {
    pub shape: Vec<usize>,
    pub sha256: String,
    /// `([channel, row, column], value)` in the `[1, 3, 640, 640]` blob.
    pub samples: Vec<([usize; 3], f32)>,
    pub source_width: u32,
    pub source_height: u32,
    pub new_width: u32,
    pub new_height: u32,
    pub pad_x: u32,
    pub pad_y: u32,
}

pub fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/opencv_cpu_v1")
}

/// The `feathertalk-scrfd` fixture is reused rather than duplicated, so it is
/// addressed as a sibling of this crate.
pub fn reference_fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../feathertalk-scrfd/tests/fixtures/opencv_cpu_v1")
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
            "arrays",
            "blobs",
            "case",
            "crops",
            "generator",
            "hashed_arrays",
            "jpeg",
            "reference_fixture",
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

    verify_source(label, &manifest)?;
    verify_generator(label, &manifest)?;
    verify_jpeg(label, &manifest)?;
    verify_arrays(label, &manifest)?;
    verify_blobs(label, &manifest)?;
    verify_crops(label, &manifest)?;
    verify_letterbox(label, &manifest)?;
    verify_reference_fixture(label, &manifest)?;

    let scalars = object(label, "scalars", &manifest["scalars"])?;
    require_keys(label, "scalars", scalars, &["level_max_scores"])?;
    let scores = reals(
        label,
        "scalars.level_max_scores",
        &manifest["scalars"]["level_max_scores"],
        3,
    )?;
    if scores.iter().any(|score| *score <= 0.0) {
        return Err(format!(
            "{label}: scalars.level_max_scores must be positive, got {scores:?}"
        ));
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
    expected_names.extend(FIXTURE_BLOBS.iter().map(|name| (*name).to_owned()));
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
        require_recorded_bytes(&path, &manifest["arrays"][name])?;
        match dtype {
            "uint8" => require_shape(&path, read_u8_array(&path)?.shape(), shape)?,
            "float32" => {
                let array = read_f32_array(&path)?;
                require_shape(&path, array.shape(), shape)?;
                check_finite(&path.display().to_string(), &array)?;
            }
            other => return Err(format!("{}: unsupported dtype {other}", path.display())),
        }
    }

    for name in FIXTURE_BLOBS {
        require_recorded_bytes(&root.join(name), &manifest["blobs"][name])?;
    }

    Ok(VerifiedFixture {
        root: root.to_owned(),
        manifest,
    })
}

/// The generator's `bgr_u8_channel_affine_v1` pattern at any size.
///
/// Unlike Task 3's helper this one is unbounded, because Task 10 needs a
/// 1280x720 frame while the committed arrays are 640x640.
pub fn pattern_bgr(width: u32, height: u32) -> BgrImage {
    let width = width as usize;
    let height = height as usize;
    let mut data = vec![0_u8; width * height * 3];
    for y in 0..height {
        for x in 0..width {
            let offset = (y * width + x) * 3;
            data[offset] = ((3 * x + 5 * y + 17) % 256) as u8;
            data[offset + 1] = ((7 * x + 11 * y + 29) % 256) as u8;
            data[offset + 2] = ((13 * x + 17 * y + 43) % 256) as u8;
        }
    }
    BgrImage::new(width as u32, height as u32, data).expect("pattern geometry is valid")
}

pub fn read_u8_array(path: &Path) -> Result<ArrayD<u8>, String> {
    let file = File::open(path).map_err(|error| format!("{}: {error}", path.display()))?;
    ArrayD::<u8>::read_npy(BufReader::new(file))
        .map_err(|error| format!("{}: {error}", path.display()))
}

pub fn read_f32_array(path: &Path) -> Result<ArrayD<f32>, String> {
    let file = File::open(path).map_err(|error| format!("{}: {error}", path.display()))?;
    ArrayD::<f32>::read_npy(BufReader::new(file))
        .map_err(|error| format!("{}: {error}", path.display()))
}

/// Read one array out of the sibling `feathertalk-scrfd` fixture.
pub fn read_reference_array(name: &str) -> ArrayD<f32> {
    let path = reference_fixture_dir().join(name);
    read_f32_array(&path).unwrap_or_else(|error| panic!("{error}"))
}

/// Row-major bytes of a fixture array, whatever its rank.
pub fn flatten_u8(array: &ArrayD<u8>) -> Vec<u8> {
    array.iter().copied().collect()
}

/// Row-major floats of a fixture array, whatever its rank.
pub fn flatten_f32(array: &ArrayD<f32>) -> Vec<f32> {
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

/// Hash a float buffer exactly the way NumPy's `tobytes` does on a
/// little-endian host, so a Rust tensor can be compared against a manifest
/// hash without committing the array.
pub fn sha256_f32_le(values: &[f32]) -> String {
    let mut digest = Sha256::new();
    for value in values {
        digest.update(value.to_le_bytes());
    }
    hex::encode(digest.finalize())
}

/// Compare two float buffers, accumulating in f64 so a long tensor cannot lose
/// the mean to rounding.
pub fn compare_f32(actual: &[f32], expected: &[f32]) -> ParityMetrics {
    assert_eq!(
        actual.len(),
        expected.len(),
        "length mismatch: {} against {}",
        actual.len(),
        expected.len()
    );
    assert!(!actual.is_empty(), "refusing to compare empty buffers");
    let mut max_abs = 0.0_f64;
    let mut sum_abs = 0.0_f64;
    let mut max_relative = 0.0_f64;
    for (left, right) in actual.iter().zip(expected) {
        let delta = (f64::from(*left) - f64::from(*right)).abs();
        sum_abs += delta;
        max_abs = max_abs.max(delta);
        max_relative = max_relative.max(delta / f64::from(right.abs()).max(1e-7));
    }
    ParityMetrics {
        max_abs,
        mean_abs: sum_abs / actual.len() as f64,
        max_relative,
    }
}

pub fn expected_detections(fixture: &VerifiedFixture) -> Vec<ExpectedDetection> {
    let document = read_json(&fixture.root.join("detections_thr002.json"));
    assert_eq!(document["schema_version"], 1);
    document["detections"]
        .as_array()
        .expect("detections must be an array")
        .iter()
        .map(|entry| {
            let raw = entry["keypoints"]
                .as_array()
                .expect("keypoints must be an array");
            assert_eq!(raw.len(), 5, "SCRFD emits five keypoints");
            let mut keypoints = [[0.0_f32; 2]; 5];
            for (slot, value) in keypoints.iter_mut().zip(raw) {
                *slot = floats::<2>(value);
            }
            ExpectedDetection {
                score: entry["score"].as_f64().expect("score must be a number") as f32,
                bbox: floats::<4>(&entry["bbox"]),
                keypoints,
            }
        })
        .collect()
}

pub fn expected_landmarks(fixture: &VerifiedFixture) -> ExpectedLandmarks {
    let document = read_json(&fixture.root.join("landmarks.json"));
    assert_eq!(document["schema_version"], 1);
    let crop = &document["crop"];
    let points = document["points"]
        .as_array()
        .expect("points must be an array")
        .iter()
        .map(|entry| {
            let pair = entry.as_array().expect("each point must be an array");
            assert_eq!(pair.len(), 2, "each point holds x and y");
            [
                pair[0].as_i64().expect("x must be an integer") as i32,
                pair[1].as_i64().expect("y must be an integer") as i32,
            ]
        })
        .collect::<Vec<_>>();
    ExpectedLandmarks {
        bbox: floats::<4>(&crop["bbox"]),
        size: unsigned_field(crop, "size"),
        origin_x: signed_field(crop, "origin_x"),
        origin_y: signed_field(crop, "origin_y"),
        points,
    }
}

pub fn crop_case(fixture: &VerifiedFixture, name: &str) -> CropCase {
    let entry = &fixture.manifest["crops"][name];
    assert!(!entry.is_null(), "crops.{name} is missing");
    let raw_padding = entry["padding"]
        .as_array()
        .expect("padding must be an array");
    assert_eq!(raw_padding.len(), 4, "padding holds four sides");
    let mut padding = [0_u32; 4];
    for (slot, value) in padding.iter_mut().zip(raw_padding) {
        *slot = value
            .as_u64()
            .expect("padding must hold unsigned integers")
            .try_into()
            .expect("padding must fit in u32");
    }
    let raw_source = entry["source"].as_array().expect("source must be an array");
    assert_eq!(raw_source.len(), 4, "source is an x, y, width, height rect");
    let mut source = [0_i64; 4];
    for (slot, value) in source.iter_mut().zip(raw_source) {
        *slot = value.as_i64().expect("source must hold integers");
    }
    CropCase {
        bbox: floats::<4>(&entry["bbox"]),
        size: unsigned_field(entry, "size"),
        origin_x: signed_field(entry, "origin_x"),
        origin_y: signed_field(entry, "origin_y"),
        padding,
        source,
        array: entry["array"]
            .as_str()
            .expect("array must be a string")
            .to_owned(),
    }
}

pub fn letterbox_pin(fixture: &VerifiedFixture) -> LetterboxPin {
    let entry = &fixture.manifest["hashed_arrays"][LETTERBOX_KEY];
    assert!(!entry.is_null(), "hashed_arrays.{LETTERBOX_KEY} is missing");
    let samples = entry["samples"]
        .as_array()
        .expect("samples must be an array")
        .iter()
        .map(|sample| {
            let values = sample.as_array().expect("each sample must be an array");
            assert_eq!(values.len(), 4, "a sample is channel, row, column, value");
            let index = [
                values[0].as_u64().expect("channel must be an integer") as usize,
                values[1].as_u64().expect("row must be an integer") as usize,
                values[2].as_u64().expect("column must be an integer") as usize,
            ];
            (
                index,
                values[3].as_f64().expect("value must be a number") as f32,
            )
        })
        .collect();
    LetterboxPin {
        shape: shape_of("fixture", LETTERBOX_KEY, &entry["shape"])
            .expect("verify_manifest checked the shape"),
        sha256: entry["sha256"]
            .as_str()
            .expect("sha256 must be a string")
            .to_owned(),
        samples,
        source_width: unsigned_field(entry, "source_width"),
        source_height: unsigned_field(entry, "source_height"),
        new_width: unsigned_field(entry, "new_width"),
        new_height: unsigned_field(entry, "new_height"),
        pad_x: unsigned_field(entry, "pad_x"),
        pad_y: unsigned_field(entry, "pad_y"),
    }
}

fn verify_source(label: &str, manifest: &Value) -> Result<(), String> {
    let source = object(label, "source", &manifest["source"])?;
    require_keys(
        label,
        "source",
        source,
        &["height", "kind", "pattern", "width"],
    )?;
    for (field, expected) in [
        ("kind", "synthetic"),
        ("pattern", "bgr_u8_channel_affine_v1"),
    ] {
        let qualified = format!("source.{field}");
        let value = text(label, &qualified, &manifest["source"][field])?;
        require_eq(label, &qualified, value, expected)?;
    }
    for field in ["height", "width"] {
        let qualified = format!("source.{field}");
        let value = number(label, &qualified, &manifest["source"][field])?;
        require_eq(label, &qualified, value, 640)?;
    }
    Ok(())
}

fn verify_generator(label: &str, manifest: &Value) -> Result<(), String> {
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
            "torch_version",
        ],
    )?;
    for (field, expected) in [
        ("backend", "opencv"),
        ("numpy_version", "2.4.6"),
        ("opencv_version", "5.0.0"),
        ("python_version", "3.11"),
        ("target", "cpu"),
        ("torch_version", "2.13.0"),
    ] {
        let qualified = format!("generator.{field}");
        let value = text(label, &qualified, &manifest["generator"][field])?;
        require_eq(label, &qualified, value, expected)?;
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
    Ok(())
}

fn verify_jpeg(label: &str, manifest: &Value) -> Result<(), String> {
    let jpeg = object(label, "jpeg", &manifest["jpeg"])?;
    require_keys(
        label,
        "jpeg",
        jpeg,
        &["optimize", "progressive", "quality", "sampling"],
    )?;
    for (field, expected) in [("optimize", 0), ("progressive", 0), ("quality", 90)] {
        let qualified = format!("jpeg.{field}");
        let value = number(label, &qualified, &manifest["jpeg"][field])?;
        require_eq(label, &qualified, value, expected)?;
    }
    require_eq(
        label,
        "jpeg.sampling",
        text(label, "jpeg.sampling", &manifest["jpeg"]["sampling"])?,
        "420",
    )?;
    Ok(())
}

fn verify_arrays(label: &str, manifest: &Value) -> Result<(), String> {
    let arrays = object(label, "arrays", &manifest["arrays"])?;
    let names = FIXTURE_ARRAYS
        .iter()
        .map(|(name, _, _)| *name)
        .collect::<Vec<_>>();
    require_keys(label, "arrays", arrays, &names)?;
    for (name, dtype, shape) in FIXTURE_ARRAYS {
        verify_array_descriptor(
            label,
            &format!("arrays.{name}"),
            &manifest["arrays"][name],
            dtype,
            shape,
        )?;
    }
    Ok(())
}

fn verify_blobs(label: &str, manifest: &Value) -> Result<(), String> {
    let blobs = object(label, "blobs", &manifest["blobs"])?;
    require_keys(label, "blobs", blobs, &FIXTURE_BLOBS)?;
    for name in FIXTURE_BLOBS {
        let qualified = format!("blobs.{name}");
        let descriptor = object(label, &qualified, &manifest["blobs"][name])?;
        require_keys(label, &qualified, descriptor, &["bytes", "sha256"])?;
        require_size_and_hash(label, &qualified, &manifest["blobs"][name])?;
    }
    Ok(())
}

fn verify_crops(label: &str, manifest: &Value) -> Result<(), String> {
    let crops = object(label, "crops", &manifest["crops"])?;
    require_keys(label, "crops", crops, &CROP_CASES)?;
    for name in CROP_CASES {
        let qualified = format!("crops.{name}");
        let entry = object(label, &qualified, &manifest["crops"][name])?;
        require_keys(
            label,
            &qualified,
            entry,
            &[
                "array", "bbox", "origin_x", "origin_y", "padding", "size", "source",
            ],
        )?;
        let value = &manifest["crops"][name];
        let array = text(label, &format!("{qualified}.array"), &value["array"])?;
        if !FIXTURE_ARRAYS
            .iter()
            .any(|(candidate, _, _)| *candidate == array)
        {
            return Err(format!(
                "{label}: {qualified}.array names an uncommitted file {array}"
            ));
        }
        reals(label, &format!("{qualified}.bbox"), &value["bbox"], 4)?;
        let size = number(label, &format!("{qualified}.size"), &value["size"])?;
        if size == 0 {
            return Err(format!("{label}: {qualified}.size must be positive"));
        }
        signed(label, &format!("{qualified}.origin_x"), &value["origin_x"])?;
        signed(label, &format!("{qualified}.origin_y"), &value["origin_y"])?;
        let padding = integers(label, &format!("{qualified}.padding"), &value["padding"], 4)?;
        if padding.iter().any(|side| *side >= size) {
            return Err(format!(
                "{label}: {qualified}.padding {padding:?} does not fit in size {size}"
            ));
        }
        let source = integers(label, &format!("{qualified}.source"), &value["source"], 4)?;
        if source[2] == 0 || source[3] == 0 {
            return Err(format!(
                "{label}: {qualified}.source must have a positive extent, got {source:?}"
            ));
        }
    }
    Ok(())
}

fn verify_letterbox(label: &str, manifest: &Value) -> Result<(), String> {
    let hashed = object(label, "hashed_arrays", &manifest["hashed_arrays"])?;
    require_keys(label, "hashed_arrays", hashed, &[LETTERBOX_KEY])?;
    let qualified = format!("hashed_arrays.{LETTERBOX_KEY}");
    let entry = &manifest["hashed_arrays"][LETTERBOX_KEY];
    let map = object(label, &qualified, entry)?;
    require_keys(
        label,
        &qualified,
        map,
        &[
            "dtype",
            "new_height",
            "new_width",
            "pad_x",
            "pad_y",
            "samples",
            "sha256",
            "shape",
            "source_height",
            "source_width",
        ],
    )?;
    require_eq(
        label,
        &format!("{qualified}.dtype"),
        text(label, &format!("{qualified}.dtype"), &entry["dtype"])?,
        "float32",
    )?;
    let shape = shape_of(label, &qualified, &entry["shape"])?;
    if shape.as_slice() != [1, 3, 640, 640] {
        return Err(format!(
            "{label}: {qualified}.shape expected [1, 3, 640, 640], got {shape:?}"
        ));
    }
    require_hash(label, &qualified, entry)?;
    for (field, expected) in [
        ("new_height", 361),
        ("new_width", 640),
        ("pad_x", 0),
        ("pad_y", 139),
        ("source_height", 720),
        ("source_width", 1280),
    ] {
        let field_label = format!("{qualified}.{field}");
        let value = number(label, &field_label, &entry[field])?;
        require_eq(label, &field_label, value, expected)?;
    }

    let samples = entry["samples"]
        .as_array()
        .ok_or_else(|| format!("{label}: {qualified}.samples must be an array"))?;
    if samples.len() != 8 {
        return Err(format!(
            "{label}: {qualified}.samples expected 8 entries, got {}",
            samples.len()
        ));
    }
    for (position, sample) in samples.iter().enumerate() {
        let field = format!("{qualified}.samples[{position}]");
        let values = sample
            .as_array()
            .filter(|values| values.len() == 4)
            .ok_or_else(|| format!("{label}: {field} must be a four element array"))?;
        for (axis, bound) in [3_u64, 640, 640].into_iter().enumerate() {
            let index = values[axis]
                .as_u64()
                .ok_or_else(|| format!("{label}: {field}[{axis}] must be an unsigned integer"))?;
            if index >= bound {
                return Err(format!(
                    "{label}: {field}[{axis}] is {index}, outside 0..{bound}"
                ));
            }
        }
        values[3]
            .as_f64()
            .filter(|value| value.is_finite())
            .ok_or_else(|| format!("{label}: {field}[3] must be a finite number"))?;
    }
    Ok(())
}

fn verify_reference_fixture(label: &str, manifest: &Value) -> Result<(), String> {
    let reference = object(label, "reference_fixture", &manifest["reference_fixture"])?;
    require_keys(
        label,
        "reference_fixture",
        reference,
        &["case", "files", "relative_path"],
    )?;
    let entry = &manifest["reference_fixture"];
    require_eq(
        label,
        "reference_fixture.case",
        text(label, "reference_fixture.case", &entry["case"])?,
        CASE,
    )?;
    require_eq(
        label,
        "reference_fixture.relative_path",
        text(
            label,
            "reference_fixture.relative_path",
            &entry["relative_path"],
        )?,
        "../feathertalk-scrfd/tests/fixtures/opencv_cpu_v1",
    )?;
    let files = object(label, "reference_fixture.files", &entry["files"])?;
    let names = REFERENCE_ARRAYS
        .iter()
        .map(|(name, _)| *name)
        .collect::<Vec<_>>();
    require_keys(label, "reference_fixture.files", files, &names)?;
    for (name, shape) in REFERENCE_ARRAYS {
        verify_array_descriptor(
            label,
            &format!("reference_fixture.files.{name}"),
            &entry["files"][name],
            "float32",
            shape,
        )?;
    }
    require_eq(
        label,
        "reference_fixture.files.input.npy.sha256",
        text(
            label,
            "reference_fixture.files.input.npy.sha256",
            &entry["files"]["input.npy"]["sha256"],
        )?,
        REFERENCE_INPUT_SHA256,
    )?;
    Ok(())
}

fn verify_array_descriptor(
    label: &str,
    qualified: &str,
    descriptor: &Value,
    dtype: &str,
    shape: &[usize],
) -> Result<(), String> {
    let map = object(label, qualified, descriptor)?;
    require_keys(
        label,
        qualified,
        map,
        &["bytes", "dtype", "sha256", "shape"],
    )?;
    require_eq(
        label,
        &format!("{qualified}.dtype"),
        text(label, &format!("{qualified}.dtype"), &descriptor["dtype"])?,
        dtype,
    )?;
    let actual = shape_of(label, qualified, &descriptor["shape"])?;
    if actual.as_slice() != shape {
        return Err(format!(
            "{label}: {qualified}.shape expected {shape:?}, got {actual:?}"
        ));
    }
    require_size_and_hash(label, qualified, descriptor)
}

fn require_size_and_hash(label: &str, qualified: &str, descriptor: &Value) -> Result<(), String> {
    number(label, &format!("{qualified}.bytes"), &descriptor["bytes"])?;
    require_hash(label, qualified, descriptor)
}

fn require_hash(label: &str, qualified: &str, descriptor: &Value) -> Result<(), String> {
    let sha256 = text(label, &format!("{qualified}.sha256"), &descriptor["sha256"])?;
    if sha256.len() != 64 || !sha256.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(format!(
            "{label}: {qualified}.sha256 is not a 64 digit hex string"
        ));
    }
    Ok(())
}

fn require_recorded_bytes(path: &Path, descriptor: &Value) -> Result<(), String> {
    let (actual_bytes, actual_sha256) = stream_hash(path)?;
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
    Ok(())
}

fn read_json(path: &Path) -> Value {
    let bytes = std::fs::read(path).unwrap_or_else(|error| panic!("{}: {error}", path.display()));
    serde_json::from_slice(&bytes).unwrap_or_else(|error| panic!("{}: {error}", path.display()))
}

fn floats<const N: usize>(value: &Value) -> [f32; N] {
    let values = value
        .as_array()
        .unwrap_or_else(|| panic!("expected an array of {N} numbers, got {value}"));
    assert_eq!(values.len(), N, "expected {N} numbers, got {values:?}");
    let mut out = [0.0_f32; N];
    for (slot, entry) in out.iter_mut().zip(values) {
        *slot = entry
            .as_f64()
            .unwrap_or_else(|| panic!("expected a number, got {entry}")) as f32;
    }
    out
}

fn unsigned_field(entry: &Value, name: &str) -> u32 {
    entry[name]
        .as_u64()
        .unwrap_or_else(|| panic!("{name} must be an unsigned integer"))
        .try_into()
        .unwrap_or_else(|_| panic!("{name} does not fit in u32"))
}

fn signed_field(entry: &Value, name: &str) -> i32 {
    entry[name]
        .as_i64()
        .unwrap_or_else(|| panic!("{name} must be an integer"))
        .try_into()
        .unwrap_or_else(|_| panic!("{name} does not fit in i32"))
}

fn check_finite(label: &str, array: &ArrayD<f32>) -> Result<(), String> {
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

fn require_shape(path: &Path, actual: &[usize], expected: &[usize]) -> Result<(), String> {
    if actual != expected {
        return Err(format!(
            "{}: decoded shape {actual:?}, expected {expected:?}",
            path.display()
        ));
    }
    Ok(())
}

fn shape_of(label: &str, qualified: &str, value: &Value) -> Result<Vec<usize>, String> {
    value
        .as_array()
        .ok_or_else(|| format!("{label}: {qualified}.shape must be an array"))?
        .iter()
        .map(|entry| {
            entry
                .as_u64()
                .filter(|entry| *entry > 0)
                .map(|entry| entry as usize)
                .ok_or_else(|| format!("{label}: {qualified}.shape must hold positive integers"))
        })
        .collect()
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

fn signed(label: &str, field: &str, value: &Value) -> Result<i64, String> {
    value
        .as_i64()
        .ok_or_else(|| format!("{label}: {field} must be an integer"))
}

fn flag(label: &str, field: &str, value: &Value) -> Result<bool, String> {
    value
        .as_bool()
        .ok_or_else(|| format!("{label}: {field} must be a boolean"))
}

fn reals(label: &str, field: &str, value: &Value, count: usize) -> Result<Vec<f64>, String> {
    let values = value
        .as_array()
        .filter(|values| values.len() == count)
        .ok_or_else(|| format!("{label}: {field} must be an array of {count} numbers"))?;
    values
        .iter()
        .map(|entry| {
            entry
                .as_f64()
                .filter(|entry| entry.is_finite())
                .ok_or_else(|| format!("{label}: {field} must hold finite numbers"))
        })
        .collect()
}

fn integers(label: &str, field: &str, value: &Value, count: usize) -> Result<Vec<u64>, String> {
    let values = value
        .as_array()
        .filter(|values| values.len() == count)
        .ok_or_else(|| format!("{label}: {field} must be an array of {count} unsigned integers"))?;
    values
        .iter()
        .map(|entry| {
            entry
                .as_u64()
                .ok_or_else(|| format!("{label}: {field} must hold unsigned integers"))
        })
        .collect()
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

// ---------------------------------------------------------------------------
// demo_frame_v1: the real video frame Tasks 12, 14 and 15 evaluate.
// ---------------------------------------------------------------------------

/// Re-exported so the contract test can size the landmark array without
/// naming `feathertalk-pfld` itself.
pub use feathertalk_pfld::PFLD_LANDMARK_COUNT;

pub const DEMO_CASE: &str = "demo_frame_v1";

/// The tracked demo clip, relative to the repository root.
pub const DEMO_VIDEO: &str = "demo/feathertalk_demo_latest_188.mp4";

pub const DEMO_VIDEO_SHA256: &str =
    "9353ad796089aa104765d651ca99f158349cfd203644923b2fa72f68b44e9ac1";

/// The zero based frame the fixture was cut from.
pub const DEMO_FRAME_INDEX: u64 = 750;

/// Committed JPEG payloads, in manifest key order.
pub const DEMO_BLOBS: [&str; 2] = ["frame.jpg", "frame_blurred.jpg"];

/// The two evaluated frames, in manifest key order.
pub const DEMO_FRAMES: [&str; 2] = ["blurred", "sharp"];

#[derive(Debug, Clone, Copy)]
pub struct DemoDetection {
    pub score: f32,
    /// `[x, y, width, height]` in source pixels.
    pub bbox: [f32; 4],
    pub keypoints: [[f32; 2]; 5],
}

#[derive(Debug, Clone, Copy)]
pub struct DemoCrop {
    pub size: u32,
    pub origin_x: i32,
    pub origin_y: i32,
    /// `[left, top, right, bottom]`.
    pub padding: [u32; 4],
    /// `[x, y, width, height]` of the clipped source rectangle.
    pub source: [i64; 4],
}

#[derive(Debug, Clone)]
pub struct DemoFrame {
    pub path: PathBuf,
    pub laplacian_variance: f64,
    pub level_max_scores: Vec<f32>,
    pub detection: DemoDetection,
    pub crop: DemoCrop,
    pub landmarks: Vec<[i32; 2]>,
}

pub fn demo_fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/demo_frame_v1")
}

/// The tracked clip lives at the repository root, three levels above
/// `rust/crates/feathertalk-frame-adapters`.
pub fn demo_video_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../..")
        .join(DEMO_VIDEO)
}

pub fn load_and_verify_demo_fixture() -> Result<VerifiedFixture, String> {
    let root = demo_fixture_dir();
    let manifest_path = root.join("fixture.json");
    let label = manifest_path.display().to_string();
    let bytes = std::fs::read(&manifest_path).map_err(|error| format!("{label}: {error}"))?;
    let manifest = verify_demo_manifest(&label, &bytes)?;

    let mut actual_names = std::fs::read_dir(&root)
        .map_err(|error| format!("{}: {error}", root.display()))?
        .map(|entry| {
            entry
                .map(|entry| entry.file_name().to_string_lossy().into_owned())
                .map_err(|error| format!("{}: {error}", root.display()))
        })
        .collect::<Result<Vec<_>, _>>()?;
    actual_names.sort();
    let mut expected_names = DEMO_BLOBS
        .iter()
        .map(|name| (*name).to_owned())
        .collect::<Vec<_>>();
    expected_names.push("fixture.json".to_owned());
    expected_names.sort();
    if actual_names != expected_names {
        return Err(format!(
            "{}: expected files {expected_names:?}, got {actual_names:?}",
            root.display()
        ));
    }

    for name in DEMO_BLOBS {
        require_recorded_bytes(&root.join(name), &manifest["blobs"][name])?;
    }

    Ok(VerifiedFixture { root, manifest })
}

/// Validate every manifest field without touching the filesystem.
///
/// The `detection_config` values are only range checked here; the contract test
/// is what compares them against the pipeline constants, so this module needs no
/// `feathertalk-frame-pipeline` import.
pub fn verify_demo_manifest(label: &str, bytes: &[u8]) -> Result<Value, String> {
    let manifest: Value =
        serde_json::from_slice(bytes).map_err(|error| format!("{label}: {error}"))?;
    let root = object(label, "manifest", &manifest)?;
    require_keys(
        label,
        "manifest",
        root,
        &[
            "blobs",
            "blur",
            "case",
            "detection_config",
            "frames",
            "generator",
            "jpeg",
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
    require_eq(
        label,
        "case",
        text(label, "case", &manifest["case"])?,
        DEMO_CASE,
    )?;

    verify_demo_source(label, &manifest)?;
    verify_generator(label, &manifest)?;
    verify_jpeg(label, &manifest)?;
    verify_demo_blur(label, &manifest)?;
    verify_demo_detection_config(label, &manifest)?;
    verify_demo_blobs(label, &manifest)?;
    verify_demo_frames(label, &manifest)?;

    Ok(manifest)
}

pub fn demo_frame(fixture: &VerifiedFixture, name: &str) -> DemoFrame {
    let entry = &fixture.manifest["frames"][name];
    assert!(entry.is_object(), "unknown demo frame {name}");

    let detection = &entry["detection"];
    let mut keypoints = [[0.0_f32; 2]; 5];
    for (slot, point) in keypoints
        .iter_mut()
        .zip(detection["keypoints"].as_array().expect("verified above"))
    {
        *slot = floats::<2>(point);
    }

    let crop = &entry["crop"];
    let mut padding = [0_u32; 4];
    for (slot, value) in padding
        .iter_mut()
        .zip(crop["padding"].as_array().expect("verified above"))
    {
        *slot = value
            .as_u64()
            .expect("verified above")
            .try_into()
            .unwrap_or_else(|_| panic!("padding does not fit in u32: {value}"));
    }
    let mut source = [0_i64; 4];
    for (slot, value) in source
        .iter_mut()
        .zip(crop["source"].as_array().expect("verified above"))
    {
        *slot = value.as_i64().expect("verified above");
    }

    DemoFrame {
        path: fixture
            .root
            .join(entry["blob"].as_str().expect("verified above")),
        laplacian_variance: entry["laplacian_variance"]
            .as_f64()
            .expect("verified above"),
        level_max_scores: floats::<3>(&entry["level_max_scores"]).to_vec(),
        detection: DemoDetection {
            score: detection["score"].as_f64().expect("verified above") as f32,
            bbox: floats::<4>(&detection["bbox"]),
            keypoints,
        },
        crop: DemoCrop {
            size: unsigned_field(crop, "size"),
            origin_x: signed_field(crop, "origin_x"),
            origin_y: signed_field(crop, "origin_y"),
            padding,
            source,
        },
        landmarks: entry["landmarks"]
            .as_array()
            .expect("verified above")
            .iter()
            .map(|point| {
                let pair = point.as_array().expect("verified above");
                [
                    signed_field_at(pair, 0, "landmark"),
                    signed_field_at(pair, 1, "landmark"),
                ]
            })
            .collect(),
    }
}

fn verify_demo_source(label: &str, manifest: &Value) -> Result<(), String> {
    let source = object(label, "source", &manifest["source"])?;
    require_keys(
        label,
        "source",
        source,
        &[
            "extraction",
            "fps",
            "frame_count",
            "frame_index",
            "height",
            "kind",
            "raw_bgr_sha256",
            "sha256",
            "video",
            "width",
        ],
    )?;
    for (field, expected) in [
        ("kind", "video_frame"),
        ("video", DEMO_VIDEO),
        ("sha256", DEMO_VIDEO_SHA256),
    ] {
        let qualified = format!("source.{field}");
        let value = text(label, &qualified, &manifest["source"][field])?;
        require_eq(label, &qualified, value, expected)?;
    }
    for (field, expected) in [
        ("width", 1280),
        ("height", 720),
        ("frame_count", 1511),
        ("frame_index", DEMO_FRAME_INDEX),
    ] {
        let qualified = format!("source.{field}");
        let value = number(label, &qualified, &manifest["source"][field])?;
        require_eq(label, &qualified, value, expected)?;
    }
    let fps = manifest["source"]["fps"]
        .as_f64()
        .ok_or_else(|| format!("{label}: source.fps must be a number"))?;
    require_eq(label, "source.fps", fps, 25.0)?;

    // `require_hash` looks for a field named `sha256`, so the raw frame digest
    // is shape checked here instead.
    let raw = text(
        label,
        "source.raw_bgr_sha256",
        &manifest["source"]["raw_bgr_sha256"],
    )?;
    if raw.len() != 64 || !raw.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(format!(
            "{label}: source.raw_bgr_sha256 is not a 64 digit hex string"
        ));
    }

    let extraction = object(
        label,
        "source.extraction",
        &manifest["source"]["extraction"],
    )?;
    require_keys(
        label,
        "source.extraction",
        extraction,
        &["arguments", "tool"],
    )?;
    require_eq(
        label,
        "source.extraction.tool",
        text(
            label,
            "source.extraction.tool",
            &manifest["source"]["extraction"]["tool"],
        )?,
        "ffmpeg",
    )?;
    let arguments = manifest["source"]["extraction"]["arguments"]
        .as_array()
        .filter(|arguments| !arguments.is_empty())
        .ok_or_else(|| format!("{label}: source.extraction.arguments must be a non-empty array"))?;
    let selector = format!("eq(n\\,{DEMO_FRAME_INDEX})");
    let mut selects_the_frame = false;
    for (index, entry) in arguments.iter().enumerate() {
        let argument = text(
            label,
            &format!("source.extraction.arguments[{index}]"),
            entry,
        )?;
        if argument.contains(&selector) {
            selects_the_frame = true;
        }
    }
    if !selects_the_frame {
        return Err(format!(
            "{label}: source.extraction.arguments must contain {selector}"
        ));
    }
    Ok(())
}

fn verify_demo_blur(label: &str, manifest: &Value) -> Result<(), String> {
    let blur = object(label, "blur", &manifest["blur"])?;
    require_keys(label, "blur", blur, &["kernel", "sigma"])?;
    // OpenCV only accepts odd Gaussian kernels; 19 is the pinned value.
    require_eq(
        label,
        "blur.kernel",
        number(label, "blur.kernel", &manifest["blur"]["kernel"])?,
        19,
    )?;
    let sigma = manifest["blur"]["sigma"]
        .as_f64()
        .ok_or_else(|| format!("{label}: blur.sigma must be a number"))?;
    require_eq(label, "blur.sigma", sigma, 3.0)
}

fn verify_demo_detection_config(label: &str, manifest: &Value) -> Result<(), String> {
    let config = object(label, "detection_config", &manifest["detection_config"])?;
    require_keys(
        label,
        "detection_config",
        config,
        &["confidence_threshold", "nms_iou_threshold"],
    )?;
    for field in ["confidence_threshold", "nms_iou_threshold"] {
        let qualified = format!("detection_config.{field}");
        if !manifest["detection_config"][field]
            .as_f64()
            .is_some_and(|value| value.is_finite() && (0.0..=1.0).contains(&value))
        {
            return Err(format!("{label}: {qualified} must be a number in 0..=1"));
        }
    }
    Ok(())
}

fn verify_demo_blobs(label: &str, manifest: &Value) -> Result<(), String> {
    let blobs = object(label, "blobs", &manifest["blobs"])?;
    require_keys(label, "blobs", blobs, &DEMO_BLOBS)?;
    for name in DEMO_BLOBS {
        let qualified = format!("blobs.{name}");
        let descriptor = object(label, &qualified, &manifest["blobs"][name])?;
        require_keys(label, &qualified, descriptor, &["bytes", "sha256"])?;
        require_size_and_hash(label, &qualified, &manifest["blobs"][name])?;
    }
    Ok(())
}

fn verify_demo_frames(label: &str, manifest: &Value) -> Result<(), String> {
    let frames = object(label, "frames", &manifest["frames"])?;
    require_keys(label, "frames", frames, &DEMO_FRAMES)?;
    for name in DEMO_FRAMES {
        let qualified = format!("frames.{name}");
        let entry = object(label, &qualified, &manifest["frames"][name])?;
        require_keys(
            label,
            &qualified,
            entry,
            &[
                "blob",
                "crop",
                "detection",
                "landmarks",
                "laplacian_variance",
                "level_max_scores",
            ],
        )?;

        let blob = text(
            label,
            &format!("{qualified}.blob"),
            &manifest["frames"][name]["blob"],
        )?;
        if !DEMO_BLOBS.contains(&blob) {
            return Err(format!(
                "{label}: {qualified}.blob must name a committed payload, got {blob}"
            ));
        }

        if !manifest["frames"][name]["laplacian_variance"]
            .as_f64()
            .is_some_and(|variance| variance.is_finite() && variance > 0.0)
        {
            return Err(format!(
                "{label}: {qualified}.laplacian_variance must be a positive number"
            ));
        }

        let scores = reals(
            label,
            &format!("{qualified}.level_max_scores"),
            &manifest["frames"][name]["level_max_scores"],
            3,
        )?;
        if scores.iter().any(|score| !(0.0..=1.0).contains(score)) {
            return Err(format!(
                "{label}: {qualified}.level_max_scores must lie in 0..=1, got {scores:?}"
            ));
        }

        verify_demo_detection(label, &qualified, &manifest["frames"][name]["detection"])?;
        verify_demo_crop(label, &qualified, &manifest["frames"][name]["crop"])?;
        verify_demo_landmarks(label, &qualified, &manifest["frames"][name]["landmarks"])?;
    }
    Ok(())
}

fn verify_demo_detection(label: &str, frame: &str, value: &Value) -> Result<(), String> {
    let qualified = format!("{frame}.detection");
    let detection = object(label, &qualified, value)?;
    require_keys(
        label,
        &qualified,
        detection,
        &["bbox", "keypoints", "score"],
    )?;
    if !value["score"]
        .as_f64()
        .is_some_and(|score| score.is_finite() && (0.0..=1.0).contains(&score))
    {
        return Err(format!(
            "{label}: {qualified}.score must be a number in 0..=1"
        ));
    }
    let bbox = reals(label, &format!("{qualified}.bbox"), &value["bbox"], 4)?;
    if bbox[2] <= 0.0 || bbox[3] <= 0.0 {
        return Err(format!(
            "{label}: {qualified}.bbox must have positive extent, got {bbox:?}"
        ));
    }
    let keypoints = value["keypoints"]
        .as_array()
        .filter(|points| points.len() == 5)
        .ok_or_else(|| format!("{label}: {qualified}.keypoints must hold 5 points"))?;
    for (index, point) in keypoints.iter().enumerate() {
        reals(label, &format!("{qualified}.keypoints[{index}]"), point, 2)?;
    }
    Ok(())
}

fn verify_demo_crop(label: &str, frame: &str, value: &Value) -> Result<(), String> {
    let qualified = format!("{frame}.crop");
    let crop = object(label, &qualified, value)?;
    require_keys(
        label,
        &qualified,
        crop,
        &["origin_x", "origin_y", "padding", "size", "source"],
    )?;
    let size = number(label, &format!("{qualified}.size"), &value["size"])?;
    if size == 0 {
        return Err(format!("{label}: {qualified}.size must be positive"));
    }
    signed(label, &format!("{qualified}.origin_x"), &value["origin_x"])?;
    signed(label, &format!("{qualified}.origin_y"), &value["origin_y"])?;
    integers(label, &format!("{qualified}.padding"), &value["padding"], 4)?;
    let source = value["source"]
        .as_array()
        .filter(|source| source.len() == 4)
        .ok_or_else(|| format!("{label}: {qualified}.source must hold 4 integers"))?;
    for (index, entry) in source.iter().enumerate() {
        signed(label, &format!("{qualified}.source[{index}]"), entry)?;
    }
    Ok(())
}

fn verify_demo_landmarks(label: &str, frame: &str, value: &Value) -> Result<(), String> {
    let qualified = format!("{frame}.landmarks");
    let points = value
        .as_array()
        .filter(|points| points.len() == PFLD_LANDMARK_COUNT)
        .ok_or_else(|| format!("{label}: {qualified} must hold {PFLD_LANDMARK_COUNT} points"))?;
    for (index, point) in points.iter().enumerate() {
        let pair = point
            .as_array()
            .filter(|pair| pair.len() == 2)
            .ok_or_else(|| format!("{label}: {qualified}[{index}] must be a pair"))?;
        for entry in pair {
            signed(label, &format!("{qualified}[{index}]"), entry)?;
        }
    }
    Ok(())
}

fn signed_field_at(pair: &[Value], index: usize, name: &str) -> i32 {
    pair[index]
        .as_i64()
        .unwrap_or_else(|| panic!("{name}[{index}] must be an integer"))
        .try_into()
        .unwrap_or_else(|_| panic!("{name}[{index}] does not fit in i32"))
}

/// The value ffmpeg receives after `flag`, as text.
///
/// A copy of the `feathertalk-frame-pipeline` test helper: test modules do not
/// cross crate boundaries.
pub fn flag_value(command: &CommandSpec, flag: &str) -> String {
    let arguments = command.arguments();
    let position = arguments
        .iter()
        .position(|argument| argument.as_os_str() == OsStr::new(flag))
        .unwrap_or_else(|| panic!("{flag} is missing from the frame command"));
    arguments
        .get(position + 1)
        .unwrap_or_else(|| panic!("{flag} carries no value"))
        .to_str()
        .unwrap_or_else(|| panic!("{flag} carries non-UTF-8 text"))
        .to_owned()
}

/// The same value parsed as a frame counter.
pub fn flag_number(command: &CommandSpec, flag: &str) -> u64 {
    flag_value(command, flag)
        .parse()
        .unwrap_or_else(|_| panic!("{flag} must carry a number"))
}

/// The frames one chunk command is expected to write, as `(index, path)` pairs.
///
/// Fake runners use this to stand in for ffmpeg's `image2` muxer: the command
/// ends with a `%06d.jpg` pattern, and `-start_number` plus `-frames:v` say
/// which indices that pattern expands to.
pub fn chunk_outputs(command: &CommandSpec) -> Vec<(u64, PathBuf)> {
    let pattern = Path::new(
        command
            .arguments()
            .last()
            .expect("the frame command ends with the output pattern"),
    );
    let directory = pattern
        .parent()
        .expect("the output pattern sits inside the frames directory")
        .to_owned();
    let first = flag_number(command, "-start_number");
    let count = flag_number(command, "-frames:v");
    (first..first + count)
        .map(|index| (index, directory.join(format!("{index:06}.jpg"))))
        .collect()
}
