use std::{
    collections::BTreeMap,
    fs::{self, File},
    io::{Read, Take},
    path::Path,
};

use burn::{
    module::Module,
    tensor::{DType, backend::Backend},
};
use burn_store::{ApplyError, ApplyResult, ModuleSnapshot, ModuleStore, SafetensorsStore};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{TrainingError, Vgg19Conv3_3};

pub const VGG19_PACKAGE_SCHEMA_VERSION: u32 = 1;
pub const VGG19_MODEL_KIND: &str = "vgg19-conv3-3";
pub const VGG19_ARCHITECTURE_VERSION: &str = "torchvision-vgg19-conv3-3-v1";
pub const VGG19_SOURCE_URL: &str = "https://download.pytorch.org/models/vgg19-dcbb9e9d.pth";

const VGG19_WEIGHT_ID: &str = "VGG19_Weights.IMAGENET1K_V1";
const MANIFEST_FILE_NAME: &str = "manifest.json";
const MODEL_FILE_NAME: &str = "model.safetensors";
const LICENSE_FILE_NAME: &str = "LICENSES.json";
const MANIFEST_MAX_BYTES: u64 = 64 * 1024;
const LICENSE_MAX_BYTES: u64 = 1024 * 1024;
const MODEL_MAX_BYTES: u64 = 16 * 1024 * 1024;
const VGG19_TENSOR_COUNT: usize = 14;
const VGG19_TOTAL_ELEMENTS: u64 = 1_735_488;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Vgg19PackageManifest {
    pub schema_version: u32,
    pub model_kind: String,
    pub architecture_version: String,
    pub source: Vgg19SourceManifest,
    pub input: Vgg19InputManifest,
    pub output_layer: String,
    pub tensor_count: usize,
    pub total_elements: u64,
    pub model: Vgg19FileManifest,
    pub licenses: Vgg19FileManifest,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Vgg19SourceManifest {
    pub framework: String,
    pub weight_id: String,
    pub url: String,
    pub sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Vgg19InputManifest {
    pub channels: usize,
    pub color_order: String,
    pub value_range: String,
    pub normalization: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Vgg19FileManifest {
    pub file_name: String,
    pub bytes: u64,
    pub sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Vgg19LicenseBundle {
    pub schema_version: u32,
    pub entries: Vec<Vgg19LicenseEntry>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Vgg19LicenseEntry {
    pub component: String,
    pub license_id: String,
    pub source_url: String,
    pub notice: String,
}

impl Vgg19PackageManifest {
    pub fn validate(&self) -> Result<(), TrainingError> {
        require_equal(
            "schema_version",
            self.schema_version,
            VGG19_PACKAGE_SCHEMA_VERSION,
        )?;
        require_equal("model_kind", self.model_kind.as_str(), VGG19_MODEL_KIND)?;
        require_equal(
            "architecture_version",
            self.architecture_version.as_str(),
            VGG19_ARCHITECTURE_VERSION,
        )?;
        require_equal(
            "source.framework",
            self.source.framework.as_str(),
            "torchvision",
        )?;
        require_equal(
            "source.weight_id",
            self.source.weight_id.as_str(),
            VGG19_WEIGHT_ID,
        )?;
        require_equal("source.url", self.source.url.as_str(), VGG19_SOURCE_URL)?;
        validate_sha256("source.sha256", &self.source.sha256)?;
        require_equal("input.channels", self.input.channels, 3)?;
        require_equal("input.color_order", self.input.color_order.as_str(), "bgr")?;
        require_equal("input.value_range", self.input.value_range.as_str(), "0..1")?;
        require_equal(
            "input.normalization",
            self.input.normalization.as_str(),
            "none",
        )?;
        require_equal("output_layer", self.output_layer.as_str(), "features.14")?;
        require_equal("tensor_count", self.tensor_count, VGG19_TENSOR_COUNT)?;
        require_equal("total_elements", self.total_elements, VGG19_TOTAL_ELEMENTS)?;
        self.model.validate("model", MODEL_FILE_NAME)?;
        self.licenses.validate("licenses", LICENSE_FILE_NAME)?;
        Ok(())
    }
}

impl Vgg19FileManifest {
    fn validate(&self, name: &str, expected_file_name: &str) -> Result<(), TrainingError> {
        require_equal(
            &format!("{name}.file_name"),
            self.file_name.as_str(),
            expected_file_name,
        )?;
        if self.bytes == 0 {
            return invalid_package(format!("{name}.bytes must be greater than zero"));
        }
        validate_sha256(&format!("{name}.sha256"), &self.sha256)
    }
}

impl Vgg19LicenseBundle {
    pub fn validate(&self) -> Result<(), TrainingError> {
        require_equal("license schema_version", self.schema_version, 1)?;
        if self.entries.is_empty() {
            return invalid_package("license bundle must contain at least one entry");
        }
        for (index, entry) in self.entries.iter().enumerate() {
            validate_non_empty(index, "component", &entry.component)?;
            validate_non_empty(index, "license_id", &entry.license_id)?;
            validate_non_empty(index, "source_url", &entry.source_url)?;
            validate_non_empty(index, "notice", &entry.notice)?;
        }
        Ok(())
    }
}

pub fn read_vgg19_manifest(
    directory: impl AsRef<Path>,
) -> Result<Vgg19PackageManifest, TrainingError> {
    let directory = directory.as_ref();
    validate_package_directory(directory)?;
    let manifest_path = directory.join(MANIFEST_FILE_NAME);
    ensure_regular_file(&manifest_path, MANIFEST_MAX_BYTES)?;
    let bytes = read_bounded(&manifest_path, MANIFEST_MAX_BYTES)?;
    let manifest: Vgg19PackageManifest = serde_json::from_slice(&bytes).map_err(|error| {
        TrainingError::InvalidPackage(format!("invalid manifest JSON: {error}"))
    })?;
    manifest.validate()?;
    Ok(manifest)
}

pub fn load_vgg19_package<B: Backend>(
    directory: impl AsRef<Path>,
    device: &B::Device,
) -> Result<Vgg19Conv3_3<B>, TrainingError> {
    let directory = directory.as_ref();
    let manifest = read_vgg19_manifest(directory)?;

    let license_path = directory.join(LICENSE_FILE_NAME);
    validate_file_integrity(&license_path, &manifest.licenses, LICENSE_MAX_BYTES)?;
    let license_bytes = read_bounded(&license_path, LICENSE_MAX_BYTES)?;
    let licenses: Vgg19LicenseBundle = serde_json::from_slice(&license_bytes).map_err(|error| {
        TrainingError::InvalidPackage(format!("invalid LICENSES.json: {error}"))
    })?;
    licenses.validate()?;

    let model_path = directory.join(MODEL_FILE_NAME);
    validate_file_integrity(&model_path, &manifest.model, MODEL_MAX_BYTES)?;
    load_strict_safetensors(&model_path, device)
}

fn load_strict_safetensors<B: Backend>(
    model_path: &Path,
    device: &B::Device,
) -> Result<Vgg19Conv3_3<B>, TrainingError> {
    let mut store = SafetensorsStore::from_file(model_path)
        .allow_partial(true)
        .validate(false);
    validate_store_snapshots(store.get_all_snapshots().map_err(store_error)?)?;

    let mut model = Vgg19Conv3_3::<B>::new_for_import(device);
    let result = model.load_from(&mut store).map_err(store_error)?;
    validate_apply_result(&result)?;
    validate_module_snapshots(&model)?;
    Ok(model.no_grad())
}

fn validate_package_directory(directory: &Path) -> Result<(), TrainingError> {
    let metadata = fs::symlink_metadata(directory)?;
    if metadata.file_type().is_symlink() {
        return invalid_package(format!(
            "package directory must not be a symbolic link: {}",
            directory.display()
        ));
    }
    if !metadata.is_dir() {
        return invalid_package(format!(
            "package path is not a directory: {}",
            directory.display()
        ));
    }

    let mut entries = fs::read_dir(directory)?
        .map(|entry| {
            entry.and_then(|entry| {
                entry.file_name().into_string().map_err(|_| {
                    std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        "package entry name is not valid UTF-8",
                    )
                })
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    entries.sort();
    let expected = [
        LICENSE_FILE_NAME.to_owned(),
        MANIFEST_FILE_NAME.to_owned(),
        MODEL_FILE_NAME.to_owned(),
    ];
    if entries != expected {
        return invalid_package(format!(
            "package directory entries must be exactly {expected:?}, got {entries:?}"
        ));
    }
    Ok(())
}

fn ensure_regular_file(path: &Path, max_bytes: u64) -> Result<fs::Metadata, TrainingError> {
    let metadata = fs::symlink_metadata(path)?;
    if metadata.file_type().is_symlink() {
        return invalid_package(format!(
            "package file must not be a symbolic link: {}",
            path.display()
        ));
    }
    if !metadata.is_file() {
        return invalid_package(format!("package entry is not a file: {}", path.display()));
    }
    if metadata.len() > max_bytes {
        return invalid_package(format!(
            "package file {} is {} bytes and exceeds {max_bytes} bytes",
            path.display(),
            metadata.len()
        ));
    }
    Ok(metadata)
}

fn validate_file_integrity(
    path: &Path,
    declared: &Vgg19FileManifest,
    max_bytes: u64,
) -> Result<(), TrainingError> {
    let metadata = ensure_regular_file(path, max_bytes)?;
    if metadata.len() != declared.bytes {
        return invalid_package(format!(
            "declared byte length for {} is {}, actual length is {}",
            declared.file_name,
            declared.bytes,
            metadata.len()
        ));
    }
    let actual = sha256_file(path)?;
    if actual != declared.sha256 {
        return Err(TrainingError::HashMismatch {
            file: declared.file_name.clone(),
            expected: declared.sha256.clone(),
            actual,
        });
    }
    Ok(())
}

fn read_bounded(path: &Path, max_bytes: u64) -> Result<Vec<u8>, TrainingError> {
    let file = File::open(path)?;
    let mut reader: Take<File> = file.take(max_bytes + 1);
    let mut bytes = Vec::new();
    reader.read_to_end(&mut bytes)?;
    if u64::try_from(bytes.len()).expect("buffer length fits u64") > max_bytes {
        return invalid_package(format!(
            "package file {} exceeds {max_bytes} bytes while reading",
            path.display()
        ));
    }
    Ok(bytes)
}

fn sha256_file(path: &Path) -> Result<String, TrainingError> {
    let mut file = File::open(path)?;
    let mut digest = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        digest.update(&buffer[..read]);
    }
    Ok(hex::encode(digest.finalize()))
}

fn validate_store_snapshots(
    snapshots: &BTreeMap<String, burn_store::TensorSnapshot>,
) -> Result<(), TrainingError> {
    let expected = expected_tensor_shapes();
    if let Some(path) = expected.keys().find(|path| !snapshots.contains_key(**path)) {
        return invalid_package(format!("missing tensor {path}"));
    }
    if let Some(path) = snapshots
        .keys()
        .find(|path| !expected.contains_key(path.as_str()))
    {
        return invalid_package(format!("unexpected tensor {path}"));
    }

    let mut total_elements = 0_u64;
    for (path, expected_shape) in &expected {
        let snapshot = snapshots
            .get(*path)
            .expect("tensor key sets were checked above");
        if snapshot.dtype != DType::F32 {
            return invalid_package(format!(
                "tensor {path} must be float32, got {:?}",
                snapshot.dtype
            ));
        }
        if snapshot.shape.as_slice() != expected_shape.as_slice() {
            return invalid_package(format!(
                "tensor {path} shape mismatch: expected {expected_shape:?}, got {:?}",
                snapshot.shape
            ));
        }
        total_elements = total_elements
            .checked_add(tensor_elements(&snapshot.shape)?)
            .ok_or_else(|| {
                TrainingError::InvalidPackage(
                    "model tensor element count overflowed u64".to_owned(),
                )
            })?;
    }
    require_equal("loaded tensor_count", snapshots.len(), VGG19_TENSOR_COUNT)?;
    require_equal(
        "loaded total_elements",
        total_elements,
        VGG19_TOTAL_ELEMENTS,
    )
}

fn validate_module_snapshots<B: Backend>(model: &Vgg19Conv3_3<B>) -> Result<(), TrainingError> {
    let snapshots = model
        .collect(None, None, false)
        .into_iter()
        .map(|snapshot| (snapshot.full_path(), snapshot))
        .collect::<BTreeMap<_, _>>();
    validate_store_snapshots(&snapshots)
}

fn validate_apply_result(result: &ApplyResult) -> Result<(), TrainingError> {
    if let Some(path) = result.missing.iter().map(|(path, _)| path).min() {
        return invalid_package(format!("missing tensor {path}"));
    }
    if let Some(error) = result.errors.first() {
        return Err(match error {
            ApplyError::ShapeMismatch { path, .. } => {
                TrainingError::InvalidPackage(format!("tensor shape mismatch: {path}"))
            }
            ApplyError::DTypeMismatch { path, .. } => {
                TrainingError::InvalidPackage(format!("tensor dtype mismatch: {path}"))
            }
            ApplyError::AdapterError { .. } | ApplyError::LoadError { .. } => {
                TrainingError::Store(error.to_string())
            }
        });
    }
    if let Some(path) = result.skipped.iter().min() {
        return invalid_package(format!("unexpected skipped tensor {path}"));
    }
    if let Some(path) = result.unused.iter().min() {
        return invalid_package(format!("unexpected tensor {path}"));
    }
    Ok(())
}

fn expected_tensor_shapes() -> BTreeMap<&'static str, Vec<usize>> {
    [
        ("conv1_1.bias", vec![64]),
        ("conv1_1.weight", vec![64, 3, 3, 3]),
        ("conv1_2.bias", vec![64]),
        ("conv1_2.weight", vec![64, 64, 3, 3]),
        ("conv2_1.bias", vec![128]),
        ("conv2_1.weight", vec![128, 64, 3, 3]),
        ("conv2_2.bias", vec![128]),
        ("conv2_2.weight", vec![128, 128, 3, 3]),
        ("conv3_1.bias", vec![256]),
        ("conv3_1.weight", vec![256, 128, 3, 3]),
        ("conv3_2.bias", vec![256]),
        ("conv3_2.weight", vec![256, 256, 3, 3]),
        ("conv3_3.bias", vec![256]),
        ("conv3_3.weight", vec![256, 256, 3, 3]),
    ]
    .into_iter()
    .collect()
}

fn tensor_elements(shape: &[usize]) -> Result<u64, TrainingError> {
    shape.iter().try_fold(1_u64, |total, dimension| {
        let dimension = u64::try_from(*dimension).map_err(|_| {
            TrainingError::InvalidPackage("tensor dimension exceeds u64".to_owned())
        })?;
        total.checked_mul(dimension).ok_or_else(|| {
            TrainingError::InvalidPackage("tensor element count overflowed u64".to_owned())
        })
    })
}

fn validate_non_empty(index: usize, name: &str, value: &str) -> Result<(), TrainingError> {
    if value.trim().is_empty() {
        return invalid_package(format!("license entry {index} {name} must be non-empty"));
    }
    Ok(())
}

fn validate_sha256(name: &str, value: &str) -> Result<(), TrainingError> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return invalid_package(format!(
            "{name} must be 64 lowercase hexadecimal characters"
        ));
    }
    Ok(())
}

fn require_equal<T>(name: &str, actual: T, expected: T) -> Result<(), TrainingError>
where
    T: PartialEq + std::fmt::Display,
{
    if actual != expected {
        return invalid_package(format!("{name} must be {expected}, got {actual}"));
    }
    Ok(())
}

fn invalid_package<T>(message: impl Into<String>) -> Result<T, TrainingError> {
    Err(TrainingError::InvalidPackage(message.into()))
}

fn store_error(error: impl std::fmt::Display) -> TrainingError {
    TrainingError::Store(error.to_string())
}
