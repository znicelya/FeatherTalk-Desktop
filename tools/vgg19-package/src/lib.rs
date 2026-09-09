use std::{
    collections::BTreeMap,
    fs::{self, File, OpenOptions},
    io::{Read, Write},
    path::{Path, PathBuf},
};

use burn::{backend::NdArray, tensor::backend::Backend};
use burn_store::ModuleSnapshot;
use feathertalk_training::{
    VGG19_ARCHITECTURE_VERSION, VGG19_MODEL_KIND, VGG19_PACKAGE_SCHEMA_VERSION, VGG19_SOURCE_URL,
    Vgg19FileManifest, Vgg19InputManifest, Vgg19LicenseBundle, Vgg19PackageManifest,
    Vgg19SourceManifest, load_vgg19_package, read_vgg19_manifest,
};
use feathertalk_weights::{
    LegacyImportRequest, LegacyModelKind, WeightImportError, import_into, save_safetensors,
};
use sha2::{Digest, Sha256};
use thiserror::Error;

type CpuBackend = NdArray<f32>;

const SOURCE_MAX_BYTES: u64 = 1024 * 1024 * 1024;
const LICENSE_MAX_BYTES: u64 = 1024 * 1024;
const MAX_TENSOR_COUNT: usize = 64;
const MAX_TOTAL_ELEMENTS: u64 = 200_000_000;
const MODEL_FILE_NAME: &str = "model.safetensors";
const LICENSE_FILE_NAME: &str = "LICENSES.json";
const MANIFEST_FILE_NAME: &str = "manifest.json";

#[derive(Debug, Clone)]
pub struct Vgg19PackageRequest {
    pub source: PathBuf,
    pub licenses: PathBuf,
    pub destination: PathBuf,
}

#[derive(Debug, Clone)]
pub struct Vgg19PackageReport {
    pub manifest: Vgg19PackageManifest,
}

#[derive(Debug, Error)]
pub enum PackageError {
    #[error("invalid package request: {0}")]
    InvalidRequest(String),
    #[error("training error: {0}")]
    Training(#[from] feathertalk_training::TrainingError),
    #[error("weight import error: {0}")]
    WeightImport(#[from] WeightImportError),
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("package publication error: {0}")]
    Publication(String),
}

pub fn build_vgg19_package(
    request: &Vgg19PackageRequest,
) -> Result<Vgg19PackageReport, PackageError> {
    build_vgg19_package_with_validation_hook(request, |_| Ok(()))
}

fn build_vgg19_package_with_validation_hook<F>(
    request: &Vgg19PackageRequest,
    validation_hook: F,
) -> Result<Vgg19PackageReport, PackageError>
where
    F: FnOnce(&Path) -> std::io::Result<()>,
{
    validate_request_paths(request)?;
    ensure_destination_absent(&request.destination)?;
    let parent = request
        .destination
        .parent()
        .ok_or_else(|| PackageError::InvalidRequest("destination has no parent".to_owned()))?;
    validate_existing_directory(parent, "destination parent")?;

    // Validate the small license input before touching the potentially large source checkpoint.
    let license_bytes = read_bounded_regular(&request.licenses, LICENSE_MAX_BYTES)?;
    let licenses: Vgg19LicenseBundle = serde_json::from_slice(&license_bytes).map_err(|error| {
        PackageError::Training(feathertalk_training::TrainingError::InvalidPackage(
            format!("invalid license JSON: {error}"),
        ))
    })?;
    licenses.validate()?;

    let source_snapshot = Snapshot::create(&request.source, SOURCE_MAX_BYTES)?;
    let device = Default::default();
    let mut candidate = feathertalk_training::Vgg19Conv3_3::<CpuBackend>::new_for_import(&device);
    let report = import_into::<CpuBackend, _>(
        &mut candidate,
        &LegacyImportRequest {
            path: source_snapshot.path.clone(),
            kind: LegacyModelKind::Vgg19Conv3_3,
            top_level_key: None,
            max_file_bytes: SOURCE_MAX_BYTES,
            max_tensor_count: MAX_TENSOR_COUNT,
            max_total_elements: MAX_TOTAL_ELEMENTS,
        },
    )?;
    if report.applied.len() != 14
        || report.ignored.len() != 24
        || report.tensor_count != 14
        || report.total_elements != 1_735_488
    {
        return Err(PackageError::InvalidRequest(format!(
            "VGG19 import report does not match the exact contract: applied {}, ignored {}, tensors {}, elements {}",
            report.applied.len(),
            report.ignored.len(),
            report.tensor_count,
            report.total_elements
        )));
    }

    let staging = tempfile::Builder::new()
        .prefix(".feathertalk-vgg19-")
        .tempdir_in(parent)?;
    let staging_path = staging.path().to_owned();
    let model_path = staging_path.join(MODEL_FILE_NAME);
    save_safetensors::<CpuBackend, _>(&candidate, &model_path)?;
    let model_manifest = file_manifest(&model_path, MODEL_FILE_NAME)?;

    let license_path = staging_path.join(LICENSE_FILE_NAME);
    copy_create_new(&license_bytes, &license_path)?;
    let license_manifest = file_manifest(&license_path, LICENSE_FILE_NAME)?;

    let manifest = Vgg19PackageManifest {
        schema_version: VGG19_PACKAGE_SCHEMA_VERSION,
        model_kind: VGG19_MODEL_KIND.to_owned(),
        architecture_version: VGG19_ARCHITECTURE_VERSION.to_owned(),
        source: Vgg19SourceManifest {
            framework: "torchvision".to_owned(),
            weight_id: "VGG19_Weights.IMAGENET1K_V1".to_owned(),
            url: VGG19_SOURCE_URL.to_owned(),
            sha256: source_snapshot.sha256.clone(),
        },
        input: Vgg19InputManifest {
            channels: 3,
            color_order: "bgr".to_owned(),
            value_range: "0..1".to_owned(),
            normalization: "none".to_owned(),
        },
        output_layer: "features.14".to_owned(),
        tensor_count: report.tensor_count,
        total_elements: report.total_elements,
        model: model_manifest,
        licenses: license_manifest,
    };
    write_manifest(&staging_path, &manifest)?;
    validation_hook(&staging_path).map_err(|error| {
        PackageError::Publication(format!("staging validation hook failed: {error}"))
    })?;
    manifest.validate()?;
    let parsed_manifest = read_vgg19_manifest(&staging_path)?;
    if parsed_manifest != manifest {
        return Err(PackageError::Publication(
            "staged manifest changed after writing".to_owned(),
        ));
    }

    let reloaded = load_vgg19_package::<CpuBackend>(&staging_path, &device)?;
    compare_module_snapshots(&candidate, &reloaded).map_err(PackageError::Publication)?;
    let entries = exact_entries(&staging_path)?;
    if entries
        != [
            LICENSE_FILE_NAME.to_owned(),
            MANIFEST_FILE_NAME.to_owned(),
            MODEL_FILE_NAME.to_owned(),
        ]
    {
        return Err(PackageError::Publication(format!(
            "unexpected staging entries: {entries:?}"
        )));
    }

    ensure_destination_absent(&request.destination)?;
    fs::rename(&staging_path, &request.destination).map_err(|error| {
        PackageError::Publication(format!(
            "rename staging {} to destination {}: {error}",
            staging_path.display(),
            request.destination.display()
        ))
    })?;
    let _persisted = staging.keep();

    Ok(Vgg19PackageReport { manifest })
}

#[derive(Debug)]
struct Snapshot {
    _directory: tempfile::TempDir,
    path: PathBuf,
    sha256: String,
}

impl Snapshot {
    fn create(source: &Path, max_bytes: u64) -> Result<Self, PackageError> {
        let metadata = fs::symlink_metadata(source)?;
        if metadata.file_type().is_symlink() || !metadata.is_file() {
            return Err(PackageError::InvalidRequest(format!(
                "source must be a regular non-symlink file: {}",
                source.display()
            )));
        }
        if metadata.len() > max_bytes {
            return Err(PackageError::InvalidRequest(format!(
                "source is {} bytes and exceeds {max_bytes} bytes",
                metadata.len()
            )));
        }
        let directory = tempfile::Builder::new()
            .prefix("feathertalk-vgg19-source-")
            .tempdir()?;
        let path = directory.path().join("source.pth");
        let mut input = File::open(source)?;
        let mut output = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)?;
        let mut digest = Sha256::new();
        let mut copied = 0_u64;
        let mut buffer = [0_u8; 64 * 1024];
        loop {
            let read = input.read(&mut buffer)?;
            if read == 0 {
                break;
            }
            copied = copied
                .checked_add(u64::try_from(read).expect("buffer length fits u64"))
                .ok_or_else(|| {
                    PackageError::InvalidRequest("source length overflowed".to_owned())
                })?;
            if copied > max_bytes {
                return Err(PackageError::InvalidRequest(format!(
                    "source exceeds {max_bytes} bytes while copying"
                )));
            }
            output.write_all(&buffer[..read])?;
            digest.update(&buffer[..read]);
        }
        output.sync_all()?;
        Ok(Self {
            _directory: directory,
            path,
            sha256: hex::encode(digest.finalize()),
        })
    }
}

fn validate_request_paths(request: &Vgg19PackageRequest) -> Result<(), PackageError> {
    if request.source.as_os_str().is_empty() {
        return Err(PackageError::InvalidRequest(
            "source path is empty".to_owned(),
        ));
    }
    if request.licenses.as_os_str().is_empty() {
        return Err(PackageError::InvalidRequest(
            "licenses path is empty".to_owned(),
        ));
    }
    if request.destination.as_os_str().is_empty() {
        return Err(PackageError::InvalidRequest(
            "destination path is empty".to_owned(),
        ));
    }
    Ok(())
}

fn validate_existing_directory(path: &Path, label: &str) -> Result<(), PackageError> {
    let metadata = fs::symlink_metadata(path).map_err(|error| {
        if error.kind() == std::io::ErrorKind::NotFound {
            PackageError::InvalidRequest(format!("{label} does not exist: {}", path.display()))
        } else {
            PackageError::Io(error)
        }
    })?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        return Err(PackageError::InvalidRequest(format!(
            "{label} must be an existing directory: {}",
            path.display()
        )));
    }
    Ok(())
}

fn ensure_destination_absent(path: &Path) -> Result<(), PackageError> {
    match fs::symlink_metadata(path) {
        Ok(_) => Err(PackageError::InvalidRequest(format!(
            "destination already exists: {}",
            path.display()
        ))),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(PackageError::Io(error)),
    }
}

fn read_bounded_regular(path: &Path, max_bytes: u64) -> Result<Vec<u8>, PackageError> {
    let metadata = fs::symlink_metadata(path)?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(PackageError::InvalidRequest(format!(
            "licenses must be a regular non-symlink file: {}",
            path.display()
        )));
    }
    if metadata.len() > max_bytes {
        return Err(PackageError::InvalidRequest(format!(
            "licenses is {} bytes and exceeds {max_bytes} bytes",
            metadata.len()
        )));
    }
    let mut file = File::open(path)?;
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)?;
    if u64::try_from(bytes.len()).expect("buffer length fits u64") > max_bytes {
        return Err(PackageError::InvalidRequest(format!(
            "licenses exceeds {max_bytes} bytes while reading"
        )));
    }
    Ok(bytes)
}

fn copy_create_new(bytes: &[u8], destination: &Path) -> Result<(), PackageError> {
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(destination)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    Ok(())
}

fn file_manifest(path: &Path, file_name: &str) -> Result<Vgg19FileManifest, PackageError> {
    let metadata = fs::symlink_metadata(path)?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(PackageError::Publication(format!(
            "staged file is not regular: {}",
            path.display()
        )));
    }
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
    Ok(Vgg19FileManifest {
        file_name: file_name.to_owned(),
        bytes: metadata.len(),
        sha256: hex::encode(digest.finalize()),
    })
}

fn write_manifest(directory: &Path, manifest: &Vgg19PackageManifest) -> Result<(), PackageError> {
    let mut bytes = serde_json::to_vec_pretty(manifest)
        .map_err(|error| PackageError::Publication(format!("serialize manifest: {error}")))?;
    bytes.push(b'\n');
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(directory.join(MANIFEST_FILE_NAME))?;
    file.write_all(&bytes)?;
    file.sync_all()?;
    Ok(())
}

fn exact_entries(directory: &Path) -> Result<Vec<String>, PackageError> {
    let mut entries = fs::read_dir(directory)?
        .map(|entry| {
            entry.and_then(|entry| {
                entry.file_name().into_string().map_err(|_| {
                    std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        "staging entry name is not valid UTF-8",
                    )
                })
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    entries.sort();
    Ok(entries)
}

fn compare_module_snapshots<B: Backend, M: ModuleSnapshot<B>>(
    expected: &M,
    actual: &M,
) -> Result<(), String> {
    let expected = expected
        .collect(None, None, false)
        .into_iter()
        .map(|snapshot| (snapshot.full_path(), snapshot))
        .collect::<BTreeMap<_, _>>();
    let actual = actual
        .collect(None, None, false)
        .into_iter()
        .map(|snapshot| (snapshot.full_path(), snapshot))
        .collect::<BTreeMap<_, _>>();
    if expected.len() != actual.len() {
        return Err(format!(
            "snapshot tensor count mismatch: {} != {}",
            expected.len(),
            actual.len()
        ));
    }
    for (path, expected) in expected {
        let actual = actual
            .get(&path)
            .ok_or_else(|| format!("reloaded module missing tensor {path}"))?;
        if expected.shape != actual.shape || expected.dtype != actual.dtype {
            return Err(format!("reloaded tensor metadata mismatch: {path}"));
        }
        if expected.to_data().map_err(|error| error.to_string())?
            != actual.to_data().map_err(|error| error.to_string())?
        {
            return Err(format!("reloaded tensor data mismatch: {path}"));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::{fs, io, path::PathBuf};

    use feathertalk_training::{Vgg19LicenseBundle, Vgg19LicenseEntry};
    use zip::ZipArchive;

    use super::{PackageError, Vgg19PackageRequest, build_vgg19_package_with_validation_hook};

    #[test]
    fn late_manifest_corruption_leaves_destination_absent() {
        let temp = tempfile::tempdir().unwrap();
        let source = extract_fixture(&temp, "vgg19-direct.pth");
        let licenses = temp.path().join("LICENSES.input.json");
        fs::write(
            &licenses,
            serde_json::to_vec_pretty(&Vgg19LicenseBundle {
                schema_version: 1,
                entries: vec![Vgg19LicenseEntry {
                    component: "synthetic VGG19 import fixture".to_owned(),
                    license_id: "LicenseRef-Test-Only".to_owned(),
                    source_url: "https://example.invalid/vgg19".to_owned(),
                    notice: "Synthetic test fixture only.".to_owned(),
                }],
            })
            .unwrap(),
        )
        .unwrap();
        let destination = temp.path().join("published");

        let error = build_vgg19_package_with_validation_hook(
            &Vgg19PackageRequest {
                source,
                licenses,
                destination: destination.clone(),
            },
            |staging| fs::write(staging.join("manifest.json"), b"{broken-json"),
        )
        .unwrap_err();

        assert!(matches!(
            error,
            PackageError::Training(_) | PackageError::Publication(_)
        ));
        assert!(!destination.exists());
    }

    fn extract_fixture(temp: &tempfile::TempDir, member: &str) -> PathBuf {
        let destination = temp.path().join(member);
        let archive_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/golden/vgg19-import-v1.zip");
        let archive = fs::File::open(archive_path).unwrap();
        let mut archive = ZipArchive::new(archive).unwrap();
        let mut source = archive.by_name(member).unwrap();
        let mut destination_file = fs::File::create(&destination).unwrap();
        io::copy(&mut source, &mut destination_file).unwrap();
        destination
    }
}
