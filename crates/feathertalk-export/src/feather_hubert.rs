use std::{
    fs::{self, File, OpenOptions},
    io::{Read, Write},
    path::{Path, PathBuf},
};

use feathertalk_models::{backend::CpuBackend, feather_hubert::FeatherHubertEncoder};
use feathertalk_weights::{inspect_feather_hubert_checkpoint, load_feather_hubert_checkpoint};
use sha2::{Digest, Sha256};

use crate::{
    MAX_SOURCE_BYTES, ModelDescription, ModelPackageManifest, PackageBuildRequest, PackageError,
    SourceManifest, TrainingManifest, package::write_model_package_with_validation_hook,
};

const SOURCE_FORMAT: &str = "pytorch-pickle-restricted";
const SOURCE_IDENTIFIER: &str = "feathertalk-feather-hubert";

#[derive(Debug, Clone)]
pub struct FeatherHubertPackageRequest {
    pub source: PathBuf,
    pub licenses: PathBuf,
    pub destination: PathBuf,
    pub created_at: String,
    pub minimum_app_version: String,
}

#[derive(Debug, Clone)]
pub struct FeatherHubertPackageReport {
    pub manifest: ModelPackageManifest,
}

pub fn build_feather_hubert_package(
    request: &FeatherHubertPackageRequest,
) -> Result<FeatherHubertPackageReport, PackageError> {
    validate_request_paths(request)?;
    let source_snapshot = SourceSnapshot::create(&request.source)?;
    let source_file_name = request
        .source
        .file_name()
        .and_then(|value| value.to_str())
        .ok_or_else(|| {
            PackageError::InvalidRequest("source file name must be valid UTF-8".to_owned())
        })?
        .to_owned();
    let source_version = request
        .source
        .file_stem()
        .and_then(|value| value.to_str())
        .filter(|value| !value.is_empty())
        .ok_or_else(|| {
            PackageError::InvalidRequest("source file must have a non-empty stem".to_owned())
        })?
        .to_owned();

    let device = Default::default();
    let inspection = inspect_feather_hubert_checkpoint(source_snapshot.path())?;
    let (model, checkpoint) =
        load_feather_hubert_checkpoint::<CpuBackend>(source_snapshot.path(), &device)?;
    if checkpoint.source_sha256() != inspection.source_sha256()
        || checkpoint.tensor_count() != inspection.tensor_count()
        || checkpoint.total_elements() != inspection.total_elements()
    {
        return Err(PackageError::InvalidRequest(
            "FeatherHuBERT import audit changed between inspection and import".to_owned(),
        ));
    }

    let config = checkpoint.config().clone();
    let description = ModelDescription::feather_hubert(config.clone());
    let source = SourceManifest {
        format: SOURCE_FORMAT.to_owned(),
        identifier: SOURCE_IDENTIFIER.to_owned(),
        version: source_version,
        file_name: source_file_name,
        sha256: source_snapshot.sha256.clone(),
        url: None,
    };
    let package_request = PackageBuildRequest {
        destination: request.destination.clone(),
        description,
        source_path: request.source.clone(),
        source,
        licenses_path: request.licenses.clone(),
        created_at: request.created_at.clone(),
        minimum_app_version: request.minimum_app_version.clone(),
        training: TrainingManifest::default(),
    };

    let source_path = request.source.clone();
    let expected_bytes = source_snapshot.bytes;
    let expected_sha256 = source_snapshot.sha256.clone();
    let report = write_model_package_with_validation_hook::<
        CpuBackend,
        FeatherHubertEncoder<CpuBackend>,
        _,
        _,
    >(
        &package_request,
        &model,
        &device,
        move |device| config.clone().init::<CpuBackend>(device),
        move || {
            let (bytes, sha256) = crate::io::sha256_file(&source_path)?;
            if bytes != expected_bytes || sha256 != expected_sha256 {
                return Err(PackageError::HashMismatch {
                    file: source_path
                        .file_name()
                        .and_then(|value| value.to_str())
                        .unwrap_or("source")
                        .to_owned(),
                    expected: expected_sha256.clone(),
                    actual: sha256,
                });
            }
            Ok(())
        },
    )?;

    Ok(FeatherHubertPackageReport {
        manifest: report.manifest,
    })
}

fn validate_request_paths(request: &FeatherHubertPackageRequest) -> Result<(), PackageError> {
    for (name, path) in [
        ("source", &request.source),
        ("licenses", &request.licenses),
        ("destination", &request.destination),
    ] {
        if path.as_os_str().is_empty() {
            return Err(PackageError::InvalidRequest(format!(
                "{name} path must not be empty"
            )));
        }
    }
    Ok(())
}

struct SourceSnapshot {
    _directory: tempfile::TempDir,
    path: PathBuf,
    bytes: u64,
    sha256: String,
}

impl SourceSnapshot {
    fn create(source: &Path) -> Result<Self, PackageError> {
        crate::io::reject_symlink_components(source)?;
        let metadata = fs::symlink_metadata(source)?;
        if metadata.file_type().is_symlink() || !metadata.is_file() {
            return Err(PackageError::InvalidRequest(format!(
                "source must be a regular non-symlink file: {}",
                source.display()
            )));
        }
        if metadata.len() > MAX_SOURCE_BYTES {
            return Err(PackageError::InvalidRequest(format!(
                "source exceeds {MAX_SOURCE_BYTES} bytes: {}",
                source.display()
            )));
        }

        let directory = tempfile::Builder::new()
            .prefix("feathertalk-export-source-")
            .tempdir()?;
        let path = directory.path().join("checkpoint.pth");
        let mut input = File::open(source)?;
        let mut output = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)?;
        let mut digest = Sha256::new();
        let mut bytes = 0_u64;
        let mut buffer = [0_u8; 64 * 1024];
        loop {
            let read = input.read(&mut buffer)?;
            if read == 0 {
                break;
            }
            bytes = bytes
                .checked_add(u64::try_from(read).expect("buffer length fits u64"))
                .ok_or_else(|| {
                    PackageError::InvalidRequest("source length overflowed u64".to_owned())
                })?;
            if bytes > MAX_SOURCE_BYTES {
                return Err(PackageError::InvalidRequest(format!(
                    "source exceeds {MAX_SOURCE_BYTES} bytes while copying"
                )));
            }
            output.write_all(&buffer[..read])?;
            digest.update(&buffer[..read]);
        }
        output.flush()?;
        output.sync_all()?;
        let sha256 = hex::encode(digest.finalize());

        let (current_bytes, current_sha256) = crate::io::sha256_file(source)?;
        if current_bytes != bytes || current_sha256 != sha256 {
            return Err(PackageError::HashMismatch {
                file: source
                    .file_name()
                    .and_then(|value| value.to_str())
                    .unwrap_or("source")
                    .to_owned(),
                expected: sha256,
                actual: current_sha256,
            });
        }

        Ok(Self {
            _directory: directory,
            path,
            bytes,
            sha256,
        })
    }

    fn path(&self) -> &Path {
        &self.path
    }
}
