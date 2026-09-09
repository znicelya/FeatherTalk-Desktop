use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
};

use burn::tensor::backend::Backend;
use burn_store::{ApplyError, ModuleSnapshot, ModuleStore, SafetensorsStore, TensorSnapshot};
use sha2::Digest;

use crate::{
    FileManifest, LicenseBundle, ModelDescription, ModelPackageManifest, PackageError,
    SourceManifest, TensorContract, TensorSpec, TrainingManifest, io,
};

#[derive(Debug, Clone)]
pub struct PackageBuildRequest {
    pub destination: PathBuf,
    pub description: ModelDescription,
    pub source_path: PathBuf,
    pub source: SourceManifest,
    pub licenses_path: PathBuf,
    pub created_at: String,
    pub minimum_app_version: String,
    pub training: TrainingManifest,
}

#[derive(Debug, Clone)]
pub struct PackageBuildReport {
    pub manifest: ModelPackageManifest,
}

pub fn write_model_package<B, M, F>(
    request: &PackageBuildRequest,
    model: &M,
    device: &B::Device,
    factory: F,
) -> Result<PackageBuildReport, PackageError>
where
    B: Backend,
    M: ModuleSnapshot<B>,
    F: Fn(&B::Device) -> M,
{
    write_model_package_with_validation_hook(request, model, device, factory, || Ok(()))
}

pub(crate) fn write_model_package_with_validation_hook<B, M, F, H>(
    request: &PackageBuildRequest,
    model: &M,
    device: &B::Device,
    factory: F,
    validation_hook: H,
) -> Result<PackageBuildReport, PackageError>
where
    B: Backend,
    M: ModuleSnapshot<B>,
    F: Fn(&B::Device) -> M,
    H: FnOnce() -> Result<(), PackageError>,
{
    request.description.validate()?;
    validate_source_snapshot(&request.source_path, &request.source)?;
    let license_bytes =
        io::read_bounded_regular(&request.licenses_path, crate::MAX_LICENSE_BYTES, "licenses")?;
    let licenses: LicenseBundle = serde_json::from_slice(&license_bytes)
        .map_err(|error| PackageError::InvalidLicense(format!("invalid license JSON: {error}")))?;
    licenses.validate()?;
    let parent = io::validate_parent(&request.destination)?;
    io::ensure_destination_absent(&request.destination)?;
    let tensors = module_tensor_contract(model)?;
    tensors.validate()?;

    let staging = io::create_staging_directory(parent)?;
    let staging_path = staging.path().to_owned();
    let model_path = staging_path.join(crate::MODEL_FILE_NAME);
    let mut store = SafetensorsStore::from_file(&model_path).overwrite(false);
    model
        .save_into(&mut store)
        .map_err(|error| PackageError::Store(error.to_string()))?;
    io::sync_regular_file(&model_path)?;
    let model_manifest = io::file_manifest(&model_path, crate::MODEL_FILE_NAME)?;
    let license_path = staging_path.join(crate::LICENSE_FILE_NAME);
    io::write_synced_create_new(&license_path, &license_bytes)?;
    let license_manifest = io::file_manifest(&license_path, crate::LICENSE_FILE_NAME)?;
    let manifest = ModelPackageManifest {
        schema_version: crate::MODEL_PACKAGE_SCHEMA_VERSION,
        model_type: request.description.model_type.clone(),
        architecture_version: request.description.architecture_version.clone(),
        configuration: request.description.configuration.clone(),
        inputs: request.description.inputs.clone(),
        outputs: request.description.outputs.clone(),
        training: request.training.clone(),
        source: request.source.clone(),
        created_at: request.created_at.clone(),
        minimum_app_version: request.minimum_app_version.clone(),
        tensors,
        model: model_manifest,
        licenses: license_manifest,
        optimizer: None,
        training_state: None,
    };
    manifest.validate()?;
    let manifest_bytes = serde_json::to_vec_pretty(&manifest)
        .map_err(|error| PackageError::Publication(format!("serialize manifest: {error}")))?;
    let manifest_path = staging_path.join(crate::MANIFEST_FILE_NAME);
    io::write_synced_create_new(&manifest_path, &manifest_bytes)?;
    io::validate_declared_file(
        &manifest_path,
        &FileManifest {
            file_name: crate::MANIFEST_FILE_NAME.to_owned(),
            bytes: u64::try_from(manifest_bytes.len()).expect("manifest length fits u64"),
            sha256: hex::encode(sha2::Sha256::digest(&manifest_bytes)),
        },
    )?;
    validate_staged_round_trip::<B, M, F>(&staging_path, &manifest, model, device, factory)?;
    validate_source_snapshot(&request.source_path, &request.source)?;
    validation_hook()?;
    io::sync_directory(&staging_path)?;
    io::publish_no_clobber(staging, &request.destination)?;
    Ok(PackageBuildReport { manifest })
}

/// Reads and validates the manifest of an inference model package.
///
/// `load_model_package` runs this first. A caller that has to know the shipped
/// hyperparameters before it can name the expected description -- the worker's
/// FeatherHuBERT loader, for one -- runs it on its own.
pub fn read_package_manifest(
    directory: impl AsRef<Path>,
) -> Result<ModelPackageManifest, PackageError> {
    let directory = directory.as_ref();
    io::validate_package_directory(directory, false)?;
    let manifest_bytes = io::read_bounded_regular(
        &directory.join(crate::MANIFEST_FILE_NAME),
        crate::MAX_MANIFEST_BYTES,
        "manifest",
    )?;
    let manifest: ModelPackageManifest =
        serde_json::from_slice(&manifest_bytes).map_err(|error| {
            PackageError::InvalidManifest(format!("invalid manifest JSON: {error}"))
        })?;
    manifest.validate()?;
    Ok(manifest)
}

pub fn load_model_package<B, M, F>(
    directory: impl AsRef<Path>,
    expected: &ModelDescription,
    device: &B::Device,
    factory: F,
) -> Result<(M, ModelPackageManifest), PackageError>
where
    B: Backend,
    M: ModuleSnapshot<B>,
    F: Fn(&B::Device) -> M,
{
    let directory = directory.as_ref();
    expected.validate()?;
    let manifest = read_package_manifest(directory)?;
    if manifest.description() != *expected {
        return Err(PackageError::InvalidRequest(
            "expected model description does not match package manifest".to_owned(),
        ));
    }
    let license_path = directory.join(crate::LICENSE_FILE_NAME);
    io::validate_declared_file(&license_path, &manifest.licenses)?;
    let license_bytes =
        io::read_bounded_regular(&license_path, crate::MAX_LICENSE_BYTES, "licenses")?;
    let licenses: LicenseBundle = serde_json::from_slice(&license_bytes)
        .map_err(|error| PackageError::InvalidLicense(format!("invalid license JSON: {error}")))?;
    licenses.validate()?;
    let model_path = directory.join(crate::MODEL_FILE_NAME);
    io::validate_declared_file(&model_path, &manifest.model)?;
    let mut store = SafetensorsStore::from_file(&model_path)
        .allow_partial(true)
        .validate(false);
    let snapshots = store
        .get_all_snapshots()
        .map_err(|error| PackageError::Store(error.to_string()))?;
    validate_snapshot_contract(snapshots, &manifest.tensors)?;
    let mut model = factory(device);
    let result = model
        .load_from(&mut store)
        .map_err(|error| PackageError::Store(error.to_string()))?;
    validate_apply_result(&result)?;
    let actual = module_tensor_contract(&model)?;
    if actual != manifest.tensors {
        return Err(PackageError::InvalidManifest(
            "loaded module tensor contract differs from manifest".to_owned(),
        ));
    }
    Ok((model, manifest))
}

fn validate_source_snapshot(path: &Path, source: &SourceManifest) -> Result<(), PackageError> {
    io::reject_symlink_components(path)?;
    let metadata = fs::symlink_metadata(path)?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(PackageError::InvalidRequest(format!(
            "source must be a regular non-symlink file: {}",
            path.display()
        )));
    }
    if metadata.len() > crate::MAX_SOURCE_BYTES {
        return Err(PackageError::InvalidRequest(format!(
            "source exceeds {} bytes",
            crate::MAX_SOURCE_BYTES
        )));
    }
    let (bytes, hash) = io::sha256_file(path)?;
    if source.file_name
        != path
            .file_name()
            .and_then(|v| v.to_str())
            .unwrap_or_default()
    {
        return Err(PackageError::InvalidRequest(
            "source manifest file_name does not match source path".to_owned(),
        ));
    }
    if bytes != metadata.len() || hash != source.sha256 {
        return Err(PackageError::HashMismatch {
            file: source.file_name.clone(),
            expected: source.sha256.clone(),
            actual: hash,
        });
    }
    Ok(())
}

fn validate_staged_round_trip<B, M, F>(
    staging: &Path,
    manifest: &ModelPackageManifest,
    original: &M,
    device: &B::Device,
    factory: F,
) -> Result<(), PackageError>
where
    B: Backend,
    M: ModuleSnapshot<B>,
    F: Fn(&B::Device) -> M,
{
    let (loaded, parsed) =
        load_model_package::<B, M, _>(staging, &manifest.description(), device, factory)?;
    if parsed != *manifest {
        return Err(PackageError::Publication(
            "staged manifest changed after writing".to_owned(),
        ));
    }
    compare_module_snapshots(original, &loaded)
}

pub(crate) fn module_tensor_contract<B: Backend, M: ModuleSnapshot<B>>(
    module: &M,
) -> Result<TensorContract, PackageError> {
    let mut entries = module
        .collect(None, None, false)
        .into_iter()
        .map(|snapshot| {
            if snapshot.dtype != burn::tensor::DType::F32 {
                return Err(PackageError::InvalidManifest(format!(
                    "tensor {} must be f32, got {:?}",
                    snapshot.full_path(),
                    snapshot.dtype
                )));
            }
            let shape = snapshot
                .shape
                .iter()
                .map(|dimension| {
                    i64::try_from(*dimension).map_err(|_| {
                        PackageError::InvalidManifest("tensor dimension exceeds i64".to_owned())
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            Ok(TensorSpec {
                name: snapshot.full_path(),
                shape,
                dtype: snapshot.dtype.name().to_owned(),
            })
        })
        .collect::<Result<Vec<_>, PackageError>>()?;
    entries.sort_by(|left, right| left.name.cmp(&right.name));
    let mut total = 0_u64;
    for entry in &entries {
        let count = entry.shape.iter().try_fold(1_u64, |value, dimension| {
            let dimension = u64::try_from(*dimension).map_err(|_| ())?;
            value.checked_mul(dimension).ok_or(())
        });
        total = total
            .checked_add(count.map_err(|_| {
                PackageError::InvalidManifest("tensor element count overflowed u64".to_owned())
            })?)
            .ok_or_else(|| {
                PackageError::InvalidManifest("tensor element count overflowed u64".to_owned())
            })?;
    }
    Ok(TensorContract {
        tensor_count: entries.len(),
        total_elements: total,
        entries,
    })
}

fn validate_snapshot_contract(
    snapshots: &BTreeMap<String, TensorSnapshot>,
    expected: &TensorContract,
) -> Result<(), PackageError> {
    let actual = snapshots
        .iter()
        .map(|(name, snapshot)| TensorSpec {
            name: name.clone(),
            shape: snapshot
                .shape
                .iter()
                .map(|dimension| i64::try_from(*dimension).unwrap_or(i64::MAX))
                .collect(),
            dtype: snapshot.dtype.name().to_owned(),
        })
        .collect::<Vec<_>>();
    if actual != expected.entries {
        return Err(PackageError::InvalidManifest(
            "safetensors tensor contract differs from manifest".to_owned(),
        ));
    }
    Ok(())
}

fn validate_apply_result(result: &burn_store::ApplyResult) -> Result<(), PackageError> {
    if let Some((path, _)) = result.missing.first() {
        return Err(PackageError::Store(format!("missing tensor {path}")));
    }
    if let Some(error) = result.errors.first() {
        return Err(match error {
            ApplyError::ShapeMismatch { path, .. } => {
                PackageError::InvalidManifest(format!("tensor shape mismatch: {path}"))
            }
            ApplyError::DTypeMismatch { path, .. } => {
                PackageError::InvalidManifest(format!("tensor dtype mismatch: {path}"))
            }
            ApplyError::AdapterError { .. } | ApplyError::LoadError { .. } => {
                PackageError::Store(error.to_string())
            }
        });
    }
    if let Some(path) = result.skipped.first() {
        return Err(PackageError::Store(format!("skipped tensor {path}")));
    }
    if let Some(path) = result.unused.first() {
        return Err(PackageError::Store(format!("unused tensor {path}")));
    }
    Ok(())
}

fn compare_module_snapshots<B: Backend, M: ModuleSnapshot<B>>(
    left: &M,
    right: &M,
) -> Result<(), PackageError> {
    let left = left
        .collect(None, None, false)
        .into_iter()
        .map(|snapshot| (snapshot.full_path(), snapshot))
        .collect::<BTreeMap<_, _>>();
    let right = right
        .collect(None, None, false)
        .into_iter()
        .map(|snapshot| (snapshot.full_path(), snapshot))
        .collect::<BTreeMap<_, _>>();
    if left.len() != right.len() {
        return Err(PackageError::Publication(
            "round-trip tensor count mismatch".to_owned(),
        ));
    }
    for (path, expected) in left {
        let actual = right.get(&path).ok_or_else(|| {
            PackageError::Publication(format!("round-trip missing tensor {path}"))
        })?;
        if expected.shape != actual.shape || expected.dtype != actual.dtype {
            return Err(PackageError::Publication(format!(
                "round-trip tensor metadata mismatch: {path}"
            )));
        }
        if expected
            .to_data()
            .map_err(|error| PackageError::Publication(error.to_string()))?
            != actual
                .to_data()
                .map_err(|error| PackageError::Publication(error.to_string()))?
        {
            return Err(PackageError::Publication(format!(
                "round-trip tensor data mismatch: {path}"
            )));
        }
    }
    Ok(())
}
