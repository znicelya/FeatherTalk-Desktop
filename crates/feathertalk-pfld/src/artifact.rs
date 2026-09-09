use std::{
    fs::{self, File},
    io::{Read, Seek},
    path::Path,
};

use burn::tensor::backend::Backend;
use burn_store::{ApplyError, ApplyResult, ModuleSnapshot, SafetensorsStore};
use feathertalk_models::{PFLD_GhostOne, PfldConfig};
use sha2::{Digest, Sha256};

use crate::{
    MAX_MANIFEST_BYTES, MAX_WEIGHT_BYTES, PFLD_EXPECTED_TENSOR_COUNT, PFLD_EXPECTED_TOTAL_ELEMENTS,
    PFLD_MODEL_BYTES, PfldRuntimeError, PfldRuntimeManifest,
};

pub(crate) fn load_artifact<B: Backend>(
    directory: &Path,
    device: &B::Device,
) -> Result<(PFLD_GhostOne<B>, PfldRuntimeManifest), PfldRuntimeError> {
    validate_directory(directory)?;
    let manifest_path = directory.join("manifest.json");
    let manifest_bytes = read_bounded(&manifest_path, MAX_MANIFEST_BYTES, "read manifest")?;
    let manifest: PfldRuntimeManifest = serde_json::from_slice(&manifest_bytes)
        .map_err(|error| PfldRuntimeError::ManifestJson(error.to_string()))?;
    manifest.validate()?;

    let weights_path = directory.join(&manifest.model.file_name);
    if weights_path.file_name().and_then(|name| name.to_str()) != Some("model.safetensors") {
        return Err(PfldRuntimeError::InvalidManifest {
            field: "model.file_name".to_owned(),
            message: "must be model.safetensors".to_owned(),
        });
    }
    let weight_bytes = read_bounded(&weights_path, MAX_WEIGHT_BYTES, "read weights")?;
    if weight_bytes.len() as u64 != PFLD_MODEL_BYTES {
        return Err(PfldRuntimeError::WeightSizeMismatch {
            expected: PFLD_MODEL_BYTES,
            actual: weight_bytes.len() as u64,
        });
    }
    let actual_hash = hex::encode(Sha256::digest(&weight_bytes));
    if actual_hash != manifest.model.sha256 {
        return Err(PfldRuntimeError::HashMismatch {
            artifact: "weights",
            expected: manifest.model.sha256,
            actual: actual_hash,
        });
    }

    let mut model = PFLD_GhostOne::<B>::new(PfldConfig::production(), device);
    let mut store = SafetensorsStore::from_bytes(Some(weight_bytes))
        .allow_partial(true)
        .validate(false);
    let result = model
        .load_from(&mut store)
        .map_err(|error| PfldRuntimeError::Store(error.to_string()))?;
    validate_apply_result(&result)?;
    let summary = module_summary::<B, _>(&model)?;
    if summary != (PFLD_EXPECTED_TENSOR_COUNT, PFLD_EXPECTED_TOTAL_ELEMENTS)
        || summary != (manifest.model.tensor_count, manifest.model.total_elements)
    {
        return Err(PfldRuntimeError::InvalidManifest {
            field: "model".to_owned(),
            message: format!("loaded tensor summary mismatch: {summary:?}"),
        });
    }
    Ok((model, manifest))
}

fn validate_directory(directory: &Path) -> Result<(), PfldRuntimeError> {
    let metadata = fs::symlink_metadata(directory).map_err(|source| PfldRuntimeError::Io {
        operation: "inspect artifact directory",
        path: directory.to_owned(),
        source,
    })?;
    if !metadata.is_dir() || metadata.file_type().is_symlink() {
        return Err(PfldRuntimeError::InvalidManifest {
            field: "artifact_directory".to_owned(),
            message: "must be a real directory".to_owned(),
        });
    }
    let mut names = Vec::new();
    for entry in fs::read_dir(directory).map_err(|source| PfldRuntimeError::Io {
        operation: "read artifact directory",
        path: directory.to_owned(),
        source,
    })? {
        let entry = entry.map_err(|source| PfldRuntimeError::Io {
            operation: "read artifact directory entry",
            path: directory.to_owned(),
            source,
        })?;
        let path = entry.path();
        let metadata = fs::symlink_metadata(&path).map_err(|source| PfldRuntimeError::Io {
            operation: "inspect artifact entry",
            path: path.clone(),
            source,
        })?;
        if !metadata.is_file() || metadata.file_type().is_symlink() {
            return Err(PfldRuntimeError::UnexpectedArtifactEntry(
                entry.file_name().to_string_lossy().into_owned(),
            ));
        }
        names.push(entry.file_name().to_string_lossy().into_owned());
    }
    names.sort();
    if names != ["manifest.json".to_owned(), "model.safetensors".to_owned()] {
        return Err(PfldRuntimeError::UnexpectedArtifactEntry(format!(
            "expected [manifest.json, model.safetensors], got {names:?}"
        )));
    }
    Ok(())
}

fn read_bounded(
    path: &Path,
    limit: u64,
    operation: &'static str,
) -> Result<Vec<u8>, PfldRuntimeError> {
    let mut file = File::open(path).map_err(|source| PfldRuntimeError::Io {
        operation,
        path: path.to_owned(),
        source,
    })?;
    let size = file
        .metadata()
        .map_err(|source| PfldRuntimeError::Io {
            operation: "inspect artifact file",
            path: path.to_owned(),
            source,
        })?
        .len();
    if size > limit {
        return if operation == "read manifest" {
            Err(PfldRuntimeError::ManifestTooLarge {
                limit,
                actual: size,
            })
        } else {
            Err(PfldRuntimeError::WeightsTooLarge {
                limit,
                actual: size,
            })
        };
    }
    let capacity = usize::try_from(size).map_err(|_| PfldRuntimeError::Io {
        operation,
        path: path.to_owned(),
        source: std::io::Error::new(std::io::ErrorKind::InvalidData, "file size exceeds usize"),
    })?;
    let mut bytes = Vec::with_capacity(capacity);
    file.seek(std::io::SeekFrom::Start(0))
        .map_err(|source| PfldRuntimeError::Io {
            operation,
            path: path.to_owned(),
            source,
        })?;
    file.take(limit + 1)
        .read_to_end(&mut bytes)
        .map_err(|source| PfldRuntimeError::Io {
            operation,
            path: path.to_owned(),
            source,
        })?;
    if bytes.len() as u64 > limit {
        return if operation == "read manifest" {
            Err(PfldRuntimeError::ManifestTooLarge {
                limit,
                actual: bytes.len() as u64,
            })
        } else {
            Err(PfldRuntimeError::WeightsTooLarge {
                limit,
                actual: bytes.len() as u64,
            })
        };
    }
    Ok(bytes)
}

fn validate_apply_result(result: &ApplyResult) -> Result<(), PfldRuntimeError> {
    if let Some(path) = result.missing.iter().map(|(path, _)| path).min() {
        return Err(PfldRuntimeError::MissingTensor(path.clone()));
    }
    if let Some(error) = result.errors.first() {
        return Err(match error {
            ApplyError::ShapeMismatch { path, .. } => PfldRuntimeError::ShapeMismatch(path.clone()),
            ApplyError::DTypeMismatch { path, .. } => PfldRuntimeError::DTypeMismatch(path.clone()),
            ApplyError::AdapterError { .. } | ApplyError::LoadError { .. } => {
                PfldRuntimeError::Store(error.to_string())
            }
        });
    }
    if let Some(path) = result.skipped.iter().min() {
        return Err(PfldRuntimeError::Store(format!("skipped tensor: {path}")));
    }
    if let Some(path) = result.unused.iter().min() {
        return Err(PfldRuntimeError::UnexpectedTensor(path.clone()));
    }
    Ok(())
}

fn module_summary<B, M>(module: &M) -> Result<(usize, u64), PfldRuntimeError>
where
    B: Backend,
    M: ModuleSnapshot<B>,
{
    let mut count = 0usize;
    let mut elements = 0u64;
    for snapshot in module.collect(None, None, false) {
        count = count
            .checked_add(1)
            .ok_or_else(|| PfldRuntimeError::Store("tensor count overflow".to_owned()))?;
        let tensor_elements = snapshot.shape.num_elements();
        elements = elements
            .checked_add(tensor_elements as u64)
            .ok_or_else(|| PfldRuntimeError::Store("tensor element count overflow".to_owned()))?;
    }
    Ok((count, elements))
}
