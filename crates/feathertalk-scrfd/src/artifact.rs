use std::{
    fs::File,
    io::{Read, Seek},
    path::Path,
};

use burn::tensor::backend::Backend;
use burn_store::{ApplyError, ApplyResult, ModuleSnapshot, SafetensorsStore};
use sha2::{Digest, Sha256};

use crate::{
    ScrfdArtifactManifest, ScrfdError,
    generated::{
        artifact_contract::{
            GENERATED_SOURCE_BYTES, GENERATED_SOURCE_SHA256, MODEL_SAFETENSORS_BYTES,
            MODEL_SAFETENSORS_SHA256,
        },
        scrfd_2_5g,
    },
};

pub const MAX_MANIFEST_BYTES: u64 = 64 * 1024;
pub const MAX_WEIGHT_BYTES: u64 = 16 * 1024 * 1024;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScrfdArtifactPaths {
    pub manifest: std::path::PathBuf,
    pub weights: std::path::PathBuf,
}

pub(crate) fn load_model<B: Backend>(
    paths: &ScrfdArtifactPaths,
    device: &B::Device,
) -> Result<(scrfd_2_5g::Model<B>, ScrfdArtifactManifest), ScrfdError> {
    let manifest_bytes = read_bounded(&paths.manifest, MAX_MANIFEST_BYTES, "read manifest")?;
    let manifest: ScrfdArtifactManifest = serde_json::from_slice(&manifest_bytes)
        .map_err(|error| ScrfdError::ManifestJson(error.to_string()))?;
    manifest.validate()?;
    validate_compiled_contract(&manifest)?;

    let weight_bytes = read_bounded(&paths.weights, MAX_WEIGHT_BYTES, "read weights")?;
    let actual = weight_bytes.len() as u64;
    if actual != manifest.weights.file_bytes {
        return Err(ScrfdError::WeightSizeMismatch {
            expected: manifest.weights.file_bytes,
            actual,
        });
    }
    if actual != MODEL_SAFETENSORS_BYTES {
        return Err(ScrfdError::WeightSizeMismatch {
            expected: MODEL_SAFETENSORS_BYTES,
            actual,
        });
    }
    let actual_hash = hex::encode(Sha256::digest(&weight_bytes));
    if actual_hash != manifest.weights.sha256 {
        return Err(ScrfdError::HashMismatch {
            artifact: "weights",
            expected: manifest.weights.sha256.clone(),
            actual: actual_hash,
        });
    }
    if actual_hash != MODEL_SAFETENSORS_SHA256 {
        return Err(ScrfdError::HashMismatch {
            artifact: "weights",
            expected: MODEL_SAFETENSORS_SHA256.to_owned(),
            actual: actual_hash,
        });
    }

    let mut model = scrfd_2_5g::Model::<B>::new(device);
    let mut store = SafetensorsStore::from_bytes(Some(weight_bytes))
        .allow_partial(true)
        .validate(false);
    let result = model
        .load_from(&mut store)
        .map_err(|error| ScrfdError::Store(error.to_string()))?;
    validate_apply_result(&result)?;
    Ok((model, manifest))
}

fn validate_compiled_contract(manifest: &ScrfdArtifactManifest) -> Result<(), ScrfdError> {
    if manifest.generated_source.file_bytes != GENERATED_SOURCE_BYTES {
        return Err(ScrfdError::ContractMismatch {
            field: "generated_source.file_bytes",
            expected: GENERATED_SOURCE_BYTES.to_string(),
            actual: manifest.generated_source.file_bytes.to_string(),
        });
    }
    if manifest.generated_source.sha256 != GENERATED_SOURCE_SHA256 {
        return Err(ScrfdError::ContractMismatch {
            field: "generated_source.sha256",
            expected: GENERATED_SOURCE_SHA256.to_owned(),
            actual: manifest.generated_source.sha256.clone(),
        });
    }
    if manifest.weights.file_bytes != MODEL_SAFETENSORS_BYTES {
        return Err(ScrfdError::ContractMismatch {
            field: "weights.file_bytes",
            expected: MODEL_SAFETENSORS_BYTES.to_string(),
            actual: manifest.weights.file_bytes.to_string(),
        });
    }
    if manifest.weights.sha256 != MODEL_SAFETENSORS_SHA256 {
        return Err(ScrfdError::ContractMismatch {
            field: "weights.sha256",
            expected: MODEL_SAFETENSORS_SHA256.to_owned(),
            actual: manifest.weights.sha256.clone(),
        });
    }
    Ok(())
}

fn read_bounded(path: &Path, limit: u64, operation: &'static str) -> Result<Vec<u8>, ScrfdError> {
    let mut file = File::open(path).map_err(|source| ScrfdError::Io {
        operation,
        path: path.to_owned(),
        source,
    })?;
    let size = file
        .metadata()
        .map_err(|source| ScrfdError::Io {
            operation: "read artifact metadata",
            path: path.to_owned(),
            source,
        })?
        .len();
    if size > limit {
        return if limit == MAX_MANIFEST_BYTES {
            Err(ScrfdError::ManifestTooLarge {
                limit,
                actual: size,
            })
        } else {
            Err(ScrfdError::WeightsTooLarge {
                limit,
                actual: size,
            })
        };
    }

    let capacity = usize::try_from(size).map_err(|_| ScrfdError::Io {
        operation,
        path: path.to_owned(),
        source: std::io::Error::new(std::io::ErrorKind::InvalidData, "file size exceeds usize"),
    })?;
    let mut bytes = Vec::with_capacity(capacity);
    file.seek(std::io::SeekFrom::Start(0))
        .map_err(|source| ScrfdError::Io {
            operation,
            path: path.to_owned(),
            source,
        })?;
    file.take(limit + 1)
        .read_to_end(&mut bytes)
        .map_err(|source| ScrfdError::Io {
            operation,
            path: path.to_owned(),
            source,
        })?;
    if bytes.len() as u64 > limit {
        return if limit == MAX_MANIFEST_BYTES {
            Err(ScrfdError::ManifestTooLarge {
                limit,
                actual: bytes.len() as u64,
            })
        } else {
            Err(ScrfdError::WeightsTooLarge {
                limit,
                actual: bytes.len() as u64,
            })
        };
    }
    Ok(bytes)
}

pub(crate) fn validate_apply_result(result: &ApplyResult) -> Result<(), ScrfdError> {
    if let Some(path) = result.missing.iter().map(|(path, _)| path).min() {
        return Err(ScrfdError::MissingTensor(path.clone()));
    }
    if let Some(error) = result.errors.first() {
        return Err(match error {
            ApplyError::ShapeMismatch {
                path,
                expected,
                found,
            } => ScrfdError::ShapeMismatch(format!("{path}: expected {expected:?}, got {found:?}")),
            ApplyError::DTypeMismatch {
                path,
                expected,
                found,
            } => ScrfdError::DTypeMismatch(format!("{path}: expected {expected:?}, got {found:?}")),
            ApplyError::AdapterError { path, message }
            | ApplyError::LoadError { path, message } => {
                ScrfdError::Store(format!("{path}: {message}"))
            }
        });
    }
    if let Some(path) = result.skipped.iter().min() {
        return Err(ScrfdError::Store(format!("skipped tensor: {path}")));
    }
    if let Some(path) = result.unused.iter().min() {
        return Err(ScrfdError::UnexpectedTensor(path.clone()));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{DType, Shape};

    #[test]
    fn committed_generated_source_matches_its_compiled_contract() {
        let path =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/generated/scrfd_2_5g.rs");
        let bytes = std::fs::read(path).unwrap();
        assert_eq!(bytes.len() as u64, GENERATED_SOURCE_BYTES);
        assert_eq!(hex::encode(Sha256::digest(bytes)), GENERATED_SOURCE_SHA256);
    }

    #[test]
    fn strict_apply_maps_every_failure_category() {
        let empty = || ApplyResult {
            applied: Vec::new(),
            skipped: Vec::new(),
            missing: Vec::new(),
            unused: Vec::new(),
            errors: Vec::new(),
        };
        let mut missing = empty();
        missing
            .missing
            .push(("a.weight".to_owned(), "Model".to_owned()));
        assert!(matches!(
            validate_apply_result(&missing),
            Err(ScrfdError::MissingTensor(path)) if path == "a.weight"
        ));

        let mut unused = empty();
        unused.unused.push("extra".to_owned());
        assert!(matches!(
            validate_apply_result(&unused),
            Err(ScrfdError::UnexpectedTensor(path)) if path == "extra"
        ));

        let mut skipped = empty();
        skipped.skipped.push("skip".to_owned());
        assert!(matches!(
            validate_apply_result(&skipped),
            Err(ScrfdError::Store(_))
        ));

        let mut shape = empty();
        shape.errors.push(ApplyError::ShapeMismatch {
            path: "shape".to_owned(),
            expected: Shape::new([1, 2]),
            found: Shape::new([2, 1]),
        });
        assert!(matches!(
            validate_apply_result(&shape),
            Err(ScrfdError::ShapeMismatch(_))
        ));

        let mut dtype = empty();
        dtype.errors.push(ApplyError::DTypeMismatch {
            path: "dtype".to_owned(),
            expected: DType::F32,
            found: DType::I32,
        });
        assert!(matches!(
            validate_apply_result(&dtype),
            Err(ScrfdError::DTypeMismatch(_))
        ));

        let mut adapter = empty();
        adapter.errors.push(ApplyError::AdapterError {
            path: "adapter".to_owned(),
            message: "bad adapter".to_owned(),
        });
        assert!(matches!(
            validate_apply_result(&adapter),
            Err(ScrfdError::Store(_))
        ));

        let mut load = empty();
        load.errors.push(ApplyError::LoadError {
            path: "load".to_owned(),
            message: "bad load".to_owned(),
        });
        assert!(matches!(
            validate_apply_result(&load),
            Err(ScrfdError::Store(_))
        ));
    }
}
