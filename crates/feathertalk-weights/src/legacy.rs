use std::{
    collections::HashSet,
    path::{Path, PathBuf},
};

use burn::tensor::backend::Backend;
use burn_store::{
    ApplyError, ApplyResult, ModuleSnapshot, ModuleStore, PytorchStore, PytorchStoreError,
    TensorSnapshot,
    pytorch::{PytorchError, PytorchReader},
};
use serde::Serialize;

use crate::{
    WeightImportError,
    key_map::{LegacyModelKind, configure_store, is_known_ignored_key_for, map_key},
    source::{
        DEFAULT_MAX_FILE_BYTES, DEFAULT_MAX_TENSOR_COUNT, DEFAULT_MAX_TOTAL_ELEMENTS, SnapshotFile,
        tensor_elements,
    },
};

#[derive(Debug, Clone)]
pub struct LegacyImportRequest {
    pub path: PathBuf,
    pub kind: LegacyModelKind,
    pub top_level_key: Option<String>,
    pub max_file_bytes: u64,
    pub max_tensor_count: usize,
    pub max_total_elements: u64,
}

impl Default for LegacyImportRequest {
    fn default() -> Self {
        Self {
            path: PathBuf::new(),
            kind: LegacyModelKind::FeatherHubert,
            top_level_key: None,
            max_file_bytes: DEFAULT_MAX_FILE_BYTES,
            max_tensor_count: DEFAULT_MAX_TENSOR_COUNT,
            max_total_elements: DEFAULT_MAX_TOTAL_ELEMENTS,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ImportReport {
    pub source_sha256: String,
    pub applied: Vec<String>,
    pub ignored: Vec<String>,
    pub tensor_count: usize,
    pub total_elements: u64,
}

pub fn import_into<B, M>(
    module: &mut M,
    request: &LegacyImportRequest,
) -> Result<ImportReport, WeightImportError>
where
    B: Backend,
    M: ModuleSnapshot<B>,
{
    let mut store = build_strict_store(request)?;
    let mut candidate = module.clone();
    let result = candidate.load_from(&mut store)?;
    validate_apply_result(request.kind, &result)?;
    let report = build_report(request, &mut store, result)?;
    *module = candidate;
    Ok(report)
}

struct StrictPytorchStore {
    store: PytorchStore,
    _snapshot: SnapshotFile,
    source_sha256: String,
    ignored: Vec<String>,
    tensor_count: usize,
    total_elements: u64,
}

impl ModuleStore for StrictPytorchStore {
    type Error = WeightImportError;

    fn collect_from<B: Backend, M: ModuleSnapshot<B>>(
        &mut self,
        module: &M,
    ) -> Result<(), Self::Error> {
        self.store.collect_from(module).map_err(store_error)
    }

    fn apply_to<B: Backend, M: ModuleSnapshot<B>>(
        &mut self,
        module: &mut M,
    ) -> Result<ApplyResult, Self::Error> {
        self.store.apply_to(module).map_err(store_error)
    }

    fn get_snapshot(&mut self, name: &str) -> Result<Option<&TensorSnapshot>, Self::Error> {
        self.store.get_snapshot(name).map_err(store_error)
    }

    fn get_all_snapshots(
        &mut self,
    ) -> Result<&std::collections::BTreeMap<String, TensorSnapshot>, Self::Error> {
        self.store.get_all_snapshots().map_err(store_error)
    }

    fn keys(&mut self) -> Result<Vec<String>, Self::Error> {
        self.store.keys().map_err(store_error)
    }
}

fn build_strict_store(
    request: &LegacyImportRequest,
) -> Result<StrictPytorchStore, WeightImportError> {
    let snapshot = SnapshotFile::copy_from(&request.path, request.max_file_bytes)?;
    let top_level_key = select_top_level_key(snapshot.path(), request)?;
    let reader = match top_level_key.as_deref() {
        Some(key) => PytorchReader::with_top_level_key(snapshot.path(), key),
        None => PytorchReader::new(snapshot.path()),
    }
    .map_err(store_error)?;
    reject_duplicate_remapped_keys(request.kind, reader.keys())?;

    let mut store = configure_store(snapshot.path(), request.kind, top_level_key.as_deref());
    let keys = store.keys().map_err(store_error)?;
    let ignored: Vec<String> = keys
        .iter()
        .filter(|key| is_known_ignored_key_for(request.kind, key))
        .cloned()
        .collect();
    let snapshots = store.get_all_snapshots().map_err(store_error)?;
    let (tensor_count, total_elements) = inspected_size(snapshots, &ignored, request)?;

    Ok(StrictPytorchStore {
        store,
        source_sha256: snapshot.sha256().to_owned(),
        _snapshot: snapshot,
        ignored,
        tensor_count,
        total_elements,
    })
}

pub(crate) fn select_top_level_key(
    path: &Path,
    request: &LegacyImportRequest,
) -> Result<Option<String>, WeightImportError> {
    let mut candidates = Vec::new();
    if let Some(key) = request.top_level_key.clone() {
        candidates.push(Some(key));
    }
    for key in ["model", "state_dict"] {
        let candidate = Some(key.to_owned());
        if !candidates.contains(&candidate) {
            candidates.push(candidate);
        }
    }
    candidates.push(None);

    for candidate in candidates {
        let mut store = configure_store(path, request.kind, candidate.as_deref());
        match store.keys() {
            Ok(keys) if !keys.is_empty() => return Ok(candidate),
            Ok(_) => continue,
            Err(PytorchStoreError::Reader(PytorchError::KeyNotFound(_))) => continue,
            Err(error) => return Err(store_error(error)),
        }
    }

    Err(WeightImportError::UnsupportedStructure(
        "no requested, model, state_dict, or direct state dictionary was found".to_owned(),
    ))
}

fn reject_duplicate_remapped_keys(
    kind: LegacyModelKind,
    keys: Vec<String>,
) -> Result<(), WeightImportError> {
    let mut remapped = HashSet::with_capacity(keys.len());
    for key in keys {
        let mapped = map_key(kind, &key);
        if !remapped.insert(mapped.clone()) {
            return Err(WeightImportError::DuplicateKey(mapped));
        }
    }
    Ok(())
}

fn inspected_size(
    snapshots: &std::collections::BTreeMap<String, TensorSnapshot>,
    ignored: &[String],
    request: &LegacyImportRequest,
) -> Result<(usize, u64), WeightImportError> {
    let ignored: HashSet<&str> = ignored.iter().map(String::as_str).collect();
    let mut count = 0usize;
    let mut total = 0u64;

    for (key, snapshot) in snapshots {
        if ignored.contains(key.as_str()) {
            continue;
        }
        count = count.checked_add(1).ok_or_else(|| {
            WeightImportError::UnsafeLimit("tensor count overflowed usize".to_owned())
        })?;
        total = total
            .checked_add(tensor_elements(snapshot)?)
            .ok_or_else(|| {
                WeightImportError::UnsafeLimit("total tensor elements overflowed u64".to_owned())
            })?;
    }

    if count > request.max_tensor_count {
        return Err(WeightImportError::UnsafeLimit(format!(
            "tensor count {count} exceeds {}",
            request.max_tensor_count
        )));
    }
    if total > request.max_total_elements {
        return Err(WeightImportError::UnsafeLimit(format!(
            "total tensor elements {total} exceeds {}",
            request.max_total_elements
        )));
    }
    Ok((count, total))
}

fn validate_apply_result(
    kind: LegacyModelKind,
    result: &ApplyResult,
) -> Result<(), WeightImportError> {
    if let Some((path, _)) = result.missing.first() {
        return Err(WeightImportError::MissingTensor(path.clone()));
    }
    if let Some(error) = result.errors.first() {
        return Err(match error {
            ApplyError::ShapeMismatch { path, .. } => {
                WeightImportError::ShapeMismatch(path.clone())
            }
            ApplyError::DTypeMismatch { path, .. } => {
                WeightImportError::DTypeMismatch(path.clone())
            }
            other => WeightImportError::Store(other.to_string()),
        });
    }
    if let Some(key) = result
        .unused
        .iter()
        .find(|key| !is_known_ignored_key_for(kind, key))
    {
        return Err(WeightImportError::UnexpectedTensor(key.clone()));
    }
    Ok(())
}

fn build_report(
    _request: &LegacyImportRequest,
    store: &mut StrictPytorchStore,
    result: ApplyResult,
) -> Result<ImportReport, WeightImportError> {
    Ok(ImportReport {
        source_sha256: store.source_sha256.clone(),
        applied: result.applied,
        ignored: store.ignored.clone(),
        tensor_count: store.tensor_count,
        total_elements: store.total_elements,
    })
}

fn store_error(error: impl std::fmt::Display) -> WeightImportError {
    WeightImportError::Store(error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn colliding_original_unet_remaps_are_rejected() {
        let error = reject_duplicate_remapped_keys(
            LegacyModelKind::OriginalUnet,
            vec![
                "block.conv.0.weight".to_owned(),
                "block.expand_conv.weight".to_owned(),
            ],
        )
        .unwrap_err();

        assert!(
            matches!(error, WeightImportError::DuplicateKey(key) if key == "block.expand_conv.weight")
        );
    }
}
