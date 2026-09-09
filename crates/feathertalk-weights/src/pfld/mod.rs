use std::{
    collections::{BTreeMap, BTreeSet},
    path::{Path, PathBuf},
};

use burn::{
    backend::Autodiff,
    module::{AutodiffModule, HasAutodiffModule},
    tensor::DType,
    tensor::backend::Backend,
};
use burn_store::{ApplyError, ApplyResult, ModuleSnapshot, PytorchStore, pytorch::PytorchReader};
use serde::{Deserialize, Serialize};

use crate::{
    WeightImportError,
    source::{
        DEFAULT_MAX_FILE_BYTES, DEFAULT_MAX_TENSOR_COUNT, DEFAULT_MAX_TOTAL_ELEMENTS, SnapshotFile,
    },
};

mod artifact;
mod envelope;
mod key_map;

use artifact::{
    ensure_destination_absent, publish_staged_artifacts, verify_staged_artifacts,
    write_staged_artifacts,
};
use envelope::{PfldInspection, inspect_checkpoint, validate_envelope};
use key_map::pfld_remapper;

pub const PFLD_CHECKPOINT_EPOCH: u64 = 335;
pub const PFLD_ARCHITECTURE_VERSION: &str = "burn-pfld-structure-v1";

#[derive(Debug, Clone)]
pub struct PfldImportRequest {
    pub checkpoint: PathBuf,
    pub destination_dir: PathBuf,
    pub max_file_bytes: u64,
    pub max_tensor_count: usize,
    pub max_total_elements: u64,
}

impl Default for PfldImportRequest {
    fn default() -> Self {
        Self {
            checkpoint: PathBuf::new(),
            destination_dir: PathBuf::new(),
            max_file_bytes: DEFAULT_MAX_FILE_BYTES,
            max_tensor_count: DEFAULT_MAX_TENSOR_COUNT,
            max_total_elements: DEFAULT_MAX_TOTAL_ELEMENTS,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TensorAudit {
    pub tensor_count: usize,
    pub total_elements: u64,
    pub keys: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TensorSummary {
    pub tensor_count: usize,
    pub total_elements: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PfldSourceManifest {
    pub file_name: String,
    pub sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PfldModelArtifact {
    pub format: String,
    pub file_name: String,
    pub sha256: String,
    pub tensor_count: usize,
    pub total_elements: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PfldIgnoredTensors {
    pub batch_norm_counters: TensorAudit,
    pub localization: TensorAudit,
    pub auxiliarynet: Option<TensorAudit>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PfldImportManifest {
    pub schema_version: u32,
    pub model_type: String,
    pub architecture_version: String,
    pub source: PfldSourceManifest,
    pub epoch: u64,
    pub backbone: TensorSummary,
    pub model: PfldModelArtifact,
    pub ignored: PfldIgnoredTensors,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PfldImportReport {
    pub destination_dir: PathBuf,
    pub manifest: PfldImportManifest,
    pub applied: Vec<String>,
}

/// Strictly imports a PFLD checkpoint, verifies its staged artifacts, and then replaces the caller.
///
/// The autodiff-module association is used only to construct an independent Burn module copy.
/// Burn 0.21's ordinary `Clone` shares `RunningState` storage such as BatchNorm statistics.
pub fn import_pfld_checkpoint<B, M>(
    module: &mut M,
    request: &PfldImportRequest,
) -> Result<PfldImportReport, WeightImportError>
where
    B: Backend,
    M: ModuleSnapshot<B> + HasAutodiffModule<Autodiff<B>>,
{
    if request.destination_dir.as_os_str().is_empty() {
        return Err(WeightImportError::ArtifactValidation(
            "destination directory must not be empty".to_owned(),
        ));
    }
    ensure_destination_absent(&request.destination_dir)?;
    let parent = request
        .destination_dir
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    if !parent.is_dir() {
        return Err(WeightImportError::ArtifactValidation(
            "destination directory parent must exist and be a directory".to_owned(),
        ));
    }

    let snapshot = SnapshotFile::copy_from(&request.checkpoint, request.max_file_bytes)?;
    let prepared = prepare_pfld_import::<B, M>(module, &snapshot, request)?;
    let staged = write_staged_artifacts::<B, M>(
        &prepared.candidate,
        snapshot.path(),
        &prepared.source_file_name,
        &prepared.source_sha256,
        &prepared.inspection,
        parent,
    )?;
    verify_staged_artifacts::<B, M>(
        module,
        &prepared.candidate,
        snapshot.path(),
        &prepared.inspection,
        &staged,
    )?;
    let manifest = publish_staged_artifacts(staged, &request.destination_dir)?;
    *module = *prepared.candidate;

    Ok(PfldImportReport {
        destination_dir: request.destination_dir.clone(),
        manifest,
        applied: prepared.applied,
    })
}

#[derive(Debug)]
struct PreparedPfldImport<M> {
    candidate: Box<M>,
    source_file_name: String,
    source_sha256: String,
    inspection: PfldInspection,
    applied: Vec<String>,
}

fn configure_pfld_store(path: &Path) -> PytorchStore {
    PytorchStore::from_file(path)
        .with_top_level_key("pfld_backbone")
        .allow_partial(true)
        .validate(false)
        .map_indices_contiguous(false)
        .remap(pfld_remapper())
}

// Derived train/valid conversion for the production graph can exceed the default Windows test
// thread stack. Returning a Box keeps the large converted module out of the caller's return slot.
const PFLD_DETACHED_CLONE_STACK_BYTES: usize = 64 * 1024 * 1024;

fn clone_module_detached<B, M>(module: &mut M) -> Result<Box<M>, WeightImportError>
where
    B: Backend,
    M: ModuleSnapshot<B> + HasAutodiffModule<Autodiff<B>>,
{
    // A mutable scoped borrow is Send under Module's existing Send contract, so callers don't need
    // an additional Sync bound even though the conversion itself runs on a dedicated stack.
    std::thread::scope(|scope| {
        let handle = std::thread::Builder::new()
            .name("feathertalk-pfld-detached-clone".to_owned())
            .stack_size(PFLD_DETACHED_CLONE_STACK_BYTES)
            .spawn_scoped(scope, move || {
                let cloned = (*module).clone();
                Box::new(cloned.train::<Autodiff<B>>().valid())
            })
            .map_err(|error| WeightImportError::Store(error.to_string()))?;
        handle
            .join()
            .map_err(|_| WeightImportError::Store("PFLD detached clone thread panicked".to_owned()))
    })
}

fn prepare_pfld_import<B, M>(
    module: &mut M,
    snapshot: &SnapshotFile,
    request: &PfldImportRequest,
) -> Result<PreparedPfldImport<M>, WeightImportError>
where
    B: Backend,
    M: ModuleSnapshot<B> + HasAutodiffModule<Autodiff<B>>,
{
    let pickle = PytorchReader::read_pickle_data(snapshot.path(), None)
        .map_err(|error| WeightImportError::InvalidPfldEnvelope(error.to_string()))?;
    let envelope = validate_envelope(pickle)?;
    let inspection = inspect_checkpoint(snapshot.path(), envelope, request)?;
    validate_target_contract::<B, M>(module, &inspection)?;
    let mut store = configure_pfld_store(snapshot.path());
    let mut candidate = clone_module_detached::<B, M>(module)?;
    let result = candidate
        .load_from(&mut store)
        .map_err(|error| WeightImportError::Store(error.to_string()))?;
    let applied = validate_pfld_apply_result(&result, &inspection)?;
    let source_file_name = request
        .checkpoint
        .file_name()
        .and_then(std::ffi::OsStr::to_str)
        .ok_or_else(|| {
            WeightImportError::Manifest("checkpoint path must end in a UTF-8 file name".to_owned())
        })?
        .to_owned();

    Ok(PreparedPfldImport {
        candidate,
        source_file_name,
        source_sha256: snapshot.sha256().to_owned(),
        inspection,
        applied,
    })
}

fn validate_target_contract<B, M>(
    module: &M,
    inspection: &PfldInspection,
) -> Result<(), WeightImportError>
where
    B: Backend,
    M: ModuleSnapshot<B>,
{
    let target = module
        .collect(None, None, false)
        .into_iter()
        .map(|snapshot| (snapshot.full_path(), snapshot))
        .collect::<BTreeMap<_, _>>();
    let target_keys = target.keys().cloned().collect::<BTreeSet<_>>();

    if let Some(source_only) = inspection.expected_applied.difference(&target_keys).next() {
        return Err(WeightImportError::UnexpectedTensor(source_only.clone()));
    }
    if let Some(target_only) = target_keys.difference(&inspection.expected_applied).next() {
        return Err(WeightImportError::MissingTensor(target_only.clone()));
    }
    if let Some((path, _)) = target
        .iter()
        .find(|(_, snapshot)| snapshot.dtype != DType::F32)
    {
        return Err(WeightImportError::DTypeMismatch(path.clone()));
    }

    Ok(())
}

fn validate_pfld_apply_result(
    result: &ApplyResult,
    inspection: &PfldInspection,
) -> Result<Vec<String>, WeightImportError> {
    if let Some(path) = result.missing.iter().map(|(path, _)| path).min() {
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
            ApplyError::AdapterError { .. } | ApplyError::LoadError { .. } => {
                WeightImportError::Store(error.to_string())
            }
        });
    }
    if let Some(path) = result.skipped.iter().min() {
        return Err(WeightImportError::Store(format!(
            "PFLD import unexpectedly skipped tensor: {path}"
        )));
    }

    let unused = result.unused.iter().cloned().collect::<BTreeSet<_>>();
    if let Some(path) = unused.difference(&inspection.expected_unused).next() {
        return Err(WeightImportError::UnexpectedTensor(path.clone()));
    }
    if let Some(path) = inspection.expected_unused.difference(&unused).next() {
        return Err(WeightImportError::InvalidPfldIgnoredSet(format!(
            "ignored tensor was unexpectedly consumed: {path}"
        )));
    }

    let applied = result.applied.iter().cloned().collect::<BTreeSet<_>>();
    if let Some(path) = inspection.expected_applied.difference(&applied).next() {
        return Err(WeightImportError::MissingTensor(path.clone()));
    }
    if let Some(path) = applied.difference(&inspection.expected_applied).next() {
        return Err(WeightImportError::UnexpectedTensor(path.clone()));
    }

    Ok(applied.into_iter().collect())
}

#[cfg(test)]
mod tests {
    use std::{
        collections::BTreeMap,
        path::{Path, PathBuf},
    };

    use burn::{
        nn::{BatchNormConfig, LinearConfig},
        tensor::{Tensor, backend::Backend},
    };
    use burn_store::ModuleSnapshot;
    use feathertalk_models::{PFLD_GhostOne, PfldConfig, backend::CpuBackend};

    use crate::{PfldImportRequest, TensorSummary, WeightImportError, source::SnapshotFile};

    use super::{clone_module_detached, prepare_pfld_import};

    fn checkpoint_path() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../../data_utils/checkpoint_epoch_335.pth.tar")
    }

    fn request_for(checkpoint: PathBuf, destination_dir: PathBuf) -> PfldImportRequest {
        PfldImportRequest {
            checkpoint,
            destination_dir,
            ..PfldImportRequest::default()
        }
    }

    fn capture_module_data<B: Backend, M: ModuleSnapshot<B>>(
        module: &M,
    ) -> BTreeMap<String, burn::tensor::TensorData> {
        module
            .collect(None, None, false)
            .into_iter()
            .map(|snapshot| (snapshot.full_path(), snapshot.to_data().unwrap()))
            .collect()
    }

    fn assert_module_data_unchanged<B: Backend, M: ModuleSnapshot<B>>(
        before: &BTreeMap<String, burn::tensor::TensorData>,
        module: &M,
    ) {
        let after = capture_module_data(module);
        assert_eq!(before.len(), after.len());
        for (key, expected) in before {
            let actual = after.get(key).expect("module tensor disappeared");
            if expected != actual {
                panic!("module tensor changed at {key}");
            }
        }
    }

    #[test]
    fn detached_clone_keeps_batch_norm_running_state_independent() {
        let device = Default::default();
        let mut original = BatchNormConfig::new(2).init::<CpuBackend>(&device);
        let candidate = clone_module_detached::<CpuBackend, _>(&mut original).unwrap();

        candidate
            .running_mean
            .update(Tensor::<CpuBackend, 1>::ones([2], &device));
        let candidate_mean = candidate.running_mean.value_sync().to_data();
        let original_mean = original.running_mean.value_sync().to_data();

        assert_eq!(
            candidate_mean,
            Tensor::<CpuBackend, 1>::ones([2], &device).to_data()
        );
        assert_eq!(
            original_mean,
            Tensor::<CpuBackend, 1>::zeros([2], &device).to_data()
        );
    }

    #[test]
    fn real_checkpoint_prepares_a_complete_candidate_without_mutating_source_model() {
        let device = Default::default();
        let mut model = PFLD_GhostOne::<CpuBackend>::new(PfldConfig::production(), &device);
        let before = capture_module_data(&model);
        let snapshot = SnapshotFile::copy_from(
            &checkpoint_path(),
            PfldImportRequest::default().max_file_bytes,
        )
        .unwrap();
        let request = request_for(checkpoint_path(), PathBuf::from("unused"));

        let prepared =
            prepare_pfld_import::<CpuBackend, _>(&mut model, &snapshot, &request).unwrap();

        assert_module_data_unchanged::<CpuBackend, _>(&before, &model);
        assert_eq!(prepared.applied.len(), 1_735);
        assert_eq!(
            prepared.inspection.applied,
            TensorSummary {
                tensor_count: 1_735,
                total_elements: 910_902
            }
        );
        assert!(prepared.applied.windows(2).all(|pair| pair[0] < pair[1]));
    }

    #[test]
    fn incompatible_module_fails_before_any_caller_mutation() {
        let device = Default::default();
        let mut model = LinearConfig::new(2, 2).init::<CpuBackend>(&device);
        let before = capture_module_data(&model);
        let snapshot = SnapshotFile::copy_from(
            &checkpoint_path(),
            PfldImportRequest::default().max_file_bytes,
        )
        .unwrap();
        let request = request_for(checkpoint_path(), PathBuf::from("unused"));

        let error =
            prepare_pfld_import::<CpuBackend, _>(&mut model, &snapshot, &request).unwrap_err();

        assert!(matches!(
            error,
            WeightImportError::MissingTensor(_) | WeightImportError::UnexpectedTensor(_)
        ));
        assert_module_data_unchanged::<CpuBackend, _>(&before, &model);
    }
}
