use std::{
    collections::BTreeMap,
    fs::{self, OpenOptions},
    io::Write,
    path::Path,
};

use burn::{backend::Autodiff, module::HasAutodiffModule, tensor::backend::Backend};
use burn_store::{ApplyError, ApplyResult, ModuleSnapshot, SafetensorsStore};

use crate::{
    PFLD_ARCHITECTURE_VERSION, PFLD_CHECKPOINT_EPOCH, PfldImportManifest, PfldModelArtifact,
    PfldSourceManifest, TensorAudit, TensorSummary, WeightImportError,
    source::{sha256_file, tensor_elements},
};

use super::{PfldInspection, clone_module_detached};

const MODEL_FILE_NAME: &str = "model.safetensors";
const MANIFEST_FILE_NAME: &str = "manifest.json";

pub(super) struct StagedArtifacts {
    directory: tempfile::TempDir,
    manifest: PfldImportManifest,
}

impl StagedArtifacts {
    fn path(&self) -> &Path {
        self.directory.path()
    }
}

pub(super) fn write_staged_artifacts<B, M>(
    candidate: &M,
    source_path: &Path,
    source_file_name: &str,
    source_sha256: &str,
    inspection: &PfldInspection,
    parent: &Path,
) -> Result<StagedArtifacts, WeightImportError>
where
    B: Backend,
    M: ModuleSnapshot<B>,
{
    let directory = tempfile::Builder::new()
        .prefix(".feathertalk-pfld-")
        .tempdir_in(parent)?;
    let current_source_sha256 = sha256_file(source_path)?;
    if current_source_sha256 != source_sha256 {
        return Err(WeightImportError::ArtifactValidation(
            "source checkpoint changed after snapshot creation".to_owned(),
        ));
    }

    let model_path = directory.path().join(MODEL_FILE_NAME);
    let mut store = SafetensorsStore::from_file(&model_path).overwrite(false);
    candidate
        .save_into(&mut store)
        .map_err(|error| WeightImportError::Store(error.to_string()))?;
    let model_sha256 = sha256_file(&model_path)?;
    let candidate_summary = module_summary::<B, M>(candidate)?;
    if candidate_summary != inspection.applied {
        return Err(WeightImportError::ArtifactValidation(format!(
            "candidate tensor summary mismatch: expected {:?}, got {:?}",
            inspection.applied, candidate_summary
        )));
    }

    let manifest = PfldImportManifest {
        schema_version: 1,
        model_type: "pfld_ghost_one".to_owned(),
        architecture_version: PFLD_ARCHITECTURE_VERSION.to_owned(),
        source: PfldSourceManifest {
            file_name: source_file_name.to_owned(),
            sha256: source_sha256.to_owned(),
        },
        epoch: PFLD_CHECKPOINT_EPOCH,
        backbone: inspection.backbone.clone(),
        model: PfldModelArtifact {
            format: "safetensors".to_owned(),
            file_name: MODEL_FILE_NAME.to_owned(),
            sha256: model_sha256,
            tensor_count: inspection.applied.tensor_count,
            total_elements: inspection.applied.total_elements,
        },
        ignored: inspection.ignored.clone(),
    };
    let mut manifest_bytes = serde_json::to_vec_pretty(&manifest)
        .map_err(|error| WeightImportError::Manifest(error.to_string()))?;
    manifest_bytes.push(b'\n');
    let manifest_path = directory.path().join(MANIFEST_FILE_NAME);
    let mut manifest_file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(manifest_path)?;
    manifest_file.write_all(&manifest_bytes)?;
    manifest_file.sync_all()?;

    Ok(StagedArtifacts {
        directory,
        manifest,
    })
}

pub(super) fn verify_staged_artifacts<B, M>(
    original: &mut M,
    candidate: &M,
    source_path: &Path,
    inspection: &PfldInspection,
    staged: &StagedArtifacts,
) -> Result<(), WeightImportError>
where
    B: Backend,
    M: ModuleSnapshot<B> + HasAutodiffModule<Autodiff<B>>,
{
    let model_path = staged.path().join(MODEL_FILE_NAME);
    let mut reloaded = clone_module_detached::<B, M>(original)?;
    let mut store = SafetensorsStore::from_file(&model_path)
        .allow_partial(true)
        .validate(false);
    let result = reloaded
        .load_from(&mut store)
        .map_err(|error| WeightImportError::Store(error.to_string()))?;
    validate_artifact_apply_result(&result, inspection)?;
    compare_module_snapshots::<B, M>(candidate, &reloaded)?;

    let manifest_path = staged.path().join(MANIFEST_FILE_NAME);
    let manifest_bytes = fs::read(manifest_path)?;
    let manifest = serde_json::from_slice::<PfldImportManifest>(&manifest_bytes)
        .map_err(|error| WeightImportError::Manifest(error.to_string()))?;
    if manifest != staged.manifest {
        return Err(WeightImportError::ArtifactValidation(
            "staged manifest content changed after writing".to_owned(),
        ));
    }

    let source_sha256 = sha256_file(source_path)?;
    if source_sha256 != manifest.source.sha256 {
        return Err(WeightImportError::ArtifactValidation(
            "source checkpoint hash does not match manifest".to_owned(),
        ));
    }
    let model_sha256 = sha256_file(&model_path)?;
    if model_sha256 != manifest.model.sha256 {
        return Err(WeightImportError::ArtifactValidation(
            "staged model hash does not match manifest".to_owned(),
        ));
    }
    validate_lower_hex_sha256("source", &manifest.source.sha256)?;
    validate_lower_hex_sha256("model", &manifest.model.sha256)?;

    let reloaded_summary = module_summary::<B, M>(&reloaded)?;
    let manifest_summary = TensorSummary {
        tensor_count: manifest.model.tensor_count,
        total_elements: manifest.model.total_elements,
    };
    if reloaded_summary != manifest_summary || reloaded_summary != inspection.applied {
        return Err(WeightImportError::ArtifactValidation(format!(
            "reloaded tensor summary mismatch: expected {:?}, manifest {:?}, got {:?}",
            inspection.applied, manifest_summary, reloaded_summary
        )));
    }
    if manifest.backbone != inspection.backbone || manifest.ignored != inspection.ignored {
        return Err(WeightImportError::ArtifactValidation(
            "manifest audit does not match checkpoint inspection".to_owned(),
        ));
    }
    validate_audit("batch_norm_counters", &manifest.ignored.batch_norm_counters)?;
    validate_audit("localization", &manifest.ignored.localization)?;
    if let Some(auxiliarynet) = &manifest.ignored.auxiliarynet {
        validate_audit("auxiliarynet", auxiliarynet)?;
    }

    let mut entries = fs::read_dir(staged.path())?
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
    if entries != [MANIFEST_FILE_NAME.to_owned(), MODEL_FILE_NAME.to_owned()] {
        return Err(WeightImportError::ArtifactValidation(format!(
            "unexpected staging entries: {entries:?}"
        )));
    }

    Ok(())
}

pub(super) fn publish_staged_artifacts(
    staged: StagedArtifacts,
    destination: &Path,
) -> Result<PfldImportManifest, WeightImportError> {
    ensure_destination_absent(destination)?;
    fs::rename(staged.path(), destination)?;
    let manifest = staged.manifest.clone();
    let _old_staging_path = staged.directory.keep();
    Ok(manifest)
}

pub(super) fn ensure_destination_absent(path: &Path) -> Result<(), WeightImportError> {
    match fs::symlink_metadata(path) {
        Ok(_) => Err(WeightImportError::ArtifactDestinationExists(
            path.to_owned(),
        )),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(WeightImportError::Io(error)),
    }
}

fn module_summary<B, M>(module: &M) -> Result<TensorSummary, WeightImportError>
where
    B: Backend,
    M: ModuleSnapshot<B>,
{
    let mut tensor_count = 0usize;
    let mut total_elements = 0u64;
    for snapshot in module.collect(None, None, false) {
        tensor_count = tensor_count.checked_add(1).ok_or_else(|| {
            WeightImportError::ArtifactValidation("module tensor count overflowed usize".to_owned())
        })?;
        let elements = tensor_elements(&snapshot)
            .map_err(|error| WeightImportError::ArtifactValidation(error.to_string()))?;
        total_elements = total_elements.checked_add(elements).ok_or_else(|| {
            WeightImportError::ArtifactValidation(
                "module tensor element count overflowed u64".to_owned(),
            )
        })?;
    }
    Ok(TensorSummary {
        tensor_count,
        total_elements,
    })
}

fn compare_module_snapshots<B, M>(expected: &M, actual: &M) -> Result<(), WeightImportError>
where
    B: Backend,
    M: ModuleSnapshot<B>,
{
    let expected = collect_snapshot_map::<B, M>(expected)?;
    let actual = collect_snapshot_map::<B, M>(actual)?;
    if let Some(path) = expected.keys().find(|path| !actual.contains_key(*path)) {
        return Err(WeightImportError::ArtifactValidation(format!(
            "reloaded module is missing tensor {path}"
        )));
    }
    if let Some(path) = actual.keys().find(|path| !expected.contains_key(*path)) {
        return Err(WeightImportError::ArtifactValidation(format!(
            "reloaded module contains unexpected tensor {path}"
        )));
    }
    for (path, expected) in expected {
        let actual = actual
            .get(&path)
            .expect("snapshot key sets were checked above");
        if expected.shape != actual.shape {
            return Err(WeightImportError::ShapeMismatch(path));
        }
        if expected.dtype != actual.dtype {
            return Err(WeightImportError::DTypeMismatch(path));
        }
        let expected_data = expected
            .to_data()
            .map_err(|error| WeightImportError::ArtifactValidation(error.to_string()))?;
        let actual_data = actual
            .to_data()
            .map_err(|error| WeightImportError::ArtifactValidation(error.to_string()))?;
        if expected_data != actual_data {
            return Err(WeightImportError::ArtifactValidation(format!(
                "reloaded tensor data mismatch: {path}"
            )));
        }
    }
    Ok(())
}

fn collect_snapshot_map<B, M>(
    module: &M,
) -> Result<BTreeMap<String, burn_store::TensorSnapshot>, WeightImportError>
where
    B: Backend,
    M: ModuleSnapshot<B>,
{
    let mut snapshots = BTreeMap::new();
    for snapshot in module.collect(None, None, false) {
        let path = snapshot.full_path();
        if snapshots.insert(path.clone(), snapshot).is_some() {
            return Err(WeightImportError::ArtifactValidation(format!(
                "duplicate module tensor path: {path}"
            )));
        }
    }
    Ok(snapshots)
}

fn validate_artifact_apply_result(
    result: &ApplyResult,
    inspection: &PfldInspection,
) -> Result<(), WeightImportError> {
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
            "safetensors reload unexpectedly skipped tensor: {path}"
        )));
    }
    if let Some(path) = result.unused.iter().min() {
        return Err(WeightImportError::UnexpectedTensor(path.clone()));
    }
    let applied = result
        .applied
        .iter()
        .cloned()
        .collect::<std::collections::BTreeSet<_>>();
    if let Some(path) = inspection.expected_applied.difference(&applied).next() {
        return Err(WeightImportError::MissingTensor(path.clone()));
    }
    if let Some(path) = applied.difference(&inspection.expected_applied).next() {
        return Err(WeightImportError::UnexpectedTensor(path.clone()));
    }
    Ok(())
}

fn validate_audit(name: &str, audit: &TensorAudit) -> Result<(), WeightImportError> {
    if audit.tensor_count != audit.keys.len() {
        return Err(WeightImportError::ArtifactValidation(format!(
            "{name} audit tensor count does not match key count"
        )));
    }
    if audit.keys.windows(2).any(|pair| pair[0] >= pair[1]) {
        return Err(WeightImportError::ArtifactValidation(format!(
            "{name} audit keys are not sorted and unique"
        )));
    }
    Ok(())
}

fn validate_lower_hex_sha256(name: &str, hash: &str) -> Result<(), WeightImportError> {
    if hash.len() != 64
        || !hash
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(WeightImportError::ArtifactValidation(format!(
            "{name} SHA-256 must be 64 lowercase hexadecimal characters"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::{collections::BTreeSet, path::PathBuf};

    use burn::nn::LinearConfig;
    use feathertalk_models::backend::CpuBackend;

    use crate::{
        PfldIgnoredTensors, TensorAudit, TensorSummary, WeightImportError, source::sha256_file,
    };

    use super::{PfldInspection, verify_staged_artifacts, write_staged_artifacts};

    struct ArtifactFixture {
        parent: tempfile::TempDir,
        destination: PathBuf,
        source_path: PathBuf,
        source_file_name: String,
        source_sha256: String,
        original: burn::nn::Linear<CpuBackend>,
        candidate: burn::nn::Linear<CpuBackend>,
        inspection: PfldInspection,
    }

    fn artifact_fixture() -> ArtifactFixture {
        let parent = tempfile::tempdir().unwrap();
        let destination = parent.path().join("published");
        let source_path = parent.path().join("source.pth");
        std::fs::write(&source_path, b"immutable-source").unwrap();
        let source_sha256 = sha256_file(&source_path).unwrap();
        let device = Default::default();
        let original = LinearConfig::new(2, 2).init::<CpuBackend>(&device);
        let candidate = original.clone();
        let summary = TensorSummary {
            tensor_count: 2,
            total_elements: 6,
        };
        let ignored = PfldIgnoredTensors {
            batch_norm_counters: TensorAudit {
                tensor_count: 0,
                total_elements: 0,
                keys: Vec::new(),
            },
            localization: TensorAudit {
                tensor_count: 0,
                total_elements: 0,
                keys: Vec::new(),
            },
            auxiliarynet: None,
        };
        let inspection = PfldInspection {
            backbone: summary.clone(),
            applied: summary,
            ignored,
            expected_applied: ["bias", "weight"].into_iter().map(str::to_owned).collect(),
            expected_unused: BTreeSet::new(),
        };

        ArtifactFixture {
            parent,
            destination,
            source_path,
            source_file_name: "source.pth".to_owned(),
            source_sha256,
            original,
            candidate,
            inspection,
        }
    }

    #[test]
    fn corrupt_safetensors_never_publishes_destination() {
        let mut fixture = artifact_fixture();
        let staged = write_staged_artifacts::<CpuBackend, _>(
            &fixture.candidate,
            &fixture.source_path,
            &fixture.source_file_name,
            &fixture.source_sha256,
            &fixture.inspection,
            fixture.parent.path(),
        )
        .unwrap();
        std::fs::write(staged.path().join("model.safetensors"), b"broken").unwrap();

        assert!(matches!(
            verify_staged_artifacts::<CpuBackend, _>(
                &mut fixture.original,
                &fixture.candidate,
                &fixture.source_path,
                &fixture.inspection,
                &staged,
            ),
            Err(WeightImportError::ArtifactValidation(_)) | Err(WeightImportError::Store(_))
        ));
        assert!(!fixture.destination.exists());
    }

    #[test]
    fn corrupt_manifest_never_publishes_destination() {
        let mut fixture = artifact_fixture();
        let staged = write_staged_artifacts::<CpuBackend, _>(
            &fixture.candidate,
            &fixture.source_path,
            &fixture.source_file_name,
            &fixture.source_sha256,
            &fixture.inspection,
            fixture.parent.path(),
        )
        .unwrap();
        std::fs::write(staged.path().join("manifest.json"), b"{not-json").unwrap();

        assert!(matches!(
            verify_staged_artifacts::<CpuBackend, _>(
                &mut fixture.original,
                &fixture.candidate,
                &fixture.source_path,
                &fixture.inspection,
                &staged,
            ),
            Err(WeightImportError::Manifest(_))
        ));
        assert!(!fixture.destination.exists());
    }
}
