use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
};

use burn::{nn::LinearConfig, tensor::backend::Backend};
use burn_store::ModuleSnapshot;
use feathertalk_models::{PFLD_GhostOne, PfldConfig, backend::CpuBackend};
use feathertalk_weights::{PfldImportRequest, WeightImportError, import_pfld_checkpoint};

fn checkpoint_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../data_utils/checkpoint_epoch_335.pth.tar")
}

fn request(destination_dir: PathBuf) -> PfldImportRequest {
    PfldImportRequest {
        checkpoint: checkpoint_path(),
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
    assert_eq!(before, &capture_module_data(module));
}

#[test]
fn existing_destination_is_rejected_without_overwrite_or_model_mutation() {
    let temp = tempfile::tempdir().unwrap();
    let destination = temp.path().join("published");
    std::fs::create_dir(&destination).unwrap();
    std::fs::write(destination.join("sentinel.txt"), b"keep").unwrap();
    let device = Default::default();
    let mut model = PFLD_GhostOne::<CpuBackend>::new(PfldConfig::production(), &device);
    let before = capture_module_data(&model);

    let error = import_pfld_checkpoint::<CpuBackend, _>(&mut model, &request(destination.clone()))
        .unwrap_err();

    assert!(matches!(
        error,
        WeightImportError::ArtifactDestinationExists(path) if path == destination
    ));
    assert_eq!(
        std::fs::read(destination.join("sentinel.txt")).unwrap(),
        b"keep"
    );
    assert_module_data_unchanged::<CpuBackend, _>(&before, &model);
}

#[test]
fn incompatible_module_leaves_destination_absent_and_module_unchanged() {
    let temp = tempfile::tempdir().unwrap();
    let destination = temp.path().join("published");
    let device = Default::default();
    let mut model = LinearConfig::new(2, 2).init::<CpuBackend>(&device);
    let before = capture_module_data(&model);

    let error = import_pfld_checkpoint::<CpuBackend, _>(&mut model, &request(destination.clone()))
        .unwrap_err();

    assert!(matches!(
        error,
        WeightImportError::MissingTensor(_) | WeightImportError::UnexpectedTensor(_)
    ));
    assert!(!destination.exists());
    assert_module_data_unchanged::<CpuBackend, _>(&before, &model);
}
