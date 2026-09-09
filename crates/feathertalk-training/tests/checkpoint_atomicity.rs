use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
};

use burn::{
    module::Module,
    nn::{Linear, LinearConfig},
    optim::{Adam, AdamConfig, adaptor::OptimizerAdaptor},
    tensor::backend::Backend,
};
use feathertalk_training::{
    CheckpointCompatibility, CheckpointDescriptor, DATA_LOADER_STATE_SCHEMA_VERSION,
    DataLoaderConfig, DataLoaderState, Provenance, RandomAlgorithm, RestoredTrainingState,
    SamplingConfig, SamplingKind, TRAINING_STATE_SCHEMA_VERSION, TrainingCheckpointState,
    TrainingConfig, TrainingError, TrainingMode, load_training_checkpoint,
    save_training_checkpoint,
};

type CpuBackend = burn::backend::NdArray<f32>;
type CpuAutodiffBackend = burn::backend::Autodiff<CpuBackend>;
type TinyOptimizer = OptimizerAdaptor<Adam, TinyModel<CpuAutodiffBackend>, CpuAutodiffBackend>;

#[derive(Module, Debug)]
struct TinyModel<B: Backend> {
    linear: Linear<B>,
}

fn model_and_optimizer(
    device: &burn::tensor::Device<CpuBackend>,
) -> (TinyModel<CpuAutodiffBackend>, TinyOptimizer) {
    (
        TinyModel {
            linear: LinearConfig::new(2, 1).init(device),
        },
        AdamConfig::new().init(),
    )
}

fn state() -> TrainingCheckpointState {
    TrainingCheckpointState {
        schema_version: TRAINING_STATE_SCHEMA_VERSION,
        epoch: 0,
        global_step: 0,
        random_seed: 7,
        data_loader: DataLoaderState {
            schema_version: DATA_LOADER_STATE_SCHEMA_VERSION,
            random_algorithm: RandomAlgorithm::Splitmix64FisherYatesV1,
            config: DataLoaderConfig {
                batch_size: 1,
                seed: 7,
                sampling: SamplingConfig {
                    kind: SamplingKind::SingleFrame,
                    temporal_stride: 0,
                },
            },
            frame_count: 2,
            epoch: 0,
            next_position: 0,
        },
        training_config: TrainingConfig {
            mode: TrainingMode::Baseline,
            batch_size: 1,
            learning_rate: 1e-2,
            total_epochs: 2,
            temporal_stride: 0,
            mouth_weight: 0.0,
            temporal_weight: 0.0,
            temporal_mouth_weight: 0.0,
            perceptual_weight: 0.01,
        },
        asset_provenance: Provenance {
            entries: BTreeMap::new(),
        },
        model_provenance: Provenance {
            entries: BTreeMap::new(),
        },
    }
}

fn oversized_state() -> TrainingCheckpointState {
    let mut value = state();
    value.asset_provenance.entries = (0..5_000)
        .map(|index| (format!("asset-{index:05}"), "a".repeat(64)))
        .collect();
    value
}

fn descriptor() -> CheckpointDescriptor {
    CheckpointDescriptor::new("tiny", "tiny-v1", "0".repeat(64))
}

fn compatibility(state: &TrainingCheckpointState) -> CheckpointCompatibility {
    let mut compatibility =
        CheckpointCompatibility::new(descriptor(), state.training_config.clone(), 2);
    compatibility.asset_provenance = state.asset_provenance.clone();
    compatibility.model_provenance = state.model_provenance.clone();
    compatibility
}

struct SavedCheckpoint {
    root: tempfile::TempDir,
    destination: PathBuf,
    device: burn::tensor::Device<CpuBackend>,
    state: TrainingCheckpointState,
}

fn saved_checkpoint() -> SavedCheckpoint {
    let root = tempfile::tempdir().unwrap();
    let destination = root.path().join("checkpoint-000001");
    let device = Default::default();
    let (model, optimizer) = model_and_optimizer(&device);
    let state = state();
    save_training_checkpoint::<CpuAutodiffBackend, _, _>(
        &destination,
        &model,
        &optimizer,
        descriptor(),
        state.clone(),
    )
    .unwrap();
    SavedCheckpoint {
        root,
        destination,
        device,
        state,
    }
}

fn load_saved(
    saved: &SavedCheckpoint,
    expected: &CheckpointCompatibility,
) -> Result<RestoredTrainingState<TinyModel<CpuAutodiffBackend>, TinyOptimizer>, TrainingError> {
    let (model, optimizer) = model_and_optimizer(&saved.device);
    load_training_checkpoint::<CpuAutodiffBackend, _, _>(
        &saved.destination,
        &model,
        &optimizer,
        &saved.device,
        expected,
    )
}

fn load_error(saved: &SavedCheckpoint, expected: &CheckpointCompatibility) -> TrainingError {
    match load_saved(saved, expected) {
        Ok(_) => panic!("checkpoint load unexpectedly succeeded"),
        Err(error) => error,
    }
}

fn invalidate_model_record(saved: &SavedCheckpoint) {
    fs::write(
        saved.destination.join("model.bin"),
        b"this is not a Burn record",
    )
    .unwrap();
}

fn create_file_symlink(target: &Path, link: &Path) -> std::io::Result<()> {
    #[cfg(windows)]
    {
        std::os::windows::fs::symlink_file(target, link)
    }
    #[cfg(unix)]
    {
        std::os::unix::fs::symlink(target, link)
    }
    #[cfg(not(any(windows, unix)))]
    {
        let _ = (target, link);
        Err(std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            "file symlinks are unsupported on this platform",
        ))
    }
}

fn create_directory_symlink(target: &Path, link: &Path) -> std::io::Result<()> {
    #[cfg(windows)]
    {
        std::os::windows::fs::symlink_dir(target, link)
    }
    #[cfg(unix)]
    {
        std::os::unix::fs::symlink(target, link)
    }
    #[cfg(not(any(windows, unix)))]
    {
        let _ = (target, link);
        Err(std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            "directory symlinks are unsupported on this platform",
        ))
    }
}

fn symlink_creation_unavailable(error: &std::io::Error) -> bool {
    error.kind() == std::io::ErrorKind::PermissionDenied
        || error.kind() == std::io::ErrorKind::Unsupported
        || (cfg!(windows) && error.raw_os_error() == Some(1314))
}

fn staging_entries(parent: &Path) -> Vec<PathBuf> {
    let mut entries = fs::read_dir(parent)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with(".checkpoint-") && name.ends_with(".staging"))
        })
        .collect::<Vec<_>>();
    entries.sort();
    entries
}

#[test]
fn save_publishes_exactly_four_files_and_returns_the_persisted_manifest() {
    let root = tempfile::tempdir().unwrap();
    let destination = root.path().join("checkpoint-000001");
    let device = Default::default();
    let (model, optimizer) = model_and_optimizer(&device);

    let manifest = save_training_checkpoint::<CpuAutodiffBackend, _, _>(
        &destination,
        &model,
        &optimizer,
        descriptor(),
        state(),
    )
    .unwrap();

    let mut names = fs::read_dir(&destination)
        .unwrap()
        .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
        .collect::<Vec<_>>();
    names.sort();
    assert_eq!(
        names,
        vec![
            "manifest.json",
            "model.bin",
            "optimizer.bin",
            "training-state.json",
        ]
    );
    assert_eq!(
        fs::read(destination.join("manifest.json")).unwrap(),
        serde_json::to_vec(&manifest).unwrap()
    );
    assert!(staging_entries(root.path()).is_empty());
}

#[test]
fn existing_destination_is_rejected_without_overwriting_it() {
    let root = tempfile::tempdir().unwrap();
    let destination = root.path().join("checkpoint-000001");
    fs::create_dir(&destination).unwrap();
    fs::write(destination.join("sentinel"), b"old").unwrap();
    let device = Default::default();
    let (model, optimizer) = model_and_optimizer(&device);

    let error = save_training_checkpoint::<CpuAutodiffBackend, _, _>(
        &destination,
        &model,
        &optimizer,
        descriptor(),
        state(),
    )
    .unwrap_err();
    assert!(matches!(error, TrainingError::CheckpointDirectory(_)));
    assert_eq!(fs::read(destination.join("sentinel")).unwrap(), b"old");
    assert!(staging_entries(root.path()).is_empty());
}

#[test]
fn failed_second_save_preserves_every_existing_checkpoint_file() {
    let root = tempfile::tempdir().unwrap();
    let destination = root.path().join("checkpoint-000001");
    let device = Default::default();
    let (model, optimizer) = model_and_optimizer(&device);
    save_training_checkpoint::<CpuAutodiffBackend, _, _>(
        &destination,
        &model,
        &optimizer,
        descriptor(),
        state(),
    )
    .unwrap();

    let before = [
        "manifest.json",
        "model.bin",
        "optimizer.bin",
        "training-state.json",
    ]
    .into_iter()
    .map(|name| (name, fs::read(destination.join(name)).unwrap()))
    .collect::<Vec<_>>();

    let (replacement_model, replacement_optimizer) = model_and_optimizer(&device);
    let error = save_training_checkpoint::<CpuAutodiffBackend, _, _>(
        &destination,
        &replacement_model,
        &replacement_optimizer,
        descriptor(),
        state(),
    )
    .unwrap_err();
    assert!(matches!(error, TrainingError::CheckpointDirectory(_)));

    for (name, expected_bytes) in before {
        assert_eq!(fs::read(destination.join(name)).unwrap(), expected_bytes);
    }
    assert!(staging_entries(root.path()).is_empty());
}

#[test]
fn invalid_state_fails_before_staging_and_leaves_no_partial_directory() {
    let root = tempfile::tempdir().unwrap();
    let destination = root.path().join("checkpoint-000001");
    let device = Default::default();
    let (model, optimizer) = model_and_optimizer(&device);
    let mut invalid = state();
    invalid.random_seed = 99;

    let error = save_training_checkpoint::<CpuAutodiffBackend, _, _>(
        &destination,
        &model,
        &optimizer,
        descriptor(),
        invalid,
    )
    .unwrap_err();
    assert!(matches!(error, TrainingError::InvalidCheckpoint(_)));
    assert!(!destination.exists());
    assert!(staging_entries(root.path()).is_empty());
}

#[test]
fn oversized_state_failure_cleans_staging_after_record_writes() {
    let root = tempfile::tempdir().unwrap();
    let destination = root.path().join("checkpoint-000001");
    let device = Default::default();
    let (model, optimizer) = model_and_optimizer(&device);

    let error = save_training_checkpoint::<CpuAutodiffBackend, _, _>(
        &destination,
        &model,
        &optimizer,
        descriptor(),
        oversized_state(),
    )
    .unwrap_err();
    assert!(matches!(error, TrainingError::InvalidCheckpoint(_)));
    assert!(!destination.exists());
    assert!(staging_entries(root.path()).is_empty());
}

#[test]
fn manifest_unknown_fields_fail_before_burn_record_decoding() {
    let saved = saved_checkpoint();
    invalidate_model_record(&saved);
    let manifest_path = saved.destination.join("manifest.json");
    let mut manifest: serde_json::Value =
        serde_json::from_slice(&fs::read(&manifest_path).unwrap()).unwrap();
    manifest["unexpected"] = true.into();
    fs::write(&manifest_path, serde_json::to_vec(&manifest).unwrap()).unwrap();

    let error = load_error(&saved, &compatibility(&saved.state));
    assert!(matches!(error, TrainingError::InvalidCheckpoint(_)));
}

#[test]
fn missing_optimizer_file_is_rejected_as_an_incomplete_directory() {
    let saved = saved_checkpoint();
    fs::remove_file(saved.destination.join("optimizer.bin")).unwrap();

    let error = load_error(&saved, &compatibility(&saved.state));
    assert!(matches!(error, TrainingError::CheckpointDirectory(_)));
}

#[test]
fn extra_checkpoint_entry_is_rejected_before_json_or_record_loading() {
    let saved = saved_checkpoint();
    fs::write(
        saved.destination.join("notes.txt"),
        b"not part of checkpoint",
    )
    .unwrap();

    let error = load_error(&saved, &compatibility(&saved.state));
    assert!(matches!(error, TrainingError::CheckpointDirectory(_)));
}

#[test]
fn modified_model_bytes_are_rejected_by_sha256() {
    let saved = saved_checkpoint();
    let model_path = saved.destination.join("model.bin");
    let mut bytes = fs::read(&model_path).unwrap();
    bytes[0] ^= 1;
    fs::write(&model_path, bytes).unwrap();

    let error = load_error(&saved, &compatibility(&saved.state));
    assert!(matches!(
        error,
        TrainingError::HashMismatch { ref file, .. } if file == "model.bin"
    ));
}

#[test]
fn model_config_mismatch_fails_before_burn_record_decoding() {
    let saved = saved_checkpoint();
    invalidate_model_record(&saved);
    let mut expected = compatibility(&saved.state);
    expected.descriptor.model_config_sha256 = "1".repeat(64);

    let error = load_error(&saved, &expected);
    assert!(matches!(error, TrainingError::CheckpointCompatibility(_)));
}

#[test]
fn asset_provenance_mismatch_fails_before_burn_record_decoding() {
    let saved = saved_checkpoint();
    invalidate_model_record(&saved);
    let mut expected = compatibility(&saved.state);
    expected
        .asset_provenance
        .entries
        .insert("assets".to_owned(), "a".repeat(64));

    let error = load_error(&saved, &expected);
    assert!(matches!(error, TrainingError::CheckpointCompatibility(_)));
}

#[test]
fn frame_count_mismatch_fails_before_burn_record_decoding() {
    let saved = saved_checkpoint();
    invalidate_model_record(&saved);
    let mut expected = compatibility(&saved.state);
    expected.frame_count = 3;

    let error = load_error(&saved, &expected);
    assert!(matches!(error, TrainingError::CheckpointCompatibility(_)));
}

#[test]
fn unsupported_optimizer_schema_fails_before_burn_record_decoding() {
    let saved = saved_checkpoint();
    invalidate_model_record(&saved);
    let manifest_path = saved.destination.join("manifest.json");
    let mut manifest: serde_json::Value =
        serde_json::from_slice(&fs::read(&manifest_path).unwrap()).unwrap();
    manifest["optimizer_schema_version"] = 2.into();
    fs::write(&manifest_path, serde_json::to_vec(&manifest).unwrap()).unwrap();

    let error = load_error(&saved, &compatibility(&saved.state));
    assert!(matches!(error, TrainingError::InvalidCheckpoint(_)));
}

#[test]
fn symlinked_model_file_is_rejected_when_the_platform_allows_symlinks() {
    let saved = saved_checkpoint();
    let model_path = saved.destination.join("model.bin");
    let target = saved.root.path().join("outside-model.bin");
    fs::rename(&model_path, &target).unwrap();
    if let Err(error) = create_file_symlink(&target, &model_path) {
        if symlink_creation_unavailable(&error) {
            eprintln!("skipping symlink assertion: {error}");
            return;
        }
        panic!("unable to create test symlink: {error}");
    }

    let error = load_error(&saved, &compatibility(&saved.state));
    assert!(matches!(error, TrainingError::CheckpointDirectory(_)));
}

#[test]
fn symlinked_destination_parent_is_rejected_without_writing_through_it() {
    let root = tempfile::tempdir().unwrap();
    let real_parent = root.path().join("real-parent");
    fs::create_dir(&real_parent).unwrap();
    let linked_parent = root.path().join("linked-parent");
    if let Err(error) = create_directory_symlink(&real_parent, &linked_parent) {
        if symlink_creation_unavailable(&error) {
            eprintln!("skipping symlink parent assertion: {error}");
            return;
        }
        panic!("unable to create test directory symlink: {error}");
    }

    let destination = linked_parent.join("checkpoint-000001");
    let device = Default::default();
    let (model, optimizer) = model_and_optimizer(&device);
    let error = save_training_checkpoint::<CpuAutodiffBackend, _, _>(
        &destination,
        &model,
        &optimizer,
        descriptor(),
        state(),
    )
    .unwrap_err();
    assert!(matches!(error, TrainingError::CheckpointDirectory(_)));
    assert!(!real_parent.join("checkpoint-000001").exists());
    assert!(staging_entries(root.path()).is_empty());
}
