use std::fs;
use std::path::PathBuf;
use std::sync::Mutex;

use feathertalk_audio::{FeatureMatrix, read_feature_file};
use feathertalk_domain::{ErrorCode, MigrateLegacyFeaturesParams, Progress, Request, TaskStage};
use feathertalk_media::CancellationToken;
use feathertalk_worker::{
    CommandOutcome, NoReporter, TaskReporter, WorkerConfig, execute,
    execute_migrate_legacy_features,
};
use ndarray::{Array2, Array3};

/// Records the stages a command reports, so a test can assert their order.
struct Recorder {
    events: Mutex<Vec<(TaskStage, Option<Progress>)>>,
}

impl Recorder {
    fn new() -> Self {
        Self {
            events: Mutex::new(Vec::new()),
        }
    }

    fn stages(&self) -> Vec<TaskStage> {
        self.events
            .lock()
            .expect("the recorder is intact")
            .iter()
            .map(|(stage, _)| stage.clone())
            .collect()
    }
}

impl TaskReporter for Recorder {
    fn report(&self, stage: TaskStage, progress: Option<Progress>) {
        self.events
            .lock()
            .expect("the recorder is intact")
            .push((stage, progress));
    }
}

fn params(source: PathBuf, destination: PathBuf) -> MigrateLegacyFeaturesParams {
    MigrateLegacyFeaturesParams {
        source,
        destination,
    }
}

fn valid_npy(path: &std::path::Path) {
    let values = (0..2048).map(|value| value as f32).collect::<Vec<_>>();
    ndarray_npy::write_npy(path, &Array3::from_shape_vec((1, 2, 1024), values).unwrap()).unwrap();
}

#[test]
fn migrates_valid_npy_and_reports_artifact() {
    let root = tempfile::tempdir().unwrap();
    let source = root.path().join("legacy.npy");
    let destination = root.path().join("features.f32");
    valid_npy(&source);

    let payload = execute_migrate_legacy_features(
        &params(source.clone(), destination.clone()),
        &CancellationToken::new(),
        &NoReporter,
    )
    .unwrap();

    let matrix = read_feature_file(&destination).unwrap();
    assert_eq!(
        matrix,
        FeatureMatrix::new(2, 1024, (0..2048).map(|v| v as f32).collect()).unwrap()
    );
    assert_eq!(payload["kind"], "migrate_legacy_features");
    assert_eq!(payload["source"], source.display().to_string());
    assert_eq!(payload["destination"], destination.display().to_string());
    assert_eq!(payload["source_shape"], serde_json::json!([1, 2, 1024]));
    assert_eq!(payload["tokens"], 2);
    assert_eq!(payload["dims"], 1024);
    assert_eq!(payload["bytes"], 8236);
    assert_eq!(payload["sha256"].as_str().unwrap().len(), 64);
}

#[test]
fn rejects_relative_source_and_existing_destination_before_reading() {
    let root = tempfile::tempdir().unwrap();
    let destination = root.path().join("features.f32");
    fs::write(&destination, b"keep").unwrap();
    let error = execute_migrate_legacy_features(
        &params(PathBuf::from("legacy.npy"), destination.clone()),
        &CancellationToken::new(),
        &NoReporter,
    )
    .unwrap_err();
    assert!(error.to_string().contains("absolute"));
    assert_eq!(fs::read(destination).unwrap(), b"keep");
}

#[test]
fn rejects_wrong_rank_without_creating_destination() {
    let root = tempfile::tempdir().unwrap();
    let source = root.path().join("rank2.npy");
    ndarray_npy::write_npy(&source, &Array2::<f32>::zeros((2, 1024))).unwrap();
    let destination = root.path().join("features.f32");
    let error = execute_migrate_legacy_features(
        &params(source, destination.clone()),
        &CancellationToken::new(),
        &NoReporter,
    )
    .unwrap_err();
    assert!(error.to_string().contains("rank 3"));
    assert!(!destination.exists());
}

#[test]
fn cancellation_before_read_is_reported_at_preparing() {
    let root = tempfile::tempdir().unwrap();
    let source = root.path().join("legacy.npy");
    valid_npy(&source);
    let destination = root.path().join("features.f32");
    let token = CancellationToken::new();
    token.cancel();
    let error =
        execute_migrate_legacy_features(&params(source, destination.clone()), &token, &NoReporter)
            .unwrap_err();
    assert!(error.is_cancelled());
    assert_eq!(error.stage(), TaskStage::Preparing);
    assert!(!destination.exists());
}

#[test]
fn the_command_reports_preparing_then_importing_and_completes() {
    let root = tempfile::tempdir().unwrap();
    let source = root.path().join("legacy.npy");
    let destination = root.path().join("features.f32");
    valid_npy(&source);
    let recorder = Recorder::new();

    let outcome = execute(
        &Request::MigrateLegacyFeatures(params(source, destination.clone())),
        &WorkerConfig::from_values(None, None, None),
        &CancellationToken::new(),
        &recorder,
    );

    let CommandOutcome::Completed(Some(payload)) = outcome else {
        panic!("expected a completed payload, got {outcome:?}");
    };
    assert_eq!(payload["kind"], "migrate_legacy_features");
    assert_eq!(payload["destination"], destination.display().to_string());
    assert_eq!(
        recorder.stages(),
        vec![
            TaskStage::Preparing,
            TaskStage::Importing,
            TaskStage::Importing
        ]
    );
}

#[test]
fn the_command_maps_a_broken_source_to_model_incompatible() {
    let root = tempfile::tempdir().unwrap();
    let source = root.path().join("broken.npy");
    fs::write(&source, b"not-an-npy-file").unwrap();
    let destination = root.path().join("features.f32");

    let outcome = execute(
        &Request::MigrateLegacyFeatures(params(source, destination.clone())),
        &WorkerConfig::from_values(None, None, None),
        &CancellationToken::new(),
        &NoReporter,
    );

    let CommandOutcome::Failed(error) = outcome else {
        panic!("expected a failure, got {outcome:?}");
    };
    assert_eq!(error.code, ErrorCode::ModelIncompatible);
    assert_eq!(error.summary, "特征迁移失败");
    assert_eq!(error.stage, TaskStage::Importing);
    error.validate().unwrap();
    assert!(!destination.exists());
}

#[test]
fn the_command_reports_cancellation_instead_of_a_failure() {
    let root = tempfile::tempdir().unwrap();
    let source = root.path().join("legacy.npy");
    let destination = root.path().join("features.f32");
    valid_npy(&source);
    let token = CancellationToken::new();
    token.cancel();

    let outcome = execute(
        &Request::MigrateLegacyFeatures(params(source, destination.clone())),
        &WorkerConfig::from_values(None, None, None),
        &token,
        &NoReporter,
    );

    assert!(matches!(outcome, CommandOutcome::Cancelled), "{outcome:?}");
    assert!(!destination.exists());
}
