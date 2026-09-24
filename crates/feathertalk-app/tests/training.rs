//! What a project's training has left on disk, as the page reads it.

use std::fs;
use std::path::Path;

use feathertalk_app::assets::AssetSurvey;
use feathertalk_app::facts::FactValue;
use feathertalk_app::training::{
    checkpoint_facts, config_facts, form_state, fresh_over_checkpoint, metric_summary,
    metrics_facts, read_checkpoint_state, read_metrics, FormState, ReadError, TrainingForm,
    TrainingSurvey, DEFAULT_EPOCHS, MAX_EPOCHS, MIN_EPOCHS,
};
use feathertalk_domain::{Request, TrainingMode, UnetVariant};
use feathertalk_project::{AssetManifest, AssetPackageState, FeatureType};

/// A directory the tests write their JSON into.
fn dir() -> tempfile::TempDir {
    tempfile::tempdir().expect("a temporary directory")
}

/// A project directory with the three directories a run writes into.
fn project() -> tempfile::TempDir {
    let dir = tempfile::tempdir().expect("a temporary directory");
    for path in [
        dir.path().join("models").join("unet"),
        dir.path().join("outputs").join("metrics"),
        dir.path().join("outputs").join("preview"),
    ] {
        fs::create_dir_all(&path).expect("a training directory");
    }
    dir
}

/// Write `body` to `path`, creating whatever directories it needs.
fn write(path: &Path, body: &str) {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).expect("the parent directory");
    }
    fs::write(path, body).expect("the file writes");
}

/// Write the checkpoint of `step`, with `body` as its training state.
fn checkpoint(project_dir: &Path, step: u64, body: &str) {
    let dir = project_dir
        .join("models")
        .join("unet")
        .join(format!("checkpoint-{step:08}"));
    fs::create_dir_all(&dir).expect("the checkpoint directory");
    write(&dir.join("training-state.json"), body);
}

/// Create a directory under `models/unet` under a name of its own.
fn checkpoint_dir(project_dir: &Path, name: &str) {
    let dir = project_dir.join("models").join("unet").join(name);
    fs::create_dir_all(&dir).expect("the directory");
}

/// Write the metrics of `step`.
fn metrics_file(project_dir: &Path, step: u64, body: &str) {
    let path = project_dir
        .join("outputs")
        .join("metrics")
        .join(format!("step-{step:08}.json"));
    write(&path, body);
}

/// Create the preview directory of `step`, which holds tensors this never reads.
fn preview_dir(project_dir: &Path, step: u64) {
    let dir = project_dir
        .join("outputs")
        .join("preview")
        .join(format!("step-{step:08}"));
    fs::create_dir_all(&dir).expect("the preview directory");
}

/// An asset survey whose package is in `state`.
///
/// The training gate asks the asset page's photograph one question -- is the
/// package locked -- so the rest is filled in with the shape a locked package has
/// and never read.
fn package(state: AssetPackageState) -> AssetSurvey {
    AssetSurvey {
        manifest: Some(AssetManifest {
            schema_version: 1,
            state,
            video_fps: 25,
            audio_sample_rate: 16_000,
            audio_channels: 1,
            frame_count: 188,
            frame_width: 160,
            frame_height: 160,
            feature_type: FeatureType::FeatherHubert,
            feature_shape: [188, 2, 1024],
            landmark_model_sha256: std::iter::repeat_n('a', 64).collect(),
            feature_model_sha256: std::iter::repeat_n('b', 64).collect(),
        }),
        ..AssetSurvey::default()
    }
}

/// The metrics file the worker writes at an epoch boundary, in mouth ROI mode.
///
/// The optional components have no `skip_serializing_if` upstream, so a mode that
/// does not compute one writes `null` instead of leaving the field out.
fn metrics_json(schema_version: u32) -> String {
    metrics_body(schema_version, 12, 2400)
}

/// The same file, at whichever point of a run a test needs it.
fn metrics_body(schema_version: u32, epoch: u64, global_step: u64) -> String {
    format!(
        r#"{{
  "schema_version": {schema_version},
  "mode": "mouth_roi",
  "epoch": {epoch},
  "global_step": {global_step},
  "total_loss": 0.4231,
  "full_loss": 0.3125,
  "perceptual_loss": 0.0182,
  "mouth_loss": 0.0924,
  "temporal_loss": null,
  "temporal_mouth_loss": null,
  "samples_seen": {global_step},
  "samples_per_second": 3.75,
  "estimated_remaining_seconds": 1830.5,
  "gpu_memory_bytes": null,
  "worker_state": "training"
}}"#
    )
}

/// The metrics of a baseline run, written by a worker that leaves the components
/// it never computes out of the file entirely.
fn baseline_metrics_json() -> String {
    r#"{
  "schema_version": 1,
  "mode": "baseline",
  "epoch": 1,
  "global_step": 200,
  "total_loss": 0.61,
  "full_loss": 0.6,
  "perceptual_loss": 0.01,
  "samples_seen": 200,
  "samples_per_second": 5.0,
  "estimated_remaining_seconds": 3600.0,
  "worker_state": "training"
}"#
    .to_owned()
}

/// A checkpoint's `training-state.json`, with only the fields a panel paints.
fn state_json(schema_version: u32) -> String {
    state_body(schema_version, 12, 2400)
}

/// The same file, at whichever point of a run a test needs it.
fn state_body(schema_version: u32, epoch: u64, global_step: u64) -> String {
    format!(
        r#"{{
  "schema_version": {schema_version},
  "epoch": {epoch},
  "global_step": {global_step},
  "random_seed": 1,
  "training_config": {{
    "mode": "mouth_roi",
    "batch_size": 1,
    "learning_rate": 0.0001,
    "total_epochs": 200,
    "temporal_stride": 1,
    "mouth_weight": 4.0,
    "temporal_weight": 0.5,
    "temporal_mouth_weight": 4.0,
    "perceptual_weight": 0.01
  }}
}}"#
    )
}

/// The whole file the worker writes: a data loader position and two provenance
/// maps sit beside the fields above.
fn full_state_json() -> String {
    r#"{
  "schema_version": 1,
  "epoch": 12,
  "global_step": 2400,
  "random_seed": 1,
  "data_loader": {
    "schema_version": 1,
    "random_algorithm": "xoshiro256plusplus",
    "config": {
      "batch_size": 1,
      "temporal_stride": 1,
      "shuffle": true
    },
    "frame_count": 200,
    "epoch": 12,
    "next_position": 0
  },
  "training_config": {
    "mode": "mouth_roi",
    "batch_size": 1,
    "learning_rate": 0.0001,
    "total_epochs": 200,
    "temporal_stride": 1,
    "mouth_weight": 4.0,
    "temporal_weight": 0.5,
    "temporal_mouth_weight": 4.0,
    "perceptual_weight": 0.01
  },
  "asset_provenance": {
    "entries": {
      "assets.json": "b1946ac92492d2347c6235b4d2611184"
    }
  },
  "model_provenance": {
    "entries": {
      "model.bin": "5891b5b522d5df086d0ff0b110fbd9d2"
    }
  }
}"#
    .to_owned()
}

#[test]
fn a_metrics_file_parses() {
    let dir = dir();
    let path = dir.path().join("step-00002400.json");
    write(&path, &metrics_json(1));

    let metrics = read_metrics(&path).expect("the metrics read");

    assert_eq!(metrics.schema_version, 1);
    assert_eq!(metrics.mode, "mouth_roi");
    assert_eq!(metrics.epoch, 12);
    assert_eq!(metrics.global_step, 2400);
    assert_eq!(metrics.total_loss, 0.4231);
    assert_eq!(metrics.full_loss, 0.3125);
    assert_eq!(metrics.perceptual_loss, 0.0182);
    assert_eq!(metrics.mouth_loss, Some(0.0924));
    assert_eq!(metrics.temporal_loss, None);
    assert_eq!(metrics.temporal_mouth_loss, None);
    assert_eq!(metrics.samples_seen, 2400);
    assert_eq!(metrics.samples_per_second, 3.75);
    assert_eq!(metrics.estimated_remaining_seconds, 1830.5);
    assert_eq!(metrics.gpu_memory_bytes, None);
}

#[test]
fn absent_loss_components_stay_absent() {
    let dir = dir();
    let path = dir.path().join("step-00000200.json");
    write(&path, &baseline_metrics_json());

    let metrics = read_metrics(&path).expect("the metrics read");

    assert_eq!(metrics.mode, "baseline");
    assert_eq!(metrics.mouth_loss, None);
    assert_eq!(metrics.temporal_loss, None);
    assert_eq!(metrics.temporal_mouth_loss, None);
}

#[test]
fn an_unknown_metrics_schema_is_refused() {
    let dir = dir();
    let path = dir.path().join("step-00002400.json");
    write(&path, &metrics_json(2));

    let error = read_metrics(&path).expect_err("an unknown schema is refused");

    match error {
        ReadError::Schema { expected, found } => {
            assert_eq!(expected, 1);
            assert_eq!(found, 2);
        }
        other => panic!("expected a schema error, got {other:?}"),
    }
}

#[test]
fn broken_metrics_json_is_refused() {
    let dir = dir();
    let path = dir.path().join("step-00002400.json");
    write(&path, "{");

    let error = read_metrics(&path).expect_err("broken JSON is refused");

    assert!(
        matches!(error, ReadError::Json(_)),
        "expected a JSON error, got {error:?}"
    );
}

#[test]
fn an_oversized_metrics_file_is_refused() {
    let dir = dir();
    let path = dir.path().join("step-00002400.json");
    write(&path, &" ".repeat(65 * 1024));

    let error = read_metrics(&path).expect_err("an oversized file is refused");

    match error {
        ReadError::TooLarge { limit } => assert_eq!(limit, 64 * 1024),
        other => panic!("expected a size error, got {other:?}"),
    }
}

#[test]
fn a_missing_file_is_an_io_error() {
    let dir = dir();
    let path = dir.path().join("step-00002400.json");

    let error = read_metrics(&path).expect_err("a missing file is refused");

    assert!(
        matches!(error, ReadError::Io(_)),
        "expected an io error, got {error:?}"
    );
}

#[test]
fn a_training_state_parses() {
    let dir = dir();
    let path = dir.path().join("training-state.json");
    write(&path, &state_json(1));

    let state = read_checkpoint_state(&path).expect("the state reads");

    assert_eq!(state.schema_version, 1);
    assert_eq!(state.epoch, 12);
    assert_eq!(state.global_step, 2400);
    assert_eq!(state.random_seed, 1);
    assert_eq!(state.training_config.mode, "mouth_roi");
    assert_eq!(state.training_config.batch_size, 1);
    assert_eq!(state.training_config.learning_rate, 0.0001);
    assert_eq!(state.training_config.total_epochs, 200);
    assert_eq!(state.training_config.temporal_stride, 1);
    assert_eq!(state.training_config.mouth_weight, 4.0);
    assert_eq!(state.training_config.temporal_weight, 0.5);
    assert_eq!(state.training_config.temporal_mouth_weight, 4.0);
    assert_eq!(state.training_config.perceptual_weight, 0.01);
}

#[test]
fn unknown_fields_are_ignored() {
    let dir = dir();
    let path = dir.path().join("training-state.json");
    write(&path, &full_state_json());

    let state = read_checkpoint_state(&path).expect("the state reads");

    assert_eq!(state.global_step, 2400);
    assert_eq!(state.training_config.total_epochs, 200);
}

#[test]
fn a_project_that_never_trained_has_nothing() {
    let dir = project();

    let survey = TrainingSurvey::inspect(dir.path());

    assert!(survey.checkpoint.is_none());
    assert_eq!(survey.checkpoint_count, 0);
    assert!(survey.state.is_none());
    assert!(survey.state_error.is_none());
    assert!(survey.metrics.is_none());
    assert!(survey.metrics_error.is_none());
    assert!(survey.preview_step.is_none());
    assert_eq!(survey.preview_count, 0);
    assert!(!survey.has_history());
}

#[test]
fn the_latest_checkpoint_wins() {
    let dir = project();
    checkpoint(dir.path(), 188, &state_body(1, 1, 188));
    checkpoint(dir.path(), 376, &state_body(1, 2, 376));

    let survey = TrainingSurvey::inspect(dir.path());

    let latest = survey.checkpoint.as_ref().expect("a checkpoint");
    assert_eq!(latest.step, 376);
    assert!(latest.path.ends_with("checkpoint-00000376"));
    assert_eq!(survey.checkpoint_count, 2);
    assert!(survey.has_history());
}

#[test]
fn names_that_are_not_ours_are_ignored() {
    let dir = project();
    for name in [
        "checkpoint-188",
        "checkpoint-0000000a",
        ".publish-1234-0",
        ".retired-1234-0",
    ] {
        checkpoint_dir(dir.path(), name);
    }

    let survey = TrainingSurvey::inspect(dir.path());

    assert!(survey.checkpoint.is_none());
    assert_eq!(survey.checkpoint_count, 0);
}

#[test]
fn a_step_past_eight_digits_still_counts() {
    let dir = project();
    checkpoint_dir(dir.path(), "checkpoint-000001880");

    let survey = TrainingSurvey::inspect(dir.path());

    let latest = survey.checkpoint.expect("a checkpoint");
    assert_eq!(latest.step, 1880);
}

#[test]
fn the_state_comes_from_the_latest_checkpoint() {
    let dir = project();
    checkpoint(dir.path(), 188, &state_body(1, 1, 188));
    checkpoint(dir.path(), 376, &state_body(1, 2, 376));

    let survey = TrainingSurvey::inspect(dir.path());

    let state = survey.state.expect("the state");
    assert_eq!(state.global_step, 376);
    assert_eq!(state.epoch, 2);
    assert!(survey.state_error.is_none());
}

#[test]
fn a_broken_state_becomes_an_error_line() {
    let dir = project();
    checkpoint(dir.path(), 188, &state_body(1, 1, 188));
    checkpoint(dir.path(), 376, "{");

    let survey = TrainingSurvey::inspect(dir.path());

    assert!(survey.state.is_none());
    assert!(survey.state_error.is_some());
}

#[test]
fn a_checkpoint_without_a_state_file_is_not_an_error() {
    let dir = project();
    checkpoint_dir(dir.path(), "checkpoint-00000376");

    let survey = TrainingSurvey::inspect(dir.path());

    assert!(survey.checkpoint.is_some());
    assert!(survey.state.is_none());
    assert!(survey.state_error.is_none());
}

#[test]
fn the_latest_metrics_file_wins() {
    let dir = project();
    metrics_file(dir.path(), 188, &metrics_body(1, 1, 188));
    metrics_file(dir.path(), 376, &metrics_body(1, 2, 376));

    let survey = TrainingSurvey::inspect(dir.path());

    let metrics = survey.metrics.as_ref().expect("the metrics");
    assert_eq!(metrics.global_step, 376);
    assert_eq!(metrics.epoch, 2);
    assert!(survey.metrics_error.is_none());
    assert!(survey.has_history());
}

#[test]
fn a_broken_metrics_file_becomes_an_error_line() {
    let dir = project();
    metrics_file(dir.path(), 188, &metrics_body(1, 1, 188));
    metrics_file(dir.path(), 376, "{");

    let survey = TrainingSurvey::inspect(dir.path());

    assert!(survey.metrics.is_none());
    assert!(survey.metrics_error.is_some());
}

#[test]
fn previews_are_counted() {
    let dir = project();
    for step in [188, 376, 564] {
        preview_dir(dir.path(), step);
    }

    let survey = TrainingSurvey::inspect(dir.path());

    assert_eq!(survey.preview_count, 3);
    assert_eq!(survey.preview_step, Some(564));
}

#[test]
fn the_facts_cover_the_three_panels() {
    let dir = project();
    checkpoint(dir.path(), 376, &state_body(1, 2, 376));
    metrics_file(dir.path(), 376, &metrics_body(1, 2, 376));
    preview_dir(dir.path(), 376);

    let survey = TrainingSurvey::inspect(dir.path());
    let metrics = survey.metrics.as_ref().expect("the metrics");
    let numbers = metrics_facts(metrics);

    assert!(numbers
        .iter()
        .any(|fact| fact.label == "training.metrics.total_loss"));
    assert!(numbers
        .iter()
        .any(|fact| fact.label == "training.metrics.samples_per_second"));
    assert!(numbers
        .iter()
        .any(|fact| fact.label == "training.metrics.gpu_memory"
            && fact.value == FactValue::Key("training.metrics.gpu_memory_none")));

    let state = survey.state.as_ref().expect("the state");
    let config = config_facts(state);

    assert!(config
        .iter()
        .any(|fact| fact.label == "training.config.batch_size"));
    assert!(config
        .iter()
        .any(|fact| fact.label == "training.config.learning_rate"));

    let checkpoints = checkpoint_facts(&survey);

    assert!(checkpoints
        .iter()
        .any(|fact| fact.label == "training.checkpoint.latest"
            && fact.value == FactValue::Text("376".to_owned())));
    assert!(checkpoints
        .iter()
        .any(|fact| fact.label == "training.checkpoint.count"
            && fact.value == FactValue::Text("1".to_owned())));
}

#[test]
fn the_form_starts_on_the_fast_preset() {
    let form = TrainingForm::default();

    assert_eq!(form.mode(), TrainingMode::Baseline);
    assert_eq!(form.variant(), UnetVariant::OriginalUnet);
    assert_eq!(form.epochs(), DEFAULT_EPOCHS);
    assert_eq!(DEFAULT_EPOCHS, 200);
    assert!(!form.resume());
}

#[test]
fn saved_metrics_show_one_based_epochs_and_keep_cumulative_counts_in_details() {
    for (epoch, expected) in [(0, "1"), (199, "200")] {
        let mut metrics: feathertalk_app::training::Metrics =
            serde_json::from_str(&metrics_body(1, epoch, 1_585_800)).unwrap();
        metrics.samples_seen = 1_585_800;
        let summary = metric_summary(&metrics);
        assert!(summary
            .iter()
            .any(|fact| fact.label == "workflow.training.metrics_epoch"
                && fact.value == FactValue::Text(expected.into())));
        assert!(!summary.iter().any(|fact| matches!(
            fact.label,
            "workflow.training.metrics_step" | "training.metrics.samples_seen"
        )));
        let details = metrics_facts(&metrics);
        for label in [
            "workflow.training.metrics_step",
            "training.metrics.samples_seen",
        ] {
            assert!(
                details
                    .iter()
                    .any(|fact| fact.label == label
                        && fact.value == FactValue::Text("1585800".into()))
            );
        }
    }
}

#[test]
fn epochs_clamp_to_the_worker_range() {
    let mut form = TrainingForm::default();

    form.set_epochs(0.0);
    assert_eq!(form.epochs(), MIN_EPOCHS);
    form.set_epochs(f64::from(MAX_EPOCHS) + 1.0);
    assert_eq!(form.epochs(), MAX_EPOCHS);
    form.set_epochs(-5.0);
    assert_eq!(form.epochs(), MIN_EPOCHS);
}

#[test]
fn a_fractional_epoch_rounds() {
    let mut form = TrainingForm::default();

    form.set_epochs(199.4);
    assert_eq!(form.epochs(), 199);
    form.set_epochs(199.6);
    assert_eq!(form.epochs(), 200);
}

#[test]
fn a_non_finite_epoch_keeps_the_last_value() {
    let mut form = TrainingForm::default();
    form.set_epochs(188.0);

    form.set_epochs(f64::NAN);
    assert_eq!(form.epochs(), 188);
    form.set_epochs(f64::INFINITY);
    assert_eq!(form.epochs(), 188);
    form.set_epochs(f64::NEG_INFINITY);
    assert_eq!(form.epochs(), 188);
}

#[test]
fn the_request_carries_the_training_form_values() {
    let dir = project();
    let mut form = TrainingForm::default();
    form.set_mode(TrainingMode::Temporal);
    form.set_variant(UnetVariant::MobileOneUnet);
    form.set_epochs(60.0);
    form.set_batch_size(4.0);
    form.set_resume(true);

    match form.request(dir.path()) {
        Request::Train(params) => {
            assert_eq!(params.project_dir.as_path(), dir.path());
            assert_eq!(params.mode, TrainingMode::Temporal);
            assert_eq!(params.variant, UnetVariant::MobileOneUnet);
            assert_eq!(params.epochs, 60);
            assert_eq!(params.batch_size, 4);
            assert!(params.resume);
        }
        other => panic!("expected a train request, got {other:?}"),
    }
}

#[test]
fn batch_size_input_submits_positive_whole_numbers() {
    for (input, expected) in [
        (0.0, 1),
        (-5.0, 1),
        (1.49, 1),
        (1.5, 2),
        (3.6, 4),
        (f64::from(u32::MAX), u32::MAX),
        (f64::MAX, u32::MAX),
    ] {
        let mut form = TrainingForm::default();
        form.set_batch_size(input);
        let Request::Train(params) = form.request(Path::new("project")) else {
            panic!("expected a train request");
        };
        assert_eq!(params.batch_size, expected, "input: {input}");
        assert_eq!(form.batch_size(), expected);
    }
}

#[test]
fn non_finite_batch_sizes_keep_the_last_submittable_value() {
    let mut form = TrainingForm::default();
    form.set_batch_size(8.0);
    for input in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        form.set_batch_size(input);
        let Request::Train(params) = form.request(Path::new("project")) else {
            panic!("expected a train request");
        };
        assert_eq!(params.batch_size, 8);
    }
}

#[test]
fn a_new_form_submits_single_sample_batches() {
    let Request::Train(params) = TrainingForm::default().request(Path::new("project")) else {
        panic!("expected a train request");
    };
    assert_eq!(params.batch_size, 1);
}

#[test]
fn an_unlocked_package_blocks() {
    let form = TrainingForm::default();
    let training = TrainingSurvey::default();

    let state = form_state(&form, &package(AssetPackageState::Preparing), &training);

    assert_eq!(state, FormState::Blocked("training.blocked.not_locked"));
    assert!(!state.is_submittable());
}

#[test]
fn resume_without_a_checkpoint_blocks() {
    let mut form = TrainingForm::default();
    form.set_resume(true);
    let training = TrainingSurvey::default();

    let state = form_state(&form, &package(AssetPackageState::Locked), &training);

    assert_eq!(state, FormState::Blocked("training.blocked.no_checkpoint"));
}

#[test]
fn resume_with_a_different_batch_size_blocks_before_submission() {
    let dir = project();
    let mut saved: serde_json::Value = serde_json::from_str(&state_body(1, 12, 2400)).unwrap();
    saved["training_config"]["batch_size"] = serde_json::json!(8);
    checkpoint(dir.path(), 2400, &saved.to_string());
    let training = TrainingSurvey::inspect(dir.path());
    let mut form = TrainingForm::default();
    form.set_resume(true);

    assert_eq!(
        form_state(&form, &package(AssetPackageState::Locked), &training),
        FormState::Blocked("training.blocked.batch_size_mismatch")
    );

    form.set_batch_size(8.0);
    assert_eq!(
        form_state(&form, &package(AssetPackageState::Locked), &training),
        FormState::Ready
    );

    form.set_batch_size(2.0);
    form.set_resume(false);
    assert_eq!(
        form_state(&form, &package(AssetPackageState::Locked), &training),
        FormState::Ready,
        "a fresh run may choose a different batch size"
    );
}

#[test]
fn resume_below_the_checkpoint_epoch_blocks() {
    let dir = project();
    checkpoint(dir.path(), 40000, &state_body(1, 200, 40000));
    let training = TrainingSurvey::inspect(dir.path());
    let mut form = TrainingForm::default();
    form.set_resume(true);

    form.set_epochs(200.0);
    assert_eq!(
        form_state(&form, &package(AssetPackageState::Locked), &training),
        FormState::Blocked("training.blocked.epoch_reached")
    );

    form.set_epochs(201.0);
    assert_eq!(
        form_state(&form, &package(AssetPackageState::Locked), &training),
        FormState::Ready
    );
}

#[test]
fn an_unreadable_state_does_not_block_resume() {
    let dir = project();
    checkpoint(dir.path(), 40000, "{");
    let training = TrainingSurvey::inspect(dir.path());
    let mut form = TrainingForm::default();
    form.set_resume(true);

    let state = form_state(&form, &package(AssetPackageState::Locked), &training);

    assert!(training.state_error.is_some());
    assert_eq!(state, FormState::Ready);
    assert!(state.is_submittable());
}

#[test]
fn a_fresh_run_over_a_checkpoint_warns() {
    let dir = project();
    checkpoint(dir.path(), 376, &state_body(1, 2, 376));
    let training = TrainingSurvey::inspect(dir.path());
    let mut form = TrainingForm::default();

    assert!(fresh_over_checkpoint(&form, &training));

    form.set_resume(true);
    assert!(!fresh_over_checkpoint(&form, &training));

    form.set_resume(false);
    assert!(!fresh_over_checkpoint(&form, &TrainingSurvey::default()));
}
