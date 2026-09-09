use std::path::{Path, PathBuf};

use feathertalk_domain::{TrainParams, TrainingMode as DomainTrainingMode, UnetVariant};
use feathertalk_export::ModelConfiguration;
use feathertalk_models::unet::{MobileOneUnetConfig, OriginalUnetConfig};
use feathertalk_training::{
    PREVIEW_TENSOR_ELEMENTS, PreviewArtifact, TrainingError, TrainingMetrics, TrainingMode,
    TrainingSample, read_preview_artifact, read_training_metrics,
};
use feathertalk_worker::{
    DEFAULT_BATCH_SIZE, DEFAULT_LEARNING_RATE, MAX_EPOCHS, TRAIN_BACKEND_NAME, TRAINING_SEED,
    TrainingPaths, WORKER_STATE, checkpoint_descriptor, latest_checkpoint, preview_sample,
    publish_checkpoint, sample_count, training_config, training_mode, write_metrics_unless_present,
    write_preview_unless_present,
};

fn params(mode: DomainTrainingMode, epochs: u32) -> TrainParams {
    TrainParams {
        project_dir: PathBuf::from("C:/tmp/project"),
        mode,
        variant: UnetVariant::OriginalUnet,
        epochs,
        batch_size: 1,
        resume: false,
    }
}

#[test]
fn the_three_request_modes_map_onto_the_training_crate() {
    assert_eq!(
        training_mode(DomainTrainingMode::Baseline),
        TrainingMode::Baseline
    );
    assert_eq!(
        training_mode(DomainTrainingMode::MouthRoi),
        TrainingMode::MouthRoi
    );
    // The request has three modes, the training crate three too, but the third
    // pair does not share a name.
    assert_eq!(
        training_mode(DomainTrainingMode::Temporal),
        TrainingMode::MouthRoiTemporal
    );
}

#[test]
fn only_the_temporal_mode_takes_a_stride_and_loses_a_sample() {
    for mode in [DomainTrainingMode::Baseline, DomainTrainingMode::MouthRoi] {
        let config = training_config(&params(mode, 2));
        assert_eq!(config.temporal_stride, 0);
        assert_eq!(sample_count(mode, 188), 188);
    }

    let config = training_config(&params(DomainTrainingMode::Temporal, 2));
    assert_eq!(config.temporal_stride, 1);
    assert_eq!(sample_count(DomainTrainingMode::Temporal, 188), 187);
    // A one-frame project starts no temporal sample at all.
    assert_eq!(sample_count(DomainTrainingMode::Temporal, 1), 0);
    assert_eq!(sample_count(DomainTrainingMode::Temporal, 0), 0);
}

#[test]
fn the_config_combines_request_values_with_worker_defaults() {
    let config = training_config(&params(DomainTrainingMode::MouthRoi, 7));

    assert_eq!(config.total_epochs, 7);
    assert_eq!(config.batch_size, DEFAULT_BATCH_SIZE);
    assert_eq!(config.learning_rate, DEFAULT_LEARNING_RATE);
    assert_eq!(config.mouth_weight, 4.0);
    assert_eq!(config.temporal_weight, 0.5);
    assert_eq!(config.temporal_mouth_weight, 4.0);
    assert_eq!(config.perceptual_weight, 0.01);
    config
        .validate()
        .expect("the assembled config is valid on its own");

    assert_eq!(DEFAULT_BATCH_SIZE, 1);
    assert_eq!(TRAINING_SEED, 1);
    assert_eq!(MAX_EPOCHS, 10_000);
    assert_eq!(WORKER_STATE, "training");
    assert_eq!(TRAIN_BACKEND_NAME, "ndarray-cpu");
}

#[test]
fn the_requested_batch_size_reaches_every_training_mode() {
    for mode in [
        DomainTrainingMode::Baseline,
        DomainTrainingMode::MouthRoi,
        DomainTrainingMode::Temporal,
    ] {
        let mut request = params(mode, 2);
        request.batch_size = 4;
        let config = training_config(&request);
        assert_eq!(config.batch_size, 4, "{mode:?}");
        config.validate().unwrap();
    }
}

#[test]
fn the_descriptor_digests_the_model_configuration() {
    let configuration = ModelConfiguration::original_unet(&OriginalUnetConfig::production());
    let descriptor = checkpoint_descriptor(&configuration).expect("the digest is computable");

    descriptor.validate().expect("64 lowercase hex characters");
    assert_eq!(descriptor.model_kind, configuration.model_type());
    assert_eq!(
        descriptor.architecture_version,
        configuration.architecture_version()
    );
    assert_eq!(descriptor.model_config_sha256.len(), 64);
    assert!(
        descriptor
            .model_config_sha256
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)),
        "{}",
        descriptor.model_config_sha256
    );
    // Same configuration, same digest: this value is what a later resume has to
    // match, so it must not depend on anything but the configuration.
    let again = checkpoint_descriptor(&configuration).unwrap();
    assert_eq!(again, descriptor);

    // The training graph is the multi-branch one; a reparameterized descriptor
    // would claim a structure training never builds.
    let mobile = ModelConfiguration::mobileone_unet(&MobileOneUnetConfig::production(), false);
    let mobile_descriptor = checkpoint_descriptor(&mobile).unwrap();
    assert_eq!(mobile_descriptor.model_kind, "mobileone_unet");
    assert_ne!(
        mobile_descriptor.model_config_sha256,
        descriptor.model_config_sha256
    );

    let reparameterized =
        ModelConfiguration::mobileone_unet(&MobileOneUnetConfig::production(), true);
    assert_ne!(
        checkpoint_descriptor(&reparameterized)
            .unwrap()
            .model_config_sha256,
        mobile_descriptor.model_config_sha256
    );

    // A micro configuration digests differently again, which is what makes the
    // offline tests' descriptors distinct from a production run's.
    assert_ne!(
        checkpoint_descriptor(&ModelConfiguration::original_unet(
            &OriginalUnetConfig::parity_micro()
        ))
        .unwrap()
        .model_config_sha256,
        descriptor.model_config_sha256
    );
}

#[test]
fn the_artifact_paths_are_step_numbered() {
    let paths = TrainingPaths::new(Path::new("C:/tmp/project"));

    assert_eq!(
        paths.checkpoints(),
        Path::new("C:/tmp/project").join("models").join("unet")
    );
    assert!(
        paths
            .checkpoint(188)
            .ends_with("models/unet/checkpoint-00000188")
    );
    assert!(
        paths
            .metrics(188)
            .ends_with("outputs/metrics/step-00000188.json")
    );
    assert!(
        paths
            .preview(188)
            .ends_with("outputs/preview/step-00000188")
    );
    // Eight digits, so a nine-digit run does not silently truncate.
    assert!(
        paths
            .checkpoint(123_456_789)
            .ends_with("checkpoint-123456789")
    );
}

#[test]
fn only_step_numbered_checkpoint_names_are_recognised() {
    assert_eq!(
        TrainingPaths::checkpoint_step("checkpoint-00000188"),
        Some(188)
    );
    assert_eq!(
        TrainingPaths::checkpoint_step("checkpoint-00000000"),
        Some(0)
    );
    // `{:08}` pads but never truncates, so a run past eight digits must still
    // find its own checkpoints.
    assert_eq!(
        TrainingPaths::checkpoint_step("checkpoint-123456789"),
        Some(123_456_789)
    );
    for name in [
        "checkpoint-188",
        "checkpoint-0000018x",
        "checkpoint-",
        "checkpoint",
        ".publish-1234-0",
        ".retired-1234-0",
        "last",
    ] {
        assert_eq!(TrainingPaths::checkpoint_step(name), None, "{name}");
    }
}

/// Behaves like `save_training_checkpoint`: it creates the destination itself
/// and refuses one that already exists.
fn fake_save(marker: &'static str) -> impl FnOnce(&Path) -> Result<(), TrainingError> {
    move |staged: &Path| {
        if staged.exists() {
            return Err(TrainingError::CheckpointDirectory(format!(
                "checkpoint destination already exists: {}",
                staged.display()
            )));
        }
        std::fs::create_dir_all(staged)?;
        std::fs::write(staged.join("manifest.json"), marker)?;
        Ok(())
    }
}

fn names(directory: &Path) -> Vec<String> {
    let mut names: Vec<String> = std::fs::read_dir(directory)
        .expect("the directory is readable")
        .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    names.sort();
    names
}

fn marker_of(checkpoint: &Path) -> String {
    std::fs::read_to_string(checkpoint.join("manifest.json")).expect("the marker is readable")
}

#[test]
fn a_published_checkpoint_lands_under_its_step_name() {
    let root = tempfile::tempdir().unwrap();
    let paths = TrainingPaths::new(root.path());

    let published =
        publish_checkpoint(&paths, 4, fake_save("first")).expect("the publish succeeds");

    assert_eq!(published, paths.checkpoint(4));
    assert_eq!(marker_of(&published), "first");
    // Nothing but the checkpoint is left behind.
    assert_eq!(names(paths.checkpoints()), vec!["checkpoint-00000004"]);
}

#[test]
fn publishing_over_an_existing_step_retires_the_old_directory() {
    let root = tempfile::tempdir().unwrap();
    let paths = TrainingPaths::new(root.path());
    let occupied = paths.checkpoint(8);
    std::fs::create_dir_all(&occupied).unwrap();
    std::fs::write(occupied.join("manifest.json"), "stale").unwrap();
    std::fs::write(occupied.join("leftover.bin"), "junk").unwrap();

    let published =
        publish_checkpoint(&paths, 8, fake_save("fresh")).expect("the publish succeeds");

    assert_eq!(published, occupied);
    assert_eq!(marker_of(&published), "fresh");
    // The staged directory replaced the old one wholesale rather than merging
    // into it, so the stale file is gone.
    assert!(!published.join("leftover.bin").exists());
    assert_eq!(names(paths.checkpoints()), vec!["checkpoint-00000008"]);
}

#[test]
fn a_panicking_save_discards_staging_and_preserves_the_previous_checkpoint() {
    let root = tempfile::tempdir().unwrap();
    let paths = TrainingPaths::new(root.path());
    publish_checkpoint(&paths, 4, fake_save("previous")).unwrap();
    let panic = std::panic::catch_unwind(|| {
        let _ = publish_checkpoint(&paths, 4, |staged: &Path| {
            std::fs::create_dir_all(staged)?;
            std::fs::write(staged.join("partial.bin"), "half")?;
            panic!("GPU readback panicked");
        });
    });
    assert!(panic.is_err());
    assert_eq!(marker_of(&paths.checkpoint(4)), "previous");
    assert_eq!(names(paths.checkpoints()), vec!["checkpoint-00000004"]);
}

#[test]
fn a_failed_save_leaves_neither_a_destination_nor_a_staging_directory() {
    let root = tempfile::tempdir().unwrap();
    let paths = TrainingPaths::new(root.path());

    let error = publish_checkpoint(&paths, 4, |staged: &Path| {
        std::fs::create_dir_all(staged)?;
        std::fs::write(staged.join("partial.bin"), "half").unwrap();
        Err(TrainingError::Store("record write failed".to_owned()))
    })
    .expect_err("a failing save fails the publish");

    assert!(matches!(error, TrainingError::Store(_)), "{error:?}");
    assert!(!paths.checkpoint(4).exists());
    assert!(names(paths.checkpoints()).is_empty());
}

#[test]
fn the_latest_checkpoint_is_the_largest_step() {
    let root = tempfile::tempdir().unwrap();
    let paths = TrainingPaths::new(root.path());
    for step in [4_u64, 12, 8] {
        publish_checkpoint(&paths, step, fake_save("body")).unwrap();
    }
    // Decoys: a hand-made short name, a file wearing a checkpoint name, and a
    // staging directory a crashed run left behind.
    std::fs::create_dir_all(paths.checkpoints().join("checkpoint-16")).unwrap();
    std::fs::write(paths.checkpoints().join("checkpoint-00000020"), "not a dir").unwrap();
    std::fs::create_dir_all(paths.checkpoints().join(".publish-1-0")).unwrap();

    let latest = latest_checkpoint(&paths)
        .expect("the directory is readable")
        .expect("three checkpoints exist");

    assert_eq!(latest, paths.checkpoint(12));
}

#[test]
fn a_project_that_never_trained_has_no_latest_checkpoint() {
    let root = tempfile::tempdir().unwrap();
    let paths = TrainingPaths::new(root.path());

    assert_eq!(
        latest_checkpoint(&paths).expect("a missing directory is not an error"),
        None
    );

    // An empty directory is the same answer.
    std::fs::create_dir_all(paths.checkpoints()).unwrap();
    assert_eq!(latest_checkpoint(&paths).unwrap(), None);
}

const MODEL_KIND: &str = "original_unet";

fn metrics_fixture(global_step: u64, total_loss: f64) -> TrainingMetrics {
    TrainingMetrics::new(
        TrainingMode::Baseline,
        1,
        global_step,
        total_loss,
        total_loss,
        0.01,
        None,
        None,
        None,
        global_step,
        2.5,
        12.0,
        None,
        WORKER_STATE,
    )
    .expect("the fixture is valid")
}

fn preview_fixture(global_step: u64, prediction: f32) -> PreviewArtifact {
    PreviewArtifact::new(
        0,
        2,
        1,
        global_step,
        MODEL_KIND,
        "a".repeat(64),
        WORKER_STATE,
        vec![prediction; PREVIEW_TENSOR_ELEMENTS],
        vec![0.5; PREVIEW_TENSOR_ELEMENTS],
        vec![1.0; PREVIEW_TENSOR_ELEMENTS],
    )
    .expect("the fixture is valid")
}

#[test]
fn metrics_are_written_once_per_step() {
    let root = tempfile::tempdir().unwrap();
    let paths = TrainingPaths::new(root.path());

    assert!(write_metrics_unless_present(&paths, 4, &metrics_fixture(4, 0.5)).unwrap());
    // A resumed run reaches step 4 again with a different loss.
    assert!(!write_metrics_unless_present(&paths, 4, &metrics_fixture(4, 0.25)).unwrap());

    let written = read_training_metrics(paths.metrics(4)).expect("the file reads back");
    // The first write is the one kept.
    assert_eq!(written.total_loss, 0.5);
    assert_eq!(written.worker_state, WORKER_STATE);
}

#[test]
fn previews_are_written_once_per_step() {
    let root = tempfile::tempdir().unwrap();
    let paths = TrainingPaths::new(root.path());
    let sha = "a".repeat(64);

    assert!(write_preview_unless_present(&paths, 8, &preview_fixture(8, 0.25)).unwrap());
    assert!(!write_preview_unless_present(&paths, 8, &preview_fixture(8, 0.75)).unwrap());

    let (artifact, manifest) =
        read_preview_artifact(paths.preview(8), MODEL_KIND, &sha).expect("the artifact reads back");
    assert_eq!(manifest.global_step, 8);
    assert_eq!(artifact.prediction()[0], 0.25);
}

#[test]
fn the_preview_pairs_the_first_frame_with_the_middle_one() {
    assert_eq!(
        preview_sample(8),
        TrainingSample::SingleFrame {
            target_index: 0,
            reference_index: 4,
        }
    );
    // A one-frame project has to reference itself, which the dataset allows.
    assert_eq!(
        preview_sample(1),
        TrainingSample::SingleFrame {
            target_index: 0,
            reference_index: 0,
        }
    );
}
