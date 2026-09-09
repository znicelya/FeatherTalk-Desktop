use std::path::PathBuf;

use feathertalk_domain::{TrainingMode as DomainTrainingMode, UnetVariant};
use feathertalk_export::ModelConfiguration;
use feathertalk_models::unet::{MobileOneUnetConfig, OriginalUnetConfig};
use feathertalk_training::CheckpointDescriptor;
use feathertalk_worker::{TRAIN_BACKEND_NAME, TrainSummary, checkpoint_descriptor, train_to_json};
use serde_json::{Value, json};

fn descriptor() -> CheckpointDescriptor {
    let configuration = ModelConfiguration::original_unet(&OriginalUnetConfig::parity_micro());
    checkpoint_descriptor(&configuration).expect("the configuration serialises")
}

/// A minimal summary, for the tests that only care about one field.
///
/// The descriptor stays `original_unet` even when the variant does not: what is
/// under test here is the enum mapping, not the model.
fn payload_of(mode: DomainTrainingMode, variant: UnetVariant) -> Value {
    let descriptor = descriptor();
    train_to_json(&TrainSummary {
        mode,
        variant,
        descriptor: &descriptor,
        frame_count: 4,
        epochs_requested: 1,
        epochs_completed: 1,
        global_step: 4,
        samples_seen: 4,
        total_loss: Some(1.0),
        resumed_from: None,
        checkpoint_dir: None,
        checkpoints_written: 1,
        metrics_written: 1,
        previews_written: 1,
    })
}

#[test]
fn a_finished_run_reports_every_field_the_design_lists() {
    let descriptor = descriptor();
    let checkpoint = PathBuf::from("C:/tmp/project/models/unet/checkpoint-00000376");

    let payload = train_to_json(&TrainSummary {
        mode: DomainTrainingMode::MouthRoi,
        variant: UnetVariant::OriginalUnet,
        descriptor: &descriptor,
        frame_count: 188,
        epochs_requested: 2,
        epochs_completed: 2,
        global_step: 376,
        samples_seen: 376,
        total_loss: Some(0.0412),
        resumed_from: None,
        checkpoint_dir: Some(&checkpoint),
        checkpoints_written: 2,
        metrics_written: 2,
        previews_written: 2,
    });

    assert_eq!(
        payload,
        json!({
            "mode": "mouth_roi",
            "variant": "original_unet",
            "backend": TRAIN_BACKEND_NAME,
            "model_kind": "original_unet",
            "architecture_version": descriptor.architecture_version,
            "model_config_sha256": descriptor.model_config_sha256,
            "frame_count": 188,
            "epochs_requested": 2,
            "epochs_completed": 2,
            "global_step": 376,
            "samples_seen": 376,
            "total_loss": 0.0412,
            "resumed_from": null,
            "checkpoint_dir": "C:/tmp/project/models/unet/checkpoint-00000376",
            "checkpoints_written": 2,
            "metrics_written": 2,
            "previews_written": 2,
        })
    );
}

#[test]
fn a_resume_with_nothing_left_to_do_invents_nothing() {
    let descriptor = descriptor();
    let resumed = PathBuf::from("C:/tmp/project/models/unet/checkpoint-00000748");

    let payload = train_to_json(&TrainSummary {
        mode: DomainTrainingMode::Temporal,
        variant: UnetVariant::MobileOneUnet,
        descriptor: &descriptor,
        frame_count: 188,
        epochs_requested: 4,
        epochs_completed: 4,
        global_step: 748,
        samples_seen: 0,
        total_loss: None,
        resumed_from: Some(&resumed),
        checkpoint_dir: None,
        checkpoints_written: 0,
        metrics_written: 0,
        previews_written: 0,
    });

    // The checkpoint had already finished all four epochs, so the loop never
    // ran: no loss was observed and no checkpoint was published. Both stay null
    // rather than being filled with a plausible zero.
    assert_eq!(payload["total_loss"], json!(null));
    assert_eq!(payload["checkpoint_dir"], json!(null));
    assert_eq!(payload["samples_seen"], json!(0));
    // The step the run was already at is still reported.
    assert_eq!(payload["global_step"], json!(748));
    assert_eq!(
        payload["resumed_from"],
        json!(resumed.display().to_string())
    );
}

#[test]
fn the_reported_mode_and_variant_use_the_command_line_slugs() {
    for (mode, slug) in [
        (DomainTrainingMode::Baseline, "baseline"),
        (DomainTrainingMode::MouthRoi, "mouth_roi"),
        (DomainTrainingMode::Temporal, "temporal"),
    ] {
        assert_eq!(
            payload_of(mode, UnetVariant::OriginalUnet)["mode"],
            json!(slug)
        );
        // For the mode, the request's own spelling is the same string.
        assert_eq!(serde_json::to_value(mode).unwrap(), json!(slug));
    }

    // For the variant it is not: the protocol splits `MobileOneUnet` into
    // `mobile_one_unet`, while the checkpoint manifest, the ONNX export and the
    // `--variant mobileone-unet` flag all say `mobileone`. The payload follows
    // the model, so it cannot contradict the `model_kind` beside it.
    for (variant, slug) in [
        (UnetVariant::OriginalUnet, "original_unet"),
        (UnetVariant::MobileOneUnet, "mobileone_unet"),
    ] {
        let payload = payload_of(DomainTrainingMode::Baseline, variant);
        assert_eq!(payload["variant"], json!(slug));
    }

    let mobileone = MobileOneUnetConfig::parity_micro();
    let configuration = ModelConfiguration::mobileone_unet(&mobileone, false);
    assert_eq!(configuration.model_type(), "mobileone_unet");
}
