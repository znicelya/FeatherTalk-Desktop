use feathertalk_training::{
    PREVIEW_ARTIFACT_SCHEMA_VERSION, PREVIEW_TENSOR_SHAPE, PreviewArtifact,
    TRAINING_METRICS_SCHEMA_VERSION, TrainingError, TrainingMetrics, TrainingMode,
};

fn metrics(mode: TrainingMode) -> TrainingMetrics {
    TrainingMetrics {
        schema_version: TRAINING_METRICS_SCHEMA_VERSION,
        mode,
        epoch: 2,
        global_step: 17,
        total_loss: 1.5,
        full_loss: 1.0,
        perceptual_loss: 0.5,
        mouth_loss: (mode != TrainingMode::Baseline).then_some(0.25),
        temporal_loss: (mode == TrainingMode::MouthRoiTemporal).then_some(0.1),
        temporal_mouth_loss: (mode == TrainingMode::MouthRoiTemporal).then_some(0.2),
        samples_seen: 34,
        samples_per_second: 12.5,
        estimated_remaining_seconds: 8.0,
        gpu_memory_bytes: Some(4_000_000),
        worker_state: "training".to_owned(),
    }
}

#[test]
fn metrics_json_is_strict_and_round_trips() {
    let value = metrics(TrainingMode::MouthRoiTemporal);
    value.validate().unwrap();
    let json = serde_json::to_string(&value).unwrap();
    assert!(json.contains("\"schema_version\":1"));
    assert_eq!(
        serde_json::from_str::<TrainingMetrics>(&json).unwrap(),
        value
    );

    let mut unknown = serde_json::to_value(value).unwrap();
    unknown["unexpected"] = true.into();
    assert!(serde_json::from_value::<TrainingMetrics>(unknown).is_err());
}

#[test]
fn metrics_mode_components_and_numbers_are_validated() {
    assert!(metrics(TrainingMode::Baseline).validate().is_ok());
    assert!(metrics(TrainingMode::MouthRoi).validate().is_ok());
    assert!(metrics(TrainingMode::MouthRoiTemporal).validate().is_ok());

    let mut wrong = metrics(TrainingMode::Baseline);
    wrong.mouth_loss = Some(1.0);
    assert!(matches!(
        wrong.validate(),
        Err(TrainingError::InvalidCheckpoint(_))
    ));

    let mut non_finite = metrics(TrainingMode::Baseline);
    non_finite.total_loss = f64::NAN;
    assert!(matches!(
        non_finite.validate(),
        Err(TrainingError::InvalidCheckpoint(_))
    ));
}

#[test]
fn preview_value_requires_three_fixed_arrays_and_strict_metadata() {
    let values = vec![0.25_f32; PREVIEW_TENSOR_SHAPE.iter().product::<u32>() as usize];
    let preview = PreviewArtifact::new(
        4,
        9,
        2,
        17,
        "original-unet",
        "a".repeat(64),
        "training",
        values.clone(),
        values.clone(),
        values,
    )
    .unwrap();
    preview.validate().unwrap();
    assert_eq!(preview.shape(), PREVIEW_TENSOR_SHAPE);
    assert_eq!(preview.prediction().len(), 76_800);

    let invalid = PreviewArtifact::new(
        4,
        9,
        2,
        17,
        "original-unet",
        "a".repeat(64),
        "training",
        vec![0.25; 10],
        vec![0.25; 76_800],
        vec![0.25; 76_800],
    );
    assert!(matches!(invalid, Err(TrainingError::InvalidCheckpoint(_))));
}

#[test]
fn schema_versions_are_fixed() {
    assert_eq!(TRAINING_METRICS_SCHEMA_VERSION, 1);
    assert_eq!(PREVIEW_ARTIFACT_SCHEMA_VERSION, 1);
}
