use feathertalk_parity::{
    archive::GoldenArchive,
    fixture::{
        ForwardCase, run_cpu_forward, run_cpu_train_step, validate_forward_fixture,
        validate_train_step_fixture,
    },
    metrics::{ParityError, compare_f32},
};
use ndarray::{ArrayD, IxDyn, array};
use serde_json::json;
use std::fs::File;
use zip::ZipWriter;

fn golden_archive() -> GoldenArchive {
    let root = env!("CARGO_MANIFEST_DIR");
    GoldenArchive::open(format!("{root}/../../tests/golden/burn-feasibility-v1.zip")).unwrap()
}

#[test]
fn cpu_forward_rejects_a_missing_archive_sidecar_before_fixture_load() {
    let temp = tempfile::tempdir().unwrap();
    let archive_path = temp.path().join("missing-sidecar.zip");
    ZipWriter::new(File::create(&archive_path).unwrap())
        .finish()
        .unwrap();
    let archive = GoldenArchive::open(archive_path).unwrap();

    assert!(matches!(
        run_cpu_forward(&archive, ForwardCase::FeatherMicro),
        Err(ParityError::ArchiveVerification(_))
    ));
}

#[test]
fn feather_contract_rejects_a_kind_mismatch_without_forward() {
    let archive = golden_archive();
    let mut fixture = archive.load_fixture("feather_micro_eval").unwrap();
    fixture.kind = "original_unet".to_owned();

    assert!(matches!(
        validate_forward_fixture(&fixture, ForwardCase::FeatherMicro),
        Err(ParityError::FixtureContract { field: "kind", .. })
    ));
}

#[test]
fn feather_contract_rejects_a_weight_member_mismatch_without_forward() {
    let archive = golden_archive();
    let mut fixture = archive.load_fixture("feather_micro_eval").unwrap();
    fixture.weights_entry = "weights/unet_production.pth".to_owned();

    assert!(matches!(
        validate_forward_fixture(&fixture, ForwardCase::FeatherMicro),
        Err(ParityError::FixtureContract {
            field: "weights_entry",
            ..
        })
    ));
}

#[test]
fn feather_contract_rejects_a_fixture_id_mismatch_without_forward() {
    let archive = golden_archive();
    let mut fixture = archive.load_fixture("feather_micro_eval").unwrap();
    fixture.id = "renamed_feather_eval".to_owned();

    assert!(matches!(
        validate_forward_fixture(&fixture, ForwardCase::FeatherMicro),
        Err(ParityError::FixtureContract {
            field: "fixture_id",
            ..
        })
    ));
}

#[test]
fn feather_contract_rejects_shape_preserving_config_changes_without_forward() {
    let archive = golden_archive();
    let mut fixture = archive.load_fixture("feather_micro_eval").unwrap();
    fixture.config.insert("dropout".to_owned(), json!(0.5));

    assert!(matches!(
        validate_forward_fixture(&fixture, ForwardCase::FeatherMicro),
        Err(ParityError::FixtureContract {
            field: "config",
            ..
        })
    ));
}

#[test]
fn unet_contract_rejects_shape_preserving_config_changes_without_forward() {
    let archive = golden_archive();
    let mut fixture = archive.load_fixture("unet_production_eval").unwrap();
    fixture.config.insert("mode".to_owned(), json!("other"));

    assert!(matches!(
        validate_forward_fixture(&fixture, ForwardCase::UnetProduction),
        Err(ParityError::FixtureContract {
            field: "config",
            ..
        })
    ));
}

#[test]
fn feather_contract_rejects_wrong_waveform_rank_without_forward() {
    let archive = golden_archive();
    let mut fixture = archive.load_fixture("feather_micro_eval").unwrap();
    fixture
        .inputs
        .insert("waveform".to_owned(), ArrayD::zeros(IxDyn(&[1360])));

    assert!(matches!(
        validate_forward_fixture(&fixture, ForwardCase::FeatherMicro),
        Err(ParityError::FixtureArrayShape {
            name: "waveform",
            ..
        })
    ));
}

#[test]
fn unet_contract_rejects_wrong_image_shape_without_forward() {
    let archive = golden_archive();
    let mut fixture = archive.load_fixture("unet_production_eval").unwrap();
    fixture
        .inputs
        .insert("image".to_owned(), ArrayD::zeros(IxDyn(&[1, 6, 160, 159])));

    assert!(matches!(
        validate_forward_fixture(&fixture, ForwardCase::UnetProduction),
        Err(ParityError::FixtureArrayShape { name: "image", .. })
    ));
}

#[test]
fn unet_contract_rejects_wrong_audio_rank_without_forward() {
    let archive = golden_archive();
    let mut fixture = archive.load_fixture("unet_production_eval").unwrap();
    fixture
        .inputs
        .insert("audio".to_owned(), ArrayD::zeros(IxDyn(&[16, 32, 32])));

    assert!(matches!(
        validate_forward_fixture(&fixture, ForwardCase::UnetProduction),
        Err(ParityError::FixtureArrayShape { name: "audio", .. })
    ));
}

#[test]
fn unet_contract_rejects_wrong_output_shape_without_forward() {
    let archive = golden_archive();
    let mut fixture = archive.load_fixture("unet_production_eval").unwrap();
    fixture
        .expected
        .insert("output".to_owned(), ArrayD::zeros(IxDyn(&[1, 3, 160, 159])));

    assert!(matches!(
        validate_forward_fixture(&fixture, ForwardCase::UnetProduction),
        Err(ParityError::FixtureArrayShape { name: "output", .. })
    ));
}

#[test]
fn forward_contract_rejects_extra_array_names_without_forward() {
    let archive = golden_archive();
    let mut fixture = archive.load_fixture("feather_micro_eval").unwrap();
    fixture
        .inputs
        .insert("unused".to_owned(), ArrayD::zeros(IxDyn(&[1])));

    assert!(matches!(
        validate_forward_fixture(&fixture, ForwardCase::FeatherMicro),
        Err(ParityError::FixtureArraySet { role: "input", .. })
    ));
}

#[test]
fn forward_contract_rejects_unknown_config_metadata_without_forward() {
    let archive = golden_archive();
    let mut fixture = archive.load_fixture("feather_micro_eval").unwrap();
    fixture.config.insert("future_flag".to_owned(), json!(true));

    assert!(matches!(
        validate_forward_fixture(&fixture, ForwardCase::FeatherMicro),
        Err(ParityError::FixtureContract {
            field: "config",
            ..
        })
    ));
}

#[test]
fn golden_forward_contracts_validate_without_forward() {
    let archive = golden_archive();
    let feather = archive.load_fixture("feather_micro_eval").unwrap();
    let unet = archive.load_fixture("unet_production_eval").unwrap();

    validate_forward_fixture(&feather, ForwardCase::FeatherMicro).unwrap();
    validate_forward_fixture(&unet, ForwardCase::UnetProduction).unwrap();
}

#[test]
fn golden_train_step_contract_validates_without_forward() {
    let fixture = golden_archive()
        .load_fixture("unet_micro_train_step")
        .unwrap();
    validate_train_step_fixture(&fixture).unwrap();
}

#[test]
fn golden_train_fixture_records_l1_cusp_adjustments() {
    let fixture = golden_archive()
        .load_fixture("unet_micro_train_step")
        .unwrap();
    let adjusted = fixture
        .metrics
        .get("l1_cusp_adjusted_elements")
        .copied()
        .expect("generator must record L1 cusp adjustments");
    assert!(adjusted > 0.0, "target should avoid at least one L1 cusp");
}

#[test]
fn train_step_contract_rejects_missing_l1_cusp_metrics() {
    let archive = golden_archive();
    let mut fixture = archive.load_fixture("unet_micro_train_step").unwrap();
    fixture.metrics.remove("initial_l1_residual_min_abs");

    assert!(matches!(
        validate_train_step_fixture(&fixture),
        Err(ParityError::FixtureContract {
            field: "training_metrics",
            ..
        })
    ));
}

#[test]
fn train_step_contract_rejects_l1_cusp_metric_boundaries() {
    let archive = golden_archive();

    let mut fixture = archive.load_fixture("unet_micro_train_step").unwrap();
    fixture.metrics.insert(
        "initial_l1_residual_min_abs".to_owned(),
        1e-3 - f64::EPSILON,
    );
    assert!(matches!(
        validate_train_step_fixture(&fixture),
        Err(ParityError::FixtureContract {
            field: "training_metrics",
            ..
        })
    ));

    let mut fixture = archive.load_fixture("unet_micro_train_step").unwrap();
    fixture
        .metrics
        .insert("l1_cusp_adjusted_elements".to_owned(), 1.5);
    assert!(matches!(
        validate_train_step_fixture(&fixture),
        Err(ParityError::FixtureContract {
            field: "training_metrics",
            ..
        })
    ));
}

#[test]
fn train_step_contract_rejects_input_provenance_changes() {
    let archive = golden_archive();
    let mut fixture = archive.load_fixture("unet_micro_train_step").unwrap();
    let generator = fixture
        .generator
        .as_mut()
        .expect("training fixture generator metadata");
    generator.insert("train_input_seed".to_owned(), serde_json::json!(4));

    assert!(matches!(
        validate_train_step_fixture(&fixture),
        Err(ParityError::FixtureContract {
            field: "generator",
            ..
        })
    ));
}

#[test]
fn train_step_contract_rejects_identity_kind_and_config_changes_without_forward() {
    let archive = golden_archive();

    let mut fixture = archive.load_fixture("unet_micro_train_step").unwrap();
    fixture.id = "renamed_train_step".to_owned();
    assert!(matches!(
        validate_train_step_fixture(&fixture),
        Err(ParityError::FixtureContract {
            field: "fixture_id",
            ..
        })
    ));

    let mut fixture = archive.load_fixture("unet_micro_train_step").unwrap();
    fixture.kind = "original_unet".to_owned();
    assert!(matches!(
        validate_train_step_fixture(&fixture),
        Err(ParityError::FixtureContract { field: "kind", .. })
    ));

    let mut fixture = archive.load_fixture("unet_micro_train_step").unwrap();
    fixture
        .config
        .insert("channels".to_owned(), json!([2, 4, 8, 16, 31]));
    assert!(matches!(
        validate_train_step_fixture(&fixture),
        Err(ParityError::FixtureContract {
            field: "config",
            ..
        })
    ));
}

#[test]
fn train_step_contract_rejects_optimizer_loss_and_mode_changes_without_forward() {
    let archive = golden_archive();

    for field in [
        "type",
        "learning_rate",
        "beta1",
        "beta2",
        "epsilon",
        "weight_decay",
    ] {
        let mut fixture = archive.load_fixture("unet_micro_train_step").unwrap();
        fixture
            .optimizer
            .as_mut()
            .unwrap()
            .insert(field.to_owned(), json!("wrong"));
        assert!(matches!(
            validate_train_step_fixture(&fixture),
            Err(ParityError::FixtureContract {
                field: "optimizer",
                ..
            })
        ));
    }

    let mut fixture = archive.load_fixture("unet_micro_train_step").unwrap();
    fixture.loss = Some("mean_squared_error".to_owned());
    assert!(matches!(
        validate_train_step_fixture(&fixture),
        Err(ParityError::FixtureContract { field: "loss", .. })
    ));

    let mut fixture = archive.load_fixture("unet_micro_train_step").unwrap();
    fixture.expected_mode = Some("train".to_owned());
    assert!(matches!(
        validate_train_step_fixture(&fixture),
        Err(ParityError::FixtureContract {
            field: "expected_mode",
            ..
        })
    ));
}

#[test]
fn train_step_contract_rejects_input_parameter_and_batch_norm_changes_without_forward() {
    let archive = golden_archive();

    let mut fixture = archive.load_fixture("unet_micro_train_step").unwrap();
    fixture
        .inputs
        .insert("target".to_owned(), ArrayD::zeros(IxDyn(&[1, 3, 160, 159])));
    assert!(matches!(
        validate_train_step_fixture(&fixture),
        Err(ParityError::FixtureArrayShape { name: "target", .. })
    ));

    let mut fixture = archive.load_fixture("unet_micro_train_step").unwrap();
    fixture.expected.remove("inc.inconv.conv.0.weight");
    assert!(matches!(
        validate_train_step_fixture(&fixture),
        Err(ParityError::FixtureArraySet {
            role: "selected_parameter",
            ..
        })
    ));

    let mut fixture = archive.load_fixture("unet_micro_train_step").unwrap();
    fixture.expected.insert(
        "inc.inconv.conv.0.weight".to_owned(),
        ArrayD::zeros(IxDyn(&[12, 6, 1, 2])),
    );
    assert!(matches!(
        validate_train_step_fixture(&fixture),
        Err(ParityError::FixtureArrayShape {
            role: "selected_parameter",
            name: "inc.inconv.conv.0.weight",
            ..
        })
    ));

    let mut fixture = archive.load_fixture("unet_micro_train_step").unwrap();
    fixture
        .expected
        .remove("audio_model.conv1.conv.1.running_mean");
    assert!(matches!(
        validate_train_step_fixture(&fixture),
        Err(ParityError::FixtureArraySet {
            role: "batch_norm_state",
            ..
        })
    ));

    let mut fixture = archive.load_fixture("unet_micro_train_step").unwrap();
    fixture.expected.insert(
        "audio_model.conv1.conv.1.running_var".to_owned(),
        ArrayD::zeros(IxDyn(&[31])),
    );
    assert!(matches!(
        validate_train_step_fixture(&fixture),
        Err(ParityError::FixtureArrayShape {
            role: "batch_norm_state",
            name: "audio_model.conv1.conv.1.running_var",
            ..
        })
    ));
}

#[test]
fn exact_arrays_have_zero_error() {
    let values = array![[1.0_f32, -2.0], [3.0, 4.0]].into_dyn();
    let metrics = compare_f32(values.view(), values.view()).unwrap();
    assert_eq!(metrics.max_abs, 0.0);
    assert_eq!(metrics.mean_abs, 0.0);
    assert_eq!(metrics.max_relative, 0.0);
}

#[test]
fn metrics_report_max_mean_and_relative_error() {
    let actual = array![1.0_f32, 3.0].into_dyn();
    let expected = array![1.0_f32, 1.0].into_dyn();
    let metrics = compare_f32(actual.view(), expected.view()).unwrap();
    assert_eq!(metrics.max_abs, 2.0);
    assert_eq!(metrics.mean_abs, 1.0);
    assert_eq!(metrics.max_relative, 2.0);
}

#[test]
fn shape_mismatch_is_an_error() {
    let actual = array![1.0_f32, 2.0].into_dyn();
    let expected = array![[1.0_f32, 2.0]].into_dyn();
    assert!(matches!(
        compare_f32(actual.view(), expected.view()),
        Err(ParityError::ShapeMismatch { .. })
    ));
}

#[test]
fn empty_arrays_are_an_error() {
    let actual = ndarray::Array1::<f32>::zeros(0).into_dyn();
    let expected = ndarray::Array1::<f32>::zeros(0).into_dyn();
    assert!(matches!(
        compare_f32(actual.view(), expected.view()),
        Err(ParityError::EmptyArray)
    ));
}

#[test]
fn non_finite_values_are_an_error() {
    let actual = array![f32::NAN].into_dyn();
    let expected = array![0.0_f32].into_dyn();
    assert!(matches!(
        compare_f32(actual.view(), expected.view()),
        Err(ParityError::NonFinite { .. })
    ));
}

#[test]
fn expected_non_finite_values_are_an_error() {
    for expected_value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let actual = array![0.0_f32].into_dyn();
        let expected = array![expected_value].into_dyn();
        assert!(matches!(
            compare_f32(actual.view(), expected.view()),
            Err(ParityError::NonFinite { .. })
        ));
    }
}

#[test]
fn mean_accumulation_stays_finite_for_finite_f32_metrics() {
    let actual = array![f32::MAX, f32::MAX].into_dyn();
    let expected = array![1.0_f32, 1.0].into_dyn();
    let metrics = compare_f32(actual.view(), expected.view()).unwrap();
    assert!(metrics.mean_abs.is_finite(), "{metrics:?}");
    assert_eq!(metrics.mean_abs, f32::MAX);
}

#[test]
fn finite_subtraction_overflow_is_an_error() {
    let actual = array![f32::MAX].into_dyn();
    let expected = array![-f32::MAX].into_dyn();
    assert!(matches!(
        compare_f32(actual.view(), expected.view()),
        Err(ParityError::MetricOverflow {
            metric: "max_abs",
            ..
        })
    ));
}

#[test]
fn finite_relative_error_overflow_is_an_error() {
    let actual = array![f32::MAX].into_dyn();
    let expected = array![0.0_f32].into_dyn();
    assert!(matches!(
        compare_f32(actual.view(), expected.view()),
        Err(ParityError::MetricOverflow {
            metric: "max_relative",
            ..
        })
    ));
}

#[test]
fn feather_micro_matches_python_on_cpu() {
    let metrics = run_cpu_forward(&golden_archive(), ForwardCase::FeatherMicro).unwrap();
    println!("feather_micro {metrics:?}");
    assert!(metrics.max_abs <= 1e-4, "{metrics:?}");
}

#[test]
fn unet_production_matches_python_on_cpu() {
    let metrics = run_cpu_forward(&golden_archive(), ForwardCase::UnetProduction).unwrap();
    println!("unet_production {metrics:?}");
    assert!(metrics.max_abs <= 1e-4, "{metrics:?}");
}

#[test]
fn unet_micro_train_step_matches_python_on_cpu() {
    let result = run_cpu_train_step(&golden_archive()).unwrap();
    println!("unet_micro_train_step {result:?}");
    assert!(result.initial_loss_relative <= 1e-3, "{result:?}");
    assert!(result.post_step_loss_relative <= 1e-3, "{result:?}");
    for (name, relative_error) in &result.selected_parameter_relative {
        assert!(*relative_error <= 1e-3, "{name}: {relative_error}");
    }
    for (name, relative_error) in &result.batch_norm_state_relative {
        assert!(*relative_error <= 1e-3, "{name}: {relative_error}");
    }
}
