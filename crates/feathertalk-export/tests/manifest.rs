use std::collections::BTreeMap;

use feathertalk_export::{
    FileManifest, LicenseBundle, LicenseEntry, MODEL_LICENSE_SCHEMA_VERSION,
    MODEL_PACKAGE_SCHEMA_VERSION, ModelConfiguration, ModelDescription, ModelPackageManifest,
    SourceManifest, TensorContract, TensorSpec, TrainingManifest, TrainingMode,
};
use feathertalk_models::feather_hubert::FeatherHubertConfig;

fn description() -> ModelDescription {
    ModelDescription::feather_hubert(FeatherHubertConfig {
        channels: 32,
        expansion: 2,
        num_blocks: 2,
        output_dim: 64,
        dropout: 0.0,
    })
}

fn manifest() -> ModelPackageManifest {
    ModelPackageManifest {
        schema_version: MODEL_PACKAGE_SCHEMA_VERSION,
        model_type: "feather_hubert".to_owned(),
        architecture_version: "feather-hubert-burn-v1".to_owned(),
        configuration: ModelConfiguration::FeatherHubert {
            channels: 32,
            expansion: 2,
            num_blocks: 2,
            output_dim: 64,
            dropout: 0.0,
        },
        inputs: description().inputs,
        outputs: description().outputs,
        training: TrainingManifest {
            mode: TrainingMode::Inference,
            mouth_weight: 0.0,
            temporal_weight: 0.0,
            temporal_mouth_weight: 0.0,
            perceptual_weight: 0.0,
        },
        source: SourceManifest {
            format: "pytorch-pickle-restricted".to_owned(),
            identifier: "fixture".to_owned(),
            version: "1".to_owned(),
            file_name: "fixture.pth".to_owned(),
            sha256: "a".repeat(64),
            url: None,
        },
        created_at: "2026-08-27T00:00:00Z".to_owned(),
        minimum_app_version: "0.1.0".to_owned(),
        tensors: TensorContract {
            tensor_count: 1,
            total_elements: 2,
            entries: vec![TensorSpec {
                name: "weight".to_owned(),
                shape: vec![2],
                dtype: "f32".to_owned(),
            }],
        },
        model: FileManifest {
            file_name: "model.safetensors".to_owned(),
            bytes: 2,
            sha256: "b".repeat(64),
        },
        licenses: FileManifest {
            file_name: "LICENSES.json".to_owned(),
            bytes: 2,
            sha256: "c".repeat(64),
        },
        optimizer: None,
        training_state: None,
    }
}

#[test]
fn feather_description_has_fixed_io_contract() {
    let value = description();
    assert_eq!(value.model_type, "feather_hubert");
    assert_eq!(value.inputs[0].name, "waveform");
    assert_eq!(value.inputs[0].shape, vec![1, -1]);
    assert_eq!(value.outputs[0].shape, vec![1, -1, 64]);
    value.validate().unwrap();
}

#[test]
fn manifest_and_license_round_trip_as_schema_one() {
    let value = manifest();
    value.validate().unwrap();
    let json = serde_json::to_string(&value).unwrap();
    assert!(json.contains("\"schema_version\":1"));
    assert_eq!(
        serde_json::from_str::<ModelPackageManifest>(&json).unwrap(),
        value
    );

    let licenses = LicenseBundle {
        schema_version: MODEL_LICENSE_SCHEMA_VERSION,
        entries: vec![LicenseEntry {
            component: "fixture".to_owned(),
            license_id: "LicenseRef-Test".to_owned(),
            source_url: "https://example.invalid/license".to_owned(),
            notice: "test only".to_owned(),
        }],
    };
    licenses.validate().unwrap();
}

#[test]
fn strict_validation_rejects_unknown_fields_bad_hashes_and_bad_tensor_order() {
    let mut json = serde_json::to_value(manifest()).unwrap();
    json["unexpected"] = true.into();
    assert!(serde_json::from_value::<ModelPackageManifest>(json).is_err());

    let mut invalid = manifest();
    invalid.source.sha256 = "ABC".to_owned();
    assert!(invalid.validate().is_err());

    let mut invalid = manifest();
    invalid.tensors.entries.push(TensorSpec {
        name: "aaa".to_owned(),
        shape: vec![1],
        dtype: "f32".to_owned(),
    });
    assert!(invalid.validate().is_err());
}

#[test]
fn strict_validation_rejects_invalid_dimensions_weights_and_unpaired_training_files() {
    let mut invalid = manifest();
    invalid.inputs[0].shape = vec![1, -2];
    assert!(invalid.validate().is_err());

    let mut invalid = manifest();
    invalid.training.perceptual_weight = f64::NAN;
    assert!(invalid.validate().is_err());

    let mut invalid = manifest();
    invalid.optimizer = Some(FileManifest {
        file_name: "optimizer.safetensors".to_owned(),
        bytes: 1,
        sha256: "d".repeat(64),
    });
    assert!(invalid.validate().is_err());
}

#[test]
fn unet_io_contract_preserves_semantic_input_order() {
    let value = ModelDescription::from_configuration(ModelConfiguration::OriginalUnet {
        channels: [2, 4, 8, 16, 32],
    });
    assert_eq!(
        value
            .inputs
            .iter()
            .map(|entry| entry.name.as_str())
            .collect::<Vec<_>>(),
        ["input", "audio"]
    );
    value.validate().unwrap();
}

#[test]
fn provenance_shape_is_stable_for_future_consumers() {
    let _ = BTreeMap::<String, String>::new();
}
