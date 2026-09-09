use feathertalk_scrfd::{
    SCRFD_ANCHORS, SCRFD_ARCHITECTURE_VERSION, SCRFD_INPUT_SHAPE, SCRFD_MODEL_KIND,
    SCRFD_SCHEMA_VERSION, SCRFD_SOURCE_ONNX_BYTES, SCRFD_SOURCE_ONNX_SHA256, SCRFD_SOURCE_OPSET,
    SCRFD_STRIDES, ScrfdArtifactManifest, ScrfdError, ScrfdFileManifest, ScrfdGeneratorManifest,
    ScrfdInputManifest, ScrfdLevelManifest, ScrfdLicenseManifest, ScrfdOutputManifest,
    ScrfdSourceManifest, ScrfdWeightManifest,
};

fn output(name: &str, source: &[usize], public: &[usize]) -> ScrfdOutputManifest {
    ScrfdOutputManifest {
        onnx_name: name.to_owned(),
        source_shape: source.to_vec(),
        public_shape: public.to_vec(),
    }
}

fn valid_manifest() -> ScrfdArtifactManifest {
    ScrfdArtifactManifest {
        schema_version: 1,
        model_kind: "scrfd_2.5g_kps".to_owned(),
        architecture_version: 1,
        source: ScrfdSourceManifest {
            format: "onnx".to_owned(),
            file_name: "scrfd_2.5g_kps.onnx".to_owned(),
            file_bytes: 3_291_017,
            sha256: "32d20c77b9e2dc1d07e94c2ab9d25bdd5cd05eddbe0b46e7b38e7a1eca22e99a".to_owned(),
            opset: 12,
            input_name: "images".to_owned(),
            output_names: [
                "out0", "out1", "out2", "out3", "out4", "out5", "out6", "out7", "out8",
            ]
            .map(str::to_owned),
        },
        generator: ScrfdGeneratorManifest {
            burn: "0.21.0".to_owned(),
            burn_onnx: "0.21.0".to_owned(),
            burn_store: "0.21.0".to_owned(),
            simplify: true,
            load_strategy: "none".to_owned(),
        },
        input: ScrfdInputManifest {
            dtype: "float32".to_owned(),
            shape: [1, 3, 640, 640],
            scale: 1.0 / 128.0,
            mean: [127.5; 3],
            swap_rb: true,
        },
        levels: [
            ScrfdLevelManifest {
                stride: 8,
                anchors: 12_800,
                score: output("out0", &[1, 12_800, 1], &[1, 12_800]),
                bbox: output("out3", &[1, 12_800, 4], &[1, 12_800, 4]),
                keypoints: output("out6", &[1, 12_800, 10], &[1, 12_800, 10]),
            },
            ScrfdLevelManifest {
                stride: 16,
                anchors: 3_200,
                score: output("out1", &[1, 3_200, 1], &[1, 3_200]),
                bbox: output("out4", &[1, 3_200, 4], &[1, 3_200, 4]),
                keypoints: output("out7", &[1, 3_200, 10], &[1, 3_200, 10]),
            },
            ScrfdLevelManifest {
                stride: 32,
                anchors: 800,
                score: output("out2", &[1, 800, 1], &[1, 800]),
                bbox: output("out5", &[1, 800, 4], &[1, 800, 4]),
                keypoints: output("out8", &[1, 800, 10], &[1, 800, 10]),
            },
        ],
        generated_source: ScrfdFileManifest {
            file_name: "scrfd_2_5g.rs".to_owned(),
            file_bytes: 123,
            sha256: "a".repeat(64),
        },
        weights: ScrfdWeightManifest {
            format: "safetensors".to_owned(),
            file_name: "model.safetensors".to_owned(),
            file_bytes: 456,
            sha256: "b".repeat(64),
        },
        license: ScrfdLicenseManifest {
            license_id: "NOASSERTION".to_owned(),
            redistribution_approved: false,
            evidence: "repository does not provide a verifiable model-weight license".to_owned(),
        },
    }
}

#[test]
fn fixed_constants_match_the_approved_source() {
    assert_eq!(SCRFD_SCHEMA_VERSION, 1);
    assert_eq!(SCRFD_ARCHITECTURE_VERSION, 1);
    assert_eq!(SCRFD_MODEL_KIND, "scrfd_2.5g_kps");
    assert_eq!(SCRFD_SOURCE_ONNX_BYTES, 3_291_017);
    assert_eq!(SCRFD_SOURCE_ONNX_SHA256.len(), 64);
    assert_eq!(SCRFD_SOURCE_OPSET, 12);
    assert_eq!(SCRFD_INPUT_SHAPE, [1, 3, 640, 640]);
    assert_eq!(SCRFD_STRIDES, [8, 16, 32]);
    assert_eq!(SCRFD_ANCHORS, [12_800, 3_200, 800]);
}

#[test]
fn schema_one_manifest_round_trips_and_validates() {
    let manifest = valid_manifest();
    manifest.validate().unwrap();
    let json = serde_json::to_string_pretty(&manifest).unwrap();
    assert_eq!(
        serde_json::from_str::<ScrfdArtifactManifest>(&json).unwrap(),
        manifest
    );
}

fn assert_invalid_field(manifest: ScrfdArtifactManifest, field: &str) {
    match manifest.validate().unwrap_err() {
        ScrfdError::InvalidManifest { field: actual, .. } => assert_eq!(actual, field),
        error => panic!("expected InvalidManifest for {field}, got {error}"),
    }
}

#[test]
fn unknown_fields_and_changed_output_mapping_are_rejected() {
    let manifest = valid_manifest();
    let mut value = serde_json::to_value(&manifest).unwrap();
    value["future_field"] = serde_json::json!(true);
    assert!(serde_json::from_value::<ScrfdArtifactManifest>(value).is_err());

    let mut changed = manifest;
    changed.levels[1].bbox.onnx_name = "out5".to_owned();
    assert!(changed.validate().is_err());
}

#[test]
fn unsupported_manifest_and_architecture_versions_are_distinct() {
    let mut manifest = valid_manifest();
    manifest.schema_version = 2;
    assert!(matches!(
        manifest.validate(),
        Err(ScrfdError::UnsupportedSchemaVersion {
            expected: 1,
            actual: 2
        })
    ));

    let mut manifest = valid_manifest();
    manifest.architecture_version = 2;
    assert!(matches!(
        manifest.validate(),
        Err(ScrfdError::UnsupportedArchitectureVersion {
            expected: 1,
            actual: 2
        })
    ));
}

#[test]
fn zero_sizes_dimensions_and_malformed_hashes_are_rejected() {
    let mut manifest = valid_manifest();
    manifest.generated_source.file_bytes = 0;
    assert_invalid_field(manifest, "generated_source.file_bytes");

    let mut manifest = valid_manifest();
    manifest.weights.sha256 = "A".repeat(64);
    assert_invalid_field(manifest, "weights.sha256");

    let mut manifest = valid_manifest();
    manifest.levels[0].score.source_shape[1] = 0;
    assert_invalid_field(manifest, "levels[0].score.source_shape");

    let mut manifest = valid_manifest();
    manifest.source.opset = 0;
    assert_invalid_field(manifest, "source.opset");
}
