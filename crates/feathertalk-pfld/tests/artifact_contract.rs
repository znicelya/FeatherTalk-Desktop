use std::path::Path;

use feathertalk_pfld::{
    PFLD_ARCHITECTURE_VERSION, PFLD_INPUT_SHAPE, PFLD_MODEL_SHA256, PFLD_OUTPUT_SHAPE,
    PFLD_RUNTIME_SCHEMA_VERSION, PFLD_SOURCE_SHA256, PfldLicenseManifest, PfldRuntimeError,
    PfldRuntimeManifest, PfldTensorSpec,
};

fn valid_manifest() -> PfldRuntimeManifest {
    PfldRuntimeManifest::approved(
        "checkpoint_epoch_335.pth.tar".to_owned(),
        PFLD_SOURCE_SHA256.to_owned(),
        PFLD_MODEL_SHA256.to_owned(),
        1735,
        910902,
    )
}

#[test]
fn approved_manifest_round_trips_and_exposes_fixed_contract() {
    let manifest = valid_manifest();
    assert_eq!(manifest.schema_version, PFLD_RUNTIME_SCHEMA_VERSION);
    assert_eq!(manifest.architecture_version, PFLD_ARCHITECTURE_VERSION);
    assert_eq!(
        manifest.input,
        PfldTensorSpec::new("input", PFLD_INPUT_SHAPE)
    );
    assert_eq!(
        manifest.output,
        PfldTensorSpec::new("landmarks", PFLD_OUTPUT_SHAPE)
    );
    assert_eq!(
        manifest.license,
        PfldLicenseManifest {
            spdx: "NOASSERTION".to_owned(),
            redistribution_approved: false,
        }
    );
    let encoded = serde_json::to_vec(&manifest).unwrap();
    let decoded: PfldRuntimeManifest = serde_json::from_slice(&encoded).unwrap();
    assert_eq!(decoded, manifest);
    manifest.validate().unwrap();
}

#[test]
fn manifest_rejects_unknown_fields_and_contract_changes() {
    let mut value: serde_json::Value = serde_json::to_value(valid_manifest()).unwrap();
    value["future_field"] = serde_json::json!(true);
    assert!(serde_json::from_value::<PfldRuntimeManifest>(value).is_err());

    for (field, replacement) in [
        ("schema_version", serde_json::json!(99)),
        ("model_type", serde_json::json!("other")),
        ("architecture_version", serde_json::json!("other")),
        ("epoch", serde_json::json!(334)),
    ] {
        let mut value: serde_json::Value = serde_json::to_value(valid_manifest()).unwrap();
        value[field] = replacement;
        let parsed: PfldRuntimeManifest = serde_json::from_value(value).unwrap();
        assert!(matches!(
            parsed.validate(),
            Err(PfldRuntimeError::InvalidManifest { .. })
                | Err(PfldRuntimeError::UnsupportedSchemaVersion { .. })
                | Err(PfldRuntimeError::UnsupportedArchitectureVersion { .. })
        ));
    }
}

#[test]
fn manifest_rejects_bad_hashes_shapes_and_license() {
    let cases = [
        ("source.sha256", serde_json::json!("A".repeat(64))),
        ("model.sha256", serde_json::json!("short")),
        ("input.shape", serde_json::json!([1, 3, 192, 191])),
        ("output.shape", serde_json::json!([1, 220, 1])),
        ("license.redistribution_approved", serde_json::json!(true)),
    ];
    for (field, replacement) in cases {
        let mut value: serde_json::Value = serde_json::to_value(valid_manifest()).unwrap();
        let target = if field.starts_with("source.") {
            value.get_mut("source").unwrap()
        } else if field.starts_with("model.") {
            value.get_mut("model").unwrap()
        } else if field.starts_with("input.") {
            value.get_mut("input").unwrap()
        } else if field.starts_with("output.") {
            value.get_mut("output").unwrap()
        } else {
            value.get_mut("license").unwrap()
        };
        target[field.split('.').nth(1).unwrap()] = replacement;
        let parsed: PfldRuntimeManifest = serde_json::from_value(value).unwrap();
        assert!(
            parsed.validate().is_err(),
            "field {field} unexpectedly accepted"
        );
    }
}

#[test]
fn artifact_loader_accepts_only_committed_directory_entries() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("artifacts/pfld_ghost_one");
    let entries = std::fs::read_dir(root).unwrap();
    let mut names = entries
        .map(|entry| entry.unwrap().file_name().into_string().unwrap())
        .collect::<Vec<_>>();
    names.sort();
    assert_eq!(names, ["manifest.json", "model.safetensors"]);
}
