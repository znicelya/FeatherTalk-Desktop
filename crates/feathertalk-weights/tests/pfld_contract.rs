use std::path::PathBuf;

use feathertalk_weights::{
    PFLD_ARCHITECTURE_VERSION, PFLD_CHECKPOINT_EPOCH, PfldIgnoredTensors, PfldImportManifest,
    PfldImportRequest, PfldModelArtifact, PfldSourceManifest, TensorAudit, TensorSummary,
};

#[test]
fn pfld_request_defaults_match_existing_import_limits() {
    let request = PfldImportRequest::default();
    assert_eq!(request.checkpoint, PathBuf::new());
    assert_eq!(request.destination_dir, PathBuf::new());
    assert_eq!(request.max_file_bytes, 4 * 1024 * 1024 * 1024);
    assert_eq!(request.max_tensor_count, 10_000);
    assert_eq!(request.max_total_elements, 2_000_000_000);
    assert_eq!(PFLD_CHECKPOINT_EPOCH, 335);
    assert_eq!(PFLD_ARCHITECTURE_VERSION, "burn-pfld-structure-v1");
}

#[test]
fn manifest_round_trips_without_absolute_paths_or_timestamps() {
    let manifest = PfldImportManifest {
        schema_version: 1,
        model_type: "pfld_ghost_one".to_owned(),
        architecture_version: PFLD_ARCHITECTURE_VERSION.to_owned(),
        source: PfldSourceManifest {
            file_name: "checkpoint_epoch_335.pth.tar".to_owned(),
            sha256: "a".repeat(64),
        },
        epoch: PFLD_CHECKPOINT_EPOCH,
        backbone: TensorSummary {
            tensor_count: 2_090,
            total_elements: 913_663,
        },
        model: PfldModelArtifact {
            format: "safetensors".to_owned(),
            file_name: "model.safetensors".to_owned(),
            sha256: "b".repeat(64),
            tensor_count: 1_735,
            total_elements: 910_902,
        },
        ignored: PfldIgnoredTensors {
            batch_norm_counters: TensorAudit {
                tensor_count: 1,
                total_elements: 1,
                keys: vec!["conv1.rbr_conv.0.bn.num_batches_tracked".to_owned()],
            },
            localization: TensorAudit {
                tensor_count: 4,
                total_elements: 2_410,
                keys: vec![
                    "localization.0.bias".to_owned(),
                    "localization.0.weight".to_owned(),
                    "localization.3.bias".to_owned(),
                    "localization.3.weight".to_owned(),
                ],
            },
            auxiliarynet: None,
        },
    };

    let json = serde_json::to_string_pretty(&manifest).unwrap();
    assert!(!json.contains(r#""destination_dir""#));
    assert!(!json.contains(r#""timestamp""#));
    assert_eq!(
        serde_json::from_str::<PfldImportManifest>(&json).unwrap(),
        manifest
    );
}
