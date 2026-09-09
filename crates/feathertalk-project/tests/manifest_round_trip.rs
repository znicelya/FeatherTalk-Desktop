use feathertalk_project::{
    AssetManifest, AssetPackageState, FeatureType, ModelSelection, ProjectManifest,
    TaskHistoryEntry, TaskHistoryStatus,
};

#[test]
fn project_manifest_round_trips_with_snake_case_enums() {
    let manifest = ProjectManifest {
        schema_version: 1,
        project_id: "demo_01".to_owned(),
        display_name: "Demo".to_owned(),
        asset_package: "assets/assets.json".to_owned(),
        default_model: ModelSelection::OriginalUnet,
        task_history: vec![TaskHistoryEntry {
            task_id: "task-1".to_owned(),
            kind: "preprocess".to_owned(),
            status: TaskHistoryStatus::Completed,
            updated_at: "2026-08-20T10:00:00Z".to_owned(),
        }],
    };
    let json = serde_json::to_string(&manifest).unwrap();
    assert!(json.contains("original_unet"));
    assert!(json.contains("completed"));
    let decoded: ProjectManifest = serde_json::from_str(&json).unwrap();
    assert_eq!(decoded, manifest);
}

#[test]
fn asset_manifest_round_trips_lifecycle_and_feature_type() {
    let manifest = AssetManifest {
        schema_version: 1,
        state: AssetPackageState::Preparing,
        video_fps: 0,
        audio_sample_rate: 0,
        audio_channels: 0,
        frame_count: 0,
        frame_width: 0,
        frame_height: 0,
        feature_type: FeatureType::FeatherHubert,
        feature_shape: [0, 0, 0],
        landmark_model_sha256: String::new(),
        feature_model_sha256: String::new(),
    };
    let json = serde_json::to_string(&manifest).unwrap();
    assert!(json.contains("preparing"));
    assert!(json.contains("feather_hubert"));
    assert_eq!(
        serde_json::from_str::<AssetManifest>(&json).unwrap(),
        manifest
    );
}
