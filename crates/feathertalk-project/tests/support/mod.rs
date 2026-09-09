use feathertalk_project::{
    AssetManifest, AssetPackageState, FeatureType, ModelSelection, ProjectManifest,
    TaskHistoryEntry, TaskHistoryStatus,
};

pub fn preparing_manifest() -> AssetManifest {
    AssetManifest {
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
    }
}

pub fn locked_manifest() -> AssetManifest {
    AssetManifest {
        schema_version: 1,
        state: AssetPackageState::Locked,
        video_fps: 25,
        audio_sample_rate: 16_000,
        audio_channels: 1,
        frame_count: 12,
        frame_width: 160,
        frame_height: 160,
        feature_type: FeatureType::FeatherHubert,
        feature_shape: [12, 2, 1024],
        landmark_model_sha256: "a".repeat(64),
        feature_model_sha256: "b".repeat(64),
    }
}

pub fn valid_project() -> ProjectManifest {
    ProjectManifest {
        schema_version: 1,
        project_id: "demo".into(),
        display_name: "Demo".into(),
        asset_package: "assets/assets.json".into(),
        default_model: ModelSelection::OriginalUnet,
        task_history: vec![TaskHistoryEntry {
            task_id: "task-1".into(),
            kind: "preprocess".into(),
            status: TaskHistoryStatus::Completed,
            updated_at: "2026-08-20T10:00:00Z".into(),
        }],
    }
}
