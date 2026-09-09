use feathertalk_project::{
    AssetPackageState, lock_asset_package, read_asset_manifest, validate_project_dir,
    write_asset_manifest_atomic, write_project_manifest_atomic,
};

fn fixture() -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(dir.path().join("assets/frames")).unwrap();
    std::fs::create_dir_all(dir.path().join("assets/landmarks")).unwrap();
    std::fs::create_dir_all(dir.path().join("assets/features")).unwrap();
    for f in [
        "assets/video_25fps.mp4",
        "assets/audio_16k_mono.wav",
        "assets/features/feather_hubert.f32",
    ] {
        std::fs::write(dir.path().join(f), b"x").unwrap();
    }
    dir
}
fn preparing() -> feathertalk_project::AssetManifest {
    feathertalk_project::AssetManifest {
        schema_version: 1,
        state: AssetPackageState::Preparing,
        video_fps: 0,
        audio_sample_rate: 0,
        audio_channels: 0,
        frame_count: 0,
        frame_width: 0,
        frame_height: 0,
        feature_type: feathertalk_project::FeatureType::FeatherHubert,
        feature_shape: [0, 0, 0],
        landmark_model_sha256: String::new(),
        feature_model_sha256: String::new(),
    }
}
fn locked() -> feathertalk_project::AssetManifest {
    feathertalk_project::AssetManifest {
        schema_version: 1,
        state: AssetPackageState::Locked,
        video_fps: 25,
        audio_sample_rate: 16000,
        audio_channels: 1,
        frame_count: 1,
        frame_width: 160,
        frame_height: 160,
        feature_type: feathertalk_project::FeatureType::FeatherHubert,
        feature_shape: [1, 2, 1024],
        landmark_model_sha256: "a".repeat(64),
        feature_model_sha256: "b".repeat(64),
    }
}
fn project() -> feathertalk_project::ProjectManifest {
    feathertalk_project::ProjectManifest {
        schema_version: 1,
        project_id: "demo".into(),
        display_name: "Demo".into(),
        asset_package: "assets/assets.json".into(),
        default_model: feathertalk_project::ModelSelection::OriginalUnet,
        task_history: vec![],
    }
}

#[test]
fn crate_root_api_supports_read_write_lock_and_validation() {
    let dir = fixture();
    let prep = preparing();
    let prep_path = dir.path().join("assets/preparing.json");
    write_asset_manifest_atomic(&prep_path, &prep).unwrap();
    assert_eq!(read_asset_manifest(&prep_path).unwrap(), prep);
    write_project_manifest_atomic(&dir.path().join("project.json"), &project()).unwrap();
    let package = lock_asset_package(dir.path(), locked()).unwrap();
    let validated = validate_project_dir(dir.path()).unwrap();
    let canonical_root = std::fs::canonicalize(dir.path()).unwrap();
    assert_eq!(package.root(), canonical_root);
    assert_eq!(package.manifest().state, AssetPackageState::Locked);
    assert_eq!(validated.root(), canonical_root);
    assert_eq!(
        validated.asset_package().manifest().state,
        AssetPackageState::Locked
    );
}
