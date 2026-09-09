use std::fs;

use feathertalk_audio::{
    AudioError, FeatureCommitSpec, FeatureMatrix, commit_feature_artifact, read_feature_file,
};
use feathertalk_project::{
    AssetManifest, AssetPackageState, FeatureType, read_asset_manifest, write_asset_manifest_atomic,
};

fn matrix(frame_count: usize) -> FeatureMatrix {
    FeatureMatrix::new(frame_count * 2, 1024, vec![0.25; frame_count * 2 * 1024]).unwrap()
}

fn spec(root: &std::path::Path, frames: u64) -> FeatureCommitSpec {
    FeatureCommitSpec {
        project_root: root.to_owned(),
        frame_count: frames,
        frame_width: 160,
        frame_height: 160,
        landmark_model_sha256: "a".repeat(64),
        feature_model_sha256: "b".repeat(64),
    }
}

fn project_root() -> tempfile::TempDir {
    let root = tempfile::tempdir().unwrap();
    fs::create_dir_all(root.path().join("assets/features")).unwrap();
    fs::create_dir_all(root.path().join("assets/frames")).unwrap();
    fs::create_dir_all(root.path().join("assets/landmarks")).unwrap();
    fs::write(root.path().join("assets/video_25fps.mp4"), b"video").unwrap();
    fs::write(root.path().join("assets/audio_16k_mono.wav"), b"audio").unwrap();
    root
}

fn preparing() -> AssetManifest {
    AssetManifest {
        schema_version: 1,
        state: AssetPackageState::Preparing,
        video_fps: 25,
        audio_sample_rate: 16_000,
        audio_channels: 1,
        frame_count: 1,
        frame_width: 160,
        frame_height: 160,
        feature_type: FeatureType::FeatherHubert,
        feature_shape: [1, 2, 1024],
        landmark_model_sha256: String::new(),
        feature_model_sha256: String::new(),
    }
}

fn locked() -> AssetManifest {
    AssetManifest {
        schema_version: 1,
        state: AssetPackageState::Locked,
        video_fps: 25,
        audio_sample_rate: 16_000,
        audio_channels: 1,
        frame_count: 1,
        frame_width: 160,
        frame_height: 160,
        feature_type: FeatureType::FeatherHubert,
        feature_shape: [1, 2, 1024],
        landmark_model_sha256: "a".repeat(64),
        feature_model_sha256: "b".repeat(64),
    }
}

#[test]
fn commits_feature_and_locked_manifest_with_matching_shape_and_hash() {
    let root = project_root();
    let artifact = commit_feature_artifact(&spec(root.path(), 1), &matrix(1)).unwrap();
    assert_eq!(artifact.tokens(), 2);
    assert_eq!(artifact.dims(), 1024);
    assert_eq!(read_feature_file(artifact.path()).unwrap(), matrix(1));
    let manifest = read_asset_manifest(&root.path().join("assets/assets.json")).unwrap();
    assert_eq!(manifest.state, AssetPackageState::Locked);
    assert_eq!(manifest.feature_shape, [1, 2, 1024]);
    assert_eq!(manifest.feature_model_sha256, "b".repeat(64));
}

#[test]
fn replaces_preparing_manifest_and_existing_feature_atomically() {
    let root = project_root();
    let manifest_path = root.path().join("assets/assets.json");
    write_asset_manifest_atomic(&manifest_path, &preparing()).unwrap();
    let feature_path = root.path().join("assets/features/feather_hubert.f32");
    fs::write(&feature_path, b"old-feature").unwrap();
    commit_feature_artifact(&spec(root.path(), 1), &matrix(1)).unwrap();
    assert_ne!(fs::read(feature_path).unwrap(), b"old-feature");
    assert_eq!(
        read_asset_manifest(&manifest_path).unwrap().state,
        AssetPackageState::Locked
    );
}

#[test]
fn locked_manifest_rejects_commit_and_preserves_old_feature() {
    let root = project_root();
    let manifest_path = root.path().join("assets/assets.json");
    write_asset_manifest_atomic(&manifest_path, &locked()).unwrap();
    let feature_path = root.path().join("assets/features/feather_hubert.f32");
    fs::write(&feature_path, b"old-feature").unwrap();
    assert!(matches!(
        commit_feature_artifact(&spec(root.path(), 1), &matrix(1)),
        Err(AudioError::LockedAssetMutation { .. })
    ));
    assert_eq!(fs::read(feature_path).unwrap(), b"old-feature");
}

#[test]
fn late_manifest_failure_rolls_back_new_feature_and_keeps_old_manifest() {
    let root = project_root();
    let manifest_path = root.path().join("assets/assets.json");
    write_asset_manifest_atomic(&manifest_path, &preparing()).unwrap();
    let feature_path = root.path().join("assets/features/feather_hubert.f32");
    fs::write(&feature_path, b"old-feature").unwrap();
    fs::remove_dir_all(root.path().join("assets/landmarks")).unwrap();
    assert!(commit_feature_artifact(&spec(root.path(), 1), &matrix(1)).is_err());
    assert_eq!(fs::read(feature_path).unwrap(), b"old-feature");
    assert_eq!(read_asset_manifest(&manifest_path).unwrap(), preparing());
}

#[test]
fn invalid_matrix_or_staging_collision_does_not_touch_outputs() {
    let root = project_root();
    let invalid = FeatureMatrix::new(1, 1024, vec![0.0; 1024]).unwrap();
    assert!(matches!(
        commit_feature_artifact(&spec(root.path(), 1), &invalid),
        Err(AudioError::FeatureShapeMismatch { .. })
    ));
    assert!(
        !root
            .path()
            .join("assets/features/feather_hubert.f32")
            .exists()
    );
}

#[test]
fn committed_manifest_and_feature_header_agree_on_public_shape() {
    let root = project_root();
    let artifact = commit_feature_artifact(&spec(root.path(), 1), &matrix(1)).unwrap();
    let manifest = read_asset_manifest(&root.path().join("assets/assets.json")).unwrap();
    let feature = read_feature_file(artifact.path()).unwrap();
    assert_eq!(manifest.feature_shape, [1, 2, 1024]);
    assert_eq!(feature.tokens(), manifest.feature_shape[0] as usize * 2);
    assert_eq!(feature.dims(), manifest.feature_shape[2] as usize);
}
