#[path = "support/mod.rs"]
mod support;
use feathertalk_project::{
    AssetPackageState, lock_asset_package, validate_project_dir, write_asset_manifest_atomic,
    write_project_manifest_atomic,
};
use support::*;

fn create_complete_project() -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(dir.path().join("assets/frames")).unwrap();
    std::fs::create_dir_all(dir.path().join("assets/landmarks")).unwrap();
    std::fs::create_dir_all(dir.path().join("assets/features")).unwrap();
    for file in [
        "assets/video_25fps.mp4",
        "assets/audio_16k_mono.wav",
        "assets/features/feather_hubert.f32",
    ] {
        std::fs::write(dir.path().join(file), b"x").unwrap();
    }
    dir
}

#[test]
fn locks_complete_non_empty_asset_package() {
    let dir = create_complete_project();
    let package = lock_asset_package(dir.path(), locked_manifest()).unwrap();
    assert_eq!(package.manifest().state, AssetPackageState::Locked);
}
#[test]
fn lock_rejects_missing_empty_and_wrong_type_artifacts() {
    let dir = tempfile::tempdir().unwrap();
    assert!(lock_asset_package(dir.path(), locked_manifest()).is_err());
}
#[test]
fn lock_writes_assets_manifest_last_and_validate_project_dir_round_trips() {
    let dir = create_complete_project();
    write_project_manifest_atomic(&dir.path().join("project.json"), &valid_project()).unwrap();
    lock_asset_package(dir.path(), locked_manifest()).unwrap();
    assert!(validate_project_dir(dir.path()).is_ok());
}
#[test]
fn validate_project_rejects_unlocked_assets_manifest() {
    let dir = create_complete_project();
    write_project_manifest_atomic(&dir.path().join("project.json"), &valid_project()).unwrap();
    write_asset_manifest_atomic(
        &dir.path().join("assets/assets.json"),
        &preparing_manifest(),
    )
    .unwrap();
    assert!(validate_project_dir(dir.path()).is_err());
}
