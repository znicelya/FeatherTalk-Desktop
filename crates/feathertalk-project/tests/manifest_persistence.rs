#[path = "support/mod.rs"]
mod support;

use feathertalk_project::{
    ProjectError, read_asset_manifest, read_project_manifest, write_asset_manifest_atomic,
    write_project_manifest_atomic,
};
use support::*;

#[test]
fn reads_and_writes_preparing_manifest_with_one_newline() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("assets.json");
    let manifest = preparing_manifest();
    write_asset_manifest_atomic(&path, &manifest).unwrap();
    assert!(std::fs::read(&path).unwrap().ends_with(b"\n"));
    assert_eq!(read_asset_manifest(&path).unwrap(), manifest);
}

#[test]
fn rejects_manifest_larger_than_one_mib() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("project.json");
    std::fs::write(&path, vec![b' '; 1_048_577]).unwrap();
    assert!(matches!(
        read_project_manifest(&path),
        Err(ProjectError::ManifestTooLarge { .. })
    ));
}

#[test]
fn rejects_invalid_utf8_before_json_parsing() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("project.json");
    std::fs::write(&path, [0xff]).unwrap();
    assert!(matches!(
        read_project_manifest(&path),
        Err(ProjectError::InvalidUtf8 { .. })
    ));
}

#[test]
fn replaces_existing_preparing_manifest_without_truncating_on_success() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("assets.json");
    let mut first = preparing_manifest();
    write_asset_manifest_atomic(&path, &first).unwrap();
    first.video_fps = 25;
    write_asset_manifest_atomic(&path, &first).unwrap();
    assert_eq!(read_asset_manifest(&path).unwrap().video_fps, 25);
}

#[test]
fn failed_validation_leaves_existing_manifest_unchanged() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("assets.json");
    let first = preparing_manifest();
    write_asset_manifest_atomic(&path, &first).unwrap();
    let mut invalid = first.clone();
    invalid.feature_shape = [1, 0, 1024];
    assert!(write_asset_manifest_atomic(&path, &invalid).is_err());
    assert_eq!(read_asset_manifest(&path).unwrap(), first);
}

#[test]
fn asset_writer_rejects_existing_locked_manifest() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("assets.json");
    let locked = locked_manifest();
    write_asset_manifest_atomic(&path, &locked).unwrap();
    assert!(matches!(
        write_asset_manifest_atomic(&path, &preparing_manifest()),
        Err(ProjectError::LockedAssetMutation { .. })
    ));
}

#[test]
fn project_writer_round_trips() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("project.json");
    let project = valid_project();
    write_project_manifest_atomic(&path, &project).unwrap();
    assert_eq!(read_project_manifest(&path).unwrap(), project);
}
