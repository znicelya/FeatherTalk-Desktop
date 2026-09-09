#[path = "support/mod.rs"]
mod support;

use feathertalk_project::{AssetPackageState, ProjectManifest};
use support::*;

#[test]
fn project_rejects_unknown_fields() {
    let json = r#"{"schema_version":1,"project_id":"demo","display_name":"Demo","asset_package":"assets/assets.json","default_model":"original_unet","task_history":[],"extra":true}"#;
    assert!(serde_json::from_str::<ProjectManifest>(json).is_err());
}

#[test]
fn project_rejects_bad_identifier_and_duplicate_task_ids() {
    let mut project = valid_project();
    project.project_id = "bad/id".into();
    assert!(project.validate().is_err());
    project.project_id = "demo".into();
    project.task_history.push(project.task_history[0].clone());
    assert!(project.validate().is_err());
}

#[test]
fn project_rejects_non_rfc3339_timestamp_and_unsafe_asset_path() {
    let mut project = valid_project();
    project.task_history[0].updated_at = "tomorrow".into();
    assert!(project.validate().is_err());
    project.task_history[0].updated_at = "2026-08-20T10:00:00Z".into();
    project.asset_package = "../assets.json".into();
    assert!(project.validate().is_err());
}

#[test]
fn preparing_manifest_accepts_empty_progress_metadata() {
    assert!(preparing_manifest().validate_preparing().is_ok());
}

#[test]
fn preparing_manifest_rejects_partial_feature_shape() {
    let mut manifest = preparing_manifest();
    manifest.feature_shape = [12, 0, 1024];
    assert!(manifest.validate_preparing().is_err());
}

#[test]
fn preparing_manifest_rejects_invalid_progress_media_and_hashes() {
    let mut manifest = preparing_manifest();
    manifest.video_fps = 24;
    assert!(manifest.validate_preparing().is_err());
    manifest.video_fps = 25;
    manifest.feature_model_sha256 = "A".repeat(64);
    assert!(manifest.validate_preparing().is_err());
}

#[test]
fn locked_manifest_requires_exact_media_and_feature_contract() {
    let mut manifest = locked_manifest();
    manifest.video_fps = 24;
    assert!(manifest.validate_locked().is_err());
}

#[test]
fn locked_manifest_rejects_uppercase_or_short_sha256() {
    let mut manifest = locked_manifest();
    manifest.landmark_model_sha256 = "A".repeat(64);
    assert!(manifest.validate_locked().is_err());
    manifest.landmark_model_sha256 = "a".repeat(63);
    assert!(manifest.validate_locked().is_err());
}

#[test]
fn locked_manifest_rejects_frame_count_shape_mismatch() {
    let mut manifest = locked_manifest();
    manifest.frame_count = 11;
    assert!(manifest.validate_locked().is_err());
}

#[test]
fn locked_state_is_exposed_by_schema() {
    assert_eq!(locked_manifest().state, AssetPackageState::Locked);
}
