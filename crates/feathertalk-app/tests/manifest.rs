//! The `project.json` a submission makes sure is there.

use std::fs;
use std::path::{Path, PathBuf};

use feathertalk_app::manifest::{
    display_name_from_dir, ensure_manifest, manifest_path, project_id_from_dir, Bootstrap,
    ManifestError,
};
use feathertalk_project::{
    read_project_manifest, write_project_manifest_atomic, ModelSelection, ProjectManifest,
    TaskHistoryEntry, TaskHistoryStatus,
};

/// A project path ending in `name`.
///
/// Nothing is created: the two derivations read the path's last component and
/// never touch the disk, and a 200 character directory name is easier to write
/// than to create on Windows.
fn path_named(name: &str) -> PathBuf {
    Path::new("projects").join(name)
}

/// A real directory named `name`, with the temporary root that owns it.
fn dir_named(name: &str) -> (tempfile::TempDir, PathBuf) {
    let root = tempfile::tempdir().expect("a temporary directory");
    let dir = root.path().join(name);
    fs::create_dir_all(&dir).expect("the project directory");
    (root, dir)
}

/// A manifest a person has already worked with: one task in its history.
fn used_manifest() -> ProjectManifest {
    ProjectManifest {
        schema_version: 1,
        project_id: "kanghui".to_owned(),
        display_name: "康辉训练素材".to_owned(),
        asset_package: "assets/assets.json".to_owned(),
        default_model: ModelSelection::MobileOneUnet,
        task_history: vec![TaskHistoryEntry {
            task_id: "01JQ0000000000000000000000".to_owned(),
            kind: "normalize_media".to_owned(),
            status: TaskHistoryStatus::Completed,
            updated_at: "2026-09-05T10:00:00Z".to_owned(),
        }],
    }
}

#[test]
fn an_ascii_name_becomes_the_project_id() {
    assert_eq!(
        project_id_from_dir(&path_named("my-project.01")),
        "my-project.01"
    );
}

#[test]
fn illegal_characters_are_dropped() {
    assert_eq!(project_id_from_dir(&path_named("我的 项目 (1)")), "1");
    assert_eq!(project_id_from_dir(&path_named("kanghui视频")), "kanghui");
}

#[test]
fn a_name_with_nothing_usable_falls_back() {
    assert_eq!(project_id_from_dir(&path_named("我的项目")), "project");
}

#[test]
fn an_overlong_id_is_truncated() {
    let name: String = std::iter::repeat_n('a', 200).collect();

    let id = project_id_from_dir(&path_named(&name));

    assert_eq!(id.len(), 128);
    assert!(id.bytes().all(|byte| byte == b'a'));
}

#[test]
fn the_display_name_keeps_the_original() {
    let path = path_named("我的项目");
    let id = project_id_from_dir(&path);

    assert_eq!(id, "project");
    assert_eq!(display_name_from_dir(&path, &id), "我的项目");
    assert_eq!(display_name_from_dir(&path_named(" 项目 "), &id), "项目");

    let long: String = std::iter::repeat_n('好', 300).collect();
    let display = display_name_from_dir(&path_named(&long), &id);

    assert_eq!(display.chars().count(), 256);
    assert_eq!(display.trim(), display);
}

#[test]
fn a_missing_manifest_is_written() {
    let (_root, dir) = dir_named("demo-project");

    let bootstrap = ensure_manifest(&dir).expect("the manifest is written");

    assert_eq!(bootstrap, Bootstrap::Created);
    let manifest = read_project_manifest(&manifest_path(&dir)).expect("the manifest reads");
    assert_eq!(manifest.schema_version, 1);
    assert_eq!(manifest.project_id, "demo-project");
    assert_eq!(manifest.display_name, "demo-project");
    assert_eq!(manifest.asset_package, "assets/assets.json");
    assert_eq!(manifest.default_model, ModelSelection::OriginalUnet);
    assert!(manifest.task_history.is_empty());
}

#[test]
fn an_existing_manifest_is_left_alone() {
    let (_root, dir) = dir_named("demo-project");
    let path = manifest_path(&dir);
    write_project_manifest_atomic(&path, &used_manifest()).expect("the manifest writes");
    let before = fs::read(&path).expect("the bytes");

    let bootstrap = ensure_manifest(&dir).expect("the manifest is already there");

    assert_eq!(bootstrap, Bootstrap::Present);
    assert_eq!(fs::read(&path).expect("the bytes"), before);
}

#[test]
fn a_broken_manifest_is_not_overwritten() {
    let (_root, dir) = dir_named("demo-project");
    let path = manifest_path(&dir);
    fs::write(&path, "{").expect("the file writes");

    let error = ensure_manifest(&dir).expect_err("a manifest that cannot be read is refused");

    assert!(
        matches!(error, ManifestError::Existing(_)),
        "expected an existing manifest error, got {error:?}"
    );
    assert_eq!(fs::read(&path).expect("the bytes"), b"{");
}
