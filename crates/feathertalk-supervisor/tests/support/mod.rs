//! Helpers shared by this crate's integration tests.
//!
//! Declared with `mod support;` from each test file, so every test binary gets
//! its own copy. No single test uses all of it, hence the blanket `dead_code`
//! allowance.

#![allow(dead_code)]

use std::path::Path;

use feathertalk_domain::TaskId;
use feathertalk_project::{
    ModelSelection, ProjectManifest, TaskHistoryEntry, TaskHistoryStatus, read_project_manifest,
    write_project_manifest_atomic,
};
use tempfile::TempDir;

/// A task id in the domain's wire format, built from a millisecond value so
/// tests can control the ordering they assert on.
pub fn task_id(millis: u64, suffix: u32) -> TaskId {
    TaskId::parse(&format!("{millis:013}-{suffix:08x}")).expect("a well formed task id")
}

pub fn entry(
    task_id: &str,
    kind: &str,
    status: TaskHistoryStatus,
    updated_at: &str,
) -> TaskHistoryEntry {
    TaskHistoryEntry {
        task_id: task_id.to_owned(),
        kind: kind.to_owned(),
        status,
        updated_at: updated_at.to_owned(),
    }
}

pub fn manifest(entries: Vec<TaskHistoryEntry>) -> ProjectManifest {
    ProjectManifest {
        schema_version: 1,
        project_id: "demo".to_owned(),
        display_name: "Demo".to_owned(),
        asset_package: "assets/assets.json".to_owned(),
        default_model: ModelSelection::OriginalUnet,
        task_history: entries,
    }
}

/// A directory holding one valid `project.json` and nothing else. The journal
/// and the startup scan only read that file, so the asset package is not needed.
pub fn project_with_history(entries: Vec<TaskHistoryEntry>) -> TempDir {
    let dir = tempfile::tempdir().expect("a temporary directory");
    write_manifest(dir.path(), &manifest(entries));
    dir
}

pub fn write_manifest(project_dir: &Path, manifest: &ProjectManifest) {
    write_project_manifest_atomic(&project_dir.join("project.json"), manifest)
        .expect("the manifest is valid and writable");
}

pub fn read_manifest(project_dir: &Path) -> ProjectManifest {
    read_project_manifest(&project_dir.join("project.json")).expect("the manifest reads back")
}

pub fn find_entry(manifest: &ProjectManifest, task_id: &str) -> TaskHistoryEntry {
    manifest
        .task_history
        .iter()
        .find(|entry| entry.task_id == task_id)
        .cloned()
        .unwrap_or_else(|| panic!("{task_id} is missing from the history"))
}
