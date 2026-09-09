use feathertalk_domain::{TaskId, TaskKind, TaskStage, TaskStatus};
use feathertalk_project::{ModelSelection, ProjectManifest, TaskHistoryEntry, TaskHistoryStatus};

const SAMPLE_TASK_ID: &str = "1787900000000-0000000a";

fn manifest(entries: Vec<TaskHistoryEntry>) -> ProjectManifest {
    ProjectManifest {
        schema_version: 1,
        project_id: "demo-project".to_owned(),
        display_name: "Demo".to_owned(),
        asset_package: "assets/assets.json".to_owned(),
        default_model: ModelSelection::OriginalUnet,
        task_history: entries,
    }
}

fn entry(task_id: &str, kind: &str) -> TaskHistoryEntry {
    TaskHistoryEntry {
        task_id: task_id.to_owned(),
        kind: kind.to_owned(),
        status: TaskHistoryStatus::Running,
        updated_at: "2026-08-28T09:00:00Z".to_owned(),
    }
}

#[test]
fn the_five_state_vocabulary_has_not_drifted() {
    let domain: Vec<String> = TaskStatus::ALL
        .into_iter()
        .map(|status| serde_json::to_string(&status).unwrap())
        .collect();
    let persisted: Vec<String> = [
        TaskHistoryStatus::Queued,
        TaskHistoryStatus::Running,
        TaskHistoryStatus::Completed,
        TaskHistoryStatus::Failed,
        TaskHistoryStatus::Cancelled,
    ]
    .into_iter()
    .map(|status| serde_json::to_string(&status).unwrap())
    .collect();
    assert_eq!(domain, persisted);
    assert_eq!(domain.len(), 5);
}

#[test]
fn every_task_kind_slug_is_accepted_by_the_real_project_validator() {
    for (index, kind) in TaskKind::ALL.into_iter().enumerate() {
        let task_id = format!("178790000{:04}-0000000a", index);
        let manifest = manifest(vec![entry(&task_id, kind.as_slug())]);
        manifest.validate().unwrap_or_else(|error| {
            panic!(
                "slug {:?} rejected by ProjectManifest: {error}",
                kind.as_slug()
            )
        });
    }
}

#[test]
fn the_canonical_task_id_shape_is_accepted_by_the_real_project_validator() {
    let task_id = TaskId::parse(SAMPLE_TASK_ID).unwrap();
    let manifest = manifest(vec![entry(task_id.as_str(), TaskKind::Train.as_slug())]);
    manifest.validate().unwrap();
}

#[test]
fn a_stage_projection_reaches_every_persisted_status() {
    let mut reached: Vec<TaskStatus> = TaskStage::ALL_UNIT_SAMPLES
        .into_iter()
        .map(|stage| stage.status())
        .collect();
    reached.sort_by_key(|status| serde_json::to_string(status).unwrap());
    reached.dedup();
    assert_eq!(
        reached.len(),
        5,
        "projection does not cover all five states"
    );
}
