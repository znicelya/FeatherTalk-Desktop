mod support;

use feathertalk_domain::{TaskKind, TaskStatus};
use feathertalk_project::TaskHistoryStatus;
use feathertalk_supervisor::SupervisorError;
use feathertalk_supervisor::journal::{MAX_JOURNAL_ENTRIES, TaskJournal};
use support::{entry, find_entry, project_with_history, read_manifest, task_id};
use time::OffsetDateTime;
use time::macros::datetime;

fn moment(minute: u8) -> OffsetDateTime {
    datetime!(2026-09-05 10:00 UTC) + time::Duration::minutes(i64::from(minute))
}

#[test]
fn a_task_moves_from_queued_to_running_to_completed() {
    let project = project_with_history(Vec::new());
    let journal = TaskJournal::new(project.path());
    let id = task_id(1_756_000_000_000, 0xabcd);

    for (index, status) in [
        TaskStatus::Queued,
        TaskStatus::Running,
        TaskStatus::Completed,
    ]
    .into_iter()
    .enumerate()
    {
        journal
            .record(&id, TaskKind::Train, status, moment(index as u8))
            .expect("the record lands");
    }

    let manifest = read_manifest(project.path());
    assert_eq!(manifest.task_history.len(), 1);
    let recorded = find_entry(&manifest, id.as_str());
    assert_eq!(recorded.status, TaskHistoryStatus::Completed);
    assert_eq!(recorded.kind, "train");
    assert_eq!(recorded.updated_at, "2026-09-05T10:02:00Z");
}

#[test]
fn the_manifest_still_validates_after_a_record() {
    let project = project_with_history(Vec::new());
    let journal = TaskJournal::new(project.path());
    let id = task_id(1_756_000_000_001, 1);

    journal
        .record(&id, TaskKind::ProbeMedia, TaskStatus::Running, moment(0))
        .expect("the record lands");

    // `read_manifest` goes through `read_project_manifest`, which validates.
    let manifest = read_manifest(project.path());
    assert!(manifest.validate().is_ok());
}

#[test]
fn two_tasks_keep_their_own_entries() {
    let project = project_with_history(Vec::new());
    let journal = TaskJournal::new(project.path());
    let first = task_id(1_756_000_000_000, 1);
    let second = task_id(1_756_000_000_500, 2);

    journal
        .record(
            &first,
            TaskKind::ExtractFrames,
            TaskStatus::Completed,
            moment(0),
        )
        .expect("the first record lands");
    journal
        .record(&second, TaskKind::Train, TaskStatus::Running, moment(1))
        .expect("the second record lands");

    let manifest = read_manifest(project.path());
    assert_eq!(manifest.task_history.len(), 2);
    assert_eq!(
        find_entry(&manifest, first.as_str()).status,
        TaskHistoryStatus::Completed
    );
    assert_eq!(
        find_entry(&manifest, second.as_str()).status,
        TaskHistoryStatus::Running
    );
}

#[test]
fn a_full_history_drops_the_oldest_terminal_entry() {
    let mut entries = Vec::new();
    for index in 0..MAX_JOURNAL_ENTRIES {
        let status = if index < 10 {
            TaskHistoryStatus::Running
        } else {
            TaskHistoryStatus::Completed
        };
        entries.push(entry(
            task_id(1_756_000_000_000 + index as u64, index as u32).as_str(),
            "train",
            status,
            "2026-09-05T09:00:00Z",
        ));
    }
    let oldest_terminal = entries[10].task_id.clone();
    let project = project_with_history(entries);
    let journal = TaskJournal::new(project.path());
    let fresh = task_id(1_756_000_999_999, 0xffff);

    journal
        .record(&fresh, TaskKind::Train, TaskStatus::Queued, moment(0))
        .expect("the record lands");

    let manifest = read_manifest(project.path());
    assert_eq!(manifest.task_history.len(), MAX_JOURNAL_ENTRIES);
    assert!(
        !manifest
            .task_history
            .iter()
            .any(|entry| entry.task_id == oldest_terminal)
    );
    assert_eq!(
        manifest
            .task_history
            .iter()
            .filter(|entry| entry.status == TaskHistoryStatus::Running)
            .count(),
        10,
        "unfinished entries must survive the trim"
    );
    assert_eq!(
        find_entry(&manifest, fresh.as_str()).status,
        TaskHistoryStatus::Queued
    );
}

#[test]
fn a_history_full_of_incomplete_tasks_is_an_error() {
    let entries = (0..MAX_JOURNAL_ENTRIES)
        .map(|index| {
            entry(
                task_id(1_756_000_000_000 + index as u64, index as u32).as_str(),
                "train",
                TaskHistoryStatus::Running,
                "2026-09-05T09:00:00Z",
            )
        })
        .collect();
    let project = project_with_history(entries);
    let journal = TaskJournal::new(project.path());
    let fresh = task_id(1_756_000_999_999, 0xffff);

    let error = journal
        .record(&fresh, TaskKind::Train, TaskStatus::Queued, moment(0))
        .expect_err("a full history of unfinished tasks cannot take another");
    assert!(matches!(
        error,
        SupervisorError::HistoryFull { limit } if limit == MAX_JOURNAL_ENTRIES
    ));

    let manifest = read_manifest(project.path());
    assert_eq!(manifest.task_history.len(), MAX_JOURNAL_ENTRIES);
    assert!(
        !manifest
            .task_history
            .iter()
            .any(|entry| entry.task_id == fresh.as_str())
    );
}
