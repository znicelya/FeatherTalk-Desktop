mod support;

use feathertalk_domain::TaskKind;
use feathertalk_project::TaskHistoryStatus;
use feathertalk_supervisor::SupervisorError;
use feathertalk_supervisor::recovery::{
    Resolution, apply_resolutions, resolve_project, scan_incomplete, scan_project,
};
use support::{entry, find_entry, manifest, project_with_history, read_manifest, task_id};
use time::macros::datetime;

fn now() -> time::OffsetDateTime {
    datetime!(2026-09-05 12:00 UTC)
}

#[test]
fn only_queued_and_running_entries_are_incomplete() {
    let statuses = [
        TaskHistoryStatus::Queued,
        TaskHistoryStatus::Running,
        TaskHistoryStatus::Completed,
        TaskHistoryStatus::Failed,
        TaskHistoryStatus::Cancelled,
    ];
    let entries = statuses
        .iter()
        .enumerate()
        .map(|(index, status)| {
            entry(
                task_id(1_756_000_000_000 + index as u64, index as u32).as_str(),
                "train",
                status.clone(),
                "2026-09-05T09:00:00Z",
            )
        })
        .collect();

    let incomplete = scan_incomplete(&manifest(entries));

    assert_eq!(incomplete.len(), 2);
    assert_eq!(incomplete[0].status, TaskHistoryStatus::Queued);
    assert_eq!(incomplete[1].status, TaskHistoryStatus::Running);
}

#[test]
fn incomplete_tasks_come_back_in_time_order() {
    let later = task_id(1_756_000_000_900, 9);
    let earlier = task_id(1_756_000_000_100, 1);
    let entries = vec![
        entry(
            later.as_str(),
            "train",
            TaskHistoryStatus::Running,
            "2026-09-05T09:00:00Z",
        ),
        entry(
            earlier.as_str(),
            "extract_frames",
            TaskHistoryStatus::Queued,
            "2026-09-05T08:00:00Z",
        ),
    ];

    let incomplete = scan_incomplete(&manifest(entries));

    assert_eq!(incomplete[0].task_id, earlier.as_str());
    assert_eq!(incomplete[1].task_id, later.as_str());
    assert_eq!(incomplete[0].kind, Some(TaskKind::ExtractFrames));
}

#[test]
fn an_unknown_kind_slug_survives_the_scan() {
    let id = task_id(1_756_000_000_000, 7);
    let entries = vec![entry(
        id.as_str(),
        "future_command",
        TaskHistoryStatus::Running,
        "2026-09-05T09:00:00Z",
    )];

    let incomplete = scan_incomplete(&manifest(entries));

    assert_eq!(incomplete.len(), 1);
    assert_eq!(incomplete[0].task_id, id.as_str());
    assert_eq!(incomplete[0].kind, None);
}

#[test]
fn resuming_puts_a_task_back_in_the_queue() {
    let id = task_id(1_756_000_000_000, 3);
    let project = project_with_history(vec![entry(
        id.as_str(),
        "train",
        TaskHistoryStatus::Running,
        "2026-09-05T09:00:00Z",
    )]);

    apply_resolutions(
        project.path(),
        &[(id.as_str().to_owned(), Resolution::Resume)],
        now(),
    )
    .expect("the resolution lands");

    let recorded = find_entry(&read_manifest(project.path()), id.as_str());
    assert_eq!(recorded.status, TaskHistoryStatus::Queued);
    assert_eq!(recorded.updated_at, "2026-09-05T12:00:00Z");
}

#[test]
fn discarding_cancels_the_task() {
    let id = task_id(1_756_000_000_000, 4);
    let project = project_with_history(vec![entry(
        id.as_str(),
        "train",
        TaskHistoryStatus::Queued,
        "2026-09-05T09:00:00Z",
    )]);

    apply_resolutions(
        project.path(),
        &[(id.as_str().to_owned(), Resolution::Discard)],
        now(),
    )
    .expect("the resolution lands");

    assert_eq!(
        find_entry(&read_manifest(project.path()), id.as_str()).status,
        TaskHistoryStatus::Cancelled
    );
}

#[test]
fn a_resolution_for_an_unknown_task_is_an_error() {
    let known = task_id(1_756_000_000_000, 5);
    let project = project_with_history(vec![entry(
        known.as_str(),
        "train",
        TaskHistoryStatus::Running,
        "2026-09-05T09:00:00Z",
    )]);

    let error = apply_resolutions(
        project.path(),
        &[
            (known.as_str().to_owned(), Resolution::Resume),
            ("1756000000001-00000006".to_owned(), Resolution::Discard),
        ],
        now(),
    )
    .expect_err("an unknown task id is refused");

    assert!(matches!(error, SupervisorError::UnknownTask { .. }));
    // Nothing at all was written: the known entry keeps its original status.
    assert_eq!(
        find_entry(&read_manifest(project.path()), known.as_str()).status,
        TaskHistoryStatus::Running
    );
}

#[test]
fn a_resolution_for_a_finished_task_is_refused() {
    let id = task_id(1_756_000_000_000, 6);
    let project = project_with_history(vec![entry(
        id.as_str(),
        "train",
        TaskHistoryStatus::Completed,
        "2026-09-05T09:00:00Z",
    )]);

    let error = apply_resolutions(
        project.path(),
        &[(id.as_str().to_owned(), Resolution::Discard)],
        now(),
    )
    .expect_err("a finished task cannot be resumed or discarded");

    assert!(matches!(error, SupervisorError::AlreadyFinished { .. }));
    assert_eq!(
        find_entry(&read_manifest(project.path()), id.as_str()).status,
        TaskHistoryStatus::Completed
    );
}

#[test]
fn a_project_directory_scans_without_the_caller_reading_it() {
    let running = task_id(1_756_000_000_000, 1);
    let done = task_id(1_756_000_000_001, 2);
    let project = project_with_history(vec![
        entry(
            running.as_str(),
            "train",
            TaskHistoryStatus::Running,
            "2026-09-05T12:00:00Z",
        ),
        entry(
            done.as_str(),
            "render",
            TaskHistoryStatus::Completed,
            "2026-09-05T12:01:00Z",
        ),
    ]);

    let tasks = scan_project(project.path()).expect("the manifest reads back");

    assert_eq!(tasks.len(), 1);
    assert_eq!(tasks[0].task_id, running.as_str());
}

#[test]
fn a_directory_without_a_manifest_cannot_be_scanned() {
    let empty = tempfile::tempdir().expect("a temporary directory");

    assert!(matches!(
        scan_project(empty.path()),
        Err(SupervisorError::Project(_))
    ));
}

#[test]
fn a_resolution_is_applied_without_the_caller_owning_a_clock() {
    let id = task_id(1_756_000_000_000, 1);
    let project = project_with_history(vec![entry(
        id.as_str(),
        "train",
        TaskHistoryStatus::Running,
        "2026-09-05T12:00:00Z",
    )]);

    resolve_project(
        project.path(),
        &[(id.as_str().to_owned(), Resolution::Discard)],
    )
    .expect("the manifest writes back");

    assert_eq!(
        find_entry(&read_manifest(project.path()), id.as_str()).status,
        TaskHistoryStatus::Cancelled
    );
    let tasks = scan_project(project.path()).expect("the manifest reads back");
    assert!(tasks.is_empty(), "a discarded task is no longer open");
}
