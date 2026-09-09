use std::path::PathBuf;

use feathertalk_app::pipeline::TaskUpdate;
use feathertalk_app::tasks::{
    CRASHED_KEY, REJECTED_KEY, Summary, TaskCenter, UNAVAILABLE_KEY, UNSUPPORTED_KEY, blocked_key,
    kind_key, latest_row, recovery_key, stage_key, status_key,
};
use feathertalk_client::{CancelToken, ClientError};
use feathertalk_domain::{
    ErrorCode, Event, Progress, Recovery, TaskError, TaskId, TaskKind, TaskStage, TaskStatus,
};
use feathertalk_supervisor::TaskHistoryStatus;
use feathertalk_supervisor::crash_log::CrashLogOutcome;
use feathertalk_supervisor::recovery::IncompleteTask;
use feathertalk_supervisor::supervisor::{SupervisedOutcome, SupervisionReport};

/// Every recovery suggestion the protocol defines. `Recovery` has no `ALL`, so
/// the list is spelled out; a new variant makes the key test fail here.
const RECOVERIES: [Recovery; 7] = [
    Recovery::Retry,
    Recovery::ResumeFromCheckpoint,
    Recovery::FreeDiskSpace,
    Recovery::SelectDifferentAdapter,
    Recovery::ExcludeBadFrames,
    Recovery::ReimportModel,
    Recovery::NotRecoverable,
];

fn task_id() -> TaskId {
    TaskId::parse("1756000000000-00000001").expect("a well formed task id")
}

fn submitted() -> TaskCenter {
    let mut center = TaskCenter::default();
    center.begin(task_id(), TaskKind::ValidateProject, CancelToken::new());
    center
}

fn progress(stage: TaskStage) -> TaskUpdate {
    TaskUpdate::Progress(Box::new(Event::new(
        task_id(),
        "2026-09-05T13:00:00Z",
        stage,
    )))
}

fn finished(outcome: SupervisedOutcome, attempts: u32) -> TaskUpdate {
    finished_for(task_id(), outcome, attempts)
}

/// The same report, for a session with more than one row in it.
fn finished_for(task_id: TaskId, outcome: SupervisedOutcome, attempts: u32) -> TaskUpdate {
    TaskUpdate::Finished(Box::new(SupervisionReport {
        task_id,
        attempts,
        outcome,
        crash_logs: Vec::new(),
        journal_errors: Vec::new(),
    }))
}

fn failure() -> TaskError {
    TaskError::new(
        ErrorCode::MediaInvalid,
        "输入文件无法解析，请确认它是完整的视频",
        "ffprobe exited with status 1",
        TaskStage::Preparing,
    )
}

fn incomplete(task_id: &str, status: TaskHistoryStatus) -> IncompleteTask {
    IncompleteTask {
        task_id: task_id.to_owned(),
        kind: Some(TaskKind::Train),
        status,
        updated_at: "2026-09-05T12:00:00Z".to_owned(),
    }
}

#[test]
fn every_key_is_distinct_and_namespaced() {
    let mut keys = Vec::new();
    for status in TaskStatus::ALL {
        keys.push(status_key(status));
    }
    for stage in TaskStage::ALL_UNIT_SAMPLES {
        keys.push(stage_key(&stage));
    }
    for kind in TaskKind::ALL {
        keys.push(kind_key(kind));
    }
    for recovery in RECOVERIES {
        keys.push(recovery_key(recovery));
    }
    assert_eq!(keys.len(), 38);
    for key in &keys {
        assert!(key.starts_with("task."), "{key} is not namespaced");
    }
    let mut sorted = keys.clone();
    sorted.sort_unstable();
    sorted.dedup();
    assert_eq!(sorted.len(), keys.len(), "two entries share one key");
}

#[test]
fn a_submitted_task_is_queued_and_busy() {
    let center = submitted();

    assert_eq!(center.rows().len(), 1);
    assert_eq!(center.rows()[0].status, TaskStatus::Queued);
    assert_eq!(center.rows()[0].attempts, 0);
    assert!(center.is_busy());
}

#[test]
fn a_completed_task_keeps_its_worker_result_for_the_ui() {
    let mut center = submitted();
    let result = serde_json::json!({
        "schema_version": 1,
        "model_type": "original_unet",
        "parameter_count": 1234
    });
    center.apply(finished(
        SupervisedOutcome::Completed {
            result: Some(result.clone()),
        },
        1,
    ));
    assert_eq!(center.rows()[0].result.as_ref(), Some(&result));
    assert_eq!(center.rows()[0].status, TaskStatus::Completed);
}

#[test]
fn a_terminal_event_does_not_release_admission_before_the_supervisor_finishes() {
    let mut center = submitted();
    center.apply(progress(TaskStage::Completed));
    assert!(
        center.is_busy(),
        "worker shutdown and journal updates are still in flight"
    );
    assert_ne!(center.rows()[0].status, TaskStatus::Completed);
    center.apply(finished(SupervisedOutcome::Completed { result: None }, 1));
    assert!(!center.is_busy());
}

#[test]
fn changing_projects_discards_old_session_results_but_cannot_drop_an_active_task() {
    let mut center = submitted();
    assert!(!center.adopt_project(Vec::new()));
    assert_eq!(center.rows().len(), 1);
    center.apply(finished(
        SupervisedOutcome::Completed {
            result: Some(serde_json::json!({"old_project": true})),
        },
        1,
    ));
    center.note(feathertalk_app::tasks::Note::new(
        "tasks.note.scan_failed",
        None,
    ));
    assert!(center.adopt_project(vec![incomplete(
        "1756000000009-00000009",
        TaskHistoryStatus::Running
    )]));
    assert!(center.rows().is_empty());
    assert!(center.notes().is_empty());
    assert_eq!(center.incomplete().len(), 1);
}

#[test]
fn progress_moves_the_row_to_running() {
    let mut center = submitted();
    let mut update = progress(TaskStage::ExtractingFrames);
    if let TaskUpdate::Progress(event) = &mut update {
        event.progress = Some(Progress {
            completed: 3,
            total: Some(10),
        });
    }

    center.apply(update);

    let row = &center.rows()[0];
    assert_eq!(row.status, TaskStatus::Running);
    assert_eq!(stage_key(&row.stage), "task.stage.extracting_frames");
    assert_eq!(row.progress.map(|progress| progress.completed), Some(3));
    assert!(center.is_busy());
}

#[test]
fn a_completed_report_finishes_the_row() {
    let mut center = submitted();

    center.apply(finished(SupervisedOutcome::Completed { result: None }, 1));

    let row = &center.rows()[0];
    assert_eq!(row.status, TaskStatus::Completed);
    assert_eq!(row.attempts, 1);
    assert!(row.failure.is_none());
    assert!(!center.is_busy());
}

#[test]
fn a_failed_report_keeps_the_workers_own_summary() {
    let mut center = submitted();

    center.apply(finished(SupervisedOutcome::Failed(failure()), 1));

    let row = &center.rows()[0];
    assert_eq!(row.status, TaskStatus::Failed);
    assert_eq!(stage_key(&row.stage), "task.stage.preparing");
    let detail = row.failure.as_ref().expect("a failed row has a failure");
    assert_eq!(
        detail.summary,
        Summary::Worker("输入文件无法解析，请确认它是完整的视频".to_owned())
    );
    assert_eq!(detail.detail, "ffprobe exited with status 1");
    assert_eq!(detail.code, Some(ErrorCode::MediaInvalid));
    // `TaskError::new` fills `recovery` from `ErrorCode::default_recovery`.
    assert_eq!(detail.recovery, Some(Recovery::Retry));
}

#[test]
fn a_crash_report_falls_back_to_the_shells_summary() {
    let mut center = submitted();

    center.apply(finished(
        SupervisedOutcome::Crashed {
            stage: TaskStage::Training {
                epoch: 3,
                step: 120,
                loss: 0.5,
            },
            recovery: Recovery::ResumeFromCheckpoint,
            detail: "the worker exited with status 101".to_owned(),
        },
        2,
    ));

    let row = &center.rows()[0];
    assert_eq!(row.status, TaskStatus::Failed);
    // The stage the worker died in, not the terminal `failed`: it is the only
    // clue about what was lost.
    assert_eq!(stage_key(&row.stage), "task.stage.training");
    let detail = row.failure.as_ref().expect("a crashed row has a failure");
    assert_eq!(detail.summary, Summary::Key(CRASHED_KEY));
    assert_eq!(detail.detail, "the worker exited with status 101");
    assert_eq!(detail.code, None);
    assert_eq!(detail.recovery, Some(Recovery::ResumeFromCheckpoint));
}

#[test]
fn an_unavailable_report_offers_no_recovery() {
    let mut center = submitted();
    let error = ClientError::WorkerNotFound { probed: Vec::new() };
    let text = error.to_string();

    center.apply(finished(SupervisedOutcome::Unavailable(error), 1));

    let row = &center.rows()[0];
    assert_eq!(row.status, TaskStatus::Failed);
    let detail = row
        .failure
        .as_ref()
        .expect("an unavailable row has a failure");
    assert_eq!(detail.summary, Summary::Key(UNAVAILABLE_KEY));
    // English, straight from the client crate: no restart fixes an installation.
    assert_eq!(detail.detail, text);
    assert_eq!(detail.recovery, None);
}

#[test]
fn a_rejected_request_says_the_worker_refused() {
    let mut center = submitted();
    let error = ClientError::Rejected {
        reason: "命令 normalize_media 需要可用的媒体工具链".to_owned(),
    };

    center.apply(finished(SupervisedOutcome::Unavailable(error), 1));

    let detail = center.rows()[0]
        .failure
        .as_ref()
        .expect("a rejected row has a failure");
    assert_eq!(detail.summary, Summary::Key(REJECTED_KEY));
    // The worker wrote the reason for the user; the shell passes it through.
    assert_eq!(detail.detail, "命令 normalize_media 需要可用的媒体工具链");
}

#[test]
fn an_unsupported_command_says_this_build_lacks_it() {
    let mut center = submitted();
    let error = ClientError::UnsupportedCommand {
        requested: "extract_features",
        supported: vec!["validate_project"],
    };
    let text = error.to_string();

    center.apply(finished(SupervisedOutcome::Unavailable(error), 1));

    let detail = center.rows()[0]
        .failure
        .as_ref()
        .expect("an unsupported row has a failure");
    assert_eq!(detail.summary, Summary::Key(UNSUPPORTED_KEY));
    assert_eq!(detail.detail, text);
}

#[test]
fn the_latest_row_of_a_kind_is_the_newest_one() {
    let mut center = TaskCenter::default();
    for (id, kind) in [
        ("1756000000000-00000001", TaskKind::NormalizeMedia),
        ("1756000000001-00000002", TaskKind::ExtractFrames),
        ("1756000000002-00000003", TaskKind::NormalizeMedia),
    ] {
        let task_id = TaskId::parse(id).expect("a well formed task id");
        center.begin(task_id, kind, CancelToken::new());
    }

    let latest = latest_row(center.rows(), TaskKind::NormalizeMedia)
        .expect("this session ran a normalisation");

    assert_eq!(latest.task_id.as_str(), "1756000000002-00000003");
    assert!(
        latest_row(center.rows(), TaskKind::Train).is_none(),
        "a command this session never ran has no row"
    );
}

#[test]
fn a_cancelled_report_marks_the_row_cancelled() {
    let mut center = submitted();

    center.apply(finished(SupervisedOutcome::Cancelled, 1));

    let row = &center.rows()[0];
    assert_eq!(row.status, TaskStatus::Cancelled);
    assert_eq!(stage_key(&row.stage), "task.stage.cancelled");
    assert!(row.failure.is_none());
    assert!(!center.is_busy());
}

#[test]
fn a_restart_shows_up_as_two_attempts() {
    let mut center = submitted();

    center.apply(finished(SupervisedOutcome::Completed { result: None }, 2));

    assert_eq!(center.rows()[0].attempts, 2);
}

#[test]
fn a_lost_log_and_a_journal_error_become_notes() {
    let mut center = submitted();
    let written = PathBuf::from("C:/projects/demo/logs/1756000000000-00000001-1.json");

    center.apply(TaskUpdate::Finished(Box::new(SupervisionReport {
        task_id: task_id(),
        attempts: 2,
        outcome: SupervisedOutcome::Completed { result: None },
        crash_logs: vec![
            CrashLogOutcome::Written(written.clone()),
            CrashLogOutcome::Failed {
                reason: "access is denied".to_owned(),
            },
        ],
        journal_errors: vec!["the manifest is read-only".to_owned()],
    })));

    let row = &center.rows()[0];
    assert_eq!(row.crash_logs, vec![written]);
    // The task completed; neither a lost log nor an unwritten history hides that.
    assert_eq!(row.status, TaskStatus::Completed);
    let notes = center.notes();
    assert_eq!(notes.len(), 2);
    assert_eq!(notes[0].key, "tasks.note.log_failed");
    assert_eq!(notes[0].detail.as_deref(), Some("access is denied"));
    assert_eq!(notes[1].key, "tasks.note.journal_failed");
    assert_eq!(
        notes[1].detail.as_deref(),
        Some("the manifest is read-only")
    );
}

#[test]
fn an_update_for_an_unknown_task_changes_nothing() {
    let mut center = TaskCenter::default();

    center.apply(progress(TaskStage::Preparing));
    center.apply(finished(SupervisedOutcome::Completed { result: None }, 1));

    assert!(center.rows().is_empty());
    assert!(!center.is_busy());
}

#[test]
fn the_startup_scan_is_resolved_one_row_at_a_time() {
    let mut center = TaskCenter::default();

    center.adopt_scan(vec![
        incomplete("1756000000000-00000001", TaskHistoryStatus::Running),
        incomplete("1756000000001-00000002", TaskHistoryStatus::Queued),
    ]);

    assert_eq!(center.incomplete().len(), 2);
    // A row from the last run is not a running task: the gate stays open.
    assert!(!center.is_busy());

    center.resolve("1756000000000-00000001");

    assert_eq!(center.incomplete().len(), 1);
    assert_eq!(center.incomplete()[0].task_id, "1756000000001-00000002");
}

#[test]
fn technical_detail_starts_collapsed_and_toggles() {
    let mut center = submitted();
    center.apply(finished(SupervisedOutcome::Failed(failure()), 1));

    assert!(
        !center.rows()[0].detail_open,
        "raw diagnostics do not open themselves"
    );

    center.toggle_detail("1756000000000-00000001");
    assert!(center.rows()[0].detail_open);

    center.toggle_detail("1756000000000-00000001");
    assert!(!center.rows()[0].detail_open);

    // An unknown id is a no-op, like every other update for a row that is gone.
    center.toggle_detail("1756000000009-00000009");
    assert!(!center.rows()[0].detail_open);
}

#[test]
fn the_submit_gate_names_the_first_blocking_reason() {
    assert_eq!(
        blocked_key(false, false, false),
        Some("tasks.blocked.no_project")
    );
    assert_eq!(
        blocked_key(true, false, false),
        Some("tasks.blocked.no_worker")
    );
    assert_eq!(blocked_key(true, true, true), Some("tasks.blocked.busy"));
    assert_eq!(blocked_key(true, true, false), None);
}

#[test]
fn stopping_asks_every_unfinished_task_once() {
    let tokens = [CancelToken::new(), CancelToken::new(), CancelToken::new()];
    let mut center = TaskCenter::default();
    for (index, (id, kind)) in [
        ("1756000000000-00000001", TaskKind::ExtractFrames),
        ("1756000000001-00000002", TaskKind::Train),
        ("1756000000002-00000003", TaskKind::Render),
    ]
    .into_iter()
    .enumerate()
    {
        let task_id = TaskId::parse(id).expect("a well formed task id");
        center.begin(task_id, kind, tokens[index].clone());
    }
    let ended = TaskId::parse("1756000000000-00000001").expect("a well formed task id");
    center.apply(finished_for(
        ended,
        SupervisedOutcome::Completed { result: None },
        1,
    ));

    let asked = center.request_stop();

    assert_eq!(asked, 2, "the two that are still running");
    assert_eq!(
        tokens[0].count(),
        0,
        "a task that has already ended has nobody to ask"
    );
    assert_eq!(tokens[1].count(), 1);
    assert_eq!(tokens[2].count(), 1);
}

/// One request, not two. The count is the difference between stopping a run and
/// killing it: the first press asks the worker to save a checkpoint and end, the
/// second ends the process where it stands.
#[test]
fn stopping_twice_would_kill_so_it_only_happens_once() {
    let token = CancelToken::new();
    let mut center = TaskCenter::default();
    center.begin(task_id(), TaskKind::Train, token.clone());

    assert_eq!(center.request_stop(), 1);

    assert_eq!(token.count(), 1, "asked once, not killed");
}

#[test]
fn stopping_an_empty_centre_asks_nothing() {
    let center = TaskCenter::default();

    assert_eq!(center.request_stop(), 0);
}
