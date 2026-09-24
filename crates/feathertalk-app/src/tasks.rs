//! What the task page knows.
//!
//! The five statuses and the thirteen stages are the protocol's own, so this
//! module invents no enum of its own: it maps them to catalog keys and folds
//! updates into one row per task. No `gpui` types, so the state machine is
//! testable without a window.

use std::path::PathBuf;

use feathertalk_client::{CancelToken, ClientError};
use feathertalk_domain::{
    ErrorCode, Event, Progress, Recovery, TaskId, TaskKind, TaskStage, TaskStatus,
};
use feathertalk_supervisor::crash_log::CrashLogOutcome;
use feathertalk_supervisor::recovery::IncompleteTask;
use feathertalk_supervisor::supervisor::{SupervisedOutcome, SupervisionReport};

use crate::pipeline::TaskUpdate;
use crate::training_progress::TrainingProgress;

/// The summary for a worker that died without saying why.
pub const CRASHED_KEY: &str = "tasks.failure.crashed";

/// The summary for an installation that no restart will fix.
pub const UNAVAILABLE_KEY: &str = "tasks.failure.unavailable";

/// The summary for a worker that ran and refused the command.
pub const REJECTED_KEY: &str = "tasks.failure.rejected";

/// The summary for a command this worker build never advertised.
pub const UNSUPPORTED_KEY: &str = "tasks.failure.unsupported";

/// The catalog key of one task status.
pub fn status_key(status: TaskStatus) -> &'static str {
    match status {
        TaskStatus::Queued => "task.status.queued",
        TaskStatus::Running => "task.status.running",
        TaskStatus::Completed => "task.status.completed",
        TaskStatus::Failed => "task.status.failed",
        TaskStatus::Cancelled => "task.status.cancelled",
    }
}

/// The catalog key of one task stage.
pub fn stage_key(stage: &TaskStage) -> &'static str {
    match stage {
        TaskStage::Queued => "task.stage.queued",
        TaskStage::Preparing => "task.stage.preparing",
        TaskStage::ExtractingAudio => "task.stage.extracting_audio",
        TaskStage::ExtractingFrames => "task.stage.extracting_frames",
        TaskStage::DetectingFaces => "task.stage.detecting_faces",
        TaskStage::ExtractingFeatures => "task.stage.extracting_features",
        TaskStage::Training { .. } => "task.stage.training",
        TaskStage::Importing => "task.stage.importing",
        TaskStage::Exporting => "task.stage.exporting",
        TaskStage::Rendering { .. } => "task.stage.rendering",
        TaskStage::Completed => "task.stage.completed",
        TaskStage::Failed { .. } => "task.stage.failed",
        TaskStage::Cancelled => "task.stage.cancelled",
    }
}

/// The catalog key of one command name.
pub fn kind_key(kind: TaskKind) -> &'static str {
    match kind {
        TaskKind::ProbeMedia => "task.kind.probe_media",
        TaskKind::NormalizeMedia => "task.kind.normalize_media",
        TaskKind::ValidateProject => "task.kind.validate_project",
        TaskKind::LockAssetPackage => "task.kind.lock_asset_package",
        TaskKind::ExtractFrames => "task.kind.extract_frames",
        TaskKind::ExtractFeatures => "task.kind.extract_features",
        TaskKind::Train => "task.kind.train",
        TaskKind::Render => "task.kind.render",
        TaskKind::InspectModel => "task.kind.inspect_model",
        TaskKind::ImportLegacyModel => "task.kind.import_legacy_model",
        TaskKind::ExportModelPackage => "task.kind.export_model_package",
        TaskKind::ExportOnnx => "task.kind.export_onnx",
        TaskKind::MigrateLegacyFeatures => "task.kind.migrate_legacy_features",
    }
}

/// The catalog key of one recovery suggestion.
pub fn recovery_key(recovery: Recovery) -> &'static str {
    match recovery {
        Recovery::Retry => "task.recovery.retry",
        Recovery::ResumeFromCheckpoint => "task.recovery.resume_from_checkpoint",
        Recovery::FreeDiskSpace => "task.recovery.free_disk_space",
        Recovery::SelectDifferentAdapter => "task.recovery.select_different_adapter",
        Recovery::ExcludeBadFrames => "task.recovery.exclude_bad_frames",
        Recovery::ReimportModel => "task.recovery.reimport_model",
        Recovery::NotRecoverable => "task.recovery.not_recoverable",
    }
}

/// Where the user-facing summary of a failure comes from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Summary {
    /// Text the worker wrote for the user, already in Chinese.
    Worker(String),
    /// A catalog key, for a failure the worker never got to describe.
    Key(&'static str),
}

/// A failure in the four parts the migration design asks for: a readable
/// summary, technical detail, the stage it happened in, and what to try next.
///
/// The stage lives on [`TaskRow`] rather than here: a row shows one stage, and
/// keeping two of them invites the page to paint the wrong one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Failure {
    pub summary: Summary,
    /// English. It comes from `ClientError` or the worker's own `detail`, and
    /// rewriting raw diagnostics only gets in the way of whoever debugs this.
    pub detail: String,
    pub code: Option<ErrorCode>,
    pub recovery: Option<Recovery>,
}

/// One line of the task list.
#[derive(Debug, Clone)]
pub struct TaskRow {
    pub task_id: TaskId,
    pub kind: TaskKind,
    pub status: TaskStatus,
    /// The last event's stage, or the stage the task broke in once it has.
    pub stage: TaskStage,
    pub progress: Option<Progress>,
    /// Submitted epoch target and latest batch, retained after the task ends.
    pub training: Option<TrainingProgress>,
    /// The final worker payload, retained for inspection and export summaries.
    pub result: Option<serde_json::Value>,
    /// Worker processes spent on this task, including the first. Zero until the
    /// report arrives; more than one means the worker was restarted.
    pub attempts: u32,
    pub failure: Option<Failure>,
    /// Crash logs the supervisor managed to write.
    pub crash_logs: Vec<PathBuf>,
    /// Whether the technical detail is expanded. Collapsed by default: raw
    /// diagnostics are for whoever debugs this, not for whoever is waiting.
    pub detail_open: bool,
    /// Asking once is polite, twice kills the worker — the client's rule, not
    /// the page's.
    pub cancel: CancelToken,
}

/// Something the page has to say that belongs to no single task.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Note {
    pub key: &'static str,
    /// English technical detail, when there is one.
    pub detail: Option<String>,
}

impl Note {
    pub fn new(key: &'static str, detail: Option<String>) -> Self {
        Self { key, detail }
    }
}

/// Every task this session knows about, plus what the startup scan found.
#[derive(Debug, Default)]
pub struct TaskCenter {
    rows: Vec<TaskRow>,
    incomplete: Vec<IncompleteTask>,
    notes: Vec<Note>,
}

impl TaskCenter {
    /// Newest first.
    pub fn rows(&self) -> &[TaskRow] {
        &self.rows
    }

    /// What was still open when the application last stopped, oldest first.
    pub fn incomplete(&self) -> &[IncompleteTask] {
        &self.incomplete
    }

    pub fn notes(&self) -> &[Note] {
        &self.notes
    }

    /// Whether a task is still running. One at a time is the smallest honest
    /// reading of "one training or inference task per GPU adapter".
    pub fn is_busy(&self) -> bool {
        self.rows.iter().any(|row| row.status.is_incomplete())
    }

    /// Add the queued row of a task that was just submitted.
    pub fn begin(&mut self, task_id: TaskId, kind: TaskKind, cancel: CancelToken) {
        self.rows.insert(
            0,
            TaskRow {
                task_id,
                kind,
                status: TaskStatus::Queued,
                stage: TaskStage::Queued,
                progress: None,
                training: None,
                result: None,
                attempts: 0,
                failure: None,
                crash_logs: Vec::new(),
                detail_open: false,
                cancel,
            },
        );
    }

    /// Capture training parameters before later edits to the form can change them.
    pub fn begin_training(&mut self, task_id: TaskId, total_epochs: u32, cancel: CancelToken) {
        self.begin(task_id, TaskKind::Train, cancel);
        self.rows[0].training = Some(TrainingProgress::new(total_epochs));
    }

    /// Fold one update into the row it belongs to.
    ///
    /// An update for a task with no row is dropped. `begin` runs on the line
    /// above `submit`, so getting here means a bug, and inventing a row out of
    /// half the information would hide it.
    pub fn apply(&mut self, update: TaskUpdate) {
        match update {
            TaskUpdate::Progress(event) => self.progress(*event),
            TaskUpdate::Finished(report) => self.finish(*report),
        }
    }

    /// Adopt the startup scan's result.
    pub fn adopt_scan(&mut self, tasks: Vec<IncompleteTask>) {
        self.incomplete = tasks;
    }

    /// Switch the visible project only after its previous worker has finished.
    /// Session results and notes belong to that project, just like its history.
    pub fn adopt_project(&mut self, tasks: Vec<IncompleteTask>) -> bool {
        if self.is_busy() {
            return false;
        }
        self.rows.clear();
        self.notes.clear();
        self.incomplete = tasks;
        true
    }

    /// Drop one unfinished entry once its resolution is on disk.
    pub fn resolve(&mut self, task_id: &str) {
        self.incomplete.retain(|task| task.task_id != task_id);
    }

    pub fn note(&mut self, note: Note) {
        self.notes.push(note);
    }

    /// Expand or collapse one row's technical detail.
    pub fn toggle_detail(&mut self, task_id: &str) {
        if let Some(row) = self.row_mut(task_id) {
            row.detail_open = !row.detail_open;
        }
    }

    /// Ask every unfinished task to stop, once.
    ///
    /// Once is the whole contract: the token counts, and a second request kills
    /// the worker instead of letting it publish a checkpoint. The answer is how
    /// many rows were asked, which is all there is to report -- nothing here waits
    /// for the tasks to end, because the caller is the application quitting and
    /// there is nobody left to tell.
    ///
    /// Not only `Train`: an extraction or a render left running past the window's
    /// life is a worker process nobody is watching either.
    pub fn request_stop(&self) -> usize {
        let mut asked = 0;
        for row in self.rows.iter().filter(|row| row.status.is_incomplete()) {
            row.cancel.request();
            asked += 1;
        }
        asked
    }

    fn progress(&mut self, event: Event) {
        let Some(row) = self.row_mut(event.task_id.as_str()) else {
            return;
        };
        if let Some(training) = &mut row.training {
            training.observe(&event);
        }
        // Terminal worker events can precede shutdown/journal completion. Keep
        // admission occupied until the supervisor supplies the final outcome.
        if event.stage.status().is_incomplete() {
            row.status = event.stage.status();
        }
        row.stage = event.stage.clone();
        row.progress = event.progress;
        if let Some(error) = event.error {
            // The event's own stage is the terminal `failed`; the error carries
            // the stage the task was actually in, which is what to show.
            row.stage = error.stage.clone();
            row.failure = Some(Failure {
                summary: Summary::Worker(error.summary),
                detail: error.detail,
                code: Some(error.code),
                recovery: Some(error.recovery),
            });
        }
    }

    fn finish(&mut self, report: SupervisionReport) {
        let SupervisionReport {
            task_id,
            attempts,
            outcome,
            crash_logs,
            journal_errors,
        } = report;
        let mut written = Vec::new();
        for log in crash_logs {
            match log {
                CrashLogOutcome::Written(path) => written.push(path),
                CrashLogOutcome::Failed { reason } => self
                    .notes
                    .push(Note::new("tasks.note.log_failed", Some(reason))),
            }
        }
        // The task's own result is never withheld because the history could not
        // be written, so this is a note rather than a failure.
        for error in journal_errors {
            self.notes
                .push(Note::new("tasks.note.journal_failed", Some(error)));
        }
        let Some(row) = self.row_mut(task_id.as_str()) else {
            return;
        };
        row.attempts = attempts;
        row.crash_logs = written;
        match outcome {
            SupervisedOutcome::Completed { result } => {
                row.status = TaskStatus::Completed;
                row.stage = TaskStage::Completed;
                row.result = result;
            }
            SupervisedOutcome::Cancelled => {
                row.status = TaskStatus::Cancelled;
                row.stage = TaskStage::Cancelled;
            }
            SupervisedOutcome::Failed(error) => {
                row.status = TaskStatus::Failed;
                row.stage = error.stage.clone();
                row.failure = Some(Failure {
                    summary: Summary::Worker(error.summary),
                    detail: error.detail,
                    code: Some(error.code),
                    recovery: Some(error.recovery),
                });
            }
            SupervisedOutcome::Crashed {
                stage,
                recovery,
                detail,
            } => {
                row.status = TaskStatus::Failed;
                row.stage = stage;
                row.failure = Some(Failure {
                    summary: Summary::Key(CRASHED_KEY),
                    detail,
                    code: None,
                    recovery: Some(recovery),
                });
            }
            SupervisedOutcome::Unavailable(error) => {
                row.status = TaskStatus::Failed;
                row.failure = Some(unavailable_failure(&error));
            }
        }
    }

    fn row_mut(&mut self, task_id: &str) -> Option<&mut TaskRow> {
        self.rows
            .iter_mut()
            .find(|row| row.task_id.as_str() == task_id)
    }
}

/// Fold a session-level `ClientError` into what the page shows.
///
/// Two of the nine variants are not "the worker could not start": a rejection
/// means the worker ran and refused the command, and an unsupported command means
/// this worker never advertised it in its `ready` frame. Both are what happens
/// when the media toolchain or a model directory is not configured, and that is a
/// different thing to go fix than a missing executable.
fn unavailable_failure(error: &ClientError) -> Failure {
    let (summary, detail) = match error {
        // The worker wrote this for the user, the way a `TaskError` summary is
        // written for the user.
        ClientError::Rejected { reason } => (Summary::Key(REJECTED_KEY), reason.clone()),
        ClientError::UnsupportedCommand { .. } => {
            (Summary::Key(UNSUPPORTED_KEY), error.to_string())
        }
        ClientError::WorkerNotFound { .. }
        | ClientError::Spawn { .. }
        | ClientError::Handshake { .. }
        | ClientError::ProtocolVersion { .. }
        | ClientError::Protocol(_)
        | ClientError::Io(_)
        | ClientError::WorkerGone { .. } => (Summary::Key(UNAVAILABLE_KEY), error.to_string()),
    };
    Failure {
        summary,
        detail,
        code: None,
        recovery: None,
    }
}

/// The most recent row of `kind`, or `None` when this session has not run it.
///
/// `begin` inserts at the front, so the first match is the newest. The asset page
/// uses it to put one command's progress next to the step that submitted it.
pub fn latest_row(rows: &[TaskRow], kind: TaskKind) -> Option<&TaskRow> {
    rows.iter().find(|row| row.kind == kind)
}

/// Why the submit button is disabled, or `None` when it is not.
///
/// Order is deliberate: without a project directory there is nothing to submit,
/// so that reason outranks a missing worker, and both outrank "busy". The page
/// shows exactly one reason, and showing the one furthest from being fixed is the
/// only ordering that does not send the user chasing the wrong knob.
pub fn blocked_key(has_project: bool, worker_ready: bool, busy: bool) -> Option<&'static str> {
    if !has_project {
        return Some("tasks.blocked.no_project");
    }
    if !worker_ready {
        return Some("tasks.blocked.no_worker");
    }
    if busy {
        return Some("tasks.blocked.busy");
    }
    None
}
