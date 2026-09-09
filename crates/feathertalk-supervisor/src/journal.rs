use std::path::{Path, PathBuf};

use feathertalk_domain::{TaskId, TaskKind, TaskStatus};
use feathertalk_project::{
    ProjectManifest, TaskHistoryEntry, read_project_manifest, write_project_manifest_atomic,
};
use time::OffsetDateTime;
use time::format_description::well_known::Rfc3339;

use crate::error::SupervisorError;
use crate::status::{history_status, task_status};

/// How many history entries the supervisor is willing to keep.
///
/// `feathertalk-project` enforces its own, much larger ceiling, but that constant
/// is private. Staying well below it keeps the manifest valid without reaching
/// into another crate's internals, and five hundred finished tasks is already
/// more history than an interface can show usefully.
pub const MAX_JOURNAL_ENTRIES: usize = 500;

/// Writes task status into one project's `project.json`.
///
/// This is the only writer of `task_history` in the workspace. The startup scan
/// depends on it: an entry that was never recorded as running cannot be offered
/// for recovery after a crash.
#[derive(Debug, Clone)]
pub struct TaskJournal {
    manifest_path: PathBuf,
}

impl TaskJournal {
    pub fn new(project_dir: &Path) -> Self {
        Self {
            manifest_path: project_dir.join("project.json"),
        }
    }

    pub fn manifest_path(&self) -> &Path {
        &self.manifest_path
    }

    /// Upsert one task's status.
    ///
    /// Read, edit, atomic write. Both ends of that round trip validate the
    /// manifest, so a record can never leave behind a file the reader rejects.
    pub fn record(
        &self,
        task_id: &TaskId,
        kind: TaskKind,
        status: TaskStatus,
        now: OffsetDateTime,
    ) -> Result<(), SupervisorError> {
        let updated_at = now.format(&Rfc3339)?;
        let mut manifest = read_project_manifest(&self.manifest_path)?;

        match manifest
            .task_history
            .iter_mut()
            .find(|entry| entry.task_id == task_id.as_str())
        {
            Some(existing) => {
                existing.kind = kind.as_slug().to_owned();
                existing.status = history_status(status);
                existing.updated_at = updated_at;
            }
            None => {
                make_room(&mut manifest)?;
                manifest.task_history.push(TaskHistoryEntry {
                    task_id: task_id.as_str().to_owned(),
                    kind: kind.as_slug().to_owned(),
                    status: history_status(status),
                    updated_at,
                });
            }
        }

        write_project_manifest_atomic(&self.manifest_path, &manifest)?;
        Ok(())
    }
}

/// Free one slot by dropping the oldest *finished* entries.
///
/// Unfinished entries are exactly what a startup scan is looking for, so they are
/// never the ones discarded. A history made entirely of unfinished tasks is a
/// real problem — something is starting tasks and never resolving them — and is
/// reported rather than papered over.
fn make_room(manifest: &mut ProjectManifest) -> Result<(), SupervisorError> {
    if manifest.task_history.len() < MAX_JOURNAL_ENTRIES {
        return Ok(());
    }
    manifest
        .task_history
        .sort_by(|left, right| left.task_id.cmp(&right.task_id));
    while manifest.task_history.len() >= MAX_JOURNAL_ENTRIES {
        let oldest_finished = manifest
            .task_history
            .iter()
            .position(|entry| !task_status(entry.status.clone()).is_incomplete());
        match oldest_finished {
            Some(index) => {
                manifest.task_history.remove(index);
            }
            None => {
                return Err(SupervisorError::HistoryFull {
                    limit: MAX_JOURNAL_ENTRIES,
                });
            }
        }
    }
    Ok(())
}
