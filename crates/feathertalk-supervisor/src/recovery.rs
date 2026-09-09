use std::path::Path;

use feathertalk_domain::TaskKind;
use feathertalk_project::{
    ProjectManifest, TaskHistoryStatus, read_project_manifest, write_project_manifest_atomic,
};
use time::OffsetDateTime;
use time::format_description::well_known::Rfc3339;

use crate::error::SupervisorError;
use crate::status::task_status;

/// One task that was still open when the application last stopped.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IncompleteTask {
    pub task_id: String,
    /// `None` when the manifest names a command this build does not know. The
    /// entry is still listed: a slug from a newer version is not a reason to hide
    /// unfinished work, or to fail the whole manifest.
    pub kind: Option<TaskKind>,
    pub status: TaskHistoryStatus,
    pub updated_at: String,
}

/// What the user decided about one unfinished task.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Resolution {
    /// Put it back in the queue so the interface can re-issue the command,
    /// continuing from the last checkpoint where the command supports it.
    Resume,
    /// Give up on it and mark it cancelled.
    Discard,
}

/// Every unfinished task in one manifest, oldest first.
///
/// "Unfinished" is `TaskStatus::is_incomplete`, the predicate the protocol crate
/// defines for exactly this scan, rather than a second list of statuses that
/// could drift from it. Task ids sort as time order, so `sort` is enough.
pub fn scan_incomplete(manifest: &ProjectManifest) -> Vec<IncompleteTask> {
    let mut incomplete: Vec<IncompleteTask> = manifest
        .task_history
        .iter()
        .filter(|entry| task_status(entry.status.clone()).is_incomplete())
        .map(|entry| IncompleteTask {
            task_id: entry.task_id.clone(),
            kind: TaskKind::from_slug(&entry.kind),
            status: entry.status.clone(),
            updated_at: entry.updated_at.clone(),
        })
        .collect();
    incomplete.sort_by(|left, right| left.task_id.cmp(&right.task_id));
    incomplete
}

/// Apply the user's decisions to `project.json` in one atomic write.
///
/// Every resolution is checked before anything is written. A half-applied set of
/// decisions would be harder to explain to the user than a rejected set, and a
/// stale id — say a task another process finished in the meantime — is a sign the
/// list on screen is out of date rather than something to silently absorb.
pub fn apply_resolutions(
    project_dir: &Path,
    resolutions: &[(String, Resolution)],
    now: OffsetDateTime,
) -> Result<(), SupervisorError> {
    if resolutions.is_empty() {
        return Ok(());
    }
    let updated_at = now.format(&Rfc3339)?;
    let manifest_path = project_dir.join("project.json");
    let mut manifest = read_project_manifest(&manifest_path)?;

    for (task_id, _) in resolutions {
        match manifest
            .task_history
            .iter()
            .find(|entry| &entry.task_id == task_id)
        {
            None => {
                return Err(SupervisorError::UnknownTask {
                    task_id: task_id.clone(),
                });
            }
            Some(entry) if !task_status(entry.status.clone()).is_incomplete() => {
                return Err(SupervisorError::AlreadyFinished {
                    task_id: task_id.clone(),
                });
            }
            Some(_) => {}
        }
    }

    for (task_id, resolution) in resolutions {
        if let Some(entry) = manifest
            .task_history
            .iter_mut()
            .find(|entry| &entry.task_id == task_id)
        {
            entry.status = match resolution {
                Resolution::Resume => TaskHistoryStatus::Queued,
                Resolution::Discard => TaskHistoryStatus::Cancelled,
            };
            entry.updated_at = updated_at.clone();
        }
    }

    write_project_manifest_atomic(&manifest_path, &manifest)?;
    Ok(())
}

/// Every unfinished task in the project at `project_dir`, oldest first.
///
/// [`scan_incomplete`] takes a manifest the caller already read, while
/// [`apply_resolutions`] reads `project.json` itself. This closes the asymmetry:
/// a caller that only wants the list would otherwise need `feathertalk-project`
/// in its dependency graph for one read.
pub fn scan_project(project_dir: &Path) -> Result<Vec<IncompleteTask>, SupervisorError> {
    let manifest = read_project_manifest(&project_dir.join("project.json"))?;
    Ok(scan_incomplete(&manifest))
}

/// Apply `resolutions` to the project at `project_dir`, stamped now.
///
/// [`apply_resolutions`] takes the timestamp so tests can pin it. Callers that
/// are not tests have no reason to own a clock, and asking an interface for one
/// buys nothing but a dependency on `time`.
pub fn resolve_project(
    project_dir: &Path,
    resolutions: &[(String, Resolution)],
) -> Result<(), SupervisorError> {
    apply_resolutions(project_dir, resolutions, OffsetDateTime::now_utc())
}
