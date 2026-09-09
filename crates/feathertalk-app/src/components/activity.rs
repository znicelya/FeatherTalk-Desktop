//! What the last task of one kind means for the thing showing it.
//!
//! The asset page's four step rows asked this first -- is anything happening to
//! this command right now, and if it ended, did it end badly -- and the training
//! page asks the same about `Train`. The answer is a judgement about task history
//! rather than about steps, so it is made once here instead of once per page.

use feathertalk_domain::{TaskKind, TaskStatus};

use crate::tasks::{TaskRow, latest_row};

/// What the last task of one kind is owed on screen.
pub(crate) enum Activity<'a> {
    /// Queued or running: the stage, a bar, and a way to stop it.
    Live(&'a TaskRow),
    /// The last attempt failed: one sentence and where to read the rest.
    Broken(&'a TaskRow),
    /// Nothing to add; what is on disk is the whole story.
    Quiet,
}

/// Find the newest task of `kind` and decide what it is owed.
pub(crate) fn activity(rows: &[TaskRow], kind: TaskKind) -> Activity<'_> {
    match latest_row(rows, kind) {
        Some(row) => match row.status {
            TaskStatus::Queued | TaskStatus::Running => Activity::Live(row),
            TaskStatus::Failed => Activity::Broken(row),
            // A finished or cancelled task has already changed the disk and the
            // snapshot has already been retaken, so the snapshot is newer than
            // anything the row could say.
            TaskStatus::Completed | TaskStatus::Cancelled => Activity::Quiet,
        },
        None => Activity::Quiet,
    }
}
