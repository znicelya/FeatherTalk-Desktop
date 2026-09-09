use feathertalk_project::ProjectError;

/// Everything the supervisor itself can refuse to do.
///
/// A worker that fails a task is *not* an error here: that outcome travels in
/// `SupervisionReport`. This enum is only for the supervisor's own bookkeeping —
/// reading and rewriting the project manifest.
///
/// `Display` is English on purpose. The desktop shell renders Chinese copy from
/// these variants, exactly as it already does for `feathertalk-client`.
#[derive(Debug, thiserror::Error)]
pub enum SupervisorError {
    #[error("the project manifest could not be read or written: {0}")]
    Project(#[from] ProjectError),
    #[error("no task history entry matches task id {task_id}")]
    UnknownTask { task_id: String },
    #[error("task {task_id} already reached a final status and cannot be resolved")]
    AlreadyFinished { task_id: String },
    #[error("the task history holds {limit} unfinished entries and cannot take another")]
    HistoryFull { limit: usize },
    #[error("a timestamp could not be formatted as RFC 3339: {0}")]
    Timestamp(#[from] time::error::Format),
}
