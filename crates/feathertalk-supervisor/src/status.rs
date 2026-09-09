use feathertalk_domain::TaskStatus;
use feathertalk_project::TaskHistoryStatus;

/// Project the protocol's coarse task status onto the shape `project.json`
/// stores.
///
/// The two enums list the same five states but live in different crates, and
/// `feathertalk-domain` deliberately does not depend on `feathertalk-project`.
/// The bridge therefore belongs here, written as an exhaustive `match` in both
/// directions so that a new state upstream breaks this build instead of quietly
/// landing on the wrong disk value.
pub fn history_status(status: TaskStatus) -> TaskHistoryStatus {
    match status {
        TaskStatus::Queued => TaskHistoryStatus::Queued,
        TaskStatus::Running => TaskHistoryStatus::Running,
        TaskStatus::Completed => TaskHistoryStatus::Completed,
        TaskStatus::Failed => TaskHistoryStatus::Failed,
        TaskStatus::Cancelled => TaskHistoryStatus::Cancelled,
    }
}

/// The inverse of [`history_status`], so `TaskStatus::is_incomplete` stays the
/// single authority on which entries a startup scan has to offer.
pub fn task_status(status: TaskHistoryStatus) -> TaskStatus {
    match status {
        TaskHistoryStatus::Queued => TaskStatus::Queued,
        TaskHistoryStatus::Running => TaskStatus::Running,
        TaskHistoryStatus::Completed => TaskStatus::Completed,
        TaskHistoryStatus::Failed => TaskStatus::Failed,
        TaskHistoryStatus::Cancelled => TaskStatus::Cancelled,
    }
}
