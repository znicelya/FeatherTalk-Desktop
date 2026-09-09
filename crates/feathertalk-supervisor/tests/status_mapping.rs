use feathertalk_domain::TaskStatus;
use feathertalk_project::TaskHistoryStatus;
use feathertalk_supervisor::status::{history_status, task_status};

#[test]
fn the_five_statuses_round_trip_both_ways() {
    for status in TaskStatus::ALL {
        assert_eq!(
            task_status(history_status(status)),
            status,
            "{status:?} did not survive the round trip"
        );
    }
}

#[test]
fn the_incomplete_predicate_survives_the_mapping() {
    for status in TaskStatus::ALL {
        let mapped = task_status(history_status(status));
        assert_eq!(
            mapped.is_incomplete(),
            status.is_incomplete(),
            "{status:?} changed its incompleteness"
        );
    }
    assert!(task_status(TaskHistoryStatus::Queued).is_incomplete());
    assert!(task_status(TaskHistoryStatus::Running).is_incomplete());
    assert!(!task_status(TaskHistoryStatus::Completed).is_incomplete());
    assert!(!task_status(TaskHistoryStatus::Failed).is_incomplete());
    assert!(!task_status(TaskHistoryStatus::Cancelled).is_incomplete());
}
