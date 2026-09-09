use feathertalk_app::ui::{ProgressPresentation, TaskFilter, progress_presentation};
use feathertalk_domain::{Progress, TaskStatus};

#[test]
fn task_filters_include_only_their_relevant_statuses() {
    for status in TaskStatus::ALL {
        assert!(TaskFilter::All.includes(status));
        assert_eq!(
            TaskFilter::Active.includes(status),
            matches!(status, TaskStatus::Queued | TaskStatus::Running)
        );
        assert_eq!(
            TaskFilter::Completed.includes(status),
            status == TaskStatus::Completed
        );
        assert_eq!(
            TaskFilter::Failed.includes(status),
            status == TaskStatus::Failed
        );
        assert_eq!(
            TaskFilter::Cancelled.includes(status),
            status == TaskStatus::Cancelled
        );
    }
}

#[test]
fn terminal_tasks_never_keep_an_indeterminate_progress_animation() {
    let uncounted = Some(Progress {
        completed: 4,
        total: None,
    });
    assert_eq!(
        progress_presentation(TaskStatus::Running, uncounted),
        ProgressPresentation::Indeterminate
    );
    assert_eq!(
        progress_presentation(TaskStatus::Completed, uncounted),
        ProgressPresentation::Complete
    );
    for status in [TaskStatus::Failed, TaskStatus::Cancelled] {
        assert_eq!(
            progress_presentation(status, uncounted),
            ProgressPresentation::Empty
        );
    }
    assert_eq!(
        progress_presentation(
            TaskStatus::Running,
            Some(Progress {
                completed: 3,
                total: Some(10)
            })
        ),
        ProgressPresentation::Counted {
            completed: 3,
            total: 10
        }
    );
}
