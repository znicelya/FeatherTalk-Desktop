use feathertalk_domain::{DomainError, ErrorCode, TaskLifecycle, TaskStage};

#[test]
fn a_new_lifecycle_starts_queued_and_is_not_terminal() {
    let lifecycle = TaskLifecycle::new();
    assert_eq!(lifecycle.current(), &TaskStage::Queued);
    assert!(!lifecycle.is_terminal());
}

#[test]
fn a_normal_render_run_advances_through_every_stage() {
    let mut lifecycle = TaskLifecycle::new();
    lifecycle.advance(TaskStage::Preparing).unwrap();
    lifecycle
        .advance(TaskStage::Rendering {
            frame: 1,
            total: 900,
        })
        .unwrap();
    lifecycle
        .advance(TaskStage::Rendering {
            frame: 2,
            total: 900,
        })
        .unwrap();
    lifecycle.advance(TaskStage::Completed).unwrap();
    assert!(lifecycle.is_terminal());
}

#[test]
fn advancing_out_of_a_terminal_stage_is_rejected() {
    let mut lifecycle = TaskLifecycle::new();
    lifecycle.advance(TaskStage::Completed).unwrap();
    assert!(matches!(
        lifecycle.advance(TaskStage::Cancelled),
        Err(DomainError::InvalidTransition {
            from: "completed",
            to: "cancelled"
        })
    ));
    assert_eq!(lifecycle.current(), &TaskStage::Completed);
}

#[test]
fn repeated_cancel_is_idempotent_and_yields_one_cancelled() {
    let mut lifecycle = TaskLifecycle::new();
    lifecycle.advance(TaskStage::Preparing).unwrap();
    assert!(lifecycle.request_cancel().unwrap());
    assert_eq!(lifecycle.current(), &TaskStage::Cancelled);
    for _ in 0..5 {
        assert!(!lifecycle.request_cancel().unwrap());
        assert_eq!(lifecycle.current(), &TaskStage::Cancelled);
    }
}

#[test]
fn cancel_after_completion_does_not_overwrite_the_outcome() {
    let mut lifecycle = TaskLifecycle::new();
    lifecycle.advance(TaskStage::Completed).unwrap();
    assert!(!lifecycle.request_cancel().unwrap());
    assert_eq!(lifecycle.current(), &TaskStage::Completed);

    let mut failed = TaskLifecycle::new();
    failed
        .advance(TaskStage::Failed {
            code: ErrorCode::DiskSpaceLow,
            message: "磁盘空间不足".to_owned(),
        })
        .unwrap();
    assert!(!failed.request_cancel().unwrap());
    assert!(matches!(failed.current(), TaskStage::Failed { .. }));
}

#[test]
fn queued_cannot_be_re_entered() {
    let mut lifecycle = TaskLifecycle::new();
    lifecycle.advance(TaskStage::Preparing).unwrap();
    assert!(matches!(
        lifecycle.advance(TaskStage::Queued),
        Err(DomainError::InvalidTransition {
            from: "preparing",
            to: "queued"
        })
    ));
}
