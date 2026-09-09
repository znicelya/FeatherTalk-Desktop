use std::path::Path;

use feathertalk_app::{
    pipeline::TaskUpdate, tasks::TaskCenter, training::TrainingForm,
    training_progress::TrainingProgress,
};
use feathertalk_client::CancelToken;
use feathertalk_domain::{
    ErrorCode, Event, Progress, Request, TaskError, TaskId, TaskStage, TaskStatus,
};
use feathertalk_supervisor::supervisor::{SupervisedOutcome, SupervisionReport};

fn task_id() -> TaskId {
    TaskId::parse("1756000000000-00000001").unwrap()
}

fn event(epoch: u32, step: u64, total: Option<u64>) -> Event {
    let mut event = Event::new(
        task_id(),
        "2026-09-08T00:00:00Z",
        TaskStage::Training {
            epoch,
            step,
            loss: 0.25,
        },
    );
    // Match the worker's clamping: conversion must use stage.step so a bad
    // counter is unknown instead of being disguised as a complete epoch.
    event.progress = Some(Progress {
        completed: total.map_or(step, |total| step.min(total)),
        total,
    });
    event
}

fn counted(completed: u64, total: u64) -> Option<Progress> {
    Some(Progress {
        completed,
        total: Some(total),
    })
}

#[test]
fn epoch_boundaries_keep_the_last_batch_in_the_epoch_that_produced_it() {
    let mut progress = TrainingProgress::new(200);
    for (epoch, step, displayed_epoch, epoch_step) in [
        (0, 1, 1, 1),
        (0, 7929, 1, 7929),
        (1, 7930, 2, 1),
        (199, 1_585_800, 200, 7929),
    ] {
        progress.observe(&event(epoch, step, Some(1_585_800)));
        assert_eq!(progress.epoch(), Some(displayed_epoch));
        assert_eq!(progress.steps(), counted(epoch_step, 7929));
    }
}

#[test]
fn batched_progress_counts_optimizer_steps_including_the_tail_batch() {
    let mut progress = TrainingProgress::new(200);
    // 7,929 frames at batch size 4: 1,982 full batches plus one tail batch.
    for (epoch, step, expected) in [(0, 1983, 1983), (1, 1984, 1), (1, 2106, 123)] {
        progress.observe(&event(epoch, step, Some(396_600)));
        assert_eq!(progress.steps(), counted(expected, 1983));
    }
    // Three frames, batch size 2 and two epochs gives four optimizer steps.
    let mut small = TrainingProgress::new(2);
    small.observe(&event(1, 4, Some(4)));
    assert_eq!(small.steps(), counted(2, 2));
}

#[test]
fn temporal_progress_uses_the_worker_total_for_frame_pairs() {
    let mut progress = TrainingProgress::new(200);
    // 7,929 frames yield 7,928 temporal pairs at batch size 1.
    progress.observe(&event(1, 7929, Some(1_585_600)));
    assert_eq!(progress.steps(), counted(1, 7928));
}

#[test]
fn a_single_batch_epoch_is_full_instead_of_zero() {
    let mut progress = TrainingProgress::new(200);
    for (epoch, step) in [(0, 1), (1, 2), (199, 200)] {
        progress.observe(&event(epoch, step, Some(200)));
        assert_eq!(progress.steps(), counted(1, 1));
    }
}

#[test]
fn resume_can_start_mid_epoch_and_retry_can_return_to_an_older_checkpoint() {
    let mut progress = TrainingProgress::new(200);
    progress.observe(&event(1, 8052, Some(1_585_800)));
    assert_eq!(progress.epoch(), Some(2));
    assert_eq!(progress.steps(), counted(123, 7929));
    progress.observe(&event(0, 7800, Some(1_585_800)));
    assert_eq!(progress.epoch(), Some(1));
    assert_eq!(progress.steps(), counted(7800, 7929));
}

#[test]
fn no_position_is_invented_before_the_first_training_event() {
    let mut progress = TrainingProgress::new(200);
    let mut preparing = Event::new(task_id(), "2026-09-08T00:00:00Z", TaskStage::Preparing);
    preparing.progress = counted(1, 2);
    progress.observe(&preparing);
    progress.observe(&Event::new(
        task_id(),
        "2026-09-08T00:00:01Z",
        TaskStage::Completed,
    ));
    assert_eq!(progress.epoch(), None);
    assert_eq!(progress.steps(), None);
}

#[test]
fn missing_or_inconsistent_totals_do_not_become_per_epoch_counts() {
    for (epochs, epoch, step, total) in [
        (200, 1, 8052, None),
        (200, 1, 8052, Some(0)),
        (200, 1, 8052, Some(1_585_801)),
        (0, 0, 1, Some(1)),
        (200, 200, 1_585_801, Some(1_585_800)),
        (200, 1, 7928, Some(1_585_800)),
        (200, 1, 7929, Some(1_585_800)),
        (200, 0, 7930, Some(1_585_800)),
        (200, 199, 1_585_801, Some(1_585_800)),
        (200, 0, 0, Some(1_585_800)),
        (1, u32::MAX, u64::MAX, Some(u64::MAX)),
    ] {
        let mut progress = TrainingProgress::new(epochs);
        progress.observe(&event(epoch, step, total));
        assert_eq!(
            progress.steps(),
            None,
            "{epochs}, {epoch}, {step}, {total:?}"
        );
    }
    let mut progress = TrainingProgress::new(200);
    progress.observe(&event(1, 8052, Some(1_585_800)));
    let mut without_progress = event(1, 8053, None);
    without_progress.progress = None;
    progress.observe(&without_progress);
    assert_eq!(progress.epoch(), Some(2));
    assert_eq!(progress.steps(), None, "do not reuse a stale total");
}

#[test]
fn large_valid_counters_are_converted_without_overflow() {
    let mut progress = TrainingProgress::new(u32::MAX);
    progress.observe(&event(u32::MAX - 1, u64::MAX, Some(u64::MAX)));
    assert_eq!(progress.epoch(), Some(u32::MAX));
    assert_eq!(progress.steps(), counted(4_294_967_297, 4_294_967_297));
}

#[test]
fn terminal_events_and_reports_retain_the_last_training_position() {
    let failure = TaskError::new(
        ErrorCode::WorkerCrashed,
        "训练中断",
        "worker stopped",
        TaskStage::Preparing,
    );
    for (stage, outcome, status) in [
        (
            TaskStage::Completed,
            SupervisedOutcome::Completed { result: None },
            TaskStatus::Completed,
        ),
        (
            TaskStage::Cancelled,
            SupervisedOutcome::Cancelled,
            TaskStatus::Cancelled,
        ),
        (
            TaskStage::Failed {
                code: ErrorCode::WorkerCrashed,
                message: "训练中断".into(),
            },
            SupervisedOutcome::Failed(failure),
            TaskStatus::Failed,
        ),
    ] {
        let mut center = TaskCenter::default();
        center.begin_training(task_id(), 200, CancelToken::new());
        center.apply(TaskUpdate::Progress(Box::new(event(
            1,
            8052,
            Some(1_585_800),
        ))));
        center.apply(TaskUpdate::Progress(Box::new(Event::new(
            task_id(),
            "2026-09-08T00:00:01Z",
            stage,
        ))));
        center.apply(TaskUpdate::Finished(Box::new(SupervisionReport {
            task_id: task_id(),
            attempts: 1,
            outcome,
            crash_logs: Vec::new(),
            journal_errors: Vec::new(),
        })));
        let row = &center.rows()[0];
        assert_eq!(row.status, status);
        assert!(row.progress.is_none());
        let progress = row.training.as_ref().unwrap();
        assert_eq!(progress.epoch(), Some(2));
        assert_eq!(progress.steps(), counted(123, 7929));
    }
}

#[test]
fn the_epoch_denominator_belongs_to_the_submitted_request() {
    let mut form = TrainingForm::default();
    let Request::Train(params) = form.request(Path::new("project")) else {
        unreachable!()
    };
    let mut center = TaskCenter::default();
    center.begin_training(task_id(), params.epochs, CancelToken::new());
    form.set_epochs(400.);
    form.set_batch_size(4.);
    center.apply(TaskUpdate::Progress(Box::new(event(
        1,
        8052,
        Some(1_585_800),
    ))));
    let progress = center.rows()[0].training.as_ref().unwrap();
    assert_eq!(progress.total_epochs(), 200);
    assert_eq!(progress.steps(), counted(123, 7929));
}
