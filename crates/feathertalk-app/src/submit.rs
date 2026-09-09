//! The one path from a button to a supervised worker.
//!
//! Every page's command lands here: the request differs, the plumbing does not.
//! The task page held this inline while it was the only caller; the asset page is
//! the second, so it moved out.

use std::path::Path;

use feathertalk_client::{SessionOptions, WorkerLocator, generate_task_id};
use feathertalk_domain::{Request, TaskKind};
use feathertalk_supervisor::journal::TaskJournal;
use feathertalk_supervisor::policy::RestartPolicy;
use feathertalk_supervisor::runner::WorkerRunner;
use gpui::{App, Entity};

use crate::manifest::{self, Bootstrap};
use crate::pipeline::{self, Job};
use crate::project::ProjectState;
use crate::state::AppState;
use crate::tasks::{Note, TaskCenter};

/// Submit `request` as `kind` and stream its updates into the task centre.
///
/// `project_dir` is where the task history lives and `log_dir` is where crash logs
/// go; both are passed in rather than read back out of `project`, so an enabled
/// button is the only thing that can reach this function and there is no missing
/// project to handle here.
pub fn submit(
    center: &Entity<TaskCenter>,
    project: &Entity<ProjectState>,
    kind: TaskKind,
    request: Request,
    project_dir: &Path,
    log_dir: &Path,
    cx: &mut App,
) {
    // The displayed form can lag behind a queued click or a device refresh.
    // Recheck admission before creating a manifest or starting a worker.
    let state = cx.global::<AppState>();
    let gate = crate::tasks::blocked_key(
        project.read(cx).dir() == Some(project_dir),
        state.worker.read(cx).is_ready(),
        center.read(cx).is_busy(),
    )
    .or_else(|| state.compute.read(cx).command_blocked_key(kind));
    if let Some(key) = gate {
        center.update(cx, |center, cx| {
            center.note(Note::new(key, None));
            cx.notify();
        });
        return;
    }
    // Snapshot the current selection once. WorkerRunner retains these overrides
    // across retries, even if the user selects another device for later tasks.
    let compute = cx.global::<AppState>().compute.clone();
    let env = match compute.read(cx).environment_for(kind) {
        Ok(env) => env,
        Err(error) => return note(center, "compute.invalid", error, cx),
    };
    let task_id = match generate_task_id() {
        Ok(task_id) => task_id,
        Err(error) => return note(center, "tasks.note.task_id_failed", error.to_string(), cx),
    };
    // A task that cannot be recorded still has to run: the manifest is what the
    // journal and the training admission need, not what the worker needs, so a
    // missing history is worth a note rather than a refused button. Every command
    // goes through here, which is why the first one a project sees is enough.
    match manifest::ensure_manifest(project_dir) {
        Ok(Bootstrap::Present | Bootstrap::Created) => {}
        Err(error) => note(center, "tasks.note.manifest_failed", error.to_string(), cx),
    }
    let runner =
        WorkerRunner::new(WorkerLocator::from_env(None), SessionOptions::default()).with_env(env);
    let training_epochs = match &request {
        Request::Train(params) => Some(params.epochs),
        _ => None,
    };
    let job = Job {
        task_id: task_id.clone(),
        kind,
        request,
        policy: RestartPolicy::default(),
        log_dir: log_dir.to_path_buf(),
        journal: Some(TaskJournal::new(project_dir)),
    };
    let submission = match pipeline::submit(runner, job) {
        Ok(submission) => submission,
        Err(error) => return note(center, "tasks.note.submit_failed", error.to_string(), cx),
    };
    // The row exists before the first update can arrive, which is what lets
    // `TaskCenter::apply` treat an unknown task as a bug rather than a race.
    center.update(cx, |center, cx| {
        if let Some(epochs) = training_epochs {
            center.begin_training(task_id, epochs, submission.cancel.clone());
        } else {
            center.begin(task_id, kind, submission.cancel.clone());
        }
        cx.notify();
    });
    let entity = center.clone();
    let selection = project.clone();
    let updates = submission.updates;
    cx.spawn(async move |cx| {
        while let Ok(update) = updates.recv().await {
            // The entity is gone once the window closes; the supervision thread
            // keeps running to its own end.
            if entity
                .update(cx, |center, cx| {
                    center.apply(update);
                    cx.notify();
                })
                .is_err()
            {
                break;
            }
        }
        // The task is over, so whatever it wrote to the project is on disk now.
        // This is the only place the asset page and the task page touch: the task
        // page does not need the re-survey, the asset page does, and neither one
        // polls the filesystem.
        let _ = selection.update(cx, |project, cx| {
            project.refresh();
            cx.notify();
        });
    })
    .detach();
}

/// Record one note on the task centre and repaint.
pub fn note(center: &Entity<TaskCenter>, key: &'static str, detail: String, cx: &mut App) {
    center.update(cx, |center, cx| {
        center.note(Note::new(key, Some(detail)));
        cx.notify();
    });
}
