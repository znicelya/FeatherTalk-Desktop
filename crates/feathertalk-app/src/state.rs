//! The shell's global state.
//!
//! State lives in `Entity<T>` handles rather than locks: the render layer tracks
//! which entities a frame read, so mutating one and calling `cx.notify()` is all
//! an event handler owes the window.

use feathertalk_client::{ComputeOptions, WorkerLocator};
use feathertalk_supervisor::scan_project;
use gpui::{App, AppContext, Entity, Global};

use crate::args::LaunchOptions;
use crate::compute::ComputeState;
use crate::generate::GenerateForm;
use crate::models::ModelForm;
use crate::navigation::Navigation;
use crate::project::{has_task_history, ProjectState};
use crate::tasks::{Note, TaskCenter};
use crate::training::TrainingForm;
use crate::ui::UiState;
use crate::worker_status::WorkerStatus;

/// Everything the workbench shell reads while rendering.
pub struct AppState {
    /// Which of the five pages is showing.
    pub navigation: Entity<Navigation>,
    /// Whether the worker executable the shell would drive exists.
    pub worker: Entity<WorkerStatus>,
    /// Discovered devices and the choice retained for later task submissions.
    pub compute: Entity<ComputeState>,
    /// Submitted tasks, the startup scan, and anything the page has to say.
    pub tasks: Entity<TaskCenter>,
    /// The project this session works on: `--project` at startup, whatever the
    /// asset page's picker chose after that.
    pub project: Entity<ProjectState>,
    /// What the training page's controls currently say.
    pub form: Entity<TrainingForm>,
    /// What the generate page's controls currently say.
    pub generate: Entity<GenerateForm>,
    pub ui: Entity<UiState>,
    pub models: Entity<ModelForm>,
}

impl AppState {
    /// Build the initial state: probe for the worker once, then look for tasks
    /// that were still open when the application last stopped.
    ///
    /// A failed scan is a note rather than a refusal to start. An unreadable
    /// manifest says nothing about whether new work can be submitted, and four of
    /// the five pages do not touch the task history at all.
    pub fn new(cx: &mut App, options: LaunchOptions) -> Self {
        let worker = WorkerStatus::probe(&WorkerLocator::from_env(None));
        let project = ProjectState::opened(options.project_dir);
        let mut center = TaskCenter::default();
        // A directory with no `project.json` is a project with no history yet, not
        // a project whose history could not be read. Only the second one is worth
        // a note, and the picker makes the first one common.
        if let Some(dir) = project.dir().filter(|dir| has_task_history(dir)) {
            match scan_project(dir) {
                Ok(tasks) => center.adopt_scan(tasks),
                Err(error) => {
                    center.note(Note::new("tasks.note.scan_failed", Some(error.to_string())));
                }
            }
        }
        Self {
            navigation: cx.new(|_| Navigation::default()),
            worker: cx.new(|_| worker),
            compute: cx.new(|_| ComputeState::new(ComputeOptions::from_env())),
            tasks: cx.new(|_| center),
            project: cx.new(|_| project),
            form: cx.new(|_| TrainingForm::default()),
            // The preview length is kept here rather than on the project: how
            // long a preview should be is a habit that follows the user, not a
            // property of whichever directory happens to be open.
            generate: cx.new(|_| GenerateForm::default()),
            ui: cx.new(|_| UiState::default()),
            models: cx.new(|_| ModelForm::default()),
        }
    }
}

impl Global for AppState {}
