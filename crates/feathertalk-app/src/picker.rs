//! The file dialogs the asset and generate pages open.
//!
//! `prompt_for_paths` answers on a oneshot channel wrapped twice: the outer layer
//! is whether the channel survived, the inner one is what the dialog reported,
//! and only then comes `None` for "the user closed it". All four cases are
//! handled here so no caller has to.
//! `prompt_for_new_path` has one fewer layer: it answers `Result<Option<PathBuf>>`
//! because there is no list to choose from.
//!
//! These dialogs need a window, so they have no unit tests, same as `state` and
//! `components`.

use std::path::{Path, PathBuf};

use feathertalk_supervisor::scan_project;
use gpui::{App, AsyncApp, Entity, PathPromptOptions};

use crate::generate::{self, PickedCheckpoint};
use crate::project::{ProjectState, has_task_history};
use crate::state::AppState;
use crate::tasks::{Note, TaskCenter};

/// The note a generate-page dialog failure records.
const GENERATE_PICK_FAILED: &str = "generate.note.pick_failed";

/// Native dialogs are asynchronous; their answers belong to the project that
/// opened them, and must not change inputs after a task has started.
pub(crate) struct PickerContext {
    project: Entity<ProjectState>,
    center: Entity<TaskCenter>,
    origin: Option<PathBuf>,
}

impl PickerContext {
    pub(crate) fn current(cx: &App) -> Option<Self> {
        Self::capture(&cx.global::<AppState>().project, cx)
    }

    fn capture(project: &Entity<ProjectState>, cx: &App) -> Option<Self> {
        let center = cx.global::<AppState>().tasks.clone();
        if center.read(cx).is_busy() {
            return None;
        }
        Some(Self {
            project: project.clone(),
            center,
            origin: project.read(cx).dir().map(Path::to_path_buf),
        })
    }

    pub(crate) fn is_current(&self, cx: &mut AsyncApp) -> bool {
        self.project
            .update(cx, |project, cx| {
                project.dir() == self.origin.as_deref() && !self.center.read(cx).is_busy()
            })
            .unwrap_or(false)
    }
}

/// Ask for a project directory, adopt it, and load whatever history it has.
pub fn pick_project_dir(project: &Entity<ProjectState>, center: &Entity<TaskCenter>, cx: &mut App) {
    let Some(context) = PickerContext::capture(project, cx) else {
        return;
    };
    // Windows toggles between files and folders (`FOS_PICKFOLDERS`), which is why
    // `can_select_mixed_files_and_dirs` is false there: one dialog per kind.
    let paths = cx.prompt_for_paths(PathPromptOptions {
        files: false,
        directories: true,
        multiple: false,
        prompt: None,
    });
    let project = project.clone();
    let center = center.clone();
    cx.spawn(async move |cx| {
        let answer = paths.await;
        if !context.is_current(cx) {
            return;
        }
        match answer {
            Ok(Ok(Some(paths))) => {
                let Some(dir) = paths.into_iter().next() else {
                    return;
                };
                // The window is gone once the shell closes, and so is the reason to
                // read the directory that was just chosen.
                let Ok(adopted) = project.update(cx, |project, cx| {
                    let adopted = project.select_dir(dir);
                    cx.notify();
                    adopted
                }) else {
                    return;
                };
                adopt_history(&center, &adopted, cx);
            }
            // The dialog was closed without a choice.
            Ok(Ok(None)) => {}
            Ok(Err(error)) => {
                note(&project, "assets.note.pick_failed", error.to_string(), cx);
            }
            Err(error) => note(&project, "assets.note.pick_failed", error.to_string(), cx),
        }
    })
    .detach();
}

/// Ask for the video normalisation will read.
///
/// The video is remembered and nothing else happens: the file is only read when
/// the user submits normalisation, and the survey describes the project
/// directory, which this choice does not change.
pub fn pick_source_video(project: &Entity<ProjectState>, cx: &mut App) {
    let Some(context) = PickerContext::capture(project, cx) else {
        return;
    };
    let paths = cx.prompt_for_paths(PathPromptOptions {
        files: true,
        directories: false,
        multiple: false,
        prompt: None,
    });
    let project = project.clone();
    cx.spawn(async move |cx| {
        let answer = paths.await;
        if !context.is_current(cx) {
            return;
        }
        match answer {
            Ok(Ok(Some(paths))) => {
                let Some(video) = paths.into_iter().next() else {
                    return;
                };
                let _ = project.update(cx, |project, cx| {
                    project.select_video(video);
                    cx.notify();
                });
            }
            Ok(Ok(None)) => {}
            Ok(Err(error)) => {
                note(&project, "assets.note.pick_failed", error.to_string(), cx);
            }
            Err(error) => note(&project, "assets.note.pick_failed", error.to_string(), cx),
        }
    })
    .detach();
}

/// Ask for the checkpoint a render reads, and read what it says.
pub fn pick_checkpoint_dir(project: &Entity<ProjectState>, cx: &mut App) {
    let Some(context) = PickerContext::capture(project, cx) else {
        return;
    };
    let paths = cx.prompt_for_paths(PathPromptOptions {
        files: false,
        directories: true,
        multiple: false,
        prompt: None,
    });
    let project = project.clone();
    cx.spawn(async move |cx| {
        let answer = paths.await;
        if !context.is_current(cx) {
            return;
        }
        match answer {
            Ok(Ok(Some(paths))) => {
                let Some(dir) = paths.into_iter().next() else {
                    return;
                };
                // The state file is read here rather than while rendering, the same
                // place `pick_project_dir` surveys the directory it was handed.
                let picked = PickedCheckpoint::adopt(dir);
                let failure = picked.error.clone();
                let _ = project.update(cx, |project, cx| {
                    project.select_checkpoint(picked);
                    // Unreadable is a note, not a refusal: whether the directory holds
                    // a usable model is the worker's verdict to give.
                    if let Some(detail) = failure {
                        project.note(Note::new("generate.note.checkpoint_failed", Some(detail)));
                    }
                    cx.notify();
                });
            }
            Ok(Ok(None)) => {}
            Ok(Err(error)) => {
                note(&project, GENERATE_PICK_FAILED, error.to_string(), cx);
            }
            Err(error) => note(&project, GENERATE_PICK_FAILED, error.to_string(), cx),
        }
    })
    .detach();
}

/// Ask for the track that goes into the container.
///
/// Nothing is read: the file is FFmpeg's second input and the survey describes the
/// project directory, which this choice does not change. What it does not change
/// either is the mouth, and the page says so in a sentence that is always there.
pub fn pick_audio_track(project: &Entity<ProjectState>, cx: &mut App) {
    let Some(context) = PickerContext::capture(project, cx) else {
        return;
    };
    // `files: true` here where the checkpoint dialog asks for a directory: Windows
    // toggles between the two and cannot offer both at once.
    let paths = cx.prompt_for_paths(PathPromptOptions {
        files: true,
        directories: false,
        multiple: false,
        prompt: None,
    });
    let project = project.clone();
    cx.spawn(async move |cx| {
        let answer = paths.await;
        if !context.is_current(cx) {
            return;
        }
        match answer {
            Ok(Ok(Some(paths))) => {
                let Some(track) = paths.into_iter().next() else {
                    return;
                };
                let _ = project.update(cx, |project, cx| {
                    project.select_audio(track);
                    cx.notify();
                });
            }
            Ok(Ok(None)) => {}
            Ok(Err(error)) => {
                note(&project, GENERATE_PICK_FAILED, error.to_string(), cx);
            }
            Err(error) => note(&project, GENERATE_PICK_FAILED, error.to_string(), cx),
        }
    })
    .detach();
}

/// Ask where the render should be written.
///
/// The dialog opens where the renders live, once there are any: `is_dir` is a
/// filesystem question, which is why it is asked in this handler and not while
/// rendering.
pub fn pick_output_path(
    project: &Entity<ProjectState>,
    project_dir: &Path,
    suggested_name: &str,
    cx: &mut App,
) {
    let Some(context) = PickerContext::capture(project, cx) else {
        return;
    };
    if context.origin.as_deref() != Some(project_dir) {
        return;
    }
    let renders = generate::renders_dir(project_dir);
    let start = if renders.is_dir() {
        renders
    } else {
        project_dir.to_path_buf()
    };
    let chosen = cx.prompt_for_new_path(&start, Some(suggested_name));
    let project = project.clone();
    cx.spawn(async move |cx| {
        let answer = chosen.await;
        if !context.is_current(cx) {
            return;
        }
        match answer {
            Ok(Ok(Some(path))) => {
                let _ = project.update(cx, |project, cx| {
                    // The container comes from the extension, so a path with none would
                    // only fail inside FFmpeg. `.mkv` and `.MP4` are left as they are.
                    project.select_output(generate::with_video_extension(path));
                    cx.notify();
                });
            }
            Ok(Ok(None)) => {}
            Ok(Err(error)) => {
                note(&project, GENERATE_PICK_FAILED, error.to_string(), cx);
            }
            Err(error) => note(&project, GENERATE_PICK_FAILED, error.to_string(), cx),
        }
    })
    .detach();
}

/// Replace the task list with what `dir` was left in the middle of.
///
/// The list is replaced rather than extended even when there is nothing to read:
/// the rows that are on screen belong to the project that was open a moment ago,
/// and their task identifiers mean nothing in this one.
fn adopt_history(center: &Entity<TaskCenter>, dir: &Path, cx: &mut AsyncApp) {
    let scanned = if has_task_history(dir) {
        scan_project(dir)
    } else {
        // No manifest is a project with no history yet, not a history that could
        // not be read, so this is not worth a note.
        Ok(Vec::new())
    };
    let _ = center.update(cx, |center, cx| {
        match scanned {
            Ok(tasks) => {
                center.adopt_project(tasks);
            }
            Err(error) => {
                center.adopt_project(Vec::new());
                center.note(Note::new("tasks.note.scan_failed", Some(error.to_string())));
            }
        }
        cx.notify();
    });
}

/// Record a dialog that did not answer, on the page that opened it.
fn note(project: &Entity<ProjectState>, key: &'static str, detail: String, cx: &mut AsyncApp) {
    let _ = project.update(cx, |project, cx| {
        project.note(Note::new(key, Some(detail)));
        cx.notify();
    });
}
