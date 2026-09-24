//! Native model source and output pickers. Cancellation never clears a form.

use std::path::Path;

use gpui::{App, AsyncApp, Entity, PathPromptOptions};
use yororen_ui::i18n::Translate;

use crate::models::{ModelForm, ModelIssue, ModelOperation};
use crate::picker::PickerContext;

pub fn pick_source(form: &Entity<ModelForm>, cx: &mut App) {
    let Some(context) = PickerContext::current(cx) else {
        return;
    };
    let operation = form.read(cx).operation();
    let paths = cx.prompt_for_paths(PathPromptOptions {
        files: operation.source_is_file(),
        directories: !operation.source_is_file(),
        multiple: false,
        prompt: Some(cx.t(operation.source_label_key())),
    });
    let form = form.clone();
    cx.spawn(async move |cx| {
        let answer = paths.await;
        if !context.is_current(cx) {
            return;
        }
        match answer {
            Ok(Ok(Some(paths))) => {
                let picked = paths.into_iter().next();
                let _ = form.update(cx, |form, cx| {
                    if form.apply_source_pick(operation, picked) {
                        cx.notify();
                    }
                });
            }
            Ok(Ok(None)) => {}
            Ok(Err(error)) => picker_failed(&form, operation, error.to_string(), cx),
            Err(error) => picker_failed(&form, operation, error.to_string(), cx),
        }
    })
    .detach();
}

pub fn pick_destination(form: &Entity<ModelForm>, project_dir: &Path, cx: &mut App) {
    let Some(context) = PickerContext::current(cx) else {
        return;
    };
    let current = form.read(cx);
    let operation = current.operation();
    if !operation.needs_destination() {
        return;
    }
    if operation.destination_is_directory() {
        pick_destination_parent(form, cx);
        return;
    }
    let start = current
        .destination()
        .and_then(Path::parent)
        .filter(|parent| parent.is_dir())
        .unwrap_or(project_dir)
        .to_path_buf();
    let suggested = current.suggested_destination_name();
    let chosen = cx.prompt_for_new_path(&start, Some(&suggested));
    let form = form.clone();
    cx.spawn(async move |cx| {
        let answer = chosen.await;
        if !context.is_current(cx) {
            return;
        }
        match answer {
            Ok(Ok(path)) => {
                let _ = form.update(cx, |form, cx| {
                    if form.apply_destination_pick(operation, path) {
                        cx.notify();
                    }
                });
            }
            Ok(Err(error)) => picker_failed(&form, operation, error.to_string(), cx),
            Err(error) => picker_failed(&form, operation, error.to_string(), cx),
        }
    })
    .detach();
}

/// The worker creates the final package directory itself. A native folder
/// picker therefore chooses its existing parent, never an already-created
/// destination that the no-clobber publisher would reject.
fn pick_destination_parent(form: &Entity<ModelForm>, cx: &mut App) {
    let Some(context) = PickerContext::current(cx) else {
        return;
    };
    let operation = form.read(cx).operation();
    let paths = cx.prompt_for_paths(PathPromptOptions {
        files: false,
        directories: true,
        multiple: false,
        prompt: Some(cx.t("models.destination.pick_parent")),
    });
    let form = form.clone();
    cx.spawn(async move |cx| {
        let answer = paths.await;
        if !context.is_current(cx) {
            return;
        }
        match answer {
            Ok(Ok(Some(paths))) => {
                let Some(parent) = paths.into_iter().next() else {
                    return;
                };
                let _ = form.update(cx, |form, cx| {
                    if form.operation() == operation {
                        form.select_destination_parent(&parent);
                        cx.notify();
                    }
                });
            }
            Ok(Ok(None)) => {}
            Ok(Err(error)) => picker_failed(&form, operation, error.to_string(), cx),
            Err(error) => picker_failed(&form, operation, error.to_string(), cx),
        }
    })
    .detach();
}

fn picker_failed(
    form: &Entity<ModelForm>,
    operation: ModelOperation,
    detail: String,
    cx: &mut AsyncApp,
) {
    let _ = form.update(cx, |form, cx| {
        if form.operation() == operation {
            form.issue = Some(ModelIssue::with_detail("models.error.picker", detail));
            cx.notify();
        }
    });
}
