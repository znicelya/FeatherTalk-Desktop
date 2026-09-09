//! Readable task records with filters, real results and explicit stop feedback.

use std::path::{Path, PathBuf};

use feathertalk_domain::{ProjectDirParams, Request, TaskKind, TaskStatus};
use feathertalk_supervisor::recovery::{IncompleteTask, Resolution, resolve_project};
use feathertalk_supervisor::status::task_status;
use gpui::{
    App, ClipboardItem, Div, Entity, FontWeight, InteractiveElement, ParentElement, SharedString,
    Stateful, StatefulInteractiveElement, Styled, div, px,
};
use yororen_ui::ActionVariantKind;
use yororen_ui::headless::badge::{BadgeVariant, badge};
use yororen_ui::headless::button::button;
use yororen_ui::headless::progress::progress;
use yororen_ui::i18n::Translate;

use crate::components::ui::{muted, page_frame, path_field, section};
use crate::project::ProjectState;
use crate::state::AppState;
use crate::submit::{note, submit};
use crate::tasks::{
    Failure, Note, Summary, TaskCenter, TaskRow, blocked_key, kind_key, recovery_key, stage_key,
    status_key,
};
use crate::theme::color;
use crate::ui::{ProgressPresentation, TaskFilter, progress_presentation};

pub fn tasks_page(cx: &mut App) -> Div {
    let state = cx.global::<AppState>();
    let center = state.tasks.clone();
    let project = state.project.clone();
    let ui = state.ui.clone();
    let filter = ui.read(cx).task_filter;
    let dir = project.read(cx).dir().map(Path::to_path_buf);
    let target = dir.clone().zip(project.read(cx).log_dir());
    let tasks = center.read(cx);
    let busy = tasks.is_busy();
    let rows = tasks.rows().to_vec();
    let incomplete = tasks.incomplete().to_vec();
    let notes = tasks.notes().to_vec();
    let blocked =
        blocked_key(dir.is_some(), state.worker.read(cx).is_ready(), busy).or_else(|| {
            state
                .compute
                .read(cx)
                .command_blocked_key(TaskKind::ValidateProject)
        });
    let mut tabs = div().flex().flex_wrap().gap_1();
    for choice in TaskFilter::ALL {
        let ui = ui.clone();
        let count = rows
            .iter()
            .filter(|row| choice.includes(row.status))
            .count();
        let caption = format!("{}  {count}", cx.t(choice.key()));
        tabs = tabs.child(
            button(
                SharedString::from(format!("task-filter-{}", choice.key())),
                cx,
            )
            .caption(caption)
            .variant(if filter == choice {
                ActionVariantKind::Primary
            } else {
                ActionVariantKind::Neutral
            })
            .on_click(move |_event, _window, cx| {
                ui.update(cx, |ui, cx| {
                    ui.task_filter = choice;
                    cx.notify();
                });
            })
            .render(cx),
        );
    }
    let mut body = div().flex().flex_col().gap_5().min_w(px(0.)).child(
        div()
            .flex()
            .flex_wrap()
            .justify_between()
            .items_center()
            .gap_3()
            .child(tabs)
            .child(validate_action(&center, &project, target, blocked, cx)),
    );
    if let Some(key) = blocked {
        body = body.child(muted(cx.t(key), cx));
    }
    if !incomplete.is_empty() {
        body = body.child(incomplete_section(
            &center,
            dir.as_deref(),
            &incomplete,
            busy,
            cx,
        ));
    }
    let visible: Vec<&TaskRow> = rows
        .iter()
        .filter(|row| filter.includes(row.status))
        .collect();
    if visible.is_empty() {
        let (title, description) = if rows.is_empty() {
            ("tasks.empty.title", "ui.tasks.empty_description")
        } else {
            ("ui.tasks.no_match", "ui.tasks.no_match_hint")
        };
        body = body.child(section("tasks-empty", cx.t(title), cx.t(description), cx).py_8());
    } else {
        let mut list = div()
            .flex()
            .flex_col()
            .w_full()
            .border_1()
            .border_color(color(cx, "border.default"))
            .rounded(px(8.))
            .overflow_hidden()
            .bg(color(cx, "surface.raised"));
        for row in visible {
            list = list.child(task_record(&center, row, cx));
        }
        body = body.child(list);
    }
    if !notes.is_empty() {
        body = body.child(notes_section(&notes, cx));
    }
    page_frame(
        "tasks-page",
        cx.t("page.tasks.title"),
        cx.t("ui.tasks.description"),
        body,
        cx,
    )
}

fn validate_action(
    center: &Entity<TaskCenter>,
    project: &Entity<ProjectState>,
    target: Option<(PathBuf, PathBuf)>,
    blocked: Option<&'static str>,
    cx: &mut App,
) -> Stateful<Div> {
    let mut action = button("tasks-submit", cx).caption(cx.t("tasks.submit"));
    if let (None, Some((dir, log_dir))) = (blocked, target) {
        let center = center.clone();
        let project = project.clone();
        action = action.on_click(move |_event, _window, cx| {
            submit(
                &center,
                &project,
                TaskKind::ValidateProject,
                Request::ValidateProject(ProjectDirParams {
                    project_dir: dir.clone(),
                }),
                &dir,
                &log_dir,
                cx,
            );
        });
    } else {
        action = action.disabled(true);
    }
    action.render(cx)
}

fn incomplete_section(
    center: &Entity<TaskCenter>,
    dir: Option<&Path>,
    tasks: &[IncompleteTask],
    busy: bool,
    cx: &mut App,
) -> Stateful<Div> {
    let mut content = section(
        "tasks-incomplete",
        cx.t("tasks.incomplete.title"),
        cx.t("tasks.incomplete.description"),
        cx,
    );
    for task in tasks {
        let kind = cx.t(task
            .kind
            .map(kind_key)
            .unwrap_or("tasks.incomplete.unknown_kind"));
        let mut row = div()
            .flex()
            .flex_wrap()
            .items_center()
            .gap_3()
            .py_2()
            .child(
                div()
                    .flex_1()
                    .min_w(px(200.))
                    .flex()
                    .flex_col()
                    .gap_1()
                    .child(div().font_weight(FontWeight::MEDIUM).child(kind))
                    .child(muted(format!("{} · {}", task.task_id, task.updated_at), cx)),
            )
            .child(
                badge(
                    row_id("incomplete-status", &task.task_id),
                    cx.t(status_key(task_status(task.status.clone()))),
                    cx,
                )
                .render(cx),
            );
        if let Some(dir) = dir {
            for (resolution, key, id) in [
                (
                    Resolution::Resume,
                    "tasks.incomplete.resume",
                    "incomplete-resume",
                ),
                (
                    Resolution::Discard,
                    "tasks.incomplete.discard",
                    "incomplete-discard",
                ),
            ] {
                let center = center.clone();
                let dir = dir.to_path_buf();
                let task_id = task.task_id.clone();
                row = row.child(
                    button(row_id(id, &task.task_id), cx)
                        .caption(cx.t(key))
                        .disabled(busy)
                        .on_click(move |_event, _window, cx| {
                            resolve_one(&center, &dir, task_id.clone(), resolution, cx)
                        })
                        .render(cx),
                );
            }
        }
        content = content.child(row);
    }
    content
}

fn resolve_one(
    center: &Entity<TaskCenter>,
    dir: &Path,
    task_id: String,
    resolution: Resolution,
    cx: &mut App,
) {
    if center.read(cx).is_busy() || cx.global::<AppState>().project.read(cx).dir() != Some(dir) {
        return;
    }
    match resolve_project(dir, &[(task_id.clone(), resolution)]) {
        Ok(()) => center.update(cx, |center, cx| {
            center.resolve(&task_id);
            cx.notify();
        }),
        Err(error) => note(center, "tasks.note.resolve_failed", error.to_string(), cx),
    }
}

fn task_record(center: &Entity<TaskCenter>, row: &TaskRow, cx: &mut App) -> Stateful<Div> {
    let id = row.task_id.as_str().to_owned();
    let task_id = id.clone();
    let entity = center.clone();
    let details = button(row_id("task-detail", &id), cx)
        .caption(cx.t(if row.detail_open {
            "ui.tasks.hide_details"
        } else {
            "ui.tasks.details"
        }))
        .on_click(move |_event, _window, cx| {
            entity.update(cx, |center, cx| {
                center.toggle_detail(&task_id);
                cx.notify();
            });
        })
        .render(cx);
    let mut content = div()
        .id(row_id("task", &id))
        .flex()
        .flex_col()
        .gap_3()
        .p_5()
        .border_b_1()
        .border_color(color(cx, "border.muted"))
        .child(
            div()
                .flex()
                .flex_wrap()
                .items_center()
                .gap_3()
                .child(
                    div()
                        .flex_1()
                        .min_w(px(160.))
                        .flex()
                        .flex_col()
                        .gap_1()
                        .child(
                            div()
                                .font_weight(FontWeight::MEDIUM)
                                .child(cx.t(kind_key(row.kind))),
                        )
                        .child(muted(cx.t(stage_key(&row.stage)), cx)),
                )
                .child(
                    badge(row_id("task-status", &id), cx.t(status_key(row.status)), cx)
                        .variant(badge_variant(row.status))
                        .render(cx),
                )
                .child(details),
        );
    if row.status.is_incomplete() || row.progress.is_some() {
        content = content.child(progress_bar(&id, row, cx));
    }
    if row.status.is_incomplete() {
        content = content.child(cancel_row(&id, row, cx));
    }
    if let Some(failure) = &row.failure {
        content = content.child(failure_summary(failure, cx));
    }
    if row.detail_open {
        let mut detail = div()
            .flex()
            .flex_col()
            .gap_3()
            .pt_2()
            .child(path_field(cx.t("ui.tasks.identifier"), id.clone(), cx))
            .child(muted(
                format!("{}：{}", cx.t("tasks.attempts"), row.attempts),
                cx,
            ));
        if let Some(result) = &row.result {
            let raw = serde_json::to_string_pretty(result).unwrap_or_else(|_| result.to_string());
            let clipboard = raw.clone();
            detail = detail
                .child(
                    div()
                        .text_size(px(13.))
                        .font_weight(FontWeight::MEDIUM)
                        .child(cx.t("ui.tasks.result")),
                )
                .child(raw_detail("task-result", &id, raw, cx))
                .child(
                    div().child(
                        button(row_id("task-copy-result", &id), cx)
                            .caption(cx.t("ui.tasks.copy_result"))
                            .on_click(move |_event, _window, cx| {
                                cx.write_to_clipboard(ClipboardItem::new_string(clipboard.clone()))
                            })
                            .render(cx),
                    ),
                );
        }
        if let Some(failure) = &row.failure {
            if let Some(code) = failure.code {
                detail = detail.child(muted(
                    format!("{}：{}", cx.t("tasks.error_code"), code.as_wire()),
                    cx,
                ));
            }
            let clipboard = failure.detail.clone();
            detail = detail
                .child(raw_detail("task-error", &id, failure.detail.clone(), cx))
                .child(
                    div().child(
                        button(row_id("task-copy-error", &id), cx)
                            .caption(cx.t("ui.tasks.copy_error"))
                            .on_click(move |_event, _window, cx| {
                                cx.write_to_clipboard(ClipboardItem::new_string(clipboard.clone()))
                            })
                            .render(cx),
                    ),
                );
        }
        for (index, path) in row.crash_logs.iter().enumerate() {
            let path = path.clone();
            detail = detail.child(
                div()
                    .flex()
                    .flex_wrap()
                    .items_center()
                    .gap_2()
                    .child(path_field(
                        cx.t("tasks.crash_log"),
                        path.display().to_string(),
                        cx,
                    ))
                    .child(
                        button(row_id(&format!("task-log-{index}"), &id), cx)
                            .caption(cx.t("generate.renders.reveal"))
                            .on_click(move |_event, _window, cx| cx.reveal_path(&path))
                            .render(cx),
                    ),
            );
        }
        content = content.child(detail);
    }
    content
}

fn raw_detail(prefix: &str, id: &str, text: String, cx: &App) -> Stateful<Div> {
    div()
        .id(row_id(prefix, id))
        .max_h(px(280.))
        .overflow_y_scroll()
        .overflow_x_scroll()
        .p_3()
        .rounded(px(6.))
        .bg(color(cx, "surface.canvas"))
        .border_1()
        .border_color(color(cx, "border.muted"))
        .text_size(px(12.))
        .font_family("Consolas")
        .child(text)
}

fn failure_summary(failure: &Failure, cx: &App) -> Div {
    let summary = match &failure.summary {
        Summary::Worker(text) => text.clone(),
        Summary::Key(key) => cx.t(key).to_string(),
    };
    let mut view = div()
        .flex()
        .flex_col()
        .gap_1()
        .rounded(px(6.))
        .p_3()
        .bg(color(cx, "status.danger.bg"))
        .text_color(color(cx, "status.danger.fg"))
        .text_size(px(13.))
        .child(summary);
    if let Some(recovery) = failure.recovery {
        view = view.child(div().text_size(px(12.)).child(cx.t(recovery_key(recovery))));
    }
    view
}

fn row_id(name: &str, task_id: &str) -> SharedString {
    SharedString::from(format!("{name}-{task_id}"))
}

pub(crate) fn badge_variant(status: TaskStatus) -> BadgeVariant {
    match status {
        TaskStatus::Queued => BadgeVariant::Neutral,
        TaskStatus::Running => BadgeVariant::Info,
        TaskStatus::Completed => BadgeVariant::Success,
        TaskStatus::Failed => BadgeVariant::Danger,
        TaskStatus::Cancelled => BadgeVariant::Neutral,
    }
}

pub(crate) fn progress_bar(id: &str, row: &TaskRow, cx: &mut App) -> Div {
    progress_bar_with(id, progress_presentation(row.status, row.progress), cx)
}

pub(crate) fn progress_bar_with(id: &str, presentation: ProgressPresentation, cx: &mut App) -> Div {
    let bar = progress(row_id("task-progress", id), cx);
    let bar = match presentation {
        ProgressPresentation::Indeterminate => bar.indeterminate(true),
        ProgressPresentation::Counted { completed, total } => {
            bar.value(completed as f32).max(total as f32)
        }
        ProgressPresentation::Complete => bar.value(1.).max(1.),
        ProgressPresentation::Empty => bar.value(0.),
    };
    div().w_full().child(bar.render(cx))
}

pub(crate) fn cancel_row(id: &str, row: &TaskRow, cx: &mut App) -> Div {
    let cancel = row.cancel.clone();
    let count = cancel.count();
    let center = cx.global::<AppState>().tasks.clone();
    let caption = cx.t(match count {
        0 => "tasks.cancel",
        1 => "ui.tasks.force_cancel",
        _ => "ui.tasks.force_requested",
    });
    let hint = cx.t(if count == 0 {
        "ui.tasks.cancel_hint"
    } else {
        "ui.tasks.force_cancel_hint"
    });
    div()
        .debug_selector(|| format!("task-cancel-row-{id}"))
        .flex()
        .flex_wrap()
        .items_center()
        .gap_3()
        .child(
            button(row_id("task-cancel", id), cx)
                .caption(caption)
                .variant(ActionVariantKind::Danger)
                .disabled(count >= 2)
                .on_click(move |_event, _window, cx| {
                    cancel.request();
                    center.update(cx, |_center, cx| cx.notify());
                })
                .render(cx),
        )
        .child(
            muted(hint, cx)
                .flex_1()
                .flex_basis(px(240.))
                .debug_selector(|| format!("task-cancel-hint-{id}")),
        )
}

fn notes_section(notes: &[Note], cx: &App) -> Stateful<Div> {
    let mut view = section("tasks-notes", cx.t("tasks.notes.title"), "", cx);
    for note in notes {
        view = view.child(muted(
            match &note.detail {
                Some(detail) => format!("{}：{detail}", cx.t(note.key)),
                None => cx.t(note.key).to_string(),
            },
            cx,
        ));
    }
    view
}
