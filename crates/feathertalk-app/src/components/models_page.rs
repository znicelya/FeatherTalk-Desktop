//! One native model operation form and its most recent supervised result.

use std::path::Path;

use feathertalk_domain::TaskStatus;
use gpui::{
    App, ClipboardItem, Div, Entity, FontWeight, InteractiveElement, ParentElement, Stateful,
    StatefulInteractiveElement, Styled, div, px,
};
use yororen_ui::ActionVariantKind;
use yororen_ui::headless::badge::{BadgeVariant, badge};
use yororen_ui::headless::button::button;
use yororen_ui::headless::disclosure::disclosure;
use yororen_ui::headless::form_field::form_field;
use yororen_ui::headless::radio_group::radio_group;
use yororen_ui::headless::toggle_button::toggle_button;
use yororen_ui::i18n::Translate;

use crate::components::tasks_page::{cancel_row, progress_bar};
use crate::components::ui::{muted, page_frame, path_field, section};
use crate::facts::FactValue;
use crate::model_picker::{pick_destination, pick_source};
use crate::models::{ModelForm, ModelIssue, ModelOperation, result_facts};
use crate::navigation::Page;
use crate::state::AppState;
use crate::submit::submit;
use crate::tasks::{Summary, TaskRow, blocked_key, kind_key, stage_key, status_key};
use crate::theme::color;
use crate::ui::TaskFilter;

pub fn models_page(cx: &mut App) -> Div {
    let state = cx.global::<AppState>();
    let form = state.models.clone();
    let current = form.read(cx).clone();
    let dir = state.project.read(cx).dir().map(Path::to_path_buf);
    let tasks = state.tasks.read(cx);
    let busy = tasks.is_busy();
    let latest = tasks
        .rows()
        .iter()
        .find(|row| ModelOperation::from_task_kind(row.kind).is_some())
        .cloned();
    let gate = blocked_key(dir.is_some(), state.worker.read(cx).is_ready(), busy).or_else(|| {
        state
            .compute
            .read(cx)
            .command_blocked_key(current.operation().task_kind())
    });
    let mut operation_form = section(
        "models-form",
        cx.t("models.form.title"),
        cx.t("models.form.description"),
        cx,
    )
    .child(operation_selector(&form, &current, busy, cx))
    .child(muted(cx.t(current.operation().description_key()), cx));
    if !current.operation().model_kinds().is_empty() {
        operation_form = operation_form.child(kind_field(&form, &current, busy, cx));
    }
    operation_form = operation_form.child(source_field(&form, &current, busy, cx));
    if current.operation().needs_destination() {
        operation_form =
            operation_form.child(destination_field(&form, &current, dir.as_deref(), busy, cx));
    }
    operation_form = operation_form.child(submit_row(&current, gate, cx));
    if let Some(issue) = &current.issue {
        operation_form = operation_form.child(issue_block(issue, cx));
    }
    let content = div()
        .min_w(px(0.))
        .flex()
        .flex_col()
        .gap_6()
        .child(operation_form)
        .child(result_section(&form, &current, latest.as_ref(), cx));
    page_frame(
        "models-page",
        cx.t("page.models.title"),
        cx.t("models.description"),
        content,
        cx,
    )
}

fn operation_selector(
    form: &Entity<ModelForm>,
    current: &ModelForm,
    busy: bool,
    cx: &mut App,
) -> Stateful<Div> {
    let mut group = radio_group("models-operation-group", cx)
        .name(cx.t("models.form.title"))
        .selected(
            ModelOperation::ALL
                .iter()
                .position(|operation| *operation == current.operation())
                .unwrap_or(0),
        )
        .render(cx)
        .min_w(px(0.))
        .flex_wrap()
        .gap_2();
    for operation in ModelOperation::ALL {
        let form = form.clone();
        group = group.child(
            toggle_button(operation.element_id(), cx)
                .caption(cx.t(operation.label_key()))
                .selected(operation == current.operation())
                .disabled(busy)
                .on_toggle(move |_selected, _event, _window, cx| {
                    form.update(cx, |form, cx| {
                        if form.set_operation(operation) {
                            cx.notify();
                        }
                    });
                })
                .render(cx),
        );
    }
    group
}

fn kind_field(
    form: &Entity<ModelForm>,
    current: &ModelForm,
    busy: bool,
    cx: &mut App,
) -> Stateful<Div> {
    let kinds = current.operation().model_kinds();
    let mut group = radio_group("models-kind-group", cx)
        .name(cx.t("models.kind.label"))
        .selected(
            kinds
                .iter()
                .position(|kind| *kind == current.kind())
                .unwrap_or(0),
        )
        .render(cx)
        .flex_wrap()
        .gap_2();
    for &kind in kinds {
        let form = form.clone();
        group = group.child(
            toggle_button(kind.element_id(), cx)
                .caption(cx.t(kind.label_key()))
                .selected(current.kind() == kind)
                .disabled(busy)
                .on_toggle(move |_selected, _event, _window, cx| {
                    form.update(cx, |form, cx| {
                        form.set_kind(kind);
                        cx.notify();
                    });
                })
                .render(cx),
        );
    }
    let help = if current.operation() == ModelOperation::ImportLegacy {
        "models.kind.import_hint"
    } else {
        "models.kind.onnx_hint"
    };
    form_field("models-kind-field", "kind", cx)
        .label(cx.t("models.kind.label"))
        .required(true)
        .help(cx.t(help))
        .input(group)
        .render(cx)
}

fn source_field(
    form: &Entity<ModelForm>,
    current: &ModelForm,
    busy: bool,
    cx: &mut App,
) -> Stateful<Div> {
    let entity = form.clone();
    let caption = if current.operation().source_is_file() {
        "models.source.pick_file"
    } else {
        "models.source.pick_directory"
    };
    let mut row = div()
        .flex()
        .flex_wrap()
        .items_center()
        .gap_2()
        .min_w(px(0.))
        .child(path_value(current.source(), cx))
        .child(
            button("models-source-pick", cx)
                .caption(cx.t(caption))
                .disabled(busy)
                .on_click(move |_event, _window, cx| pick_source(&entity, cx))
                .render(cx),
        );
    if let Some(path) = current.source() {
        row = row.child(copy_path_button("models-source-copy", path, cx));
    }
    form_field("models-source-field", "source", cx)
        .label(cx.t(current.operation().source_label_key()))
        .required(true)
        .input(row)
        .render(cx)
}

fn destination_field(
    form: &Entity<ModelForm>,
    current: &ModelForm,
    dir: Option<&Path>,
    busy: bool,
    cx: &mut App,
) -> Stateful<Div> {
    let entity = form.clone();
    let start = dir.map(Path::to_path_buf);
    let (caption, help) = if current.operation().destination_is_directory() {
        (
            "models.destination.pick_parent",
            "models.destination.parent_hint",
        )
    } else {
        (
            "models.destination.pick_file",
            "models.destination.file_hint",
        )
    };
    let mut row = div()
        .flex()
        .flex_wrap()
        .items_center()
        .gap_2()
        .min_w(px(0.))
        .child(path_value(current.destination(), cx))
        .child(
            button("models-destination-pick", cx)
                .caption(cx.t(caption))
                .disabled(busy || start.is_none())
                .on_click(move |_event, _window, cx| {
                    if let Some(start) = &start {
                        pick_destination(&entity, start, cx);
                    }
                })
                .render(cx),
        );
    if let Some(path) = current.destination() {
        row = row.child(copy_path_button("models-destination-copy", path, cx));
    }
    form_field("models-destination-field", "destination", cx)
        .label(cx.t(current.operation().destination_label_key()))
        .required(true)
        .help(cx.t(help))
        .input(row)
        .render(cx)
}

fn path_value(path: Option<&Path>, cx: &App) -> Div {
    let value = path
        .map(|path| path.display().to_string())
        .unwrap_or_else(|| cx.t("models.path.unset").to_string());
    let display = value.replace('\\', "\\\u{200b}").replace('/', "/\u{200b}");
    div()
        .flex_1()
        .min_w(px(180.))
        .min_h(px(36.))
        .p_2()
        .overflow_hidden()
        .rounded(px(6.))
        .border_1()
        .border_color(color(cx, "border.default"))
        .bg(color(cx, "surface.sunken"))
        .text_size(px(13.))
        .text_color(color(
            cx,
            if path.is_some() {
                "content.primary"
            } else {
                "content.tertiary"
            },
        ))
        .child(display)
}

fn copy_path_button(id: &'static str, path: &Path, cx: &mut App) -> Stateful<Div> {
    let value = path.display().to_string();
    button(id, cx)
        .caption(cx.t("models.path.copy"))
        .on_click(move |_event, _window, cx| {
            cx.write_to_clipboard(ClipboardItem::new_string(value.clone()))
        })
        .render(cx)
}

fn submit_row(current: &ModelForm, gate: Option<&'static str>, cx: &mut App) -> Div {
    let validation = current.request().err();
    let reason = gate.or_else(|| validation.as_ref().map(|issue| issue.key));
    let mut row = div()
        .flex()
        .flex_wrap()
        .items_center()
        .gap_3()
        .min_w(px(0.))
        .child(
            button("models-submit", cx)
                .caption(cx.t(current.operation().label_key()))
                .variant(ActionVariantKind::Primary)
                .disabled(reason.is_some())
                .on_click(move |_event, _window, cx| submit_current(cx))
                .render(cx),
        );
    row = row.child(muted(cx.t(reason.unwrap_or("models.project_hint")), cx));
    row
}

/// Admission is read again at click time; project paths and request fields are
/// captured together immediately before the shared supervisor is called.
fn submit_current(cx: &mut App) {
    let state = cx.global::<AppState>();
    let form = state.models.clone();
    let project = state.project.clone();
    let center = state.tasks.clone();
    let current = form.read(cx).clone();
    let dir = project.read(cx).dir().map(Path::to_path_buf);
    let log_dir = project.read(cx).log_dir();
    let gate = blocked_key(
        dir.is_some(),
        state.worker.read(cx).is_ready(),
        center.read(cx).is_busy(),
    )
    .or_else(|| {
        state
            .compute
            .read(cx)
            .command_blocked_key(current.operation().task_kind())
    });
    if let Some(key) = gate {
        set_issue(&form, ModelIssue::new(key), cx);
        return;
    }
    let (Some(dir), Some(log_dir)) = (dir, log_dir) else {
        set_issue(&form, ModelIssue::new("tasks.blocked.no_project"), cx);
        return;
    };
    let request = match current.preflight() {
        Ok(request) => request,
        Err(issue) => {
            set_issue(&form, issue, cx);
            return;
        }
    };
    form.update(cx, |form, cx| {
        form.issue = None;
        cx.notify();
    });
    submit(
        &center,
        &project,
        request.kind(),
        request,
        &dir,
        &log_dir,
        cx,
    );
}

fn set_issue(form: &Entity<ModelForm>, issue: ModelIssue, cx: &mut App) {
    form.update(cx, |form, cx| {
        form.issue = Some(issue);
        cx.notify();
    });
}

fn issue_block(issue: &ModelIssue, cx: &App) -> Div {
    let mut block = div()
        .flex()
        .flex_col()
        .gap_1()
        .min_w(px(0.))
        .p_3()
        .rounded(px(6.))
        .bg(color(cx, "status.danger.bg"))
        .text_color(color(cx, "status.danger.fg"))
        .text_size(px(13.))
        .child(cx.t(issue.key));
    if let Some(detail) = &issue.detail {
        block = block.child(
            div()
                .min_w(px(0.))
                .overflow_hidden()
                .text_size(px(12.))
                .child(detail.clone()),
        );
    }
    block
}

fn result_section(
    form: &Entity<ModelForm>,
    current: &ModelForm,
    row: Option<&TaskRow>,
    cx: &mut App,
) -> Stateful<Div> {
    let mut block = section(
        "models-result",
        cx.t("models.result.title"),
        cx.t("models.result.description"),
        cx,
    );
    let Some(row) = row else {
        return block
            .child(muted(cx.t("models.result.empty"), cx))
            .child(tasks_link(cx));
    };
    let status = cx.t(status_key(row.status)).to_string();
    let variant = match row.status {
        TaskStatus::Completed => BadgeVariant::Success,
        TaskStatus::Failed => BadgeVariant::Danger,
        TaskStatus::Cancelled => BadgeVariant::Warning,
        TaskStatus::Queued | TaskStatus::Running => BadgeVariant::Info,
    };
    block = block.child(
        div()
            .flex()
            .flex_wrap()
            .items_center()
            .gap_3()
            .child(
                div()
                    .font_weight(FontWeight::SEMIBOLD)
                    .text_color(color(cx, "content.primary"))
                    .child(cx.t(kind_key(row.kind))),
            )
            .child(
                badge("models-result-status", status, cx)
                    .variant(variant)
                    .render(cx),
            ),
    );
    if row.status.is_incomplete() {
        block = block
            .child(muted(cx.t(stage_key(&row.stage)), cx))
            .child(progress_bar(row.task_id.as_str(), row, cx))
            .child(cancel_row(row.task_id.as_str(), row, cx));
    }
    if let Some(failure) = &row.failure {
        let summary = match &failure.summary {
            Summary::Worker(summary) => summary.clone(),
            Summary::Key(key) => cx.t(key).to_string(),
        };
        block = block.child(
            div()
                .text_color(color(cx, "status.danger.fg"))
                .text_size(px(13.))
                .child(summary),
        );
    } else if row.status == TaskStatus::Cancelled {
        block = block.child(muted(cx.t("models.result.cancelled"), cx));
    }
    if row.status == TaskStatus::Completed {
        if let Some(result) = &row.result {
            let facts = result_facts(row.kind, result);
            let mut values = div().min_w(px(0.)).flex().flex_col().gap_3();
            for fact in &facts {
                let value = match &fact.value {
                    FactValue::Text(value) => value.clone(),
                    FactValue::Key(key) => cx.t(key).to_string(),
                };
                if matches!(
                    fact.label,
                    "models.result.source" | "models.result.destination"
                ) {
                    values = values.child(path_field(cx.t(fact.label), value, cx));
                } else {
                    values = values.child(
                        div()
                            .flex()
                            .gap_4()
                            .min_w(px(0.))
                            .child(
                                div()
                                    .w(px(112.))
                                    .flex_shrink_0()
                                    .child(muted(cx.t(fact.label), cx)),
                            )
                            .child(
                                div()
                                    .flex_1()
                                    .min_w(px(0.))
                                    .text_size(px(13.))
                                    .text_color(color(cx, "content.primary"))
                                    .child(value),
                            ),
                    );
                }
            }
            if facts.is_empty() {
                values = values.child(muted(cx.t("models.result.no_payload"), cx));
            }
            block = block.child(values).child(result_actions(result, cx));
            let entity = form.clone();
            let focus = color(cx, "border.focus");
            block = block.child(
                disclosure(
                    "models-result-disclosure",
                    cx.t("models.result.details").to_string(),
                    cx,
                )
                .open(current.result_details_open)
                .on_toggle(move |_event, _window, cx| {
                    entity.update(cx, |form, cx| {
                        form.result_details_open = !form.result_details_open;
                        cx.notify();
                    });
                })
                .render(cx)
                .focusable()
                .tab_index(0)
                .tab_stop(true)
                .p_2()
                .border_1()
                .border_color(color(cx, "surface.base"))
                .focus_visible(move |style| style.border_color(focus)),
            );
            if current.result_details_open {
                let json =
                    serde_json::to_string_pretty(result).unwrap_or_else(|_| result.to_string());
                block = block.child(
                    div()
                        .id("models-result-json")
                        .min_w(px(0.))
                        .max_h(px(280.))
                        .overflow_y_scroll()
                        .overflow_x_scroll()
                        .p_3()
                        .rounded(px(6.))
                        .bg(color(cx, "surface.sunken"))
                        .text_color(color(cx, "content.secondary"))
                        .text_size(px(12.))
                        .font_family("Consolas")
                        .child(json),
                );
            }
        } else {
            block = block.child(muted(cx.t("models.result.no_payload"), cx));
        }
    }
    block.child(tasks_link(cx))
}

fn result_actions(result: &serde_json::Value, cx: &mut App) -> Div {
    let json = serde_json::to_string_pretty(result).unwrap_or_else(|_| result.to_string());
    let mut actions = div().flex().flex_wrap().gap_2().child(
        button("models-result-copy", cx)
            .caption(cx.t("models.result.copy"))
            .on_click(move |_event, _window, cx| {
                cx.write_to_clipboard(ClipboardItem::new_string(json.clone()))
            })
            .render(cx),
    );
    if let Some(destination) = result
        .get("destination")
        .and_then(serde_json::Value::as_str)
    {
        let path = std::path::PathBuf::from(destination);
        actions = actions.child(
            button("models-result-reveal", cx)
                .caption(cx.t("models.result.reveal"))
                .on_click(move |_event, _window, cx| cx.reveal_path(&path))
                .render(cx),
        );
    }
    actions
}

fn tasks_link(cx: &mut App) -> Stateful<Div> {
    button("models-view-tasks", cx)
        .caption(cx.t("models.result.tasks"))
        .on_click(move |_event, _window, cx| {
            let state = cx.global::<AppState>();
            let navigation = state.navigation.clone();
            let ui = state.ui.clone();
            ui.update(cx, |ui, cx| {
                ui.task_filter = TaskFilter::All;
                cx.notify();
            });
            navigation.update(cx, |navigation, cx| {
                if navigation.select(Page::Tasks) {
                    cx.notify();
                }
            });
        })
        .render(cx)
}
