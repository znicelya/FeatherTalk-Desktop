//! Training choices and saved observations stay separate from live task progress.
//! All mutable controls still mirror the existing TrainingForm.

use std::path::Path;

use feathertalk_domain::{Progress, TaskKind, TaskStage, TaskStatus, TrainingMode};
use gpui::{
    App, Div, Entity, FontWeight, InteractiveElement, ParentElement, SharedString, Stateful,
    Styled, Window, div, px,
};
use yororen_ui::ActionVariantKind;
use yororen_ui::headless::badge::badge;
use yororen_ui::headless::button::button;
use yororen_ui::headless::number_input::number_input;
use yororen_ui::headless::radio::radio;
use yororen_ui::headless::switch::switch;
use yororen_ui::i18n::Translate;

use crate::components::compute_control::selected_summary;
use crate::components::control_renderers::RadioControlExt;
use crate::components::facts::facts_list;
use crate::components::tasks_page::{badge_variant, cancel_row, progress_bar_with};
use crate::components::ui::{
    inline_muted, label_separator, muted, page_frame, path_field, section,
};
use crate::facts::{Fact, FactValue};
use crate::navigation::Page;
use crate::project::ProjectState;
use crate::state::AppState;
use crate::submit::submit;
use crate::tasks::{
    Note, Summary, TaskCenter, TaskRow, blocked_key, latest_row, stage_key, status_key,
};
use crate::theme::color;
use crate::training::{
    ALL_MODES, ALL_VARIANTS, FormState, MAX_EPOCHS, MIN_EPOCHS, TrainingForm, TrainingSurvey,
    checkpoint_facts, checkpoints_dir, config_facts, form_state, fresh_over_checkpoint,
    metric_summary, metrics_facts, mode_element_id, mode_hint_key, mode_label_key,
    variant_element_id, variant_label_key,
};
use crate::ui::ProgressPresentation;

pub fn training_page(window: &mut Window, cx: &mut App) -> Div {
    let state = cx.global::<AppState>();
    let project = state.project.clone();
    let center = state.tasks.clone();
    let form = state.form.clone();
    let worker_ready = state.worker.read(cx).is_ready();
    let view = project.read(cx);
    let dir = view.dir().map(Path::to_path_buf);
    let log_dir = view.log_dir();
    let assets = view.survey().clone();
    let training = view.training().clone();
    let notes = view.notes().to_vec();
    let tasks = center.read(cx);
    let rows = tasks.rows().to_vec();
    let current = form.read(cx).clone();
    let gate = blocked_key(dir.is_some(), worker_ready, tasks.is_busy())
        .or_else(|| state.compute.read(cx).command_blocked_key(TaskKind::Train));
    let admission = form_state(&current, &assets, &training);
    let warn = fresh_over_checkpoint(&current, &training);
    let parameters = FormContext {
        project: &project,
        center: &center,
        form: &form,
        dir: dir.as_deref(),
        log_dir: log_dir.as_deref(),
        current: &current,
        saved_batch_size: training
            .state
            .as_ref()
            .map(|state| state.training_config.batch_size),
        admission,
        gate,
        warn,
    };
    let mut body = div()
        .w_full()
        .min_w(px(0.))
        .flex()
        .flex_col()
        .gap_5()
        .child(form_section(&parameters, window, cx))
        .child(monitoring_section(
            &project,
            &training,
            &rows,
            dir.is_some(),
            cx,
        ))
        .child(details_section(&training, &current, cx));
    if training.metrics_error.is_some() || training.state_error.is_some() || !notes.is_empty() {
        body = body.child(notes_card(&notes, &training, cx));
    }
    page_frame(
        "training-page",
        cx.t("page.training.title").to_string(),
        cx.t("workflow.training.description").to_string(),
        body,
        cx,
    )
}

struct FormContext<'a> {
    project: &'a Entity<ProjectState>,
    center: &'a Entity<TaskCenter>,
    form: &'a Entity<TrainingForm>,
    dir: Option<&'a Path>,
    log_dir: Option<&'a Path>,
    current: &'a TrainingForm,
    saved_batch_size: Option<u64>,
    admission: FormState,
    gate: Option<&'static str>,
    warn: bool,
}

fn form_section(ctx: &FormContext<'_>, window: &mut Window, cx: &mut App) -> Stateful<Div> {
    let mut content = section(
        "training-form",
        cx.t("training.form.title").to_string(),
        cx.t("workflow.training.form_description").to_string(),
        cx,
    )
    .child(preset_group(ctx, cx))
    .child(variant_group(ctx, cx))
    .child(
        div()
            .grid()
            .grid_cols(2)
            .gap_5()
            .child(epochs_row(ctx, window, cx))
            .child(batch_size_row(ctx, window, cx)),
    )
    .child(resume_row(ctx, cx));
    let compute = cx.global::<AppState>().compute.clone();
    let mut locations = div().grid().grid_cols(2).gap_5().child(path_field(
        cx.t("training.device.label").to_string(),
        selected_summary(compute.read(cx), cx),
        cx,
    ));
    if let Some(dir) = ctx.dir {
        locations = locations.child(path_field(
            cx.t("training.output.label").to_string(),
            checkpoints_dir(dir).display().to_string(),
            cx,
        ));
    }
    content = content.child(locations);
    if ctx.warn {
        content = content.child(
            div()
                .rounded(px(6.))
                .p_3()
                .bg(color(cx, "status.warning.bg"))
                .text_size(px(13.))
                .text_color(color(cx, "status.warning.fg"))
                .child(cx.t("training.warn.fresh_over_checkpoint").to_string()),
        );
    }
    content.child(submit_row(ctx, cx))
}

fn preset_group(ctx: &FormContext<'_>, cx: &mut App) -> Div {
    let mut choices = div().flex().flex_wrap().gap_3();
    for mode in ALL_MODES {
        choices = choices.child(preset_row(ctx, mode, cx));
    }
    div()
        .flex()
        .flex_col()
        .gap_2()
        .child(muted(cx.t("training.form.preset").to_string(), cx))
        .child(choices)
}

fn preset_row(ctx: &FormContext<'_>, mode: TrainingMode, cx: &mut App) -> Stateful<Div> {
    let form = ctx.form.clone();
    let selected = ctx.current.mode() == mode;
    let hint_key = match mode {
        TrainingMode::Baseline => "workflow.training.fast_hint",
        TrainingMode::MouthRoi => "workflow.training.mouth_hint",
        TrainingMode::Temporal => "workflow.training.temporal_hint",
    };
    let surface = choice_surface(selected, cx)
        .flex_1()
        .min_w(px(174.))
        .flex_col()
        .gap_2()
        .child(
            div()
                .flex()
                .items_center()
                .gap_2()
                .child(choice_indicator(selected, cx))
                .child(
                    div()
                        .font_weight(FontWeight::SEMIBOLD)
                        .child(cx.t(mode_label_key(mode)).to_string()),
                ),
        )
        .child(muted(cx.t(hint_key).to_string(), cx));
    let hover = color(cx, "surface.hover");
    let focus = color(cx, "action.primary.bg");
    radio(training_id("training-preset", mode_element_id(mode)), cx)
        .checked(selected)
        .on_toggle(move |_checked, _event, _window, cx| {
            form.update(cx, |form, cx| {
                form.set_mode(mode);
                cx.notify();
            });
        })
        .apply_focusable(surface)
        .hover(move |style| style.bg(hover))
        .focus_visible(move |style| style.border_color(focus))
}

fn variant_group(ctx: &FormContext<'_>, cx: &mut App) -> Div {
    let mut choices = div().flex().flex_wrap().gap_3();
    for variant in ALL_VARIANTS {
        let form = ctx.form.clone();
        let selected = ctx.current.variant() == variant;
        let surface = choice_surface(selected, cx)
            .flex_1()
            .min_w(px(220.))
            .items_center()
            .gap_2()
            .child(choice_indicator(selected, cx))
            .child(cx.t(variant_label_key(variant)).to_string());
        let hover = color(cx, "surface.hover");
        let focus = color(cx, "action.primary.bg");
        choices = choices.child(
            radio(
                training_id("training-variant", variant_element_id(variant)),
                cx,
            )
            .checked(selected)
            .on_toggle(move |_checked, _event, _window, cx| {
                form.update(cx, |form, cx| {
                    form.set_variant(variant);
                    cx.notify();
                });
            })
            .apply_focusable(surface)
            .hover(move |style| style.bg(hover))
            .focus_visible(move |style| style.border_color(focus)),
        );
    }
    div()
        .flex()
        .flex_col()
        .gap_2()
        .child(muted(cx.t("training.form.variant").to_string(), cx))
        .child(choices)
}

fn choice_surface(selected: bool, cx: &App) -> Div {
    div()
        .flex()
        .p_3()
        .rounded(px(6.))
        .border_1()
        .cursor_pointer()
        .bg(color(
            cx,
            if selected {
                "surface.sunken"
            } else {
                "surface.raised"
            },
        ))
        .border_color(color(
            cx,
            if selected {
                "action.primary.bg"
            } else {
                "border.default"
            },
        ))
}

fn choice_indicator(selected: bool, cx: &App) -> Div {
    let mut ring = div()
        .size(px(16.))
        .flex_shrink_0()
        .rounded_full()
        .border_1()
        .border_color(color(
            cx,
            if selected {
                "action.primary.bg"
            } else {
                "border.default"
            },
        ))
        .flex()
        .items_center()
        .justify_center();
    if selected {
        ring = ring.child(
            div()
                .size(px(8.))
                .rounded_full()
                .bg(color(cx, "action.primary.bg")),
        );
    }
    ring
}

/// Yororen owns the text/caret state; on_change only mirrors the accepted value.
fn epochs_row(ctx: &FormContext<'_>, window: &mut Window, cx: &mut App) -> Div {
    let form = ctx.form.clone();
    let field = number_input("training-epochs")
        .min(f64::from(MIN_EPOCHS))
        .max(f64::from(MAX_EPOCHS))
        .step(1.0)
        .value(f64::from(ctx.current.epochs()))
        .on_change(move |value, _window, cx| {
            form.update(cx, |form, cx| {
                form.set_epochs(value);
                cx.notify();
            });
        })
        .render(cx, window);
    div()
        .min_w(px(0.))
        .flex()
        .flex_col()
        .gap_2()
        .child(muted(cx.t("training.form.epochs").to_string(), cx))
        .child(div().w(px(156.)).child(field))
        .child(muted(cx.t("workflow.training.epochs_hint").to_string(), cx))
}

fn batch_size_row(ctx: &FormContext<'_>, window: &mut Window, cx: &mut App) -> Div {
    let form = ctx.form.clone();
    let field = number_input("training-batch-size")
        .min(1.0)
        .max(f64::from(u32::MAX))
        .step(1.0)
        .value(f64::from(ctx.current.batch_size()))
        .on_change(move |value, _window, cx| {
            form.update(cx, |form, cx| {
                form.set_batch_size(value);
                cx.notify();
            });
        })
        .render(cx, window);
    let mut row = div()
        .min_w(px(0.))
        .flex()
        .flex_col()
        .gap_2()
        .child(muted(cx.t("training.form.batch_size").to_string(), cx))
        .child(
            div()
                .debug_selector(|| "training-batch-size-field".into())
                .w(px(156.))
                .child(field),
        )
        .child(muted(cx.t("training.form.batch_size_hint").to_string(), cx));
    if ctx.current.resume()
        && let Some(saved) = ctx.saved_batch_size
    {
        row = row.child(muted(
            format!(
                "{}{}{saved}",
                cx.t("training.form.batch_size_resume"),
                label_separator(cx),
            ),
            cx,
        ));
    }
    row
}

fn resume_row(ctx: &FormContext<'_>, cx: &mut App) -> Div {
    let form = ctx.form.clone();
    let control = switch("training-resume", cx)
        .checked(ctx.current.resume())
        .on_toggle(move |checked, _event, _window, cx| {
            form.update(cx, |form, cx| {
                form.set_resume(checked);
                cx.notify();
            });
        })
        .render(cx);
    div()
        .min_w(px(0.))
        .flex()
        .flex_col()
        .gap_2()
        .child(muted(cx.t("training.form.resume").to_string(), cx))
        .child(
            div()
                .min_h(px(36.))
                .flex()
                .items_center()
                .gap_3()
                .child(control)
                .child(
                    cx.t(if ctx.current.resume() {
                        "workflow.training.resume_on"
                    } else {
                        "workflow.training.resume_off"
                    })
                    .to_string(),
                ),
        )
        .child(muted(cx.t("workflow.training.resume_hint").to_string(), cx))
}

fn submit_row(ctx: &FormContext<'_>, cx: &mut App) -> Div {
    let caption = cx.t(if ctx.current.resume() {
        "workflow.training.resume_submit"
    } else {
        "training.submit"
    });
    let action = button("training-submit", cx)
        .caption(caption)
        .variant(ActionVariantKind::Primary);
    let action = match (ctx.gate, ctx.admission, ctx.dir.zip(ctx.log_dir)) {
        (None, FormState::Ready, Some((dir, log_dir))) => {
            let center = ctx.center.clone();
            let project = ctx.project.clone();
            let form = ctx.form.clone();
            let dir = dir.to_path_buf();
            let log_dir = log_dir.to_path_buf();
            action.on_click(move |_event, _window, cx| {
                let request = form.read(cx).request(&dir);
                submit(
                    &center,
                    &project,
                    TaskKind::Train,
                    request,
                    &dir,
                    &log_dir,
                    cx,
                );
            })
        }
        _ => action.disabled(true),
    };
    let mut actions = div().flex().flex_wrap().gap_2().child(action.render(cx));
    if ctx.admission == FormState::Blocked("training.blocked.not_locked") {
        actions = actions.child(navigate_button(
            "training-go-assets",
            "workflow.training.go_assets",
            Page::Assets,
            cx,
        ));
    }
    let mut row = div()
        .flex()
        .flex_col()
        .gap_2()
        .pt_4()
        .border_t_1()
        .border_color(color(cx, "border.muted"))
        .child(actions);
    let reason = ctx.gate.or(match ctx.admission {
        FormState::Blocked(key) => Some(key),
        FormState::Ready => None,
    });
    if let Some(key) = reason {
        row = row.child(muted(cx.t(key).to_string(), cx));
    }
    row
}

fn monitoring_section(
    project: &Entity<ProjectState>,
    training: &TrainingSurvey,
    rows: &[TaskRow],
    has_project: bool,
    cx: &mut App,
) -> Stateful<Div> {
    let project = project.clone();
    let refresh = button("training-refresh", cx)
        .caption(cx.t("workflow.training.refresh"))
        .disabled(!has_project)
        .on_click(move |_event, _window, cx| {
            project.update(cx, |project, cx| {
                project.refresh();
                cx.notify();
            });
        })
        .render(cx);
    let mut content = section(
        "training-monitoring",
        cx.t("workflow.training.monitoring_title").to_string(),
        cx.t("workflow.training.monitoring_description").to_string(),
        cx,
    );
    if let Some(row) = latest_row(rows, TaskKind::Train) {
        let state = div()
            .flex()
            .flex_wrap()
            .items_center()
            .gap_2()
            .child(
                inline_muted(cx.t("workflow.training.latest_task").to_string(), cx)
                    .debug_selector(|| "training-latest-label".into()),
            )
            .child(
                badge(
                    "training-latest-status",
                    cx.t(status_key(row.status)).to_string(),
                    cx,
                )
                .variant(badge_variant(row.status))
                .render(cx),
            );
        content = content.child(state).child(training_position(row, cx));
        match row.status {
            TaskStatus::Queued | TaskStatus::Running => {
                content = content.child(live_block(row, cx))
            }
            TaskStatus::Failed => content = content.child(failure_block(row, cx)),
            TaskStatus::Completed | TaskStatus::Cancelled => {}
        }
    }
    if let Some(metrics) = &training.metrics {
        content = content.child(
            div()
                .flex()
                .flex_col()
                .gap_3()
                .child(
                    div()
                        .font_weight(FontWeight::MEDIUM)
                        .child(cx.t("workflow.training.saved_metrics").to_string()),
                )
                .child(facts_list("training-metrics", &metric_summary(metrics), cx)),
        );
    } else {
        content = content.child(muted(
            cx.t("workflow.training.metrics_empty").to_string(),
            cx,
        ));
    }
    let checkpoint_summary: Vec<_> = checkpoint_facts(training)
        .into_iter()
        .filter(|fact| {
            !matches!(
                fact.label,
                "training.checkpoint.latest" | "training.checkpoint.preview_count"
            )
        })
        .collect();
    content = content.child(
        div()
            .pt_4()
            .border_t_1()
            .border_color(color(cx, "border.muted"))
            .child(facts_list("training-checkpoint", &checkpoint_summary, cx)),
    );
    if let Some(checkpoint) = &training.checkpoint {
        content = content.child(path_field(
            cx.t("workflow.training.checkpoint_path").to_string(),
            checkpoint.path.display().to_string(),
            cx,
        ));
    } else {
        content = content.child(muted(cx.t("training.checkpoint.empty").to_string(), cx));
    }
    let mut actions = div().flex().flex_wrap().gap_2().child(refresh);
    if training.checkpoint.is_some() {
        actions = actions.child(navigate_button(
            "training-go-generate",
            "workflow.training.go_generate",
            Page::Generate,
            cx,
        ));
    }
    content.child(actions)
}

fn training_position(row: &TaskRow, cx: &mut App) -> Div {
    let training = row.training.as_ref();
    let epoch = training
        .and_then(|progress| progress.epoch())
        .map(|epoch| epoch.to_string())
        .unwrap_or_else(|| "—".into());
    let total_epochs = training
        .map(|progress| progress.total_epochs())
        .filter(|total| *total > 0)
        .map(|total| total.to_string())
        .unwrap_or_else(|| "—".into());
    let presentation = match training.and_then(|progress| progress.steps()) {
        Some(Progress {
            completed,
            total: Some(total),
        }) => ProgressPresentation::Counted { completed, total },
        _ if row.status.is_incomplete() => ProgressPresentation::Indeterminate,
        _ => ProgressPresentation::Empty,
    };
    let steps = match presentation {
        ProgressPresentation::Counted { completed, total } => format!("{completed} / {total}"),
        _ => "— / —".into(),
    };
    div()
        .w_full()
        .min_w(px(0.))
        .flex()
        .flex_col()
        .gap_2()
        .child(
            div()
                .flex()
                .flex_wrap()
                .items_center()
                .gap_x_4()
                .gap_y_1()
                .child(
                    inline_muted(
                        format!(
                            "{} {epoch} / {total_epochs}",
                            cx.t("workflow.training.live_epoch")
                        ),
                        cx,
                    )
                    .debug_selector(|| "training-epoch-position".into()),
                )
                .child(
                    inline_muted(
                        format!("{} {steps}", cx.t("workflow.training.live_step")),
                        cx,
                    )
                    .debug_selector(|| "training-epoch-steps".into()),
                ),
        )
        .child(progress_bar_with(row.task_id.as_str(), presentation, cx))
}

fn live_block(row: &TaskRow, cx: &mut App) -> Div {
    let id = row.task_id.as_str().to_owned();
    let mut content = div()
        .min_w(px(0.))
        .flex()
        .flex_col()
        .gap_2()
        .child(muted(cx.t(stage_key(&row.stage)).to_string(), cx));
    if let TaskStage::Training { loss, .. } = &row.stage {
        content = content.child(muted(
            format!("{} {loss:.6}", cx.t("training.metrics.total_loss")),
            cx,
        ));
    }
    content.child(cancel_row(&id, row, cx))
}

fn failure_block(row: &TaskRow, cx: &App) -> Div {
    let summary = match &row.failure {
        Some(failure) => match &failure.summary {
            Summary::Worker(written) => written.clone(),
            Summary::Key(key) => cx.t(key).to_string(),
        },
        None => cx.t(status_key(row.status)).to_string(),
    };
    div()
        .flex()
        .flex_col()
        .gap_1()
        .child(
            div()
                .w_full()
                .min_w(px(0.))
                .whitespace_normal()
                .text_color(color(cx, "status.danger.fg"))
                .child(summary),
        )
        .child(muted(cx.t("training.failure.hint").to_string(), cx))
}

fn details_section(
    training: &TrainingSurvey,
    current: &TrainingForm,
    cx: &mut App,
) -> Stateful<Div> {
    let ui = cx.global::<AppState>().ui.clone();
    let open = ui.read(cx).training_details;
    let toggle = button("training-details", cx)
        .caption(cx.t(if open {
            "workflow.details_hide"
        } else {
            "workflow.training.details_show"
        }))
        .on_click(move |_event, _window, cx| {
            ui.update(cx, |ui, cx| {
                ui.training_details = !ui.training_details;
                cx.notify();
            });
        })
        .render(cx);
    let mut content = section(
        "training-details-section",
        cx.t("workflow.training.details_title").to_string(),
        cx.t("workflow.training.details_description").to_string(),
        cx,
    )
    .child(div().flex().child(toggle));
    if !open {
        return content;
    }
    content = content.child(muted(cx.t(mode_hint_key(current.mode())).to_string(), cx));
    let checkpoint_details: Vec<_> = checkpoint_facts(training)
        .into_iter()
        .filter(|fact| fact.label == "training.checkpoint.latest")
        .collect();
    if !checkpoint_details.is_empty() {
        content = content.child(facts_list(
            "training-checkpoint-details",
            &checkpoint_details,
            cx,
        ));
    }
    if let Some(state) = &training.state {
        content = content
            .child(
                div()
                    .font_weight(FontWeight::MEDIUM)
                    .child(cx.t("training.config.title").to_string()),
            )
            .child(facts_list("training-config", &config_facts(state), cx));
    } else {
        content = content.child(muted(cx.t("training.config.empty").to_string(), cx));
    }
    if let Some(metrics) = &training.metrics {
        let mut metrics_details = vec![Fact {
            label: "training.metrics.source",
            value: FactValue::Text(format!("step-{:08}.json", metrics.global_step)),
        }];
        metrics_details.extend(metrics_facts(metrics).into_iter().filter(|fact| {
            !matches!(
                fact.label,
                "training.metrics.mode"
                    | "training.metrics.total_loss"
                    | "training.metrics.samples_per_second"
            )
        }));
        content = content
            .child(
                div()
                    .font_weight(FontWeight::MEDIUM)
                    .child(cx.t("workflow.training.metrics_details").to_string()),
            )
            .child(muted(
                cx.t("workflow.training.metrics_details_hint").to_string(),
                cx,
            ))
            .child(facts_list("training-metrics-details", &metrics_details, cx));
    }
    if training.preview_count > 0 {
        content = content.child(muted(
            format!(
                "{}{}{}",
                cx.t("workflow.training.preview_data"),
                label_separator(cx),
                training.preview_count,
            ),
            cx,
        ));
    }
    content
}

fn navigate_button(id: &'static str, key: &'static str, page: Page, cx: &mut App) -> Stateful<Div> {
    let navigation = cx.global::<AppState>().navigation.clone();
    button(id, cx)
        .caption(cx.t(key))
        .on_click(move |_event, _window, cx| {
            navigation.update(cx, |navigation, cx| {
                if navigation.select(page) {
                    cx.notify();
                }
            });
        })
        .render(cx)
}

fn notes_card(notes: &[Note], training: &TrainingSurvey, cx: &App) -> Stateful<Div> {
    let mut content = section(
        "training-notes",
        cx.t("training.notes.title").to_string(),
        "",
        cx,
    );
    if let Some(detail) = &training.metrics_error {
        content = content.child(muted(
            format!(
                "{}{}{detail}",
                cx.t("training.note.metrics_failed"),
                label_separator(cx),
            ),
            cx,
        ));
    }
    if let Some(detail) = &training.state_error {
        content = content.child(muted(
            format!(
                "{}{}{detail}",
                cx.t("training.note.state_failed"),
                label_separator(cx),
            ),
            cx,
        ));
    }
    for (index, note) in notes.iter().enumerate() {
        let line = match &note.detail {
            Some(detail) => format!("{}{}{detail}", cx.t(note.key), label_separator(cx)),
            None => cx.t(note.key).to_string(),
        };
        content =
            content.child(muted(line, cx).id(SharedString::from(format!("training-note-{index}"))));
    }
    content
}

fn training_id(name: &str, member: &str) -> SharedString {
    SharedString::from(format!("{name}-{member}"))
}
