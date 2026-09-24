//! Prepare a video, inspect real worker results, and move into training.
//! Rendering reads ProjectState's snapshot; all commands share the supervisor.

use std::path::Path;

use feathertalk_domain::{ProbeMediaParams, ProjectDirParams, Request, TaskKind, TaskStatus};
use gpui::{
    App, Div, Entity, FontWeight, InteractiveElement, ParentElement, SharedString, Stateful,
    Styled, div, px,
};
use yororen_ui::ActionVariantKind;
use yororen_ui::headless::badge::{BadgeVariant, badge};
use yororen_ui::headless::button::button;
use yororen_ui::i18n::Translate;

use crate::assets::{AssetSurvey, Step, StepState, facts, request, step_state};
use crate::components::activity::{Activity, activity};
use crate::components::facts::facts_list;
use crate::components::tasks_page::{badge_variant, cancel_row, progress_bar};
use crate::components::ui::{
    inline_muted, label_separator, muted, page_frame, path_field, section,
};
use crate::facts::{Fact, FactValue};
use crate::navigation::Page;
use crate::picker::{pick_project_dir, pick_source_video};
use crate::project::ProjectState;
use crate::state::AppState;
use crate::submit::submit;
use crate::tasks::{
    Note, Summary, TaskCenter, TaskRow, blocked_key, latest_row, stage_key, status_key,
};
use crate::theme::color;

pub fn assets_page(cx: &mut App) -> Div {
    let state = cx.global::<AppState>();
    let project = state.project.clone();
    let center = state.tasks.clone();
    let worker_ready = state.worker.read(cx).is_ready();
    let view = project.read(cx);
    let dir = view.dir().map(Path::to_path_buf);
    let log_dir = view.log_dir();
    let source_video = view.source_video().map(Path::to_path_buf);
    let survey = view.survey().clone();
    let notes = view.notes().to_vec();
    let tasks = center.read(cx);
    let rows = tasks.rows().to_vec();
    let busy = tasks.is_busy();
    let gate = blocked_key(dir.is_some(), worker_ready, busy);
    let ctx = StepContext {
        project: &project,
        center: &center,
        dir: dir.as_deref(),
        log_dir: log_dir.as_deref(),
        source_video: source_video.as_deref(),
        survey: &survey,
        rows: &rows,
        gate,
        busy,
    };
    let mut body = div()
        .w_full()
        .flex()
        .flex_col()
        .gap_5()
        .min_w(px(0.))
        .child(selection_section(&ctx, cx))
        .child(steps_section(&ctx, cx));
    if survey.is_locked() {
        body = body.child(continue_section(cx));
    }
    body = body.child(package_section(&ctx, cx));
    if survey.manifest_error.is_some() || !notes.is_empty() {
        body = body.child(notes_card(&notes, survey.manifest_error.as_deref(), cx));
    }
    page_frame(
        "assets-page",
        cx.t("page.assets.title").to_string(),
        cx.t("workflow.assets.description").to_string(),
        body,
        cx,
    )
}

struct StepContext<'a> {
    project: &'a Entity<ProjectState>,
    center: &'a Entity<TaskCenter>,
    dir: Option<&'a Path>,
    log_dir: Option<&'a Path>,
    source_video: Option<&'a Path>,
    survey: &'a AssetSurvey,
    rows: &'a [TaskRow],
    gate: Option<&'static str>,
    busy: bool,
}

fn selection_section(ctx: &StepContext<'_>, cx: &mut App) -> Stateful<Div> {
    let project = ctx.project.clone();
    let center = ctx.center.clone();
    let pick_dir = button("assets-pick-project", cx)
        .caption(cx.t("assets.project.pick"))
        .disabled(ctx.busy)
        .on_click(move |_event, _window, cx| pick_project_dir(&project, &center, cx))
        .render(cx);
    let project = ctx.project.clone();
    let refresh = button("assets-refresh", cx)
        .caption(cx.t("assets.refresh"))
        .disabled(ctx.dir.is_none() || ctx.busy)
        .on_click(move |_event, _window, cx| {
            if cx.global::<AppState>().tasks.read(cx).is_busy() {
                return;
            }
            project.update(cx, |project, cx| {
                project.refresh();
                cx.notify();
            });
        })
        .render(cx);
    let project = ctx.project.clone();
    let pick_video = button("assets-pick-video", cx)
        .caption(cx.t("assets.video.pick"))
        .disabled(ctx.dir.is_none() || ctx.busy)
        .on_click(move |_event, _window, cx| {
            if cx.global::<AppState>().tasks.read(cx).is_busy() || project.read(cx).dir().is_none()
            {
                return;
            }
            pick_source_video(&project, cx);
        })
        .render(cx);
    let probe = auxiliary_button(ctx, TaskKind::ProbeMedia, cx);
    let mut content = section(
        "assets-selection",
        cx.t("workflow.assets.source_title").to_string(),
        cx.t("workflow.assets.source_description").to_string(),
        cx,
    )
    .child(
        div()
            .flex()
            .flex_wrap()
            .items_end()
            .gap_3()
            .child(
                path_field(
                    cx.t("assets.project.label").to_string(),
                    path_text(ctx.dir, "assets.project.unset", cx),
                    cx,
                )
                .min_w(px(240.)),
            )
            .child(
                div()
                    .flex()
                    .gap_2()
                    .flex_shrink_0()
                    .child(pick_dir)
                    .child(refresh),
            ),
    )
    .child(
        div()
            .flex()
            .flex_wrap()
            .items_end()
            .gap_3()
            .pt_4()
            .border_t_1()
            .border_color(color(cx, "border.muted"))
            .child(
                path_field(
                    cx.t("assets.video.label").to_string(),
                    path_text(ctx.source_video, "assets.video.unset", cx),
                    cx,
                )
                .min_w(px(240.)),
            )
            .child(
                div()
                    .flex()
                    .gap_2()
                    .flex_shrink_0()
                    .child(pick_video)
                    .child(probe),
            ),
    );
    if let Some(reason) = auxiliary_gate(ctx, TaskKind::ProbeMedia, cx) {
        content = content.child(muted(cx.t(reason).to_string(), cx));
    }
    if let Some(row) = latest_row(ctx.rows, TaskKind::ProbeMedia) {
        content = content.child(auxiliary_feedback(row, cx));
    }
    content
}

fn path_text(path: Option<&Path>, unset: &'static str, cx: &App) -> String {
    path.map(|path| path.display().to_string())
        .unwrap_or_else(|| cx.t(unset).to_string())
}

fn steps_section(ctx: &StepContext<'_>, cx: &mut App) -> Stateful<Div> {
    let mut content = section(
        "assets-steps",
        cx.t("workflow.assets.steps_title").to_string(),
        cx.t("workflow.assets.steps_description").to_string(),
        cx,
    );
    if let Some(key) = ctx.gate {
        content = content.child(muted(cx.t(key).to_string(), cx));
    }
    for (index, step) in Step::ALL.into_iter().enumerate() {
        content = content.child(step_row(ctx, step, index, cx));
    }
    content
}

fn step_row(ctx: &StepContext<'_>, step: Step, index: usize, cx: &mut App) -> Div {
    let state = step_state(step, ctx.survey, ctx.source_video.is_some());
    let command_gate = cx
        .global::<AppState>()
        .compute
        .read(cx)
        .command_blocked_key(step.kind());
    let prerequisite = match state {
        StepState::Blocked(key) => Some(key),
        // Existing normalized files can be present without a selected source.
        // Redo still needs an input; it must not become an enabled no-op.
        StepState::Done if step == Step::Normalize && ctx.source_video.is_none() => {
            Some("assets.blocked.no_source")
        }
        StepState::Ready | StepState::Done | StepState::Locked => None,
    };
    let reason = ctx.gate.or(command_gate).or(prerequisite);
    let recent = activity(ctx.rows, step.kind());
    let (status, variant) = match recent {
        Activity::Live(task) | Activity::Broken(task) => (
            cx.t(status_key(task.status)).to_string(),
            badge_variant(task.status),
        ),
        Activity::Quiet => state_badge(state, cx),
    };
    let title = div()
        .flex()
        .flex_wrap()
        .items_center()
        .gap_2()
        .child(
            div()
                .font_weight(FontWeight::SEMIBOLD)
                .child(cx.t(step.label_key()).to_string()),
        )
        .child(
            badge(step_id("assets-state", step), status, cx)
                .variant(variant)
                .render(cx),
        );
    let mut details = div()
        .min_w(px(0.))
        .flex_1()
        .flex()
        .flex_col()
        .gap_2()
        .child(
            div()
                .flex()
                .flex_wrap()
                .items_center()
                .justify_between()
                .gap_3()
                .child(title)
                .child(run_button(ctx, step, state, reason, cx)),
        )
        .child(muted(
            cx.t(match step {
                Step::Normalize => "workflow.assets.normalize_hint",
                Step::ExtractFrames => "workflow.assets.frames_hint",
                Step::ExtractFeatures => "workflow.assets.features_hint",
                Step::Lock => "workflow.assets.lock_hint",
            })
            .to_string(),
            cx,
        ));
    if state != StepState::Locked
        && ctx.gate.is_none()
        && let Some(key) = command_gate.or(prerequisite)
    {
        details = details.child(muted(cx.t(key).to_string(), cx));
    }
    match recent {
        Activity::Live(task) => {
            details = details
                .child(muted(cx.t(stage_key(&task.stage)).to_string(), cx))
                .child(live_block(task, cx));
        }
        Activity::Broken(task) => details = details.child(failure_line(task, cx)),
        Activity::Quiet => {}
    }
    let mut row = div()
        .w_full()
        .min_w(px(0.))
        .flex()
        .items_start()
        .gap_3()
        .child(
            div()
                .size(px(30.))
                .flex_shrink_0()
                .rounded(px(6.))
                .bg(color(cx, "surface.sunken"))
                .text_color(color(cx, "content.secondary"))
                .text_size(px(12.))
                .font_weight(FontWeight::SEMIBOLD)
                .flex()
                .items_center()
                .justify_center()
                .child(format!("{:02}", index + 1)),
        )
        .child(details);
    if index > 0 {
        row = row
            .pt_4()
            .border_t_1()
            .border_color(color(cx, "border.muted"));
    }
    row
}

fn run_button(
    ctx: &StepContext<'_>,
    step: Step,
    state: StepState,
    reason: Option<&'static str>,
    cx: &mut App,
) -> Stateful<Div> {
    let caption_key = match state {
        StepState::Done => "workflow.assets.redo",
        StepState::Locked => "assets.state.locked",
        StepState::Blocked(_) | StepState::Ready => "workflow.assets.run_step",
    };
    let action = button(step_id("assets-run", step), cx)
        .caption(cx.t(caption_key))
        .variant(if state == StepState::Ready && reason.is_none() {
            ActionVariantKind::Primary
        } else {
            ActionVariantKind::Neutral
        });
    let action = match (reason, state.is_submittable(), ctx.dir.zip(ctx.log_dir)) {
        (None, true, Some((dir, log_dir))) => {
            let center = ctx.center.clone();
            let project = ctx.project.clone();
            let dir = dir.to_path_buf();
            let log_dir = log_dir.to_path_buf();
            action.on_click(move |_event, _window, cx| {
                let source = project.read(cx).source_video().map(Path::to_path_buf);
                if let Some(request) = request(step, &dir, source.as_deref()) {
                    submit(&center, &project, step.kind(), request, &dir, &log_dir, cx);
                }
            })
        }
        _ => action.disabled(true),
    };
    action.render(cx).flex_shrink_0()
}

fn state_badge(state: StepState, cx: &App) -> (String, BadgeVariant) {
    let (key, variant) = match state {
        StepState::Blocked(_) => ("workflow.assets.waiting", BadgeVariant::Neutral),
        StepState::Ready => ("assets.state.ready", BadgeVariant::Info),
        StepState::Done => ("workflow.assets.artifacts_present", BadgeVariant::Success),
        StepState::Locked => ("assets.state.locked", BadgeVariant::Success),
    };
    (cx.t(key).to_string(), variant)
}

fn continue_section(cx: &mut App) -> Div {
    let navigation = cx.global::<AppState>().navigation.clone();
    let next = button("assets-continue-training", cx)
        .caption(cx.t("workflow.assets.continue_training"))
        .variant(ActionVariantKind::Primary)
        .on_click(move |_event, _window, cx| {
            navigation.update(cx, |navigation, cx| {
                if navigation.select(Page::Training) {
                    cx.notify();
                }
            });
        })
        .render(cx);
    div()
        .flex()
        .flex_wrap()
        .items_center()
        .justify_between()
        .gap_3()
        .p_4()
        .rounded(px(8.))
        .bg(color(cx, "status.success.bg"))
        .child(
            div()
                .flex_1()
                .min_w(px(240.))
                .flex()
                .flex_col()
                .gap_1()
                .child(
                    div()
                        .font_weight(FontWeight::SEMIBOLD)
                        .text_color(color(cx, "status.success.fg"))
                        .child(cx.t("workflow.assets.ready_title").to_string()),
                )
                .child(muted(
                    cx.t("workflow.assets.ready_description").to_string(),
                    cx,
                )),
        )
        .child(next)
}

fn package_section(ctx: &StepContext<'_>, cx: &mut App) -> Stateful<Div> {
    let ui = cx.global::<AppState>().ui.clone();
    let open = ui.read(cx).asset_details;
    let toggle = button("assets-details", cx)
        .caption(cx.t(if open {
            "workflow.details_hide"
        } else {
            "workflow.assets.details_show"
        }))
        .on_click(move |_event, _window, cx| {
            ui.update(cx, |ui, cx| {
                ui.asset_details = !ui.asset_details;
                cx.notify();
            });
        })
        .render(cx);
    let validate = auxiliary_button(ctx, TaskKind::ValidateProject, cx);
    let mut content = section(
        "assets-package",
        cx.t("workflow.assets.package_title").to_string(),
        cx.t("workflow.assets.package_description").to_string(),
        cx,
    );
    let package = facts(ctx.survey);
    let summary: Vec<_> = package
        .iter()
        .filter(|fact| {
            matches!(
                fact.label,
                "assets.package.state"
                    | "assets.package.frames"
                    | "assets.package.resolution"
                    | "assets.package.duration"
            )
        })
        .cloned()
        .collect();
    if !summary.is_empty() {
        content = content.child(facts_list("assets-package", &summary, cx));
    }
    content = content.child(
        div()
            .flex()
            .flex_wrap()
            .gap_2()
            .child(validate)
            .child(toggle),
    );
    if let Some(key) = auxiliary_gate(ctx, TaskKind::ValidateProject, cx) {
        content = content.child(muted(cx.t(key).to_string(), cx));
    }
    if let Some(row) = latest_row(ctx.rows, TaskKind::ValidateProject) {
        content = content.child(auxiliary_feedback(row, cx));
    }
    if open {
        let advanced: Vec<_> = package
            .into_iter()
            .filter(|fact| {
                !matches!(
                    fact.label,
                    "assets.package.state"
                        | "assets.package.frames"
                        | "assets.package.resolution"
                        | "assets.package.duration"
                )
            })
            .collect();
        content = content
            .child(muted(cx.t("workflow.assets.fixed_format").to_string(), cx))
            .child(facts_list("assets-package-details", &advanced, cx));
    }
    content
}

fn auxiliary_gate(ctx: &StepContext<'_>, kind: TaskKind, cx: &App) -> Option<&'static str> {
    ctx.gate
        .or_else(|| {
            cx.global::<AppState>()
                .compute
                .read(cx)
                .command_blocked_key(kind)
        })
        .or_else(|| match kind {
            TaskKind::ProbeMedia if ctx.source_video.is_none() => Some("assets.blocked.no_source"),
            TaskKind::ValidateProject if !ctx.survey.is_locked() => {
                Some("workflow.assets.validate_prerequisite")
            }
            _ => None,
        })
}

/// The inspections share the same history and cancellation as preparation.
fn auxiliary_button(ctx: &StepContext<'_>, kind: TaskKind, cx: &mut App) -> Stateful<Div> {
    let (id, key) = match kind {
        TaskKind::ProbeMedia => ("assets-probe-media", "workflow.assets.inspect_video"),
        _ => (
            "assets-validate-project",
            "workflow.assets.validate_project",
        ),
    };
    let action = button(id, cx).caption(cx.t(key));
    let action = match (auxiliary_gate(ctx, kind, cx), ctx.dir.zip(ctx.log_dir)) {
        (None, Some((dir, log_dir))) => {
            let center = ctx.center.clone();
            let project = ctx.project.clone();
            let dir = dir.to_path_buf();
            let log_dir = log_dir.to_path_buf();
            action.on_click(move |_event, _window, cx| {
                let request = match kind {
                    TaskKind::ProbeMedia => {
                        let Some(input) = project.read(cx).source_video().map(Path::to_path_buf)
                        else {
                            return;
                        };
                        Request::ProbeMedia(ProbeMediaParams { input })
                    }
                    _ => Request::ValidateProject(ProjectDirParams {
                        project_dir: dir.clone(),
                    }),
                };
                submit(&center, &project, kind, request, &dir, &log_dir, cx);
            })
        }
        _ => action.disabled(true),
    };
    action.render(cx)
}

fn auxiliary_feedback(row: &TaskRow, cx: &mut App) -> Div {
    let title_key = if row.kind == TaskKind::ProbeMedia {
        "workflow.assets.probe_result"
    } else {
        "workflow.assets.validation_result"
    };
    let mut content = div()
        .flex()
        .flex_col()
        .gap_3()
        .pt_3()
        .border_t_1()
        .border_color(color(cx, "border.muted"))
        .child(
            div()
                .flex()
                .flex_wrap()
                .items_center()
                .gap_2()
                .child(
                    inline_muted(cx.t(title_key).to_string(), cx)
                        .debug_selector(|| "assets-result-title".into()),
                )
                .child(
                    badge(
                        SharedString::from(format!(
                            "assets-result-status-{}",
                            row.task_id.as_str()
                        )),
                        cx.t(status_key(row.status)).to_string(),
                        cx,
                    )
                    .variant(badge_variant(row.status))
                    .render(cx),
                ),
        );
    match row.status {
        TaskStatus::Queued | TaskStatus::Running => {
            content = content
                .child(muted(cx.t(stage_key(&row.stage)).to_string(), cx))
                .child(live_block(row, cx));
        }
        TaskStatus::Failed => content = content.child(failure_line(row, cx)),
        TaskStatus::Completed => {
            if row.kind == TaskKind::ProbeMedia {
                if let Some(result) = &row.result {
                    content =
                        content.child(facts_list("assets-probe-result", &probe_facts(result), cx));
                }
                content = content.child(muted(cx.t("workflow.assets.probe_scope").to_string(), cx));
            } else {
                content = content.child(muted(
                    cx.t("workflow.assets.validation_passed").to_string(),
                    cx,
                ));
            }
        }
        TaskStatus::Cancelled => {}
    }
    content
}

fn probe_facts(result: &serde_json::Value) -> Vec<Fact> {
    let mut facts = Vec::new();
    if let Some(duration) = result
        .pointer("/format/duration_seconds")
        .and_then(serde_json::Value::as_f64)
    {
        facts.push(Fact {
            label: "workflow.assets.probe_duration",
            value: FactValue::Text(format!("{duration:.1}")),
        });
    }
    if let Some(format) = result
        .pointer("/format/format_name")
        .and_then(serde_json::Value::as_str)
    {
        facts.push(Fact {
            label: "workflow.assets.probe_container",
            value: FactValue::Text(format.to_owned()),
        });
    }
    if let Some((width, height)) = result
        .pointer("/video/width")
        .and_then(serde_json::Value::as_u64)
        .zip(
            result
                .pointer("/video/height")
                .and_then(serde_json::Value::as_u64),
        )
    {
        facts.push(Fact {
            label: "assets.package.resolution",
            value: FactValue::Text(format!("{width} × {height}")),
        });
    }
    if let Some((numerator, denominator)) = result
        .pointer("/video/frame_rate/numerator")
        .and_then(serde_json::Value::as_f64)
        .zip(
            result
                .pointer("/video/frame_rate/denominator")
                .and_then(serde_json::Value::as_f64),
        )
        && denominator > 0.
    {
        facts.push(Fact {
            label: "assets.package.fps",
            value: FactValue::Text(format!("{:.2}", numerator / denominator)),
        });
    }
    for (pointer, label) in [
        ("/audio/sample_rate", "assets.package.sample_rate"),
        ("/audio/channels", "assets.package.channels"),
    ] {
        if let Some(value) = result.pointer(pointer).and_then(serde_json::Value::as_u64) {
            facts.push(Fact {
                label,
                value: FactValue::Text(value.to_string()),
            });
        }
    }
    facts
}

fn live_block(row: &TaskRow, cx: &mut App) -> Div {
    let id = row.task_id.as_str().to_owned();
    div()
        .flex()
        .flex_col()
        .gap_2()
        .min_w(px(0.))
        .child(progress_bar(&id, row, cx))
        .child(cancel_row(&id, row, cx))
}

fn failure_line(row: &TaskRow, cx: &App) -> Div {
    let summary = match &row.failure {
        Some(failure) => match &failure.summary {
            Summary::Worker(written) => written.clone(),
            Summary::Key(key) => cx.t(key).to_string(),
        },
        None => cx.t(status_key(row.status)).to_string(),
    };
    div()
        .min_w(px(0.))
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
        .child(muted(cx.t("assets.failure.hint").to_string(), cx))
}

fn step_id(name: &str, step: Step) -> SharedString {
    SharedString::from(format!("{name}-{}", step.element_id()))
}

fn notes_card(notes: &[Note], manifest_error: Option<&str>, cx: &App) -> Stateful<Div> {
    let mut content = section(
        "assets-notes",
        cx.t("assets.notes.title").to_string(),
        "",
        cx,
    );
    if let Some(detail) = manifest_error {
        content = content.child(muted(
            format!(
                "{}{}{detail}",
                cx.t("assets.note.manifest_failed"),
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
            content.child(muted(line, cx).id(SharedString::from(format!("assets-note-{index}"))));
    }
    content
}
