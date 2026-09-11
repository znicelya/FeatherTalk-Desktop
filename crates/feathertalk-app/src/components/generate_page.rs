//! Choose render inputs, inspect live progress, and open published videos.
//! Audio remains a muxed track: mouth features come from the locked asset package.

use std::path::{Path, PathBuf};

use feathertalk_domain::{TaskKind, TaskStage, TaskStatus};
use feathertalk_project::AssetManifest;
use gpui::{
    App, Div, Entity, FontWeight, InteractiveElement, ParentElement, SharedString, Stateful,
    Styled, Window, div, px,
};
use yororen_ui::ActionVariantKind;
use yororen_ui::headless::badge::badge;
use yororen_ui::headless::button::button;
use yororen_ui::headless::number_input::number_input;
use yororen_ui::headless::radio::radio;
use yororen_ui::i18n::Translate;

use crate::assets::AssetSurvey;
use crate::components::control_renderers::RadioControlExt;
use crate::components::facts::facts_list;
use crate::components::tasks_page::{badge_variant, cancel_row, progress_bar};
use crate::components::ui::{
    inline_muted, label_separator, muted, page_frame, path_field, section,
};
use crate::facts::{Fact, FactValue};
use crate::generate::{
    FormState, GenerateForm, MAX_PREVIEW_FRAMES, MIN_PREVIEW_FRAMES, PickedCheckpoint, RENDER_FPS,
    RenderRequest, RenderSurvey, RenderedVideo, audio_path, checkpoint_path, ensure_renders_dir,
    form_state, output_path,
};
use crate::navigation::Page;
use crate::picker::{pick_audio_track, pick_checkpoint_dir, pick_output_path};
use crate::project::ProjectState;
use crate::state::AppState;
use crate::submit::submit;
use crate::tasks::{
    Note, Summary, TaskCenter, TaskRow, blocked_key, latest_row, stage_key, status_key,
};
use crate::theme::color;
use crate::training::{TrainingSurvey, mode_value};

pub fn generate_page(window: &mut Window, cx: &mut App) -> Div {
    let state = cx.global::<AppState>();
    let project = state.project.clone();
    let center = state.tasks.clone();
    let form = state.generate.clone();
    let worker_ready = state.worker.read(cx).is_ready();
    let selection = project.read(cx);
    let dir = selection.dir().map(Path::to_path_buf);
    let log_dir = selection.log_dir();
    let assets = selection.survey().clone();
    let training = selection.training().clone();
    let picked_checkpoint = selection.checkpoint().cloned();
    let picked_audio = selection.audio().map(Path::to_path_buf);
    let picked_output = selection.output().map(Path::to_path_buf);
    let renders = selection.renders().clone();
    let notes = selection.notes().to_vec();
    let current = form.read(cx).clone();
    let tasks = center.read(cx);
    let rows = tasks.rows().to_vec();
    let busy = tasks.is_busy();
    let gate = blocked_key(dir.is_some(), worker_ready, busy)
        .or_else(|| state.compute.read(cx).command_blocked_key(TaskKind::Render));
    let admission = form_state(
        &assets,
        &training,
        picked_checkpoint.as_ref(),
        picked_audio.as_deref(),
        dir.as_deref().unwrap_or(Path::new("")),
    );
    let parameters = FormContext {
        project: &project,
        center: &center,
        form: &form,
        dir: dir.as_deref(),
        log_dir: log_dir.as_deref(),
        assets: &assets,
        training: &training,
        picked_checkpoint: picked_checkpoint.as_ref(),
        picked_audio: picked_audio.as_deref(),
        picked_output: picked_output.as_deref(),
        renders: &renders,
        current: &current,
        admission,
        gate,
        busy,
    };
    let mut body = div()
        .w_full()
        .min_w(px(0.))
        .flex()
        .flex_col()
        .gap_5()
        .child(form_card(&parameters, window, cx));
    if let Some(row) = latest_row(&rows, TaskKind::Render) {
        body = body.child(activity_section(row, cx));
    }
    body = body
        .child(renders_card(&renders, &rows, cx))
        .child(format_section(
            assets.manifest.as_ref().filter(|_| assets.is_locked()),
            cx,
        ));
    if !notes.is_empty() {
        body = body.child(notes_card(&notes, cx));
    }
    page_frame(
        "generate-page",
        cx.t("page.generate.title").to_string(),
        cx.t("workflow.generate.description").to_string(),
        body,
        cx,
    )
}

struct FormContext<'a> {
    project: &'a Entity<ProjectState>,
    center: &'a Entity<TaskCenter>,
    form: &'a Entity<GenerateForm>,
    dir: Option<&'a Path>,
    log_dir: Option<&'a Path>,
    assets: &'a AssetSurvey,
    training: &'a TrainingSurvey,
    picked_checkpoint: Option<&'a PickedCheckpoint>,
    picked_audio: Option<&'a Path>,
    picked_output: Option<&'a Path>,
    renders: &'a RenderSurvey,
    current: &'a GenerateForm,
    admission: FormState,
    gate: Option<&'static str>,
    busy: bool,
}

fn form_card(ctx: &FormContext<'_>, window: &mut Window, cx: &mut App) -> Stateful<Div> {
    let mut content = section(
        "generate-form",
        cx.t("generate.form.title").to_string(),
        cx.t("workflow.generate.form_description").to_string(),
        cx,
    )
    .child(checkpoint_row(ctx, cx))
    .child(audio_row(ctx, cx))
    .child(
        div()
            .p_3()
            .rounded(px(6.))
            .bg(color(cx, "surface.sunken"))
            .child(muted(cx.t("generate.audio.note").to_string(), cx)),
    )
    .child(preview_row(ctx, window, cx));
    if let Some(dir) = ctx.dir {
        content = content.child(output_row(ctx, dir, cx));
    }
    content.child(submit_row(ctx, cx))
}

fn checkpoint_row(ctx: &FormContext<'_>, cx: &mut App) -> Div {
    let project = ctx.project.clone();
    let pick = button("generate-checkpoint-pick", cx)
        .caption(cx.t("generate.checkpoint.pick"))
        .disabled(ctx.dir.is_none() || ctx.busy)
        .on_click(move |_event, _window, cx| {
            if cx.global::<AppState>().tasks.read(cx).is_busy() || project.read(cx).dir().is_none()
            {
                return;
            }
            pick_checkpoint_dir(&project, cx);
        })
        .render(cx);
    let project = ctx.project.clone();
    let latest = button("generate-checkpoint-latest", cx)
        .caption(cx.t("generate.checkpoint.use_latest"))
        .disabled(ctx.picked_checkpoint.is_none() || ctx.busy)
        .on_click(move |_event, _window, cx| {
            if cx.global::<AppState>().tasks.read(cx).is_busy() {
                return;
            }
            project.update(cx, |project, cx| {
                project.use_latest_checkpoint();
                cx.notify();
            });
        })
        .render(cx);
    let value = checkpoint_path(ctx.picked_checkpoint, ctx.training)
        .map(|path| path.display().to_string())
        .unwrap_or_else(|| cx.t("workflow.generate.checkpoint_unset").to_string());
    div()
        .min_w(px(0.))
        .flex()
        .flex_col()
        .gap_2()
        .child(
            div()
                .flex()
                .flex_wrap()
                .items_end()
                .gap_3()
                .child(
                    path_field(cx.t("generate.checkpoint.label").to_string(), value, cx)
                        .min_w(px(230.)),
                )
                .child(
                    div()
                        .flex()
                        .gap_2()
                        .flex_shrink_0()
                        .child(latest)
                        .child(pick),
                ),
        )
        .child(muted(checkpoint_line(ctx, cx), cx))
}

fn checkpoint_line(ctx: &FormContext<'_>, cx: &App) -> String {
    if checkpoint_path(ctx.picked_checkpoint, ctx.training).is_none() {
        return cx.t("generate.checkpoint.none").to_string();
    }
    let source = if ctx.picked_checkpoint.is_some() {
        "workflow.generate.checkpoint_picked"
    } else {
        "workflow.generate.checkpoint_latest"
    };
    let state = match ctx.picked_checkpoint {
        Some(picked) => picked.state.as_ref(),
        None => ctx.training.state.as_ref(),
    };
    let mut line = cx.t(source).to_string();
    if let Some(state) = state {
        let mode = match mode_value(&state.training_config.mode) {
            FactValue::Text(measured) => measured,
            FactValue::Key(key) => cx.t(key).to_string(),
        };
        line = format!(
            "{line} · {} {} · {} {} · {mode}",
            cx.t("generate.checkpoint.step"),
            state.global_step,
            cx.t("generate.checkpoint.epoch"),
            state.epoch,
        );
    } else if ctx.picked_checkpoint.is_some() {
        line = format!("{line} · {}", cx.t("generate.checkpoint.unreadable"));
    } else if let Some(checkpoint) = &ctx.training.checkpoint {
        line = format!(
            "{line} · {} {}",
            cx.t("generate.checkpoint.step"),
            checkpoint.step
        );
    }
    line
}

fn audio_row(ctx: &FormContext<'_>, cx: &mut App) -> Div {
    let project = ctx.project.clone();
    let resolved = ctx
        .dir
        .and_then(|dir| audio_path(ctx.picked_audio, dir, ctx.assets));
    let value = resolved
        .as_ref()
        .map(|path| path.display().to_string())
        .unwrap_or_else(|| cx.t("generate.audio.missing").to_string());
    let pick = button("generate-audio-pick", cx)
        .caption(cx.t("generate.audio.pick"))
        .disabled(ctx.dir.is_none() || ctx.busy)
        .on_click(move |_event, _window, cx| {
            if cx.global::<AppState>().tasks.read(cx).is_busy() || project.read(cx).dir().is_none()
            {
                return;
            }
            pick_audio_track(&project, cx);
        })
        .render(cx);
    div()
        .min_w(px(0.))
        .flex()
        .flex_col()
        .gap_2()
        .pt_4()
        .border_t_1()
        .border_color(color(cx, "border.muted"))
        .child(
            div()
                .flex()
                .flex_wrap()
                .items_end()
                .gap_3()
                .child(
                    path_field(cx.t("generate.audio.label").to_string(), value, cx).min_w(px(230.)),
                )
                .child(pick),
        )
        .child(muted(
            cx.t(if resolved.is_none() {
                "workflow.generate.audio_unset"
            } else if ctx.picked_audio.is_some() {
                "workflow.generate.audio_picked"
            } else {
                "workflow.generate.audio_project"
            })
            .to_string(),
            cx,
        ))
}

fn preview_row(ctx: &FormContext<'_>, window: &mut Window, cx: &mut App) -> Div {
    let mut choices = div().flex().flex_wrap().gap_3();
    for preview in [true, false] {
        let form = ctx.form.clone();
        let selected = ctx.current.preview() == preview;
        let (id, title_key, hint_key) = if preview {
            (
                "generate-preview",
                "workflow.generate.preview_title",
                "workflow.generate.preview_description",
            )
        } else {
            (
                "generate-full",
                "workflow.generate.full_title",
                "workflow.generate.full_description",
            )
        };
        let mut indicator = div()
            .size(px(16.))
            .rounded_full()
            .border_1()
            .flex_shrink_0()
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
            indicator = indicator.child(
                div()
                    .size(px(8.))
                    .rounded_full()
                    .bg(color(cx, "action.primary.bg")),
            );
        }
        let surface = div()
            .flex_1()
            .min_w(px(220.))
            .flex()
            .flex_col()
            .gap_2()
            .p_3()
            .rounded(px(6.))
            .border_1()
            .cursor(if ctx.busy {
                gpui::CursorStyle::OperationNotAllowed
            } else {
                gpui::CursorStyle::PointingHand
            })
            .opacity(if ctx.busy { 0.55 } else { 1.0 })
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
            .child(
                div().flex().items_center().gap_2().child(indicator).child(
                    div()
                        .font_weight(FontWeight::SEMIBOLD)
                        .child(cx.t(title_key).to_string()),
                ),
            )
            .child(muted(cx.t(hint_key).to_string(), cx));
        let hover = color(cx, "surface.hover");
        let focus = color(cx, "action.primary.bg");
        let busy = ctx.busy;
        choices = choices.child(
            radio(id, cx)
                .checked(selected)
                .disabled(ctx.busy)
                .on_toggle(move |_checked, _event, _window, cx| {
                    if cx.global::<AppState>().tasks.read(cx).is_busy() {
                        return;
                    }
                    form.update(cx, |form, cx| {
                        form.set_preview(preview);
                        cx.notify();
                    });
                })
                .apply_focusable(surface)
                .hover(move |style| if busy { style } else { style.bg(hover) })
                .focus_visible(move |style| style.border_color(focus)),
        );
    }
    let mut row = div()
        .min_w(px(0.))
        .flex()
        .flex_col()
        .gap_3()
        .child(muted(cx.t("workflow.generate.output_mode").to_string(), cx))
        .child(choices);
    if ctx.current.preview() {
        let manifest = ctx
            .assets
            .manifest
            .as_ref()
            .filter(|_| ctx.assets.is_locked());
        let ceiling = match manifest {
            Some(manifest) => u32::try_from(manifest.frame_count)
                .unwrap_or(MAX_PREVIEW_FRAMES)
                .clamp(MIN_PREVIEW_FRAMES, MAX_PREVIEW_FRAMES),
            None => MAX_PREVIEW_FRAMES,
        };
        let form = ctx.form.clone();
        let field = number_input("generate-preview-frames")
            .disabled(ctx.busy)
            .min(f64::from(MIN_PREVIEW_FRAMES))
            .max(f64::from(ceiling))
            .step(1.0)
            .value(f64::from(ctx.current.preview_frames()))
            .on_change(move |value, _window, cx| {
                if cx.global::<AppState>().tasks.read(cx).is_busy() {
                    return;
                }
                form.update(cx, |form, cx| {
                    form.set_preview_frames(value);
                    cx.notify();
                });
            })
            .render(cx, window);
        let estimate = format!(
            "{} {} {}",
            cx.t("workflow.generate.preview_duration"),
            ctx.current.preview_seconds(),
            cx.t("workflow.generate.seconds"),
        );
        row = row
            .child(
                div()
                    .flex()
                    .flex_wrap()
                    .items_center()
                    .gap_3()
                    .child(div().w(px(156.)).child(field))
                    .child(inline_muted(
                        cx.t("generate.preview.frames").to_string(),
                        cx,
                    ))
                    .child(
                        inline_muted(estimate, cx)
                            .debug_selector(|| "generate-preview-estimate".into()),
                    ),
            )
            .child(muted(
                cx.t("workflow.generate.preview_limit").to_string(),
                cx,
            ));
    }
    row
}

fn output_row(ctx: &FormContext<'_>, dir: &Path, cx: &mut App) -> Div {
    let project = ctx.project.clone();
    let dir = dir.to_path_buf();
    let output = output_path(ctx.picked_output, &dir, ctx.renders, ctx.current.preview());
    let suggested_name = output
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_default();
    let pick = button("generate-output-pick", cx)
        .caption(cx.t("generate.output.pick"))
        .disabled(ctx.busy)
        .on_click(move |_event, _window, cx| {
            if cx.global::<AppState>().tasks.read(cx).is_busy()
                || project.read(cx).dir() != Some(dir.as_path())
            {
                return;
            }
            pick_output_path(&project, &dir, &suggested_name, cx);
        })
        .render(cx);
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
                cx.t("generate.output.label").to_string(),
                output.display().to_string(),
                cx,
            )
            .min_w(px(230.)),
        )
        .child(pick)
}

fn submit_row(ctx: &FormContext<'_>, cx: &mut App) -> Div {
    let caption = cx.t(if ctx.current.preview() {
        "workflow.generate.submit_preview"
    } else {
        "workflow.generate.submit_full"
    });
    let action = button("generate-submit", cx)
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
                let current = form.read(cx).clone();
                let selection = project.read(cx);
                let picked_checkpoint = selection.checkpoint().cloned();
                let picked_audio = selection.audio().map(Path::to_path_buf);
                let picked_output = selection.output().map(Path::to_path_buf);
                let assets = selection.survey().clone();
                let training = selection.training().clone();
                let renders = selection.renders().clone();
                let checkpoint = checkpoint_path(picked_checkpoint.as_ref(), &training);
                let audio = audio_path(picked_audio.as_deref(), &dir, &assets);
                let output =
                    output_path(picked_output.as_deref(), &dir, &renders, current.preview());
                let Some(checkpoint) = checkpoint else {
                    return;
                };
                let Some(audio) = audio else {
                    return;
                };
                if let Err(error) = ensure_renders_dir(&dir) {
                    project.update(cx, |project, cx| {
                        project.note(Note::new(
                            "generate.note.output_dir_failed",
                            Some(error.to_string()),
                        ));
                        cx.notify();
                    });
                }
                let request = RenderRequest {
                    project_dir: &dir,
                    checkpoint,
                    audio: &audio,
                    output: &output,
                    max_output_frames: current.max_output_frames(),
                }
                .build();
                submit(
                    &center,
                    &project,
                    TaskKind::Render,
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
    let next_page = match ctx.admission {
        FormState::Blocked("generate.blocked.not_locked") => {
            Some((Page::Assets, "workflow.training.go_assets"))
        }
        FormState::Blocked("generate.blocked.no_checkpoint") => {
            Some((Page::Training, "workflow.assets.continue_training"))
        }
        _ => None,
    };
    if let Some((page, key)) = next_page {
        let navigation = cx.global::<AppState>().navigation.clone();
        actions = actions.child(
            button("generate-go-prerequisite", cx)
                .caption(cx.t(key))
                .on_click(move |_event, _window, cx| {
                    navigation.update(cx, |navigation, cx| {
                        if navigation.select(page) {
                            cx.notify();
                        }
                    });
                })
                .render(cx),
        );
    }
    let mut row = div().flex().flex_col().gap_2().child(actions);
    let reason = ctx.gate.or(match ctx.admission {
        FormState::Blocked(key) => Some(key),
        FormState::Ready => None,
    });
    if let Some(key) = reason {
        row = row.child(muted(cx.t(key).to_string(), cx));
    }
    row
}

fn activity_section(row: &TaskRow, cx: &mut App) -> Stateful<Div> {
    let id = row.task_id.as_str().to_owned();
    let mut content = section(
        "generate-live",
        cx.t("generate.live.title").to_string(),
        cx.t("workflow.generate.latest_task").to_string(),
        cx,
    )
    .child(
        div()
            .flex()
            .flex_wrap()
            .items_center()
            .gap_2()
            .child(
                badge(
                    "generate-task-status",
                    cx.t(status_key(row.status)).to_string(),
                    cx,
                )
                .variant(badge_variant(row.status))
                .render(cx),
            )
            .child(muted(cx.t(stage_key(&row.stage)).to_string(), cx)),
    );
    if row.status.is_incomplete() {
        if let TaskStage::Rendering { frame, total } = &row.stage {
            content = content.child(div().child(format!(
                "{}{}{frame} / {total}",
                cx.t("generate.live.position"),
                label_separator(cx),
            )));
        }
        content = content
            .child(progress_bar(&id, row, cx))
            .child(cancel_row(&id, row, cx));
    } else if row.status == TaskStatus::Failed {
        let summary = match &row.failure {
            Some(failure) => match &failure.summary {
                Summary::Worker(written) => written.clone(),
                Summary::Key(key) => cx.t(key).to_string(),
            },
            None => cx.t(status_key(row.status)).to_string(),
        };
        content = content
            .child(
                div()
                    .w_full()
                    .min_w(px(0.))
                    .whitespace_normal()
                    .text_color(color(cx, "status.danger.fg"))
                    .child(summary),
            )
            .child(muted(cx.t("generate.failure.hint").to_string(), cx));
    }
    content
}

fn format_section(manifest: Option<&AssetManifest>, cx: &mut App) -> Stateful<Div> {
    let ui = cx.global::<AppState>().ui.clone();
    let open = ui.read(cx).generate_details;
    let toggle = button("generate-details", cx)
        .caption(cx.t(if open {
            "workflow.details_hide"
        } else {
            "workflow.generate.details_show"
        }))
        .on_click(move |_event, _window, cx| {
            ui.update(cx, |ui, cx| {
                ui.generate_details = !ui.generate_details;
                cx.notify();
            });
        })
        .render(cx);
    let mut content = section(
        "generate-format",
        cx.t("generate.format.title").to_string(),
        cx.t("workflow.generate.format_description").to_string(),
        cx,
    )
    .child(div().flex().child(toggle));
    if open {
        let mut facts = vec![
            Fact {
                label: "generate.format.fps",
                value: FactValue::Text(RENDER_FPS.to_string()),
            },
            Fact {
                label: "generate.format.video_codec",
                value: FactValue::Text("libx264 (yuv420p)".to_owned()),
            },
            Fact {
                label: "generate.format.audio_codec",
                value: FactValue::Text("aac".to_owned()),
            },
            Fact {
                label: "generate.format.quality",
                value: FactValue::Key("generate.format.quality_default"),
            },
        ];
        if let Some(manifest) = manifest {
            facts.insert(
                0,
                Fact {
                    label: "generate.format.resolution",
                    value: FactValue::Text(format!(
                        "{} × {}",
                        manifest.frame_width, manifest.frame_height
                    )),
                },
            );
        }
        content = content.child(facts_list("generate-format", &facts, cx));
    }
    content
}

/// Session results also cover a successful render saved outside outputs/renders.
/// Those paths come only from completed worker results, never from form guesses.
fn renders_card(renders: &RenderSurvey, rows: &[TaskRow], cx: &mut App) -> Stateful<Div> {
    let mut published: Vec<(&TaskRow, PathBuf)> = Vec::new();
    for row in rows
        .iter()
        .filter(|row| row.kind == TaskKind::Render && row.status == TaskStatus::Completed)
    {
        let Some(path) = row
            .result
            .as_ref()
            .and_then(|result| result.get("output_path"))
            .and_then(serde_json::Value::as_str)
            .filter(|path| !path.is_empty())
            .map(PathBuf::from)
        else {
            continue;
        };
        if !renders.videos.iter().any(|video| video.path == path)
            && !published.iter().any(|(_, existing)| *existing == path)
        {
            published.push((row, path));
        }
    }
    let count = renders.videos.len() + published.len();
    let mut content = section(
        "generate-renders",
        cx.t("generate.renders.title").to_string(),
        cx.t("workflow.generate.renders_description").to_string(),
        cx,
    );
    if count == 0 {
        return content.child(div().py_4().child(muted(
            cx.t("workflow.generate.renders_empty").to_string(),
            cx,
        )));
    }
    for (row, path) in published.iter().take(8) {
        let id = format!("generate-result-{}", row.task_id.as_str());
        let mut output = div()
            .min_w(px(0.))
            .flex()
            .flex_col()
            .gap_2()
            .py_3()
            .border_b_1()
            .border_color(color(cx, "border.muted"))
            .child(
                div()
                    .flex()
                    .flex_wrap()
                    .items_end()
                    .gap_3()
                    .child(
                        path_field(
                            cx.t("workflow.generate.published_output").to_string(),
                            path.display().to_string(),
                            cx,
                        )
                        .min_w(px(230.)),
                    )
                    .child(video_actions(&id, path, cx)),
            );
        if let Some(frame_count) = row
            .result
            .as_ref()
            .and_then(|result| result.get("frame_count"))
            .and_then(serde_json::Value::as_u64)
        {
            output = output.child(muted(
                format!(
                    "{}{}{frame_count}",
                    cx.t("workflow.generate.actual_frames"),
                    label_separator(cx),
                ),
                cx,
            ));
        }
        content = content.child(output);
    }
    for video in renders
        .videos
        .iter()
        .take(8_usize.saturating_sub(published.len()))
    {
        content = content.child(render_row(video, cx));
    }
    content.child(muted(
        format!(
            "{}{}{count}",
            cx.t("generate.renders.count"),
            label_separator(cx),
        ),
        cx,
    ))
}

fn render_row(video: &RenderedVideo, cx: &mut App) -> Div {
    let id = format!("generate-render-{}", video.name);
    div()
        .min_w(px(0.))
        .flex()
        .flex_wrap()
        .items_center()
        .gap_3()
        .py_3()
        .border_b_1()
        .border_color(color(cx, "border.muted"))
        .child(
            div()
                .flex_1()
                .min_w(px(230.))
                .flex()
                .flex_col()
                .gap_1()
                .child(
                    div()
                        .w_full()
                        .min_w(px(0.))
                        .whitespace_normal()
                        .font_weight(FontWeight::MEDIUM)
                        .child(video.name.clone()),
                )
                .child(muted(size(video.bytes), cx)),
        )
        .child(video_actions(&id, &video.path, cx))
}

fn video_actions(id: &str, path: &Path, cx: &mut App) -> Div {
    let play_path = path.to_path_buf();
    let reveal_path = path.to_path_buf();
    let play = button(SharedString::from(format!("{id}-play")), cx)
        .caption(cx.t("generate.renders.play"))
        .on_click(move |_event, _window, cx| cx.open_with_system(&play_path))
        .render(cx);
    let reveal = button(SharedString::from(format!("{id}-reveal")), cx)
        .caption(cx.t("generate.renders.reveal"))
        .on_click(move |_event, _window, cx| cx.reveal_path(&reveal_path))
        .render(cx);
    div()
        .flex()
        .flex_shrink_0()
        .gap_2()
        .child(play)
        .child(reveal)
}

fn notes_card(notes: &[Note], cx: &App) -> Stateful<Div> {
    let mut content = section(
        "generate-notes",
        cx.t("generate.notes.title").to_string(),
        "",
        cx,
    );
    for (index, note) in notes.iter().enumerate() {
        let line = match &note.detail {
            Some(detail) => format!("{}{}{detail}", cx.t(note.key), label_separator(cx)),
            None => cx.t(note.key).to_string(),
        };
        content =
            content.child(muted(line, cx).id(SharedString::from(format!("generate-note-{index}"))));
    }
    content
}

/// Binary units and integer arithmetic preserve the existing file-size semantics.
fn size(bytes: u64) -> String {
    const KIB: u64 = 1024;
    const MIB: u64 = KIB * 1024;
    const GIB: u64 = MIB * 1024;
    let (unit, scale) = if bytes >= GIB {
        ("GiB", GIB)
    } else if bytes >= MIB {
        ("MiB", MIB)
    } else if bytes >= KIB {
        ("KiB", KIB)
    } else {
        return format!("{bytes} B");
    };
    let whole = bytes / scale;
    let tenth = (bytes % scale).saturating_mul(10) / scale;
    format!("{whole}.{tenth} {unit}")
}
