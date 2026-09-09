//! The shared compute selector and its background refresh action.

use crate::components::ui::muted;
use crate::theme::color;
use feathertalk_client::{SessionOptions, WorkerLocator, backend_name};
use feathertalk_domain::{AdapterInfo, Backend};
use gpui::{App, Div, FontWeight, ParentElement, SharedString, Styled, div, px};
use yororen_ui::headless::button::button;
use yororen_ui::i18n::Translate;

use crate::compute::{ComputeState, spawn_discovery};
use crate::state::AppState;

/// Called once after bootstrap and later by the refresh button, never by render.
pub fn refresh(cx: &mut App) {
    let state = cx.global::<AppState>();
    let compute = state.compute.clone();
    let worker = state.worker.clone();
    if compute.read(cx).is_refreshing() {
        return;
    }
    compute.update(cx, |state, cx| {
        state.begin_discovery();
        cx.notify();
    });
    let receiver = match spawn_discovery(WorkerLocator::from_env(None), SessionOptions::default()) {
        Ok(receiver) => receiver,
        Err(error) => {
            compute.update(cx, |state, cx| {
                state.finish_discovery(Err(error.to_string()));
                cx.notify();
            });
            return;
        }
    };
    cx.spawn(async move |cx| match receiver.recv().await {
        Ok(discovery) => {
            let _ = worker.update(cx, |state, cx| {
                *state = discovery.worker;
                cx.notify();
            });
            let _ = compute.update(cx, |state, cx| {
                state.finish_discovery(discovery.ready);
                cx.notify();
            });
        }
        Err(error) => {
            let _ = compute.update(cx, |state, cx| {
                state.finish_discovery(Err(error.to_string()));
                cx.notify();
            });
        }
    })
    .detach();
}

pub fn compute_control(cx: &mut App) -> Div {
    let compute = cx.global::<AppState>().compute.clone();
    let snapshot = compute.read(cx).clone();
    let selected = snapshot
        .requested()
        .and_then(|options| options.adapter.as_deref());
    let action = button("compute-refresh", cx)
        .caption(cx.t("compute.refresh"))
        .disabled(snapshot.is_refreshing())
        .on_click(|_event, _window, cx| refresh(cx))
        .render(cx);
    let mut choices = div().flex().flex_col().gap_3();
    for adapter in snapshot.adapters() {
        let entity = compute.clone();
        let id = adapter.id.clone();
        let selectable = snapshot.can_select(&id);
        let checked = selected == Some(id.as_str());
        let status = cx.t(if checked {
            "ui.compute.selected"
        } else if selectable {
            "ui.compute.available"
        } else {
            "ui.compute.unavailable"
        });
        let title = adapter.name.clone();
        let detail = if adapter.backend == Backend::Cpu {
            cx.t("ui.compute.cpu_hint").to_string()
        } else {
            adapter_summary(adapter, cx)
        };
        let control = button(SharedString::from(format!("compute-device-{id}")), cx)
            .disabled(!selectable)
            .on_click(move |_event, _window, cx| {
                entity.update(cx, |state, cx| {
                    if state.select(&id).is_ok() {
                        cx.notify();
                    }
                });
            })
            .render(cx)
            .w_full()
            .justify_start()
            .flex_col()
            .items_start()
            .gap_2()
            .p_4()
            .border_1()
            .border_color(color(
                cx,
                if checked {
                    "content.primary"
                } else {
                    "border.default"
                },
            ))
            .bg(color(cx, "surface.base"))
            .child(
                div()
                    .w_full()
                    .flex()
                    .items_center()
                    .justify_between()
                    .gap_3()
                    .child(div().font_weight(FontWeight::MEDIUM).child(title))
                    .child(
                        div()
                            .text_size(px(12.))
                            .text_color(color(cx, "content.tertiary"))
                            .child(status),
                    ),
            )
            .child(muted(detail, cx));
        choices = choices.child(control);
    }
    if snapshot.adapters().is_empty() {
        choices = choices.child(muted(
            cx.t(if snapshot.is_refreshing() {
                "compute.discovering"
            } else {
                "compute.unavailable"
            }),
            cx,
        ));
    }
    let mut view = div()
        .flex()
        .flex_col()
        .gap_4()
        .child(choices)
        .child(div().child(action));
    if let Some(error) = snapshot.error() {
        view = view.child(
            div()
                .rounded(px(6.))
                .p_3()
                .bg(color(cx, "status.danger.bg"))
                .text_size(px(12.))
                .text_color(color(cx, "status.danger.fg"))
                .child(format!("{}：{error}", cx.t("compute.invalid"))),
        );
    }
    view
}

/// The toolbar shows only a useful device name; full diagnostics live in settings.
pub fn compact_summary(state: &ComputeState, cx: &App) -> String {
    if state.is_refreshing() {
        return cx.t("ui.compute.pending").to_string();
    }
    match state.selected_adapter() {
        Ok(adapter) if adapter.backend == Backend::Cpu => "CPU".into(),
        Ok(adapter) => format!("GPU · {}", adapter.name),
        Err(_) => cx.t("ui.compute.configure").to_string(),
    }
}

pub fn selected_summary(state: &ComputeState, cx: &App) -> String {
    if let Ok(adapter) = state.selected_adapter() {
        return adapter_summary(adapter, cx);
    }
    match state.requested() {
        Some(options) => format!(
            "{} · {}",
            backend_name(options.backend).to_uppercase(),
            options
                .adapter
                .clone()
                .unwrap_or_else(|| cx.t("compute.automatic").to_string())
        ),
        None => cx.t("compute.invalid").to_string(),
    }
}

fn adapter_summary(adapter: &AdapterInfo, cx: &App) -> String {
    let vram = match adapter.vram_bytes {
        Some(bytes) => format!(
            "{} {:.1} GiB",
            cx.t("compute.vram"),
            bytes as f64 / 1_073_741_824.0
        ),
        None => cx.t("compute.vram_unavailable").to_string(),
    };
    format!(
        "{} · {} · {vram}",
        adapter.name,
        backend_name(adapter.backend).to_uppercase()
    )
}
