//! Explicit settings surface for appearance, devices and service diagnostics.

use gpui::{App, ClipboardItem, Div, ParentElement, Styled, Window, div};
use yororen_ui::ActionVariantKind;
use yororen_ui::headless::button::button;
use yororen_ui::i18n::Translate;

use crate::components::{
    compute_control::compute_control,
    ui::{muted, page_frame, path_field, section},
};
use crate::state::AppState;
use crate::ui::AppLocale;
use crate::worker_status::{WorkerStatus, source_key};

pub fn set_theme(dark: bool, cx: &mut App) {
    let ui = cx.global::<AppState>().ui.clone();
    ui.update(cx, |ui, cx| {
        ui.dark = dark;
        cx.notify();
    });
    yororen_ui::theme::install(cx, crate::theme::palette(dark));
    cx.refresh_windows();
}

pub fn set_locale(locale: AppLocale, window: &mut Window, cx: &mut App) {
    let ui = cx.global::<AppState>().ui.clone();
    ui.update(cx, |ui, cx| {
        ui.locale = locale;
        cx.notify();
    });
    crate::catalog::install(cx, locale);
    window.set_window_title(&cx.t("shell.title"));
    cx.refresh_windows();
}

pub fn settings_page(cx: &mut App) -> Div {
    let state = cx.global::<AppState>();
    let ui = state.ui.clone();
    let dark = ui.read(cx).dark;
    let locale = ui.read(cx).locale;
    let worker = state.worker.read(cx).clone();
    let back_ui = ui.clone();
    let back = button("settings-back", cx)
        .caption(cx.t("ui.settings.back"))
        .on_click(move |_event, _window, cx| {
            back_ui.update(cx, |ui, cx| {
                ui.settings_open = false;
                cx.notify();
            });
        })
        .render(cx);
    let mut choices = div().flex().gap_2();
    for (value, id, key) in [
        (false, "settings-light", "ui.theme.light"),
        (true, "settings-dark", "ui.theme.dark"),
    ] {
        choices = choices.child(
            button(id, cx)
                .caption(cx.t(key))
                .variant(if dark == value {
                    ActionVariantKind::Primary
                } else {
                    ActionVariantKind::Neutral
                })
                .on_click(move |_event, _window, cx| set_theme(value, cx))
                .render(cx),
        );
    }
    let appearance = section(
        "settings-appearance",
        cx.t("ui.theme.title"),
        cx.t("ui.theme.description"),
        cx,
    )
    .child(choices);
    let mut language_choices = div().flex().gap_2();
    for (value, id, key) in [
        (
            AppLocale::ZhCn,
            "settings-language-zh-cn",
            "ui.language.zh_cn",
        ),
        (AppLocale::En, "settings-language-en", "ui.language.en"),
    ] {
        language_choices = language_choices.child(
            button(id, cx)
                .caption(cx.t(key))
                .variant(if locale == value {
                    ActionVariantKind::Primary
                } else {
                    ActionVariantKind::Neutral
                })
                .on_click(move |_event, window, cx| set_locale(value, window, cx))
                .render(cx),
        );
    }
    let language = section(
        "settings-language",
        cx.t("ui.language.title"),
        cx.t("ui.language.description"),
        cx,
    )
    .child(language_choices);
    let compute = section(
        "settings-compute",
        cx.t("compute.label"),
        cx.t("ui.compute.description"),
        cx,
    )
    .child(compute_control(cx));
    let mut service = section(
        "settings-worker",
        cx.t("ui.service.title"),
        cx.t("ui.service.description"),
        cx,
    );
    let diagnostic = match worker {
        WorkerStatus::Ready { path } => {
            service = service.child(path_field(
                cx.t("shell.worker.path"),
                path.display().to_string(),
                cx,
            ));
            path.display().to_string()
        }
        WorkerStatus::Missing { probed } => {
            let mut lines = Vec::new();
            for candidate in probed {
                let label = cx.t(source_key(candidate.source));
                let value = candidate
                    .path
                    .map(|path| path.display().to_string())
                    .unwrap_or_else(|| cx.t("shell.worker.unset").to_string());
                lines.push(format!("{label}: {value}"));
                service = service.child(path_field(label, value, cx));
            }
            lines.join("\n")
        }
    };
    service = service
        .child(
            button("settings-copy-diagnostics", cx)
                .caption(cx.t("ui.service.copy"))
                .on_click(move |_event, _window, cx| {
                    cx.write_to_clipboard(ClipboardItem::new_string(diagnostic.clone()))
                })
                .render(cx),
        )
        .child(muted(cx.t("ui.service.refresh_hint"), cx));
    page_frame(
        "settings-page",
        cx.t("ui.settings.title"),
        cx.t("ui.settings.description"),
        div()
            .flex()
            .flex_col()
            .gap_5()
            .child(div().child(back))
            .child(appearance)
            .child(language)
            .child(compute)
            .child(service),
        cx,
    )
}
