//! Native desktop frame: compact navigation, project toolbar and scrolling pages.

use gpui::{
    App, Context, Div, InteractiveElement, IntoElement, ParentElement, Render, Styled, Window, div,
    px,
};
use yororen_ui::headless::button::button;
use yororen_ui::headless::icon::IconSource;
use yororen_ui::i18n::Translate;

use crate::components::{
    compute_control, nav_rail::nav_rail, page_body::page_body, settings::settings_page,
};
use crate::picker::pick_project_dir;
use crate::state::AppState;
use crate::theme::color;

pub struct Workbench;

/// Handle traversal before focused-node dispatch. GPUI's root-view fallback
/// does not visit the rendered div when no control has focus.
pub fn install_keyboard_navigation(cx: &mut App) {
    cx.intercept_keystrokes(|event, window, cx| {
        let modifiers = event.keystroke.modifiers;
        if window.window_handle().downcast::<Workbench>().is_none()
            || event.keystroke.key != "tab"
            || modifiers.control
            || modifiers.alt
            || modifiers.platform
            || modifiers.function
        {
            return;
        }
        if modifiers.shift {
            window.focus_prev();
        } else {
            window.focus_next();
        }
        window.prevent_default();
        cx.stop_propagation();
    })
    .detach();
}

impl Render for Workbench {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let state = cx.global::<AppState>();
        let navigation = state.navigation.clone();
        let current = navigation.read(cx).current();
        let settings = state.ui.read(cx).settings_open;
        let page = if settings {
            settings_page(cx)
        } else {
            page_body(current, window, cx)
        };
        div()
            .id("workbench")
            .tab_group()
            .size_full()
            .flex()
            .font_family("Microsoft YaHei UI")
            .text_size(px(14.))
            .text_color(color(cx, "content.primary"))
            .bg(color(cx, "surface.canvas"))
            .child(nav_rail(&navigation, cx))
            .child(
                div()
                    .flex_1()
                    .min_w(px(0.))
                    .min_h(px(0.))
                    .h_full()
                    .flex()
                    .flex_col()
                    .child(toolbar(cx))
                    .child(page),
            )
    }
}

fn toolbar(cx: &mut App) -> Div {
    let state = cx.global::<AppState>();
    let project = state.project.clone();
    let center = state.tasks.clone();
    let busy = center.read(cx).is_busy();
    let selected = project.read(cx).dir().map(|path| {
        path.file_name()
            .unwrap_or(path.as_os_str())
            .to_string_lossy()
            .into_owned()
    });
    let project_name = selected.unwrap_or_else(|| cx.t("ui.project.unset").to_string());
    let compute_name = compute_control::compact_summary(state.compute.read(cx), cx);
    let ui = state.ui.clone();
    let dark = ui.read(cx).dark;
    let choose_caption = cx.t("ui.project.change");
    let project_action = button("toolbar-project", cx)
        .caption(choose_caption)
        .icon(IconSource::Resource("app-icons/folder.svg".into()))
        .disabled(busy)
        .on_click(move |_event, _window, cx| pick_project_dir(&project, &center, cx))
        .render(cx);
    let settings_ui = ui.clone();
    let device = button("toolbar-device", cx)
        .caption(compute_name)
        .on_click(move |_event, _window, cx| {
            settings_ui.update(cx, |ui, cx| {
                ui.settings_open = true;
                cx.notify();
            });
        })
        .render(cx)
        .max_w(px(205.))
        .overflow_hidden();
    let theme_button = button("toolbar-theme", cx)
        .caption(cx.t(if dark {
            "ui.theme.light"
        } else {
            "ui.theme.dark"
        }))
        .icon(IconSource::Resource(
            if dark {
                "app-icons/sun.svg"
            } else {
                "app-icons/moon.svg"
            }
            .into(),
        ))
        .on_click(move |_event, _window, cx| crate::components::settings::set_theme(!dark, cx))
        .render(cx);
    div()
        .h(px(64.))
        .flex_shrink_0()
        .px_7()
        .flex()
        .items_center()
        .gap_3()
        .bg(color(cx, "surface.base"))
        .border_b_1()
        .border_color(color(cx, "border.default"))
        .child(
            div()
                .flex_1()
                .min_w(px(0.))
                .flex()
                .flex_col()
                .gap_1()
                .child(
                    div()
                        .text_size(px(11.))
                        .text_color(color(cx, "content.tertiary"))
                        .child(cx.t("ui.project.current")),
                )
                .child(div().text_size(px(13.)).text_ellipsis().child(project_name)),
        )
        .child(project_action)
        .child(device)
        .child(theme_button)
}
