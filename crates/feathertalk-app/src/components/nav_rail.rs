//! Persistent workflow navigation and the settings entry.

use gpui::{App, Div, Entity, FontWeight, ParentElement, Styled, div, px, svg};
use yororen_ui::headless::button::button;
use yororen_ui::headless::icon::IconSource;
use yororen_ui::i18n::Translate;

use crate::navigation::{Navigation, Page};
use crate::state::AppState;
use crate::theme::color;

pub fn nav_rail(navigation: &Entity<Navigation>, cx: &mut App) -> Div {
    let current = navigation.read(cx).current();
    let ui = cx.global::<AppState>().ui.clone();
    let settings = ui.read(cx).settings_open;
    let compute = cx.global::<AppState>().compute.read(cx);
    let ready = compute.is_available();
    let status = cx.t(if compute.is_refreshing() {
        "ui.service.connecting"
    } else if ready {
        "ui.service.ready"
    } else {
        "ui.service.unavailable"
    });
    let status_color = color(
        cx,
        if ready {
            "status.success.fg"
        } else {
            "content.tertiary"
        },
    );
    let mut rail = div()
        .w(px(212.))
        .flex_shrink_0()
        .h_full()
        .flex()
        .flex_col()
        .bg(color(cx, "surface.canvas"))
        .border_r_1()
        .border_color(color(cx, "border.default"))
        .px_3()
        .py_5()
        .gap_2()
        .child(
            div()
                .flex()
                .items_center()
                .gap_3()
                .px_2()
                .pb_5()
                .child(
                    div()
                        .size(px(32.))
                        .rounded(px(8.))
                        .flex()
                        .items_center()
                        .justify_center()
                        .bg(color(cx, "action.primary.bg"))
                        .child(
                            svg()
                                .path("app-icons/feather.svg")
                                .size(px(21.))
                                .text_color(color(cx, "action.primary.fg")),
                        ),
                )
                .child(
                    div()
                        .flex()
                        .flex_col()
                        .gap_1()
                        .child(
                            div()
                                .font_weight(FontWeight::SEMIBOLD)
                                .text_size(px(16.))
                                .child("FeatherTalk"),
                        )
                        .child(
                            div()
                                .text_size(px(11.))
                                .text_color(color(cx, "content.tertiary"))
                                .child(cx.t("ui.brand.subtitle")),
                        ),
                ),
        );
    for (index, page) in Page::ALL.into_iter().enumerate() {
        if index == 0 || index == 3 {
            rail = rail.child(
                div()
                    .px_3()
                    .pt_4()
                    .pb_1()
                    .text_size(px(11.))
                    .text_color(color(cx, "content.tertiary"))
                    .child(cx.t(if index == 0 {
                        "ui.nav.workflow"
                    } else {
                        "ui.nav.manage"
                    })),
            );
        }
        let selected = page == current && !settings;
        let entity = navigation.clone();
        let ui = ui.clone();
        let icon = match page {
            Page::Assets => "app-icons/assets.svg",
            Page::Training => "app-icons/training.svg",
            Page::Generate => "app-icons/generate.svg",
            Page::Models => "app-icons/models.svg",
            Page::Tasks => "app-icons/tasks.svg",
        };
        // The button renderer owns hover, active, and focus styles.
        rail = rail.child(
            button(page.element_id(), cx)
                .caption(cx.t(page.label_key()))
                .icon(IconSource::Resource(icon.into()))
                .on_click(move |_event, _window, cx| {
                    ui.update(cx, |ui, cx| {
                        ui.settings_open = false;
                        cx.notify();
                    });
                    entity.update(cx, |navigation, cx| {
                        navigation.select(page);
                        cx.notify();
                    });
                })
                .render(cx)
                .w_full()
                .h(px(40.))
                .justify_start()
                .gap_3()
                .font_weight(if selected {
                    FontWeight::SEMIBOLD
                } else {
                    FontWeight::NORMAL
                })
                .text_color(color(cx, "content.primary"))
                .bg(color(
                    cx,
                    if selected {
                        "surface.sunken"
                    } else {
                        "surface.canvas"
                    },
                ))
                .border_1()
                .border_color(color(
                    cx,
                    if selected {
                        "surface.sunken"
                    } else {
                        "surface.canvas"
                    },
                )),
        );
    }
    let settings_ui = ui.clone();
    rail.child(div().flex_1()).child(
        div()
            .flex()
            .flex_col()
            .gap_3()
            .border_t_1()
            .border_color(color(cx, "border.default"))
            .pt_3()
            .child(
                button("nav-settings", cx)
                    .caption(cx.t("ui.settings.title"))
                    .icon(IconSource::Resource("app-icons/settings.svg".into()))
                    .on_click(move |_event, _window, cx| {
                        settings_ui.update(cx, |ui, cx| {
                            ui.settings_open = true;
                            cx.notify();
                        });
                    })
                    .render(cx)
                    .w_full()
                    .justify_start()
                    .gap_3()
                    .border_1()
                    .border_color(color(
                        cx,
                        if settings {
                            "surface.sunken"
                        } else {
                            "surface.canvas"
                        },
                    ))
                    .bg(color(
                        cx,
                        if settings {
                            "surface.sunken"
                        } else {
                            "surface.canvas"
                        },
                    )),
            )
            .child(
                div()
                    .flex()
                    .items_center()
                    .gap_2()
                    .px_3()
                    .child(div().size(px(6.)).rounded_full().bg(status_color))
                    .child(
                        div()
                            .text_size(px(11.))
                            .text_color(color(cx, "content.tertiary"))
                            .child(status),
                    ),
            )
            .child(
                div()
                    .px_3()
                    .text_size(px(10.))
                    .text_color(color(cx, "content.tertiary"))
                    .child(cx.t("ui.local_workspace")),
            ),
    )
}
