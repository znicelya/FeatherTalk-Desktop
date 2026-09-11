//! Shared native layout primitives for the workbench pages.

use yororen_ui::i18n::I18nContext;

use gpui::{
    App, Div, ElementId, FontWeight, InteractiveElement, IntoElement, ParentElement, SharedString,
    Stateful, StatefulInteractiveElement, Styled, div, px,
};

use crate::theme::color;

pub fn page_frame(
    id: &'static str,
    title: impl Into<SharedString>,
    description: impl Into<SharedString>,
    content: impl IntoElement,
    cx: &App,
) -> Div {
    div().flex_1().min_w(px(0.)).min_h(px(0.)).h_full().child(
        div().id(id).size_full().overflow_y_scroll().child(
            div()
                .flex()
                .flex_col()
                .gap_6()
                .w_full()
                .max_w(px(1240.))
                .mx_auto()
                .p(px(28.))
                .child(
                    div()
                        .flex()
                        .flex_col()
                        .gap_2()
                        .child(
                            div()
                                .text_size(px(24.))
                                .font_weight(FontWeight::SEMIBOLD)
                                .text_color(color(cx, "content.primary"))
                                .child(title.into()),
                        )
                        .child(muted(description, cx)),
                )
                .child(content),
        ),
    )
}

pub fn section(
    id: impl Into<ElementId>,
    title: impl Into<SharedString>,
    description: impl Into<SharedString>,
    cx: &App,
) -> Stateful<Div> {
    let description = description.into();
    let mut header = div().flex().flex_col().gap_1().child(
        div()
            .text_size(px(16.))
            .font_weight(FontWeight::SEMIBOLD)
            .text_color(color(cx, "content.primary"))
            .child(title.into()),
    );
    if !description.is_empty() {
        header = header.child(muted(description, cx));
    }
    div()
        .id(id)
        .min_w(px(0.))
        .w_full()
        .flex()
        .flex_col()
        .gap_4()
        .p_5()
        .bg(color(cx, "surface.raised"))
        .border_1()
        .border_color(color(cx, "border.default"))
        .rounded(px(8.))
        .child(header)
}

pub fn muted(value: impl Into<SharedString>, cx: &App) -> Div {
    div()
        .min_w(px(0.))
        .text_size(px(13.))
        .text_color(color(cx, "content.tertiary"))
        .child(value.into())
}

/// Short labels in wrapping action rows need their intrinsic width; a zero
/// minimum lets GPUI collapse an auto-sized paragraph into a vertical column.
pub fn inline_muted(value: impl Into<SharedString>, cx: &App) -> Div {
    muted(value, cx)
        .min_w(gpui::Length::Auto)
        .flex_shrink_0()
        .whitespace_nowrap()
}

pub fn path_field(label: impl Into<SharedString>, value: impl Into<SharedString>, cx: &App) -> Div {
    let value = value.into();
    // Path separators provide natural wrap opportunities without changing the
    // path stored in the form or submitted to the worker.
    let display = value.replace('\\', "\\\u{200b}").replace('/', "/\u{200b}");
    div()
        .flex_1()
        .min_w(px(0.))
        .flex()
        .flex_col()
        .gap_1()
        .child(muted(label, cx))
        .child(
            div()
                .w_full()
                .min_w(px(0.))
                .whitespace_normal()
                .text_size(px(13.))
                .text_color(color(cx, "content.primary"))
                .child(display),
        )
}

/// Separator for composed label/value strings such as "Attempts: 2".
pub fn label_separator(cx: &App) -> &'static str {
    label_separator_for_locale(&cx.i18n().locale().to_tag())
}

fn label_separator_for_locale(locale_tag: &str) -> &'static str {
    if locale_tag
        .split('-')
        .next()
        .is_some_and(|language| language.eq_ignore_ascii_case("zh"))
    {
        "："
    } else {
        ": "
    }
}

#[cfg(test)]
mod tests {
    use super::label_separator_for_locale;

    #[test]
    fn label_separator_uses_locale_conventions() {
        assert_eq!(label_separator_for_locale("zh-CN"), "：");
        assert_eq!(label_separator_for_locale("en"), ": ");
        assert_eq!(label_separator_for_locale("en-US"), ": ");
    }
}
