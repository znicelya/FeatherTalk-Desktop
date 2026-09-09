//! Compact, bounded presentation of facts read from worker output or project files.

use gpui::{App, Div, InteractiveElement, ParentElement, SharedString, Styled, div, px};
use yororen_ui::i18n::Translate;

use crate::components::ui::muted;
use crate::facts::{Fact, FactValue};
use crate::theme::color;

/// Labels sit above their values so paths and measured numbers never compete for
/// the same line. Two columns still fit the workbench's minimum window size.
pub(crate) fn facts_list(id_prefix: &str, facts: &[Fact], cx: &App) -> Div {
    let mut list = div()
        .w_full()
        .min_w(px(0.))
        .grid()
        .grid_cols(2)
        .gap_x_6()
        .gap_y_3();
    for fact in facts {
        let value = match &fact.value {
            FactValue::Text(measured) => measured.clone(),
            FactValue::Key(key) => cx.t(key).to_string(),
        };
        list = list.child(
            div()
                .id(SharedString::from(format!("{id_prefix}-{}", fact.label)))
                .min_w(px(0.))
                .flex()
                .flex_col()
                .gap_1()
                .child(muted(cx.t(fact.label).to_string(), cx).text_size(px(12.)))
                .child(
                    div()
                        .w_full()
                        .min_w(px(0.))
                        .whitespace_normal()
                        .text_size(px(14.))
                        .text_color(color(cx, "content.primary"))
                        .child(value),
                ),
        );
    }
    list
}
