//! Native controls with persistent keyboard focus and neutral theme tokens.

use std::sync::Arc;

use gpui::{
    AnyElement, App, Div, FocusHandle, FontWeight, InteractiveElement, IntoElement, ParentElement,
    SharedString, Stateful, StatefulInteractiveElement, Styled, Window, div, px, rgb,
};
use yororen_ui::headless::{
    button::ButtonProps, icon::IconProps, number_input::NumberInputProps, radio::RadioProps,
    switch::SwitchProps, text_input::TextInputState, toggle_button::ToggleButtonProps,
};
use yororen_ui::renderer::renderers::{
    ButtonRenderer, NumberInputRenderer, SwitchRenderer, ToggleButtonRenderer,
    TokenNumberInputRenderer,
};
use yororen_ui::{ActionVariantKind, RendererContext, markers};

use crate::theme::color;

pub fn install(cx: &mut App) {
    cx.register_renderer_arc::<markers::Button, dyn ButtonRenderer>(Arc::new(WorkbenchButton));
    cx.register_renderer_arc::<markers::ToggleButton, dyn ToggleButtonRenderer>(Arc::new(
        WorkbenchToggle,
    ));
    cx.register_renderer_arc::<markers::Switch, dyn SwitchRenderer>(Arc::new(WorkbenchSwitch));
    cx.register_renderer_arc::<markers::NumberInput, dyn NumberInputRenderer>(Arc::new(
        WorkbenchNumberInput,
    ));
}

struct WorkbenchButton;

impl ButtonRenderer for WorkbenchButton {
    fn compose(&self, props: &ButtonProps, _focus: &FocusHandle, cx: &App) -> Stateful<Div> {
        let prefix = format!("action.{}", props.variant.as_str());
        let disabled = props.disabled || !props.clickable;
        let background = if props.variant == ActionVariantKind::Neutral && !disabled {
            color(cx, "surface.base")
        } else {
            color(
                cx,
                &format!("{prefix}.{}", if disabled { "disabled_bg" } else { "bg" }),
            )
        };
        let foreground = color(
            cx,
            &format!("{prefix}.{}", if disabled { "disabled_fg" } else { "fg" }),
        );
        let border = if props.variant == ActionVariantKind::Neutral {
            color(cx, "border.default")
        } else {
            background
        };
        let hover = if disabled {
            background
        } else {
            color(cx, &format!("{prefix}.hover_bg"))
        };
        let active = if disabled {
            background
        } else {
            color(cx, &format!("{prefix}.active_bg"))
        };
        let focus = color(cx, "border.focus");
        // An untracked, keyed focus handle is retained by GPUI across renders.
        // These page builders create fresh props on every frame.
        let mut view = div()
            .id(props.id.clone())
            .focusable()
            .tab_index(0)
            .tab_stop(!disabled)
            .flex()
            .items_center()
            .justify_center()
            .gap_2()
            .min_h(px(36.))
            .px(px(14.))
            .py(px(7.))
            .rounded(px(6.))
            .border_1()
            .border_color(border)
            .bg(background)
            .text_color(foreground)
            .text_size(px(13.))
            .font_weight(FontWeight::MEDIUM)
            .hover(move |style| style.bg(hover))
            .active(move |style| style.bg(active))
            .focus_visible(move |style| style.border_color(focus))
            .cursor(if disabled {
                gpui::CursorStyle::OperationNotAllowed
            } else {
                gpui::CursorStyle::PointingHand
            });
        if let Some(source) = props.icon.clone() {
            view = view.child(
                IconProps {
                    id: SharedString::from(format!("{:?}-icon", props.id)).into(),
                    source,
                    size: Some(props.icon_size),
                    color: Some(foreground),
                }
                .render(cx),
            );
        }
        if let Some(caption) = &props.caption {
            view = view.child(div().min_w(px(0.)).text_ellipsis().child(caption.clone()));
        }
        view
    }
}

struct WorkbenchToggle;

impl ToggleButtonRenderer for WorkbenchToggle {
    fn compose(&self, props: &ToggleButtonProps, focus: &FocusHandle, cx: &App) -> Stateful<Div> {
        let variant = if !props.selected {
            ActionVariantKind::Neutral
        } else if props.variant == ActionVariantKind::Neutral {
            ActionVariantKind::Primary
        } else {
            props.variant
        };
        WorkbenchButton.compose(
            &ButtonProps {
                id: props.id.clone(),
                focus_handle: focus.clone(),
                on_click: None,
                disabled: props.disabled,
                clickable: true,
                variant,
                caption: props.caption.clone(),
                icon: props.icon.clone(),
                icon_size: props.icon_size,
            },
            focus,
            cx,
        )
    }
}

struct WorkbenchSwitch;

impl SwitchRenderer for WorkbenchSwitch {
    fn compose(&self, props: &SwitchProps, _focus: &FocusHandle, cx: &App) -> Stateful<Div> {
        let background = color(
            cx,
            if props.checked {
                "action.primary.bg"
            } else {
                "border.default"
            },
        );
        let focus = color(cx, "border.focus");
        let mut track = div()
            .id(props.id.clone())
            .focusable()
            .tab_index(0)
            .tab_stop(!props.disabled)
            .w(px(36.))
            .h(px(22.))
            .flex_shrink_0()
            .flex()
            .items_center()
            .px(px(2.))
            .rounded_full()
            .border_1()
            .border_color(background)
            .bg(background)
            .opacity(if props.disabled { 0.5 } else { 1.0 })
            .cursor(if props.disabled {
                gpui::CursorStyle::OperationNotAllowed
            } else {
                gpui::CursorStyle::PointingHand
            })
            .focus_visible(move |style| style.border_color(focus));
        track = if props.checked {
            track.justify_end()
        } else {
            track.justify_start()
        };
        let thumb = if props.checked {
            color(cx, "action.primary.fg")
        } else {
            rgb(0xffffff).into()
        };
        track.child(div().size(px(16.)).rounded_full().bg(thumb))
    }
}

struct WorkbenchNumberInput;

impl NumberInputRenderer for WorkbenchNumberInput {
    fn compose(&self, props: &NumberInputProps, cx: &mut App, window: &mut Window) -> AnyElement {
        // Use the exact key owned by the pinned Yororen renderer so caret,
        // selection and IME state survive unrelated page refreshes.
        let state = window.use_keyed_state(props.id.clone(), cx, |_window, cx| {
            TextInputState::new(&mut *cx)
        });
        // FocusHandle clones share their tab metadata in GPUI 0.3.3.
        let focus = state
            .read(cx)
            .focus_handle()
            .tab_index(0)
            .tab_stop(!props.disabled);
        let value = props.value.to_string();
        if !focus.is_focused(window) && state.read(cx).value != value {
            state.update(cx, |state, _cx| state.set_value(value.clone()));
        }
        if props.disabled {
            // Yororen 0.3's stepper still reacts when disabled. A static field
            // avoids changing either the displayed value or the submitted form.
            return div()
                .id(props.id.clone())
                .min_h(px(36.))
                .px_3()
                .py_2()
                .flex()
                .items_center()
                .rounded(px(6.))
                .border_1()
                .border_color(color(cx, "border.default"))
                .bg(color(cx, "surface.sunken"))
                .text_size(px(13.))
                .text_color(color(cx, "content.tertiary"))
                .cursor(gpui::CursorStyle::OperationNotAllowed)
                .child(value)
                .into_any_element();
        }
        TokenNumberInputRenderer.compose(props, cx, window)
    }
}

/// Retain a keyed focus handle when applying a radio to a page-owned surface.
pub(crate) trait RadioControlExt {
    fn apply_focusable(self, surface: Div) -> Stateful<Div>;
}

impl RadioControlExt for RadioProps {
    fn apply_focusable(self, surface: Div) -> Stateful<Div> {
        let mut view = surface
            .id(self.id)
            .focusable()
            .tab_index(0)
            .tab_stop(!self.disabled);
        if !self.disabled
            && let Some(on_toggle) = self.on_toggle
        {
            let checked = self.checked;
            view = view
                .on_click(move |event, window, cx| on_toggle(!checked, Some(event), window, cx));
        }
        view
    }
}
