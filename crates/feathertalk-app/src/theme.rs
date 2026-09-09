//! A neutral shadcn/ui-inspired palette on top of native Yororen controls.

use gpui::{App, Hsla};
use serde_json::json;
use yororen_ui::renderer;
use yororen_ui::theme::{ActiveTheme, Theme};

pub fn palette(dark: bool) -> Theme {
    let mut theme = if dark {
        renderer::system_dark()
    } else {
        renderer::system_light()
    };
    let overrides: serde_json::Map<String, serde_json::Value> = serde_json::from_str(if dark {
        include_str!("../themes/dark.json")
    } else {
        include_str!("../themes/light.json")
    })
    .expect("the bundled palette is valid");
    for (key, value) in overrides {
        theme.set(&key, value);
    }
    for (key, value) in [
        ("tokens.control.button.min_height", 36),
        ("tokens.control.button.horizontal_padding", 14),
        ("tokens.control.button.radius", 6),
        ("tokens.control.input.min_height", 36),
        ("tokens.control.number_input.min_height", 36),
        ("tokens.control.select.min_height", 36),
        ("tokens.control.toggle_button.min_height", 36),
        ("tokens.control.badge.radius", 5),
        ("tokens.control.badge.min_height", 22),
        ("tokens.control.progress.bar_default_h", 6),
        ("tokens.radii.lg", 8),
        ("tokens.typography.font_size_md", 14),
        ("tokens.typography.font_size_sm", 13),
        ("tokens.typography.font_size_xs", 12),
    ] {
        theme.set(key, json!(value));
    }
    theme.set(
        "tokens.typography.family_default",
        json!("Microsoft YaHei UI"),
    );
    theme
}

pub fn color(cx: &App, key: &str) -> Hsla {
    cx.theme().get_color(key).unwrap_or_default()
}
