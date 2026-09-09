//! Small embedded line icons, sharing the application's asset source with Yororen.

use gpui::{AssetSource, SharedString};
use std::borrow::Cow;
use yororen_ui::assets::UiAsset;

pub struct WorkbenchAssets;

const ICONS: &[(&str, &str)] = &[
    (
        "app-icons/feather.svg",
        r#"<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M20 4c-3-3-9 0-12 4s-2 8-2 10c2 0 6 1 10-2s7-9 4-12Z"/><path d="m3 21 13-13M9 15h6M12 12V7"/></svg>"#,
    ),
    (
        "app-icons/assets.svg",
        r#"<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="18" height="18" rx="3"/><path d="m3 16 5-5 4 4 3-3 6 6"/><circle cx="15.5" cy="8" r="1.5"/></svg>"#,
    ),
    (
        "app-icons/training.svg",
        r#"<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M4 19h16M6 15V9M12 15V5M18 15v-3"/><circle cx="6" cy="6" r="1"/><circle cx="18" cy="9" r="1"/></svg>"#,
    ),
    (
        "app-icons/generate.svg",
        r#"<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="4" width="18" height="16" rx="3"/><path d="m10 8 6 4-6 4Z"/></svg>"#,
    ),
    (
        "app-icons/models.svg",
        r#"<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="m12 3 9 5-9 5-9-5Z M3 8v9l9 5 9-5V8M12 13v9M7.5 5.5l9 5"/></svg>"#,
    ),
    (
        "app-icons/tasks.svg",
        r#"<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><rect x="5" y="3" width="14" height="18" rx="2"/><path d="m8 8 1 1 2-2M13 8h3M8 13h8M8 17h5"/></svg>"#,
    ),
    (
        "app-icons/settings.svg",
        r#"<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="m10 3-.6 3-2 .9-2.6-.9-2 3.5L5 11v2l-2.2 1.5 2 3.5 2.6-.9 2 .9.6 3h4l.6-3 2-.9 2.6.9 2-3.5L19 13v-2l2.2-1.5-2-3.5-2.6.9-2-.9-.6-3Z"/><circle cx="12" cy="12" r="3"/></svg>"#,
    ),
    (
        "app-icons/folder.svg",
        r#"<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M3 7V5a2 2 0 0 1 2-2h5l2 3h7a2 2 0 0 1 2 2v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2Z"/></svg>"#,
    ),
    (
        "app-icons/moon.svg",
        r#"<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M20 14a8 8 0 0 1-10-10 8.5 8.5 0 1 0 10 10Z"/></svg>"#,
    ),
    (
        "app-icons/sun.svg",
        r#"<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M2 12h2M20 12h2M5 5l1.5 1.5M17.5 17.5 19 19M5 19l1.5-1.5M17.5 6.5 19 5"/></svg>"#,
    ),
    (
        "app-icons/chevron.svg",
        r#"<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="m9 6 6 6-6 6"/></svg>"#,
    ),
];

impl AssetSource for WorkbenchAssets {
    fn load(&self, path: &str) -> gpui::Result<Option<Cow<'static, [u8]>>> {
        if let Some((_, svg)) = ICONS.iter().find(|(name, _)| *name == path) {
            return Ok(Some(Cow::Borrowed(svg.as_bytes())));
        }
        UiAsset.load(path)
    }

    fn list(&self, path: &str) -> gpui::Result<Vec<SharedString>> {
        let mut entries = UiAsset.list(path)?;
        entries.extend(
            ICONS
                .iter()
                .filter(|(name, _)| name.starts_with(path))
                .map(|(name, _)| SharedString::from(*name)),
        );
        Ok(entries)
    }
}
