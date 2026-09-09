//! The page area.
//!
//! Five pages backed by the worker's command interface.

use gpui::{App, Div, Window};

use crate::components::assets_page::assets_page;
use crate::components::generate_page::generate_page;
use crate::components::models_page::models_page;
use crate::components::tasks_page::tasks_page;
use crate::components::training_page::training_page;
use crate::navigation::Page;

/// Build the body of `page`.
///
/// `window` is passed through for the training and generate pages: their number
/// fields are `number_input`s, whose text state is minted by
/// `window.use_keyed_state`.
pub fn page_body(page: Page, window: &mut Window, cx: &mut App) -> Div {
    match page {
        Page::Assets => assets_page(cx),
        Page::Tasks => tasks_page(cx),
        Page::Training => training_page(window, cx),
        // Listing the one instead of `_ =>` means a sixth page fails to compile
        // rather than quietly rendering a placeholder nobody meant.
        Page::Generate => generate_page(window, cx),
        Page::Models => models_page(cx),
    }
}
