use std::collections::BTreeSet;

use feathertalk_app::navigation::{Navigation, Page};

#[test]
fn the_five_pages_keep_the_navigation_order_the_design_fixes() {
    let keys: Vec<&str> = Page::ALL.iter().map(|page| page.label_key()).collect();
    assert_eq!(
        keys,
        vec![
            "nav.assets",
            "nav.training",
            "nav.generate",
            "nav.models",
            "nav.tasks"
        ]
    );
}

#[test]
fn the_shell_opens_on_the_asset_page() {
    assert_eq!(Navigation::default().current(), Page::Assets);
}

#[test]
fn selecting_another_page_reports_the_change() {
    let mut navigation = Navigation::default();
    assert!(navigation.select(Page::Tasks));
    assert_eq!(navigation.current(), Page::Tasks);
}

#[test]
fn selecting_the_current_page_changes_nothing() {
    let mut navigation = Navigation::default();
    assert!(!navigation.select(Page::Assets));
    assert_eq!(navigation.current(), Page::Assets);
}

#[test]
fn every_page_carries_a_distinct_non_empty_identity() {
    let mut seen: BTreeSet<&str> = BTreeSet::new();
    for page in Page::ALL {
        for value in [
            page.element_id(),
            page.label_key(),
            page.title_key(),
            page.pending_key(),
        ] {
            assert!(!value.is_empty(), "{page:?} has an empty identity entry");
            assert!(seen.insert(value), "{value} is used twice");
        }
    }
    assert_eq!(seen.len(), Page::ALL.len() * 4);
}
