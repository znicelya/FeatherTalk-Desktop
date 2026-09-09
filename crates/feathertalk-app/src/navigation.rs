//! The fixed workbench navigation.
//!
//! The migration design fixes the main navigation to five pages in one order.
//! `Page::ALL` is the only place that order lives, and every identity string is
//! a literal in a total `match`, so adding a page without giving it an element
//! id and its catalog keys fails to compile instead of rendering blank copy.

/// One of the five workbench pages.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum Page {
    #[default]
    Assets,
    Training,
    Generate,
    Models,
    Tasks,
}

impl Page {
    /// Every page, in navigation order.
    pub const ALL: [Page; 5] = [
        Page::Assets,
        Page::Training,
        Page::Generate,
        Page::Models,
        Page::Tasks,
    ];

    /// The stable element id of this page's navigation entry.
    ///
    /// Stateful children in a list need a stable id or focus and animation state
    /// desynchronise when the list re-renders.
    pub fn element_id(self) -> &'static str {
        match self {
            Self::Assets => "nav-assets",
            Self::Training => "nav-training",
            Self::Generate => "nav-generate",
            Self::Models => "nav-models",
            Self::Tasks => "nav-tasks",
        }
    }

    /// The catalog key of the navigation label.
    pub fn label_key(self) -> &'static str {
        match self {
            Self::Assets => "nav.assets",
            Self::Training => "nav.training",
            Self::Generate => "nav.generate",
            Self::Models => "nav.models",
            Self::Tasks => "nav.tasks",
        }
    }

    /// The catalog key of the page heading.
    pub fn title_key(self) -> &'static str {
        match self {
            Self::Assets => "page.assets.title",
            Self::Training => "page.training.title",
            Self::Generate => "page.generate.title",
            Self::Models => "page.models.title",
            Self::Tasks => "page.tasks.title",
        }
    }

    /// The catalog key of the placeholder body this slice renders.
    pub fn pending_key(self) -> &'static str {
        match self {
            Self::Assets => "page.assets.pending",
            Self::Training => "page.training.pending",
            Self::Generate => "page.generate.pending",
            Self::Models => "page.models.pending",
            Self::Tasks => "page.tasks.pending",
        }
    }
}

/// Which page the workbench is showing.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Navigation {
    current: Page,
}

impl Navigation {
    /// The page the shell is showing.
    pub fn current(&self) -> Page {
        self.current
    }

    /// Select `page`, reporting whether anything changed.
    ///
    /// The render layer notifies its entity only on a real change, so clicking
    /// the entry that is already selected costs no frame.
    pub fn select(&mut self, page: Page) -> bool {
        if self.current == page {
            return false;
        }
        self.current = page;
        true
    }
}
