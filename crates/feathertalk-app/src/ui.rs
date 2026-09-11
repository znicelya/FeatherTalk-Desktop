//! Presentation state and decisions that do not need a running window.

use feathertalk_domain::{Progress, TaskStatus};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum AppLocale {
    #[default]
    ZhCn,
    En,
}

impl AppLocale {
    pub const fn tag(self) -> &'static str {
        match self {
            Self::ZhCn => "zh-CN",
            Self::En => "en",
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum TaskFilter {
    #[default]
    All,
    Active,
    Completed,
    Failed,
    Cancelled,
}

impl TaskFilter {
    pub const ALL: [Self; 5] = [
        Self::All,
        Self::Active,
        Self::Completed,
        Self::Failed,
        Self::Cancelled,
    ];

    pub fn includes(self, status: TaskStatus) -> bool {
        match self {
            Self::All => true,
            Self::Active => status.is_incomplete(),
            Self::Completed => status == TaskStatus::Completed,
            Self::Failed => status == TaskStatus::Failed,
            Self::Cancelled => status == TaskStatus::Cancelled,
        }
    }

    pub fn key(self) -> &'static str {
        match self {
            Self::All => "ui.tasks.all",
            Self::Active => "ui.tasks.active",
            Self::Completed => "ui.tasks.completed",
            Self::Failed => "ui.tasks.failed",
            Self::Cancelled => "ui.tasks.cancelled",
        }
    }
}

#[derive(Debug, Default)]
pub struct UiState {
    pub locale: AppLocale,
    pub settings_open: bool,
    pub dark: bool,
    pub asset_details: bool,
    pub training_details: bool,
    pub generate_details: bool,
    pub task_filter: TaskFilter,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProgressPresentation {
    Indeterminate,
    Counted { completed: u64, total: u64 },
    Complete,
    Empty,
}

pub fn progress_presentation(
    status: TaskStatus,
    progress: Option<Progress>,
) -> ProgressPresentation {
    if status == TaskStatus::Completed {
        return ProgressPresentation::Complete;
    }
    if let Some(Progress {
        completed,
        total: Some(total),
    }) = progress
    {
        return ProgressPresentation::Counted {
            completed,
            total: total.max(1),
        };
    }
    if status.is_incomplete() {
        ProgressPresentation::Indeterminate
    } else {
        ProgressPresentation::Empty
    }
}
