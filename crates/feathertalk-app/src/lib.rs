//! The FeatherTalk desktop workbench.
//!
//! `args`, `catalog`, `navigation` and `worker_status` hold no gpui types, so the
//! shell's decisions are unit testable without a display. Everything that renders
//! sits on top of them.

pub mod args;
pub mod assets;
pub mod catalog;
pub mod components;
pub mod compute;
pub mod facts;
pub mod generate;
pub mod manifest;
pub mod model_picker;
pub mod models;
pub mod navigation;
pub mod picker;
pub mod pipeline;
pub mod project;
pub mod state;
pub mod submit;
pub mod tasks;
pub mod theme;
pub mod training;
pub mod training_progress;
pub mod ui;
pub mod ui_assets;
pub mod workbench;
pub mod worker_status;
