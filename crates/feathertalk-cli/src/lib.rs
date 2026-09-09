//! The FeatherTalk command line client.
//!
//! A library plus a thin binary, following `tools/onnx-validate`: `main.rs` only
//! parses arguments and exits, so every line of user-facing text is reachable
//! from a test without spawning a process.

mod cli;
mod render;
mod run;

pub use cli::{Cli, Command};
pub use render::{
    HumanSink, JsonSink, capabilities_report, event_line, failure_block, recovery_label,
    render_client_error, slug, stage_label,
};
pub use run::{EXIT_CANCELLED, EXIT_COMPLETED, EXIT_SESSION_ERROR, EXIT_TASK_FAILED, run};
