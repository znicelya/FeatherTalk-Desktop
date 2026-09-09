//! Helpers shared by this crate's integration tests.
//!
//! Included with `#[path = "support/harness.rs"] mod harness;` rather than
//! compiled as its own test target, so each test binary gets its own copy.
//! That is why the whole module allows dead code: no single test uses all of it.

#![allow(dead_code)]

use std::path::PathBuf;
use std::time::Duration;

use feathertalk_client::SessionOptions;

/// Cargo builds the fake worker before the test binary and hands us its path.
pub const FAKE_WORKER: &str = env!("CARGO_BIN_EXE_feathertalk-fake-worker");

pub fn fake_worker() -> PathBuf {
    PathBuf::from(FAKE_WORKER)
}

/// The environment that selects one fake worker scenario.
pub fn scenario(name: &str) -> Vec<(String, String)> {
    vec![("FT_FAKE_WORKER_SCENARIO".to_string(), name.to_string())]
}

/// Production deadlines are seconds long. Tests use these instead so a case
/// that is *supposed* to hit a deadline finishes in well under a second.
pub fn fast_options() -> SessionOptions {
    SessionOptions {
        handshake_timeout: Duration::from_millis(800),
        cancel_grace: Duration::from_millis(200),
        shutdown_grace: Duration::from_millis(200),
        stderr_tail_lines: 20,
    }
}
