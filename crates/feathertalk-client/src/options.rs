use std::time::Duration;

/// The deadlines and bounds of one worker session.
///
/// Every wait in this crate is bounded by one of these, so a misbehaving worker
/// can never hang the caller. Tests shorten them; the CLI uses the defaults.
#[derive(Debug, Clone)]
pub struct SessionOptions {
    /// How long the worker has to send `ready` after it is spawned.
    pub handshake_timeout: Duration,
    /// How long the worker has to react to `cancel` before it is killed.
    pub cancel_grace: Duration,
    /// How long the worker has to exit after `shutdown` before it is killed.
    pub shutdown_grace: Duration,
    /// How many of the worker's most recent stderr lines to keep for reports.
    pub stderr_tail_lines: usize,
}

impl Default for SessionOptions {
    fn default() -> Self {
        Self {
            handshake_timeout: Duration::from_secs(30),
            cancel_grace: Duration::from_secs(10),
            shutdown_grace: Duration::from_secs(5),
            stderr_tail_lines: 20,
        }
    }
}
