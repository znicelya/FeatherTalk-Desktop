use std::path::PathBuf;

use feathertalk_domain::DomainError;

/// Where a candidate worker path came from, so a failure can name the knob the
/// operator has to turn instead of just saying "not found".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkerPathSource {
    CliOption,
    EnvVar,
    SiblingOfCurrentExe,
}

impl WorkerPathSource {
    pub fn as_label(self) -> &'static str {
        match self {
            Self::CliOption => "--worker",
            Self::EnvVar => crate::ENV_WORKER_BIN,
            Self::SiblingOfCurrentExe => "sibling of the current executable",
        }
    }
}

/// One discovery attempt. `path` is `None` when the source was not set at all,
/// which reads differently to the operator than a path that was set and missed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProbedPath {
    pub source: WorkerPathSource,
    pub path: Option<PathBuf>,
}

/// Everything that can go wrong between the caller and the worker process.
///
/// `Display` is English on purpose: this crate is also the desktop shell's
/// transport, and each front end renders its own user-facing copy. The CLI
/// translates these into Chinese.
#[derive(Debug, thiserror::Error)]
pub enum ClientError {
    #[error("no worker executable was found")]
    WorkerNotFound { probed: Vec<ProbedPath> },
    #[error("failed to spawn the worker at {path}")]
    Spawn {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("the handshake failed: {reason}")]
    Handshake {
        reason: String,
        stderr_tail: Vec<String>,
    },
    #[error(
        "protocol version mismatch: this client speaks {expected}, the worker reported {actual}"
    )]
    ProtocolVersion { expected: u32, actual: u32 },
    #[error("the worker rejected the request: {reason}")]
    Rejected { reason: String },
    #[error("the worker does not support {requested}")]
    UnsupportedCommand {
        requested: &'static str,
        supported: Vec<&'static str>,
    },
    #[error("protocol error: {0}")]
    Protocol(#[from] DomainError),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("the worker exited without reporting a terminal stage")]
    WorkerGone {
        status: Option<i32>,
        stderr_tail: Vec<String>,
    },
}

impl ClientError {
    /// The last lines the worker wrote to stderr, when they were captured.
    ///
    /// A worker that dies during startup usually explains itself here and
    /// nowhere else, so every front end wants to print this.
    pub fn stderr_tail(&self) -> &[String] {
        match self {
            Self::Handshake { stderr_tail, .. } | Self::WorkerGone { stderr_tail, .. } => {
                stderr_tail
            }
            Self::WorkerNotFound { .. }
            | Self::Spawn { .. }
            | Self::ProtocolVersion { .. }
            | Self::Rejected { .. }
            | Self::UnsupportedCommand { .. }
            | Self::Protocol(_)
            | Self::Io(_) => &[],
        }
    }
}
