//! Whether the worker executable the shell would drive actually exists.
//!
//! Discovery is `feathertalk-client`'s job; this module only turns its two
//! outcomes into something the shell can paint. A `Missing` status keeps the
//! candidate list, because "configured but absent" and "never configured" are
//! different problems and the operator can only tell them apart by seeing the
//! probed paths next to the knob each one came from.

use std::path::PathBuf;

use feathertalk_client::{ProbedPath, WorkerLocator, WorkerPathSource};

/// The catalog key naming a discovery source.
///
/// `WorkerPathSource::as_label` is English on purpose -- the client crate serves
/// the CLI too -- so the shell maps the source onto its own copy instead.
pub fn source_key(source: WorkerPathSource) -> &'static str {
    match source {
        WorkerPathSource::CliOption => "shell.worker.source.cli",
        WorkerPathSource::EnvVar => "shell.worker.source.env",
        WorkerPathSource::SiblingOfCurrentExe => "shell.worker.source.sibling",
    }
}

/// The result of one worker discovery pass.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WorkerStatus {
    /// The worker executable was found at `path`.
    Ready { path: PathBuf },
    /// No usable worker executable; `probed` is every source in priority order.
    Missing { probed: Vec<ProbedPath> },
}

impl WorkerStatus {
    /// Probe once at boot or during a device refresh. Task retries and restarts
    /// still belong to the worker supervision slice.
    pub fn probe(locator: &WorkerLocator) -> Self {
        match locator.resolve() {
            Ok(path) => Self::Ready { path },
            Err(_) => Self::Missing {
                probed: locator.candidates(),
            },
        }
    }

    /// Whether a worker executable is available.
    pub fn is_ready(&self) -> bool {
        matches!(self, Self::Ready { .. })
    }

    /// The catalog key of the status badge.
    pub fn badge_key(&self) -> &'static str {
        match self {
            Self::Ready { .. } => "shell.worker.ready",
            Self::Missing { .. } => "shell.worker.missing",
        }
    }

    /// The candidates a failed probe went through; empty once a worker resolved.
    pub fn probed(&self) -> &[ProbedPath] {
        match self {
            Self::Ready { .. } => &[],
            Self::Missing { probed } => probed,
        }
    }
}
