//! Worker protocol client.
//!
//! This crate owns the worker child process and the version 2 stdio protocol:
//! discovery, spawn, handshake, one task per session, cancellation, and
//! reaping. It performs no argument parsing and writes nothing to the
//! terminal, so both the CLI and the future desktop shell can drive the same
//! transport with their own presentation.

mod compute;
mod error;
mod locator;
mod options;
mod session;
mod task_id;

pub use compute::{
    ComputeError, ComputeOptions, ENV_WORKER_ADAPTER, ENV_WORKER_BACKEND, adapter_is_eligible,
    backend_name, uses_compute_selection,
};
pub use error::{ClientError, ProbedPath, WorkerPathSource};
pub use locator::{ENV_WORKER_BIN, WORKER_FILE_STEM, WorkerLocator};
pub use options::SessionOptions;
pub use session::{CancelToken, EventSink, FrameLine, SessionOutcome, WorkerSession};
pub use task_id::generate_task_id;
