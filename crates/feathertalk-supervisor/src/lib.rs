//! Desktop-side worker supervision.
//!
//! The desktop shell must survive everything the worker can do to itself: a
//! crash, a lost GPU device, a botched install. This crate owns that policy —
//! when to restart, when to give up, what to keep on disk afterwards, and which
//! unfinished tasks a fresh launch has to offer to resume or clean up. It draws
//! no pixels and writes nothing to the terminal, so every decision here is
//! testable without a window.

pub mod crash_log;
pub mod error;
pub mod journal;
pub mod policy;
pub mod recovery;
pub mod runner;
pub mod status;
pub mod supervisor;

pub use crash_log::{CrashLogOutcome, CrashReport, write_crash_log};
pub use error::SupervisorError;
/// Re-exported because `IncompleteTask::status` is one of these: a caller that
/// reads the startup scan should not need `feathertalk-project` in its manifest
/// to name the type it just received.
pub use feathertalk_project::TaskHistoryStatus;
pub use journal::TaskJournal;
pub use policy::{Disposition, RestartPolicy};
pub use recovery::{
    IncompleteTask, Resolution, apply_resolutions, resolve_project, scan_incomplete, scan_project,
};
pub use runner::{Attempt, AttemptResult, TaskRunner, WorkerRunner};
pub use supervisor::{SupervisedOutcome, SupervisionReport, WorkerSupervisor};
