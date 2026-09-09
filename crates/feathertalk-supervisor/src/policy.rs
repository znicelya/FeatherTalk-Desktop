use std::time::Duration;

use feathertalk_client::{ClientError, SessionOutcome};
use feathertalk_domain::ErrorCode;

/// How many times one task may be attempted, and how long to wait in between.
///
/// `max_attempts` counts the first attempt, so the default of two means "try
/// once, restart once". Tests set `restart_delay` to zero; the desktop shell
/// keeps a short pause so a worker that dies instantly cannot spin the CPU.
#[derive(Debug, Clone)]
pub struct RestartPolicy {
    pub max_attempts: u32,
    pub restart_delay: Duration,
}

impl Default for RestartPolicy {
    fn default() -> Self {
        Self {
            max_attempts: 2,
            restart_delay: Duration::from_millis(500),
        }
    }
}

/// What the supervisor should do after one attempt ends.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Disposition {
    /// The task reached a terminal state the caller has to be told about.
    Finish,
    /// Spawn a fresh worker. `resume` asks the retry to continue from the last
    /// checkpoint instead of starting over.
    Restart { resume: bool },
    /// Restarting cannot help, or the attempt budget is spent.
    GiveUp,
}

/// Decide what one finished attempt means.
///
/// `attempt` is one-based. Both matches are exhaustive on purpose: when the
/// protocol or the client grows a variant, this function stops compiling instead
/// of quietly folding the new case into "give up".
pub fn classify(outcome: &SessionOutcome, attempt: u32, policy: &RestartPolicy) -> Disposition {
    match outcome {
        SessionOutcome::Completed { .. } | SessionOutcome::Cancelled => Disposition::Finish,
        SessionOutcome::Failed(error) => match error.code {
            // These two are the only codes whose default recovery is
            // `ResumeFromCheckpoint`, and the only ones a fresh process can
            // plausibly get past: the GPU is gone, or the worker died on its own.
            ErrorCode::GpuDeviceLost | ErrorCode::WorkerCrashed => {
                restart_or_give_up(true, attempt, policy)
            }
            // Everything else is a deterministic failure. Re-running it burns
            // time and scrolls the real reason off the user's screen; the error
            // already carries a `Recovery` the interface can turn into a button.
            ErrorCode::MediaInvalid
            | ErrorCode::FaceNotFound
            | ErrorCode::LandmarkInvalid
            | ErrorCode::FeatureShapeMismatch
            | ErrorCode::ModelIncompatible
            | ErrorCode::GpuOutOfMemory
            | ErrorCode::DiskSpaceLow
            | ErrorCode::TaskCancelled => Disposition::Finish,
        },
        SessionOutcome::SessionError(error) => match error {
            // The worker vanished mid-task: exactly the crash isolation case.
            ClientError::WorkerGone { .. } => restart_or_give_up(true, attempt, policy),
            // The process never got as far as running the task, so there is no
            // checkpoint to resume from, but a second spawn is cheap and a
            // transient startup failure is plausible.
            ClientError::Handshake { .. } | ClientError::Spawn { .. } => {
                restart_or_give_up(false, attempt, policy)
            }
            // Installation, version and vocabulary problems: the same executable
            // will answer the same way every time.
            ClientError::WorkerNotFound { .. }
            | ClientError::ProtocolVersion { .. }
            | ClientError::Rejected { .. }
            | ClientError::UnsupportedCommand { .. }
            | ClientError::Protocol(_)
            | ClientError::Io(_) => Disposition::GiveUp,
        },
    }
}

fn restart_or_give_up(resume: bool, attempt: u32, policy: &RestartPolicy) -> Disposition {
    if attempt >= policy.max_attempts {
        Disposition::GiveUp
    } else {
        Disposition::Restart { resume }
    }
}
