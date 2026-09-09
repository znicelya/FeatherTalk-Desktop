//! Progress and cancellation seam for the long-running pipeline stages.
//!
//! This crate reports frame counts and asks whether it should stop. It knows
//! nothing about the worker protocol, threads, or task identifiers.

/// A progress point a pipeline stage reached.
///
/// `completed` and `total` count frames -- never chunks, never percentages --
/// so a caller can forward them into its own progress type unchanged.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PipelinePhase {
    /// `completed` frames of `total` have been written and inspected.
    Extracting { completed: u64, total: u64 },
    /// Evaluation is about to run on frame number `completed` of `total`.
    Evaluating { completed: u64, total: u64 },
}

/// Receives progress points and answers cancellation questions.
///
/// Both methods run on the thread that drives the pipeline, so an
/// implementation must not block. There is deliberately no `Send + Sync`
/// bound: the worker's reporter owns an `mpsc::Sender`, which is `Send` but
/// not `Sync`, and the pipeline never moves the observer across threads.
pub trait PipelineObserver {
    /// Called once per progress point. The default drops the phase.
    fn phase(&self, _phase: PipelinePhase) {}

    /// Called before each chunk and before each evaluated frame. The default
    /// never cancels.
    fn is_cancelled(&self) -> bool {
        false
    }
}

/// The observer for callers that want neither progress nor cancellation.
#[derive(Debug, Clone, Copy, Default)]
pub struct NoObserver;

impl PipelineObserver for NoObserver {}
