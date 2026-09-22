use std::cell::RefCell;

use feathertalk_domain::{Metrics, Progress, TaskError, TaskStage};

/// How a running command reports intermediate stages.
///
/// A command never writes to stdout itself: the runtime's control loop is the
/// only owner of the writer and of task lifecycle, so a report is a message to
/// it, not an event. Terminal stages are not reported here; they are the
/// command's return value.
pub trait TaskReporter {
    fn report(&self, stage: TaskStage, progress: Option<Progress>);

    /// Existing observers can ignore measurements while the wire reporter
    /// carries real GPU allocation statistics to clients.
    fn report_metrics(&self, stage: TaskStage, progress: Option<Progress>, _metrics: Metrics) {
        self.report(stage, progress);
    }

    /// Final observer check after model readback and before publishing an
    /// artifact. GPU execution uses this boundary to surface asynchronous
    /// faults while the output is still staged and can be discarded.
    fn before_publish(&self) -> Result<(), TaskError> {
        Ok(())
    }
}

/// The reporter for callers that do not observe progress: direct library users
/// and tests that only assert the outcome.
pub struct NoReporter;

impl TaskReporter for NoReporter {
    fn report(&self, _stage: TaskStage, _progress: Option<Progress>) {}
}

/// Retains the last stage even when a model unwinds before returning an error.
pub(crate) struct TrackedReporter<'a> {
    inner: &'a dyn TaskReporter,
    stage: RefCell<TaskStage>}

impl<'a> TrackedReporter<'a> {
    pub(crate) fn new(inner: &'a dyn TaskReporter) -> Self {
        Self {
            inner,
            stage: RefCell::new(TaskStage::Preparing)}
    }

    pub(crate) fn stage(&self) -> TaskStage {
        self.stage.borrow().clone()
    }
}

impl TaskReporter for TrackedReporter<'_> {
    fn before_publish(&self) -> Result<(), TaskError> {
        self.inner.before_publish()
    }

    fn report(&self, stage: TaskStage, progress: Option<Progress>) {
        if !stage.is_terminal() {
            *self.stage.borrow_mut() = stage.clone();
        }
        self.inner.report(stage, progress);
    }

    fn report_metrics(&self, stage: TaskStage, progress: Option<Progress>, metrics: Metrics) {
        if !stage.is_terminal() {
            *self.stage.borrow_mut() = stage.clone();
        }
        self.inner.report_metrics(stage, progress, metrics);
    }
}
