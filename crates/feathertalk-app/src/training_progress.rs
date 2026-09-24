//! Per-epoch presentation of the worker's cumulative training counters.

use feathertalk_domain::{Event, Progress, TaskStage};

/// A task's submitted target and last observed batch. Kept separately from its
/// current stage, because shutdown events carry no training position.
#[derive(Debug, Clone)]
pub struct TrainingProgress {
    total_epochs: u32,
    last: Option<TrainingSnapshot>,
}

#[derive(Debug, Clone, Copy)]
struct TrainingSnapshot {
    epoch: u32,
    global_step: u64,
    total_steps: Option<u64>,
}

impl TrainingProgress {
    pub fn new(total_epochs: u32) -> Self {
        Self {
            total_epochs,
            last: None,
        }
    }

    pub fn total_epochs(&self) -> u32 {
        self.total_epochs
    }

    /// Accept rollback after an automatic checkpoint retry. Non-training events
    /// leave the last observed position intact, including terminal events.
    pub fn observe(&mut self, event: &Event) {
        if let TaskStage::Training { epoch, step, .. } = &event.stage {
            self.last = Some(TrainingSnapshot {
                epoch: *epoch,
                global_step: *step,
                total_steps: event.progress.and_then(|progress| progress.total),
            });
        }
    }

    /// The batch's one-based epoch, not the checkpoint's completed epoch count.
    pub fn epoch(&self) -> Option<u32> {
        self.last?
            .epoch
            .checked_add(1)
            .filter(|epoch| *epoch <= self.total_epochs)
    }

    /// Optimizer steps in this epoch, including the last partial batch. Unknown
    /// or inconsistent counters remain unknown instead of using a global count.
    pub fn steps(&self) -> Option<Progress> {
        let last = self.last?;
        self.epoch()?;
        let total = last.total_steps?;
        let epochs = u64::from(self.total_epochs);
        let steps_per_epoch = total.checked_div(epochs)?;
        if steps_per_epoch == 0 || !total.is_multiple_of(epochs) {
            return None;
        }
        let preceding_steps = u64::from(last.epoch).checked_mul(steps_per_epoch)?;
        let completed = last.global_step.checked_sub(preceding_steps)?;
        if completed == 0 || completed > steps_per_epoch {
            return None;
        }
        Some(Progress {
            completed,
            total: Some(steps_per_epoch),
        })
    }
}
