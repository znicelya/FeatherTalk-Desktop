use std::cell::{Cell, RefCell};

use feathertalk_domain::{Progress, TaskError, TaskStage};
use feathertalk_worker::{GpuFailure, TaskReporter};

/// A device fault first observed at an artifact publication boundary.
pub struct PublicationFault {
    checks: Cell<usize>,
    fail_on: usize,
    stage: RefCell<TaskStage>,
}

impl PublicationFault {
    pub fn on_check(fail_on: usize) -> Self {
        Self {
            checks: Cell::new(0),
            fail_on,
            stage: RefCell::new(TaskStage::Preparing),
        }
    }
}

impl TaskReporter for PublicationFault {
    fn report(&self, stage: TaskStage, _progress: Option<Progress>) {
        *self.stage.borrow_mut() = stage;
    }

    fn before_publish(&self) -> Result<(), TaskError> {
        let check = self.checks.get() + 1;
        self.checks.set(check);
        if check == self.fail_on {
            Err(
                GpuFailure::DeviceLost("injected device loss before publication".into())
                    .task_error(self.stage.borrow().clone()),
            )
        } else {
            Ok(())
        }
    }
}
