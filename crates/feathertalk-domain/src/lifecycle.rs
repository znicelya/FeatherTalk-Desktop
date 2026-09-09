use crate::{DomainError, TaskStage};

#[derive(Debug, Clone, PartialEq)]
pub struct TaskLifecycle {
    current: TaskStage,
}

impl Default for TaskLifecycle {
    fn default() -> Self {
        Self::new()
    }
}

impl TaskLifecycle {
    pub fn new() -> Self {
        Self {
            current: TaskStage::Queued,
        }
    }

    pub fn current(&self) -> &TaskStage {
        &self.current
    }

    pub fn is_terminal(&self) -> bool {
        self.current.is_terminal()
    }

    pub fn advance(&mut self, next: TaskStage) -> Result<(), DomainError> {
        if self.current.is_terminal() || matches!(next, TaskStage::Queued) {
            return Err(DomainError::InvalidTransition {
                from: self.current.as_slug(),
                to: next.as_slug(),
            });
        }
        self.current = next;
        Ok(())
    }

    pub fn request_cancel(&mut self) -> Result<bool, DomainError> {
        if self.current.is_terminal() {
            return Ok(false);
        }
        self.current = TaskStage::Cancelled;
        Ok(true)
    }
}
