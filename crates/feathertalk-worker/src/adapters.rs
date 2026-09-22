use std::collections::{BTreeMap, BTreeSet};

use feathertalk_domain::TaskId;

#[derive(Debug, thiserror::Error)]
pub enum AdapterLockError {
    #[error("unknown adapter {0}")]
    Unknown(String),
    #[error("adapter {adapter_id} is already running task {}", holder.as_str())]
    Occupied { adapter_id: String, holder: TaskId },
    #[error("adapter {0} is not locked")]
    NotHeld(String)}

/// Enforces "at most one task per adapter".
///
/// Keys are the stable IDs from the compute registry, including the default
/// CPU adapter and each discovered native GPU.
#[derive(Debug)]
pub struct AdapterLocks {
    known: BTreeSet<String>,
    occupied: BTreeMap<String, TaskId>}

impl AdapterLocks {
    pub fn new(adapter_ids: impl IntoIterator<Item = String>) -> Self {
        Self {
            known: adapter_ids.into_iter().collect(),
            occupied: BTreeMap::new()}
    }

    pub fn acquire(&mut self, adapter_id: &str, task_id: TaskId) -> Result<(), AdapterLockError> {
        self.check_known(adapter_id)?;
        if let Some(holder) = self.occupied.get(adapter_id) {
            return Err(AdapterLockError::Occupied {
                adapter_id: adapter_id.to_owned(),
                holder: holder.clone()});
        }
        self.occupied.insert(adapter_id.to_owned(), task_id);
        Ok(())
    }

    pub fn release(&mut self, adapter_id: &str) -> Result<(), AdapterLockError> {
        self.check_known(adapter_id)?;
        if self.occupied.remove(adapter_id).is_none() {
            return Err(AdapterLockError::NotHeld(adapter_id.to_owned()));
        }
        Ok(())
    }

    pub fn holder(&self, adapter_id: &str) -> Option<&TaskId> {
        self.occupied.get(adapter_id)
    }

    pub fn is_free(&self, adapter_id: &str) -> bool {
        self.known.contains(adapter_id) && !self.occupied.contains_key(adapter_id)
    }

    fn check_known(&self, adapter_id: &str) -> Result<(), AdapterLockError> {
        if self.known.contains(adapter_id) {
            Ok(())
        } else {
            Err(AdapterLockError::Unknown(adapter_id.to_owned()))
        }
    }
}
