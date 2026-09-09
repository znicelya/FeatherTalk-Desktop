#[derive(Debug, thiserror::Error)]
pub enum TrainingError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("invalid training input: {0}")]
    InvalidInput(String),
    #[error("invalid training configuration: {0}")]
    InvalidConfig(String),
    #[error("invalid data loader configuration: {0}")]
    InvalidDataLoaderConfig(String),
    #[error("invalid data loader state: {0}")]
    InvalidDataLoaderState(String),
    #[error("data loader arithmetic overflow while {operation}")]
    DataLoaderOverflow { operation: &'static str },
    #[error("unable to allocate epoch permutation for {samples} samples")]
    PermutationAllocation {
        samples: u64,
        #[source]
        source: std::collections::TryReserveError,
    },
    #[error("unable to allocate prepared batch buffers for {items} items")]
    BatchAllocation {
        items: u64,
        #[source]
        source: std::collections::TryReserveError,
    },
    #[error("prepared batch is stale or belongs to another data loader")]
    StalePreparedBatch,
    #[error("invalid VGG19 package: {0}")]
    InvalidPackage(String),
    #[error("VGG19 package hash mismatch for {file}: expected {expected}, got {actual}")]
    HashMismatch {
        file: String,
        expected: String,
        actual: String,
    },
    #[error("Burn store error: {0}")]
    Store(String),
    #[error("invalid training checkpoint: {0}")]
    InvalidCheckpoint(String),
    #[error("training checkpoint compatibility error: {0}")]
    CheckpointCompatibility(String),
    #[error("training checkpoint directory error: {0}")]
    CheckpointDirectory(String),
}
