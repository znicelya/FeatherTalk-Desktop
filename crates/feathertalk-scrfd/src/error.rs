use std::{io, path::PathBuf};

use thiserror::Error;

#[derive(Debug, Error)]
pub enum ScrfdError {
    #[error("I/O error during {operation} at {}: {source}", path.display())]
    Io {
        operation: &'static str,
        path: PathBuf,
        #[source]
        source: io::Error,
    },
    #[error("manifest exceeds {limit} bytes: {actual}")]
    ManifestTooLarge { limit: u64, actual: u64 },
    #[error("weights exceed {limit} bytes: {actual}")]
    WeightsTooLarge { limit: u64, actual: u64 },
    #[error("manifest JSON error: {0}")]
    ManifestJson(String),
    #[error("unsupported manifest schema version: expected {expected}, got {actual}")]
    UnsupportedSchemaVersion { expected: u32, actual: u32 },
    #[error("unsupported SCRFD architecture version: expected {expected}, got {actual}")]
    UnsupportedArchitectureVersion { expected: u32, actual: u32 },
    #[error("invalid manifest field {field}: {message}")]
    InvalidManifest { field: String, message: String },
    #[error("artifact contract mismatch for {field}: expected {expected}, got {actual}")]
    ContractMismatch {
        field: &'static str,
        expected: String,
        actual: String,
    },
    #[error("weight byte count mismatch: expected {expected}, got {actual}")]
    WeightSizeMismatch { expected: u64, actual: u64 },
    #[error("SHA-256 mismatch for {artifact}: expected {expected}, got {actual}")]
    HashMismatch {
        artifact: &'static str,
        expected: String,
        actual: String,
    },
    #[error("Burn store error: {0}")]
    Store(String),
    #[error("missing tensor: {0}")]
    MissingTensor(String),
    #[error("unexpected tensor: {0}")]
    UnexpectedTensor(String),
    #[error("tensor shape mismatch: {0}")]
    ShapeMismatch(String),
    #[error("tensor dtype mismatch: {0}")]
    DTypeMismatch(String),
    #[error("invalid SCRFD input shape: expected [1, 3, 640, 640], got {actual:?}")]
    InvalidInputShape { actual: [usize; 4] },
    #[error("invalid SCRFD output shape for {name}: expected {expected:?}, got {actual:?}")]
    InvalidOutputShape {
        name: &'static str,
        expected: Vec<usize>,
        actual: Vec<usize>,
    },
}
