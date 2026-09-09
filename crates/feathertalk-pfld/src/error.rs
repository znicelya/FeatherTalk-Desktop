use std::path::PathBuf;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum PfldError {
    #[error("I/O error during {operation} at {path}: {source}")]
    Io {
        operation: &'static str,
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("mean face file is not valid UTF-8: {path}")]
    InvalidUtf8 { path: PathBuf },
    #[error("invalid mean face token at index {index} in {path}")]
    InvalidMeanFaceToken { path: PathBuf, index: usize },
    #[error("invalid mean face count in {path}: expected {expected}, got {actual}")]
    InvalidMeanFaceCount {
        path: PathBuf,
        expected: usize,
        actual: usize,
    },
    #[error("invalid vector length for {field}: expected {expected}, got {actual}")]
    InvalidVectorLength {
        field: &'static str,
        expected: usize,
        actual: usize,
    },
    #[error("non-finite value for {field} at index {index}")]
    NonFiniteValue { field: &'static str, index: usize },
    #[error("crop width and height must be non-zero")]
    InvalidCropGeometry,
    #[error("decoded coordinate {axis} at landmark index {index} is outside i32 range")]
    CoordinateOutOfRange { index: usize, axis: &'static str },
}

#[derive(Debug, thiserror::Error)]
pub enum PfldRuntimeError {
    #[error("I/O error during {operation} at {}: {source}", path.display())]
    Io {
        operation: &'static str,
        path: std::path::PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("manifest exceeds {limit} bytes: {actual}")]
    ManifestTooLarge { limit: u64, actual: u64 },
    #[error("weights exceed {limit} bytes: {actual}")]
    WeightsTooLarge { limit: u64, actual: u64 },
    #[error("manifest JSON error: {0}")]
    ManifestJson(String),
    #[error("unsupported manifest schema version: expected {expected}, got {actual}")]
    UnsupportedSchemaVersion { expected: u32, actual: u32 },
    #[error("unsupported PFLD architecture version: expected {expected}, got {actual}")]
    UnsupportedArchitectureVersion { expected: String, actual: String },
    #[error("invalid manifest field {field}: {message}")]
    InvalidManifest { field: String, message: String },
    #[error("SHA-256 mismatch for {artifact}: expected {expected}, got {actual}")]
    HashMismatch {
        artifact: &'static str,
        expected: String,
        actual: String,
    },
    #[error("weight byte count mismatch: expected {expected}, got {actual}")]
    WeightSizeMismatch { expected: u64, actual: u64 },
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
    #[error("invalid PFLD input shape: expected [1, 3, 192, 192], got {actual:?}")]
    InvalidInputShape { actual: [usize; 4] },
    #[error("PFLD input contains a non-finite value")]
    NonFiniteInput,
    #[error("invalid PFLD output shape: expected [1, 220], got {actual:?}")]
    InvalidOutputShape { actual: [usize; 2] },
    #[error("PFLD output contains a non-finite value")]
    NonFiniteOutput,
    #[error("artifact directory contains an unexpected entry: {0}")]
    UnexpectedArtifactEntry(String),
}
