use thiserror::Error;

#[derive(Debug, Error)]
pub enum PackageError {
    #[error("invalid model package request: {0}")]
    InvalidRequest(String),
    #[error("invalid model package manifest: {0}")]
    InvalidManifest(String),
    #[error("invalid model package license bundle: {0}")]
    InvalidLicense(String),
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("model package hash mismatch for {file}: expected {expected}, got {actual}")]
    HashMismatch {
        file: String,
        expected: String,
        actual: String,
    },
    #[error("Burn store error: {0}")]
    Store(String),
    #[error("legacy weight import error: {0}")]
    WeightImport(#[from] feathertalk_weights::WeightImportError),
    #[error("model package publication error: {0}")]
    Publication(String),
}
