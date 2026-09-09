use thiserror::Error;

#[derive(Debug, Error)]
pub enum WeightImportError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("unsafe checkpoint limit: {0}")]
    UnsafeLimit(String),
    #[error("unsupported checkpoint structure: {0}")]
    UnsupportedStructure(String),
    #[error("missing tensor: {0}")]
    MissingTensor(String),
    #[error("unexpected tensor: {0}")]
    UnexpectedTensor(String),
    #[error("shape mismatch: {0}")]
    ShapeMismatch(String),
    #[error("dtype mismatch: {0}")]
    DTypeMismatch(String),
    #[error("duplicate remapped tensor key: {0}")]
    DuplicateKey(String),
    #[error("Burn store error: {0}")]
    Store(String),
    #[error("invalid PFLD checkpoint envelope: {0}")]
    InvalidPfldEnvelope(String),
    #[error("invalid PFLD epoch: expected {expected}, got {actual}")]
    InvalidPfldEpoch { expected: u64, actual: String },
    #[error("invalid PFLD ignored tensor set: {0}")]
    InvalidPfldIgnoredSet(String),
    #[error("artifact destination already exists: {}", .0.display())]
    ArtifactDestinationExists(std::path::PathBuf),
    #[error("artifact validation failed: {0}")]
    ArtifactValidation(String),
    #[error("manifest error: {0}")]
    Manifest(String),
}
