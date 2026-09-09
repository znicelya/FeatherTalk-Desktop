use std::path::PathBuf;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum ProjectError {
    #[error("I/O error during {operation} at {path}: {source}")]
    Io {
        operation: &'static str,
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("manifest at {path} exceeds the maximum size of {limit} bytes")]
    ManifestTooLarge { path: PathBuf, limit: usize },
    #[error("manifest at {path} is not valid UTF-8")]
    InvalidUtf8 { path: PathBuf },
    #[error("manifest at {path} contains invalid JSON: {source}")]
    InvalidJson {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },
    #[error("unsupported schema version {version} at {path}")]
    UnsupportedSchemaVersion { path: PathBuf, version: u32 },
    #[error("invalid field {field}: {message}")]
    InvalidField { field: String, message: String },
    #[error("unsafe relative path {path}")]
    UnsafeRelativePath { path: String },
    #[error("symbolic link encountered at {path}")]
    Symlink { path: PathBuf },
    #[error("missing or invalid filesystem entry at {path}")]
    InvalidFilesystemEntry { path: PathBuf },
    #[error("required artifact is empty at {path}")]
    EmptyArtifact { path: PathBuf },
    #[error("locked asset package cannot be mutated at {path}")]
    LockedAssetMutation { path: PathBuf },
    #[error("atomic replacement is unsupported at {path}")]
    AtomicReplacementUnsupported { path: PathBuf },
}
