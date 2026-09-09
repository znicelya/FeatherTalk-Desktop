use std::path::PathBuf;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum MediaError {
    #[error("I/O error during {operation} at {path}: {source}")]
    Io {
        operation: &'static str,
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("media input does not exist: {path}")]
    InputMissing { path: PathBuf },
    #[error("media input is not a regular file: {path}")]
    InputNotRegularFile { path: PathBuf },
    #[error("symbolic link is not allowed: {path}")]
    SymlinkNotAllowed { path: PathBuf },
    #[error("output directory is invalid: {path}")]
    OutputDirectoryInvalid { path: PathBuf },
    #[error("output directory contains the input: input={input}, output={output}")]
    OutputInsideInput { input: PathBuf, output: PathBuf },
    #[error("normalization output conflicts with input: {path}")]
    OutputConflictsWithInput { path: PathBuf },
    #[error("normalization output destination is invalid: {path}")]
    OutputDestinationInvalid { path: PathBuf },
    #[error("unsupported normalization target for {field}: expected {expected}, got {actual}")]
    UnsupportedTarget {
        field: &'static str,
        expected: &'static str,
        actual: String,
    },
    #[error("invalid media toolchain field {field}: {message}")]
    InvalidToolchain {
        field: &'static str,
        message: String,
    },
    #[error("media probe exceeds {limit} bytes: {actual}")]
    ProbeTooLarge { limit: usize, actual: usize },
    #[error("media probe JSON is invalid: {message}")]
    ProbeJson { message: String },
    #[error("media probe contract violation for {field}: {message}")]
    ProbeContract { field: String, message: String },
    #[error("media probe is missing required {stream} stream")]
    MissingStream { stream: &'static str },
    #[error("media probe contains multiple {stream} streams")]
    DuplicateStream { stream: &'static str },
    #[error("media tool failed during {operation}: exit={exit_code:?}, stderr={stderr}")]
    ToolFailed {
        operation: &'static str,
        exit_code: Option<i32>,
        stderr: String,
    },
    #[error("media tool timed out during {operation} after {timeout_ms} ms")]
    ToolTimedOut {
        operation: &'static str,
        timeout_ms: u64,
    },
    #[error("media tool was cancelled during {operation}")]
    ToolCancelled { operation: &'static str },
    #[error("media tool output exceeded {limit} bytes during {operation} on {stream}: {actual}")]
    ToolOutputTooLarge {
        operation: &'static str,
        stream: &'static str,
        limit: usize,
        actual: usize,
    },
    #[error("failed to spawn media tool during {operation}: {message}")]
    ToolSpawn {
        operation: &'static str,
        message: String,
    },
    #[error("normalized media verification failed for {field}: expected {expected}, got {actual}")]
    NormalizationVerificationFailed {
        field: &'static str,
        expected: String,
        actual: String,
    },
    #[error("media output commit failed during {operation}: {message}")]
    OutputCommitFailed {
        operation: &'static str,
        message: String,
    },
    #[error(
        "media output rollback failed during {operation}: primary={primary}; rollback={rollback}"
    )]
    OutputRollbackFailed {
        operation: &'static str,
        primary: String,
        rollback: String,
    },
}
