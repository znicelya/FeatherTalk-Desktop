use thiserror::Error;

#[derive(Debug, Error)]
pub enum PipelineError {
    #[error("invalid frame pipeline field {field}: {message}")]
    InvalidField {
        field: &'static str,
        message: String,
    },
    #[error("invalid quality report field {field}: {message}")]
    InvalidReport { field: String, message: String },
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
    #[error("I/O error during {operation} at {path}: {source}")]
    Io {
        operation: &'static str,
        path: std::path::PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("frame output is missing: {path}")]
    FrameMissing { path: std::path::PathBuf },
    #[error("frame output is not a regular non-symlink file: {path}")]
    FrameNotRegular { path: std::path::PathBuf },
    #[error("frame output is empty: {path}")]
    FrameEmpty { path: std::path::PathBuf },
    #[error("frame output exceeds {limit} bytes: {path} ({actual})")]
    FrameTooLarge {
        path: std::path::PathBuf,
        limit: u64,
        actual: u64,
    },
    /// A frame file exists but its JPEG header cannot be read.
    #[error("frame is not a decodable JPEG: {path} ({message})")]
    FrameUndecodable {
        path: std::path::PathBuf,
        message: String,
    },
    /// A landmark file is a symlink, a directory, or another non-regular entry.
    #[error("landmark file is not a regular non-symlink file: {path}")]
    LandmarkNotRegular { path: std::path::PathBuf },
    /// A landmark file's bytes are not what `serialize_landmarks` writes.
    #[error("landmark file is malformed: {path} ({message})")]
    InvalidLandmark {
        path: std::path::PathBuf,
        message: String,
    },
    #[error("output destination already exists: {path}")]
    OutputDestinationExists { path: std::path::PathBuf },
    #[error("{component} adapter failed: {message}")]
    Adapter {
        component: &'static str,
        message: String,
    },
    #[error("quality evaluation rejected frame artifacts: {count} anomalies")]
    QualityRejected { count: usize },
    #[error("quality report JSON is invalid: {message}")]
    ReportJson { message: String },
    #[error("quality report is not a regular non-symlink file: {path}")]
    ReportNotRegular { path: std::path::PathBuf },
    #[error("quality report exceeds {limit} bytes: {actual}")]
    ReportTooLarge { limit: usize, actual: usize },
    #[error("atomic frame artifact commit failed during {operation}: {message}")]
    PublishFailed {
        operation: &'static str,
        message: String,
    },
    #[error(
        "atomic frame artifact rollback failed during {operation}: primary={primary}; rollback={rollback}"
    )]
    PublishRollbackFailed {
        operation: &'static str,
        primary: String,
        rollback: String,
    },
    /// The observer asked the pipeline to stop.
    #[error("frame pipeline cancelled during {operation}")]
    Cancelled { operation: &'static str },
}
