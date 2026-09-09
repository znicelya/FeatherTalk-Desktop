use std::path::PathBuf;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum PreprocessError {
    #[error("I/O error during {operation} at {path}: {source}")]
    Io {
        operation: &'static str,
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("landmark file is not valid UTF-8: {path}")]
    InvalidUtf8 { path: PathBuf },
    #[error("invalid landmark line {line} in {path}: {message}")]
    InvalidLine {
        path: PathBuf,
        line: usize,
        message: String,
    },
    #[error("wrong landmark count in {path}: expected {expected}, got {actual}")]
    WrongLandmarkCount {
        path: PathBuf,
        expected: usize,
        actual: usize,
    },
    #[error("non-finite coordinate at line {line} in {path}")]
    NonFiniteCoordinate { path: PathBuf, line: usize },
    #[error("negative coordinate at line {line} in {path}")]
    NegativeCoordinate { path: PathBuf, line: usize },
    #[error("invalid geometry for {field}: {message}")]
    InvalidGeometry {
        field: &'static str,
        message: String,
    },
    #[error("frame index {frame_index} is outside frame count {frame_count}")]
    FrameIndexOutOfRange {
        frame_index: usize,
        frame_count: usize,
    },
}
