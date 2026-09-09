use thiserror::Error;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum FaceError {
    #[error("image dimensions must be non-zero")]
    InvalidImageSize,
    #[error("invalid configuration for {field}: {message}")]
    InvalidConfiguration {
        field: &'static str,
        message: String,
    },
    #[error(
        "invalid tensor length at level {level} for {field}: expected {expected}, got {actual}"
    )]
    InvalidTensorLength {
        level: usize,
        field: &'static str,
        expected: usize,
        actual: usize,
    },
    #[error("non-finite value at level {level} for {field}, index {index}")]
    NonFiniteValue {
        level: usize,
        field: &'static str,
        index: usize,
    },
    #[error("invalid detection geometry at index {index}")]
    InvalidDetectionGeometry { index: usize },
    #[error("invalid crop geometry for {field}: {message}")]
    InvalidCropGeometry {
        field: &'static str,
        message: String,
    },
}
