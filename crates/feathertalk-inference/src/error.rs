use std::path::PathBuf;

#[derive(Debug, thiserror::Error)]
pub enum InferenceError {
    #[error("invalid input directory for {field}: {path}")]
    InvalidInputDirectory { field: &'static str, path: PathBuf },
    #[error("invalid input artifact for {field} at {path}: {message}")]
    InvalidInputArtifact {
        field: &'static str,
        path: PathBuf,
        message: String,
    },
    #[error("frame index {index} is outside source frame count {count}")]
    FrameIndexOutOfRange { index: usize, count: usize },
    #[error(
        "frame {index} dimensions differ from expected {expected_width}x{expected_height}: got {actual_width}x{actual_height}"
    )]
    FrameDimensionsMismatch {
        index: usize,
        expected_width: u32,
        expected_height: u32,
        actual_width: u32,
        actual_height: u32,
    },
    #[error("frame reader failed for frame {index} at {path}: {message}")]
    FrameReader {
        index: usize,
        path: PathBuf,
        message: String,
    },
    #[error("failed to start raw video sink: {message}")]
    SinkStart { message: String },
    #[error("raw video sink write failed: {message}")]
    SinkWrite { message: String },
    #[error("raw video sink finish failed: {message}")]
    SinkFinish { message: String },
    #[error("staging output collision: {path}")]
    StagingCollision { path: PathBuf },
    #[error("staging output is invalid: {path}: {message}")]
    StagingOutputInvalid { path: PathBuf, message: String },
    #[error("atomic publish failed for {path}: {message}")]
    AtomicPublishFailed { path: PathBuf, message: String },
    #[error("tool failed during {operation}: exit_code={exit_code:?}; stderr={stderr}")]
    ToolFailed {
        operation: &'static str,
        exit_code: Option<i32>,
        stderr: String,
    },
    #[error("source frame count {actual} is too small; expected at least {minimum}")]
    FrameCountTooSmall { actual: usize, minimum: usize },
    #[error("feature frame count must be greater than zero")]
    EmptyFeatures,
    #[error("output frame index {index} is outside output count {count}")]
    OutputFrameOutOfRange { index: usize, count: usize },
    #[error("invalid inference field {field}: {message}")]
    InvalidField {
        field: &'static str,
        message: String,
    },
    #[error("arithmetic overflow while building inference plan")]
    ArithmeticOverflow,
    #[error("output destination already exists: {path}")]
    OutputExists { path: PathBuf },
    #[error("output destination is not a regular non-symlink file: {path}")]
    OutputNotRegular { path: PathBuf },
    #[error("symbolic link encountered in output path: {path}")]
    OutputSymlink { path: PathBuf },
    #[error("output parent directory is missing or invalid: {path}")]
    OutputParentInvalid { path: PathBuf },
    #[error("task id is invalid: {task_id}")]
    InvalidTaskId { task_id: String },
    #[error("FFmpeg path must be absolute: {path}")]
    FfmpegPathNotAbsolute { path: PathBuf },
    #[error("FFmpeg path must not be empty")]
    EmptyFfmpegPath,
    #[error("frame dimensions must be non-zero: {width}x{height}")]
    InvalidFrameDimensions { width: u32, height: u32 },
    #[error("BGR frame buffer length mismatch: expected {expected} bytes, got {actual}")]
    FrameBufferLengthMismatch { expected: usize, actual: usize },
    #[error("pixel ({x}, {y}) is outside frame {width}x{height}")]
    PixelOutOfRange {
        x: u32,
        y: u32,
        width: u32,
        height: u32,
    },
    #[error(
        "invalid bounding box ({xmin}, {ymin})..({xmax}, {ymax}) for frame {frame_width}x{frame_height}"
    )]
    InvalidBbox {
        xmin: i32,
        ymin: i32,
        xmax: i32,
        ymax: i32,
        frame_width: u32,
        frame_height: u32,
    },
    #[error("resize target must be non-zero: {width}x{height}")]
    InvalidResizeTarget { width: u32, height: u32 },
    #[error("tensor shape mismatch for {context}: expected {expected:?}, got {actual:?}")]
    TensorShapeMismatch {
        context: &'static str,
        expected: Vec<usize>,
        actual: Vec<usize>,
    },
    #[error("invalid feature shape: tokens={tokens}, dims={dims}")]
    InvalidFeatureShape { tokens: usize, dims: usize },
    #[error(
        "audio window slot {slot} references frame {index}, but feature frame count is {frame_count}"
    )]
    InvalidAudioWindowIndex {
        slot: usize,
        index: usize,
        frame_count: usize,
    },
    #[error("model input value at {context}[{index}] is not finite")]
    NonFiniteModelInput { context: &'static str, index: usize },
    #[error("failed to read model tensor data for {context}: {message}")]
    ModelTensorData {
        context: &'static str,
        message: String,
    },
    #[error("model output value at index {index} is not finite")]
    NonFiniteModelOutput { index: usize },
    #[error("model output value at index {index} is outside [0,1]: {value}")]
    ModelOutputOutOfRange { index: usize, value: f32 },
    #[error("prediction value at index {index} is not finite")]
    NonFinitePrediction { index: usize },
    #[error(
        "cannot paste {source_width}x{source_height} frame at ({x}, {y}) into {destination_width}x{destination_height} frame"
    )]
    PasteOutOfBounds {
        x: i32,
        y: i32,
        source_width: u32,
        source_height: u32,
        destination_width: u32,
        destination_height: u32,
    },
    #[error("allocation of {bytes} bytes failed")]
    AllocationFailure { bytes: usize },
    /// The caller stopped the render. A cancelled operation is not a failure of
    /// the render, which is why it has its own variant instead of borrowing a
    /// sink error and a sentinel message.
    #[error("cancelled during {operation}")]
    Cancelled { operation: &'static str },
}
