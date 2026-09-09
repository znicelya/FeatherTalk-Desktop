use thiserror::Error;

#[derive(Debug, Error)]
pub enum AudioError {
    #[error("wav I/O error during {operation} at {path}: {source}")]
    WavIo {
        operation: &'static str,
        path: std::path::PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("wav file is not a regular non-symlink file: {path}")]
    WavNotRegular { path: std::path::PathBuf },
    #[error("wav file exceeds {limit} bytes: {actual}")]
    WavTooLarge { limit: u64, actual: u64 },
    #[error("wav file is not a RIFF/WAVE container")]
    InvalidRiffHeader,
    #[error("wav header is invalid: {reason}")]
    InvalidWavHeader { reason: String },
    #[error("wav file is missing the {chunk:?} chunk")]
    MissingWavChunk { chunk: &'static str },
    #[error("unsupported wav format code {code}, expected 16-bit PCM")]
    UnsupportedWavFormat { code: u16 },
    #[error("unsupported wav channel count {actual}, expected mono")]
    UnsupportedWavChannels { actual: u16 },
    #[error("unsupported wav sample rate {actual}, expected {expected}")]
    UnsupportedWavSampleRate { actual: u32, expected: u32 },
    #[error("unsupported wav bit depth {actual}, expected 16")]
    UnsupportedWavBitDepth { actual: u16 },
    #[error("wav payload is truncated: expected {expected} bytes, got {actual}")]
    WavPayloadTruncated { expected: u64, actual: u64 },
    #[error("wav file has no samples")]
    EmptyWav,
    #[error("waveform is empty")]
    EmptyWaveform,
    #[error("waveform contains a non-finite value at index {index}")]
    NonFiniteWaveform { index: usize },
    #[error("waveform has zero variance")]
    ConstantWaveform,
    #[error("chunk size must be greater than zero")]
    InvalidChunkSize,
    #[error("audio arithmetic overflow")]
    ArithmeticOverflow,
    #[error("audio input is too large: {actual} chunks exceeds {limit}")]
    TooManyChunks { actual: usize, limit: usize },
    #[error("feature output dimension must be greater than zero")]
    InvalidFeatureDimension,
    #[error("feature output length {actual} is not divisible by dimension {dimension}")]
    FeatureLengthMismatch { actual: usize, dimension: usize },
    #[error(
        "feature shape does not match requested frame count {frame_count}: tokens={tokens}, dims={dims}"
    )]
    FeatureShapeMismatch {
        frame_count: u64,
        tokens: usize,
        dims: usize,
    },
    #[error("feature contains a non-finite value at index {index}")]
    NonFiniteFeature { index: usize },
    #[error("feature matrix size overflow")]
    FeatureSizeOverflow,
    #[error("feature file is not a regular non-symlink file: {path}")]
    FeatureNotRegular { path: std::path::PathBuf },
    #[error("feature file exceeds {limit} bytes: {actual}")]
    FeatureTooLarge { limit: u64, actual: u64 },
    #[error("feature file has an invalid magic header")]
    InvalidFeatureMagic,
    #[error("unsupported feature file version {version}")]
    UnsupportedFeatureVersion { version: u32 },
    #[error("feature file header is truncated: {actual} bytes")]
    FeatureHeaderTruncated { actual: usize },
    #[error("feature payload is truncated: expected {expected} bytes, got {actual}")]
    FeaturePayloadTruncated { expected: u64, actual: u64 },
    #[error("feature file has {actual} trailing bytes")]
    FeatureTrailingBytes { actual: usize },
    #[error("feature file has an invalid payload size")]
    InvalidFeaturePayloadSize,
    #[error("feature file pair width must be 2, got {actual}")]
    InvalidFeaturePairWidth { actual: u64 },
    #[error("feature I/O error during {operation} at {path}: {source}")]
    FeatureIo {
        operation: &'static str,
        path: std::path::PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("locked asset manifest cannot be mutated: {path}")]
    LockedAssetMutation { path: std::path::PathBuf },
    #[error("feature artifact commit failed during {operation}: {message}")]
    CommitFailed {
        operation: &'static str,
        message: String,
    },
    #[error(
        "feature artifact rollback failed during {operation}: primary={primary}; rollback={rollback}"
    )]
    CommitRollbackFailed {
        operation: &'static str,
        primary: String,
        rollback: String,
    },
    #[error("feature staging path already exists: {path}")]
    StagingCollision { path: std::path::PathBuf },
    #[error("{operation} was cancelled")]
    Cancelled { operation: &'static str },
}
