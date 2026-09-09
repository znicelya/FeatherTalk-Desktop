mod commands;
mod commit;
mod error;
mod execution;
mod model;
mod normalize;
mod probe;
mod process;
mod validate;

pub use commands::{
    CommandSpec, audio_normalization_command, probe_command, video_normalization_command,
};
pub use error::MediaError;
pub use execution::{probe_media, probe_media_with_runner, probe_video_with_runner};
pub use model::{
    AudioMetadata, FrameRate, MediaArtifact, MediaInput, MediaProbe, MediaToolchain,
    NormalizationSpec, NormalizedMedia, NormalizedMediaLayout, ProbeFormat, ValidatedInput,
    VideoMetadata,
};
pub use normalize::{
    NormalizePhase, normalize_media, normalize_media_observed, normalize_media_with_runner,
};
pub use probe::parse_probe_json;
pub use process::{
    CancellableProcessRunner, CancellationToken, MAX_CAPTURE_BYTES, ProcessOutput, ProcessRunner,
    SystemProcessRunner,
};
pub use validate::{validate_input, validate_normalization};
