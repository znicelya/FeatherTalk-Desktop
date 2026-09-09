use std::{
    path::{Path, PathBuf},
    time::Duration,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MediaInput {
    pub source: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NormalizationSpec {
    pub target_video_fps: u32,
    pub target_audio_sample_rate: u32,
    pub target_audio_channels: u16,
    pub output_dir: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidatedInput {
    source: PathBuf,
}

impl ValidatedInput {
    pub(crate) fn new(source: PathBuf) -> Self {
        Self { source }
    }

    pub fn source(&self) -> &Path {
        &self.source
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NormalizedMediaLayout {
    output_dir: PathBuf,
    video_path: PathBuf,
    audio_path: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MediaToolchain {
    ffmpeg: PathBuf,
    ffprobe: PathBuf,
    timeout: Duration,
}

impl MediaToolchain {
    pub fn new(
        ffmpeg: PathBuf,
        ffprobe: PathBuf,
        timeout: Duration,
    ) -> Result<Self, crate::MediaError> {
        if !ffmpeg.is_absolute() {
            return Err(crate::MediaError::InvalidToolchain {
                field: "ffmpeg",
                message: "path must be absolute".to_owned(),
            });
        }
        if ffmpeg.as_os_str().is_empty() {
            return Err(crate::MediaError::InvalidToolchain {
                field: "ffmpeg",
                message: "path must not be empty".to_owned(),
            });
        }
        if !ffprobe.is_absolute() {
            return Err(crate::MediaError::InvalidToolchain {
                field: "ffprobe",
                message: "path must be absolute".to_owned(),
            });
        }
        if ffprobe.as_os_str().is_empty() {
            return Err(crate::MediaError::InvalidToolchain {
                field: "ffprobe",
                message: "path must not be empty".to_owned(),
            });
        }
        if timeout.is_zero() || timeout > Duration::from_secs(24 * 60 * 60) {
            return Err(crate::MediaError::InvalidToolchain {
                field: "timeout",
                message: "must be greater than zero and no more than 24 hours".to_owned(),
            });
        }
        Ok(Self {
            ffmpeg,
            ffprobe,
            timeout,
        })
    }

    pub fn ffmpeg(&self) -> &Path {
        &self.ffmpeg
    }

    pub fn ffprobe(&self) -> &Path {
        &self.ffprobe
    }

    pub fn timeout(&self) -> Duration {
        self.timeout
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameRate {
    numerator: u32,
    denominator: u32,
}

impl FrameRate {
    pub fn new(numerator: u32, denominator: u32) -> Result<Self, crate::MediaError> {
        if numerator == 0 || denominator == 0 {
            return Err(crate::MediaError::InvalidToolchain {
                field: "frame_rate",
                message: "numerator and denominator must be non-zero".to_owned(),
            });
        }
        Ok(Self {
            numerator,
            denominator,
        })
    }

    pub fn numerator(self) -> u32 {
        self.numerator
    }

    pub fn denominator(self) -> u32 {
        self.denominator
    }

    pub fn frames_per_second(self) -> f64 {
        f64::from(self.numerator) / f64::from(self.denominator)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ProbeFormat {
    format_name: String,
    duration_seconds: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct VideoMetadata {
    codec_name: String,
    pixel_format: String,
    width: u32,
    height: u32,
    frame_rate: FrameRate,
    frame_count: u64,
    duration_seconds: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AudioMetadata {
    codec_name: String,
    sample_format: String,
    sample_rate: u32,
    channels: u16,
    sample_count: u64,
    duration_seconds: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MediaProbe {
    format: ProbeFormat,
    video: Option<VideoMetadata>,
    audio: Option<AudioMetadata>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MediaArtifact {
    bytes: u64,
    sha256: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct NormalizedMedia {
    layout: NormalizedMediaLayout,
    source: MediaProbe,
    video: Option<VideoMetadata>,
    audio: Option<AudioMetadata>,
    video_artifact: MediaArtifact,
    audio_artifact: MediaArtifact,
}

impl MediaProbe {
    pub(crate) fn new(
        format: ProbeFormat,
        video: Option<VideoMetadata>,
        audio: Option<AudioMetadata>,
    ) -> Self {
        Self {
            format,
            video,
            audio,
        }
    }

    pub fn format(&self) -> &ProbeFormat {
        &self.format
    }

    pub fn video(&self) -> Option<&VideoMetadata> {
        self.video.as_ref()
    }

    pub fn audio(&self) -> Option<&AudioMetadata> {
        self.audio.as_ref()
    }
}

impl ProbeFormat {
    pub(crate) fn new(format_name: String, duration_seconds: f64) -> Self {
        Self {
            format_name,
            duration_seconds,
        }
    }

    pub fn format_name(&self) -> &str {
        &self.format_name
    }

    pub fn duration_seconds(&self) -> f64 {
        self.duration_seconds
    }
}

impl VideoMetadata {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        codec_name: String,
        pixel_format: String,
        width: u32,
        height: u32,
        frame_rate: FrameRate,
        frame_count: u64,
        duration_seconds: f64,
    ) -> Self {
        Self {
            codec_name,
            pixel_format,
            width,
            height,
            frame_rate,
            frame_count,
            duration_seconds,
        }
    }

    pub fn codec_name(&self) -> &str {
        &self.codec_name
    }
    pub fn pixel_format(&self) -> &str {
        &self.pixel_format
    }
    pub fn width(&self) -> u32 {
        self.width
    }
    pub fn height(&self) -> u32 {
        self.height
    }
    pub fn frame_rate(&self) -> FrameRate {
        self.frame_rate
    }
    pub fn frame_count(&self) -> u64 {
        self.frame_count
    }
    pub fn duration_seconds(&self) -> f64 {
        self.duration_seconds
    }
}

impl AudioMetadata {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        codec_name: String,
        sample_format: String,
        sample_rate: u32,
        channels: u16,
        sample_count: u64,
        duration_seconds: f64,
    ) -> Self {
        Self {
            codec_name,
            sample_format,
            sample_rate,
            channels,
            sample_count,
            duration_seconds,
        }
    }

    pub fn codec_name(&self) -> &str {
        &self.codec_name
    }
    pub fn sample_format(&self) -> &str {
        &self.sample_format
    }
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }
    pub fn channels(&self) -> u16 {
        self.channels
    }
    pub fn sample_count(&self) -> u64 {
        self.sample_count
    }
    pub fn duration_seconds(&self) -> f64 {
        self.duration_seconds
    }
}

impl MediaArtifact {
    pub(crate) fn new(bytes: u64, sha256: String) -> Self {
        Self { bytes, sha256 }
    }
    pub fn bytes(&self) -> u64 {
        self.bytes
    }
    pub fn sha256(&self) -> &str {
        &self.sha256
    }
}

impl NormalizedMedia {
    pub(crate) fn new(
        layout: NormalizedMediaLayout,
        source: MediaProbe,
        video: Option<VideoMetadata>,
        audio: Option<AudioMetadata>,
        video_artifact: MediaArtifact,
        audio_artifact: MediaArtifact,
    ) -> Self {
        Self {
            layout,
            source,
            video,
            audio,
            video_artifact,
            audio_artifact,
        }
    }

    pub fn layout(&self) -> &NormalizedMediaLayout {
        &self.layout
    }
    pub fn source(&self) -> &MediaProbe {
        &self.source
    }
    pub fn video(&self) -> Option<&VideoMetadata> {
        self.video.as_ref()
    }
    pub fn audio(&self) -> Option<&AudioMetadata> {
        self.audio.as_ref()
    }
    pub fn video_artifact(&self) -> &MediaArtifact {
        &self.video_artifact
    }
    pub fn audio_artifact(&self) -> &MediaArtifact {
        &self.audio_artifact
    }
}

impl NormalizedMediaLayout {
    pub(crate) fn new(output_dir: PathBuf, video_path: PathBuf, audio_path: PathBuf) -> Self {
        Self {
            output_dir,
            video_path,
            audio_path,
        }
    }

    pub fn output_dir(&self) -> &Path {
        &self.output_dir
    }
    pub fn video_path(&self) -> &Path {
        &self.video_path
    }
    pub fn audio_path(&self) -> &Path {
        &self.audio_path
    }
}
