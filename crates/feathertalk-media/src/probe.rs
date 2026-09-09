use serde::Deserialize;

use crate::{AudioMetadata, FrameRate, MediaError, MediaProbe, ProbeFormat, VideoMetadata};

const MAX_PROBE_BYTES: usize = 1024 * 1024;
const MAX_DURATION_SECONDS: f64 = 24.0 * 60.0 * 60.0;
const MAX_DIMENSION: u32 = 16_384;
const MAX_COUNT: u64 = 100_000_000_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum StreamExpectation {
    AudioVideo,
    VideoOnly,
    AudioOnly,
}

#[derive(Debug, Deserialize)]
struct RawProbe {
    #[serde(default)]
    format: RawFormat,
    #[serde(default)]
    streams: Vec<RawStream>,
}

#[derive(Debug, Default, Deserialize)]
struct RawFormat {
    format_name: Option<String>,
    duration: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
struct RawStream {
    codec_type: Option<String>,
    codec_name: Option<String>,
    pix_fmt: Option<String>,
    width: Option<serde_json::Value>,
    height: Option<serde_json::Value>,
    avg_frame_rate: Option<String>,
    r_frame_rate: Option<String>,
    nb_read_frames: Option<serde_json::Value>,
    duration: Option<serde_json::Value>,
    sample_fmt: Option<String>,
    sample_rate: Option<serde_json::Value>,
    channels: Option<serde_json::Value>,
}

pub fn parse_probe_json(bytes: &[u8]) -> Result<MediaProbe, MediaError> {
    parse_probe_json_for(bytes, StreamExpectation::AudioVideo)
}

pub(crate) fn parse_probe_json_for(
    bytes: &[u8],
    expectation: StreamExpectation,
) -> Result<MediaProbe, MediaError> {
    if bytes.len() > MAX_PROBE_BYTES {
        return Err(MediaError::ProbeTooLarge {
            limit: MAX_PROBE_BYTES,
            actual: bytes.len(),
        });
    }
    let raw: RawProbe = serde_json::from_slice(bytes).map_err(|error| MediaError::ProbeJson {
        message: error.to_string(),
    })?;
    let format_name = require_name("format.format_name", raw.format.format_name.as_deref())?;
    let format_duration = parse_duration("format.duration", raw.format.duration.as_ref())?;

    let mut video = Vec::new();
    let mut audio = Vec::new();
    for stream in &raw.streams {
        match stream.codec_type.as_deref() {
            Some("video") => video.push(stream),
            Some("audio") => audio.push(stream),
            _ => {}
        }
    }
    require_count(
        "video",
        video.len(),
        expectation != StreamExpectation::AudioOnly,
    )?;
    require_count(
        "audio",
        audio.len(),
        expectation != StreamExpectation::VideoOnly,
    )?;

    let video = video
        .first()
        .map(|stream| parse_video(stream, format_duration))
        .transpose()?;
    let audio = audio
        .first()
        .map(|stream| parse_audio(stream, format_duration))
        .transpose()?;
    Ok(MediaProbe::new(
        ProbeFormat::new(format_name, format_duration),
        video,
        audio,
    ))
}

fn require_count(stream: &'static str, count: usize, required: bool) -> Result<(), MediaError> {
    if count > 1 {
        Err(MediaError::DuplicateStream { stream })
    } else if required && count == 0 {
        Err(MediaError::MissingStream { stream })
    } else if !required && count != 0 {
        Err(MediaError::ProbeContract {
            field: format!("streams.{stream}"),
            message: "stream must be absent".to_owned(),
        })
    } else {
        Ok(())
    }
}

fn parse_video(stream: &RawStream, fallback_duration: f64) -> Result<VideoMetadata, MediaError> {
    let codec = require_name("video.codec_name", stream.codec_name.as_deref())?;
    let pixel = require_name("video.pix_fmt", stream.pix_fmt.as_deref())?;
    let width = parse_u64("video.width", stream.width.as_ref())?;
    let height = parse_u64("video.height", stream.height.as_ref())?;
    let width = u32::try_from(width).map_err(|_| contract("video.width", "exceeds u32"))?;
    let height = u32::try_from(height).map_err(|_| contract("video.height", "exceeds u32"))?;
    if width == 0 || height == 0 || width > MAX_DIMENSION || height > MAX_DIMENSION {
        return Err(contract("video.dimensions", "must be within 1..=16384"));
    }
    let rate = parse_rate(
        stream
            .avg_frame_rate
            .as_deref()
            .filter(|value| *value != "0/0")
            .or(stream.r_frame_rate.as_deref()),
    )?;
    let duration = match stream.duration.as_ref() {
        Some(value) => parse_duration("video.duration", Some(value))?,
        None => fallback_duration,
    };
    let frame_count = match stream.nb_read_frames.as_ref() {
        Some(value) if value != "N/A" => parse_count("video.nb_read_frames", value)?,
        _ => checked_rounded_count("video.frame_count", duration * rate.frames_per_second())?,
    };
    Ok(VideoMetadata::new(
        codec,
        pixel,
        width,
        height,
        rate,
        frame_count,
        duration,
    ))
}

fn parse_audio(stream: &RawStream, fallback_duration: f64) -> Result<AudioMetadata, MediaError> {
    let codec = require_name("audio.codec_name", stream.codec_name.as_deref())?;
    let sample_format = require_name("audio.sample_fmt", stream.sample_fmt.as_deref())?;
    let sample_rate = parse_u64("audio.sample_rate", stream.sample_rate.as_ref())?;
    let sample_rate =
        u32::try_from(sample_rate).map_err(|_| contract("audio.sample_rate", "exceeds u32"))?;
    if sample_rate == 0 {
        return Err(contract("audio.sample_rate", "must be non-zero"));
    }
    let channels = parse_u64("audio.channels", stream.channels.as_ref())?;
    let channels =
        u16::try_from(channels).map_err(|_| contract("audio.channels", "exceeds u16"))?;
    if channels == 0 {
        return Err(contract("audio.channels", "must be non-zero"));
    }
    let duration = match stream.duration.as_ref() {
        Some(value) => parse_duration("audio.duration", Some(value))?,
        None => fallback_duration,
    };
    let sample_count =
        checked_rounded_count("audio.sample_count", duration * f64::from(sample_rate))?;
    Ok(AudioMetadata::new(
        codec,
        sample_format,
        sample_rate,
        channels,
        sample_count,
        duration,
    ))
}

fn parse_rate(value: Option<&str>) -> Result<FrameRate, MediaError> {
    let value = value.ok_or_else(|| contract("video.frame_rate", "missing"))?;
    let (numerator, denominator) = value
        .split_once('/')
        .ok_or_else(|| contract("video.frame_rate", "must be numerator/denominator"))?;
    let numerator = numerator
        .parse::<u32>()
        .map_err(|_| contract("video.frame_rate", "invalid numerator"))?;
    let denominator = denominator
        .parse::<u32>()
        .map_err(|_| contract("video.frame_rate", "invalid denominator"))?;
    FrameRate::new(numerator, denominator)
        .map_err(|_| contract("video.frame_rate", "zero component"))
}

fn parse_duration(field: &str, value: Option<&serde_json::Value>) -> Result<f64, MediaError> {
    let duration = parse_f64(field, value)?;
    if !duration.is_finite() || duration <= 0.0 || duration > MAX_DURATION_SECONDS {
        Err(contract(
            field,
            "must be finite and within 0..=86400 seconds",
        ))
    } else {
        Ok(duration)
    }
}

fn parse_f64(field: &str, value: Option<&serde_json::Value>) -> Result<f64, MediaError> {
    match value {
        Some(serde_json::Value::String(value)) => value
            .parse::<f64>()
            .map_err(|_| contract(field, "invalid number")),
        Some(serde_json::Value::Number(value)) => value
            .as_f64()
            .ok_or_else(|| contract(field, "invalid number")),
        _ => Err(contract(field, "missing number")),
    }
}

fn parse_u64(field: &str, value: Option<&serde_json::Value>) -> Result<u64, MediaError> {
    match value {
        Some(serde_json::Value::String(value)) => value
            .parse::<u64>()
            .map_err(|_| contract(field, "invalid unsigned integer")),
        Some(serde_json::Value::Number(value)) => value
            .as_u64()
            .ok_or_else(|| contract(field, "invalid unsigned integer")),
        _ => Err(contract(field, "missing unsigned integer")),
    }
}

fn parse_count(field: &str, value: &serde_json::Value) -> Result<u64, MediaError> {
    let count = parse_u64(field, Some(value))?;
    if count == 0 || count > MAX_COUNT {
        Err(contract(field, "must be within 1..=100000000000"))
    } else {
        Ok(count)
    }
}

fn checked_rounded_count(field: &str, value: f64) -> Result<u64, MediaError> {
    if !value.is_finite() || value < 0.5 || value > MAX_COUNT as f64 + 0.499_999 {
        return Err(contract(field, "derived count exceeds limits"));
    }
    let count = value.round() as u64;
    if count == 0 || count > MAX_COUNT {
        Err(contract(field, "must be within 1..=100000000000"))
    } else {
        Ok(count)
    }
}

fn require_name(field: &str, value: Option<&str>) -> Result<String, MediaError> {
    let value = value.ok_or_else(|| contract(field, "missing"))?;
    if value.is_empty()
        || value.trim() != value
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-' | b',' | b'.'))
    {
        Err(contract(
            field,
            "must be a non-empty ASCII metadata identifier",
        ))
    } else {
        Ok(value.to_owned())
    }
}

fn contract(field: impl Into<String>, message: impl Into<String>) -> MediaError {
    MediaError::ProbeContract {
        field: field.into(),
        message: message.into(),
    }
}
