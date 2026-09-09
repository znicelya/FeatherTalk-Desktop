use std::{
    fs::{self, File, OpenOptions},
    io::Read,
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};

use sha2::{Digest, Sha256};

use crate::{
    AudioMetadata, CommandSpec, MediaArtifact, MediaError, MediaProbe, MediaToolchain,
    NormalizationSpec, NormalizedMedia, ProcessOutput, ProcessRunner, SystemProcessRunner,
    VideoMetadata, audio_normalization_command,
    commands::video_normalization_command,
    commit::{SystemFileOps, commit_output_pair},
    execution::probe_media_with_runner,
    probe::{StreamExpectation, parse_probe_json_for},
    validate_normalization,
};

static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);
const MAX_OUTPUT_BYTES: u64 = 2 * 1024 * 1024 * 1024;
/// The largest video/audio duration difference the two normalized outputs may
/// carry.
///
/// The bound guards against a truncated or mis-mapped stream, not against
/// imperfectly aligned sources. A container reports a video duration that is a
/// whole number of frame periods (40 ms at 25 fps) and an audio duration that is
/// a whole number of codec frames (1024 samples for AAC, which is 128 ms at
/// 8 kHz), so two streams covering the same recording routinely differ by more
/// than a hundred milliseconds: this repository's own demo clip probes 60.44 s
/// of video against 60.48 s of audio, and normalization preserves both. What
/// this check exists to catch -- one output truncated while the other is whole
/// -- is short by seconds, far outside the bound.
///
/// This is not the product's 20 ms audio/video *sync* budget. Sync is set by the
/// two streams sharing a start time and by neither being resampled in time,
/// which the per-stream contracts above already pin down; a tail that runs a
/// frame longer than its partner shifts nothing.
const MAX_DURATION_DELTA_SECONDS: f64 = 0.200;

/// The phases of one normalization, in the order they run.
///
/// Reported to an observer immediately *before* each phase starts, so a caller
/// that displays the phase describes what is running. The observer is an output
/// channel only: it cannot fail and cannot stop the pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormalizePhase {
    Probing,
    NormalizingVideo,
    NormalizingAudio,
    Verifying,
    Committing,
}

pub fn normalize_media_observed<R: ProcessRunner + ?Sized>(
    input: &crate::ValidatedInput,
    spec: &NormalizationSpec,
    toolchain: &MediaToolchain,
    runner: &R,
    observer: &dyn Fn(NormalizePhase),
) -> Result<NormalizedMedia, MediaError> {
    // Layout validation runs before the first phase is announced: a spec that
    // names an unusable output directory never claims work it did not start.
    let layout = validate_normalization(input, spec)?;

    observer(NormalizePhase::Probing);
    let source = probe_media_with_runner(input, toolchain, runner)?;
    require_source_streams(&source)?;

    let mut video_temp = TempOutput::create(layout.output_dir(), "video", "mp4")?;
    let mut audio_temp = TempOutput::create(layout.output_dir(), "audio", "wav")?;

    observer(NormalizePhase::NormalizingVideo);
    run_tool(
        runner,
        &video_normalization_command(toolchain, input.source(), video_temp.path()),
        toolchain,
    )?;

    observer(NormalizePhase::NormalizingAudio);
    run_tool(
        runner,
        &audio_normalization_command(toolchain, input.source(), audio_temp.path()),
        toolchain,
    )?;

    observer(NormalizePhase::Verifying);
    let video = verify_video_output(video_temp.path(), toolchain, runner)?;
    let audio = verify_audio_output(audio_temp.path(), toolchain, runner)?;
    let delta = (video.duration_seconds() - audio.duration_seconds()).abs();
    if delta > MAX_DURATION_DELTA_SECONDS {
        return Err(MediaError::NormalizationVerificationFailed {
            field: "duration_delta",
            expected: format!("<= {MAX_DURATION_DELTA_SECONDS:.3} seconds"),
            actual: format!("{delta:.6} seconds"),
        });
    }

    observer(NormalizePhase::Committing);
    let video_artifact = hash_file(video_temp.path())?;
    let audio_artifact = hash_file(audio_temp.path())?;
    commit_output_pair(
        video_temp.path(),
        audio_temp.path(),
        layout.video_path(),
        layout.audio_path(),
        &SystemFileOps,
    )?;
    video_temp.disarm();
    audio_temp.disarm();
    Ok(NormalizedMedia::new(
        layout,
        source,
        Some(video),
        Some(audio),
        video_artifact,
        audio_artifact,
    ))
}

pub fn normalize_media_with_runner<R: ProcessRunner + ?Sized>(
    input: &crate::ValidatedInput,
    spec: &NormalizationSpec,
    toolchain: &MediaToolchain,
    runner: &R,
) -> Result<NormalizedMedia, MediaError> {
    normalize_media_observed(input, spec, toolchain, runner, &|_phase| {})
}

pub fn normalize_media(
    input: &crate::ValidatedInput,
    spec: &NormalizationSpec,
    toolchain: &MediaToolchain,
) -> Result<NormalizedMedia, MediaError> {
    normalize_media_with_runner(input, spec, toolchain, &SystemProcessRunner)
}

fn require_source_streams(probe: &MediaProbe) -> Result<(), MediaError> {
    if probe.video().is_none() {
        return Err(MediaError::MissingStream { stream: "video" });
    }
    if probe.audio().is_none() {
        return Err(MediaError::MissingStream { stream: "audio" });
    }
    Ok(())
}

fn run_tool<R: ProcessRunner + ?Sized>(
    runner: &R,
    command: &CommandSpec,
    toolchain: &MediaToolchain,
) -> Result<(), MediaError> {
    let output = runner.run(command, toolchain.timeout())?;
    ensure_capture_bounds(&output, command.operation())?;
    ensure_success(&output, command.operation())
}

fn verify_video_output<R: ProcessRunner + ?Sized>(
    path: &Path,
    toolchain: &MediaToolchain,
    runner: &R,
) -> Result<VideoMetadata, MediaError> {
    require_output_file(path)?;
    let output = runner.run(
        &crate::commands::probe_command(toolchain, path),
        toolchain.timeout(),
    )?;
    ensure_capture_bounds(&output, "probe")?;
    ensure_success(&output, "probe")?;
    let probe = parse_probe_json_for(output.stdout(), StreamExpectation::VideoOnly)?;
    if !format_has_name(probe.format().format_name(), "mp4") {
        return Err(verification(
            "video.format_name",
            "format list containing mp4",
            probe.format().format_name(),
        ));
    }
    let video = probe
        .video()
        .ok_or(MediaError::MissingStream { stream: "video" })?;
    if video.codec_name() != "mpeg4" {
        return Err(verification(
            "video.codec_name",
            "mpeg4",
            video.codec_name(),
        ));
    }
    if video.pixel_format() != "yuv420p" {
        return Err(verification(
            "video.pixel_format",
            "yuv420p",
            video.pixel_format(),
        ));
    }
    if video.frame_rate().numerator() != 25 || video.frame_rate().denominator() != 1 {
        return Err(verification(
            "video.frame_rate",
            "25/1",
            &format!(
                "{}/{}",
                video.frame_rate().numerator(),
                video.frame_rate().denominator()
            ),
        ));
    }
    Ok(video.clone())
}

fn verify_audio_output<R: ProcessRunner + ?Sized>(
    path: &Path,
    toolchain: &MediaToolchain,
    runner: &R,
) -> Result<AudioMetadata, MediaError> {
    require_output_file(path)?;
    let output = runner.run(
        &crate::commands::probe_command(toolchain, path),
        toolchain.timeout(),
    )?;
    ensure_capture_bounds(&output, "probe")?;
    ensure_success(&output, "probe")?;
    let probe = parse_probe_json_for(output.stdout(), StreamExpectation::AudioOnly)?;
    if !format_has_name(probe.format().format_name(), "wav") {
        return Err(verification(
            "audio.format_name",
            "format list containing wav",
            probe.format().format_name(),
        ));
    }
    let audio = probe
        .audio()
        .ok_or(MediaError::MissingStream { stream: "audio" })?;
    if audio.codec_name() != "pcm_s16le" {
        return Err(verification(
            "audio.codec_name",
            "pcm_s16le",
            audio.codec_name(),
        ));
    }
    if audio.sample_format() != "s16" {
        return Err(verification(
            "audio.sample_fmt",
            "s16",
            audio.sample_format(),
        ));
    }
    if audio.sample_rate() != 16_000 {
        return Err(verification(
            "audio.sample_rate",
            "16000",
            &audio.sample_rate().to_string(),
        ));
    }
    if audio.channels() != 1 {
        return Err(verification(
            "audio.channels",
            "1",
            &audio.channels().to_string(),
        ));
    }
    Ok(audio.clone())
}

fn ensure_success(output: &ProcessOutput, operation: &'static str) -> Result<(), MediaError> {
    if output.exit_code() != Some(0) {
        return Err(MediaError::ToolFailed {
            operation,
            exit_code: output.exit_code(),
            stderr: String::from_utf8_lossy(output.stderr()).into_owned(),
        });
    }
    Ok(())
}

fn ensure_capture_bounds(
    output: &ProcessOutput,
    operation: &'static str,
) -> Result<(), MediaError> {
    if output.stdout().len() > crate::MAX_CAPTURE_BYTES {
        return Err(MediaError::ToolOutputTooLarge {
            operation,
            stream: "stdout",
            limit: crate::MAX_CAPTURE_BYTES,
            actual: output.stdout().len(),
        });
    }
    if output.stderr().len() > crate::MAX_CAPTURE_BYTES {
        return Err(MediaError::ToolOutputTooLarge {
            operation,
            stream: "stderr",
            limit: crate::MAX_CAPTURE_BYTES,
            actual: output.stderr().len(),
        });
    }
    Ok(())
}

fn require_output_file(path: &Path) -> Result<(), MediaError> {
    let metadata = match fs::symlink_metadata(path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            return Err(MediaError::NormalizationVerificationFailed {
                field: "output_file",
                expected: "regular non-empty file".to_owned(),
                actual: format!("missing: {}", path.display()),
            });
        }
        Err(source) => {
            return Err(MediaError::Io {
                operation: "stat_normalized_output",
                path: path.to_owned(),
                source,
            });
        }
    };
    if metadata.file_type().is_symlink() || !metadata.is_file() || metadata.len() == 0 {
        return Err(MediaError::NormalizationVerificationFailed {
            field: "output_file",
            expected: "regular non-empty file".to_owned(),
            actual: path.display().to_string(),
        });
    }
    if metadata.len() > MAX_OUTPUT_BYTES {
        return Err(MediaError::NormalizationVerificationFailed {
            field: "output_file_size",
            expected: format!("<= {MAX_OUTPUT_BYTES} bytes"),
            actual: metadata.len().to_string(),
        });
    }
    OpenOptions::new()
        .read(true)
        .write(true)
        .open(path)
        .and_then(|file| file.sync_all())
        .map_err(|source| MediaError::Io {
            operation: "sync_normalized_output",
            path: path.to_owned(),
            source,
        })?;
    Ok(())
}

fn hash_file(path: &Path) -> Result<MediaArtifact, MediaError> {
    let mut file = File::open(path).map_err(|source| MediaError::Io {
        operation: "hash_normalized_output",
        path: path.to_owned(),
        source,
    })?;
    let mut digest = Sha256::new();
    let mut bytes = 0_u64;
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let read = file.read(&mut buffer).map_err(|source| MediaError::Io {
            operation: "hash_normalized_output",
            path: path.to_owned(),
            source,
        })?;
        if read == 0 {
            break;
        }
        bytes = bytes.checked_add(read as u64).ok_or_else(|| {
            MediaError::NormalizationVerificationFailed {
                field: "output_file_size",
                expected: "non-overflowing byte count".to_owned(),
                actual: "overflow".to_owned(),
            }
        })?;
        digest.update(&buffer[..read]);
    }
    Ok(MediaArtifact::new(bytes, hex::encode(digest.finalize())))
}

fn verification(field: &'static str, expected: &str, actual: &str) -> MediaError {
    MediaError::NormalizationVerificationFailed {
        field,
        expected: expected.to_owned(),
        actual: actual.to_owned(),
    }
}

fn format_has_name(format_names: &str, expected: &str) -> bool {
    format_names
        .split(',')
        .any(|format_name| format_name.trim() == expected)
}

struct TempOutput {
    path: PathBuf,
    armed: bool,
}

pub(crate) fn next_temp_id() -> u64 {
    TEMP_COUNTER.fetch_add(1, Ordering::Relaxed)
}

impl TempOutput {
    fn create(dir: &Path, stem: &str, extension: &str) -> Result<Self, MediaError> {
        for _ in 0..32 {
            let id = next_temp_id();
            let path = dir.join(format!(
                ".feathertalk-{stem}-{}-{id}.tmp.{extension}",
                std::process::id()
            ));
            match OpenOptions::new()
                .write(true)
                .read(true)
                .create_new(true)
                .open(&path)
            {
                Ok(_) => return Ok(Self { path, armed: true }),
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(source) => {
                    return Err(MediaError::Io {
                        operation: "create_normalization_temp",
                        path,
                        source,
                    });
                }
            }
        }
        Err(MediaError::OutputCommitFailed {
            operation: "create_temp",
            message: "unable to allocate unique temp path".to_owned(),
        })
    }

    fn path(&self) -> &Path {
        &self.path
    }

    fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for TempOutput {
    fn drop(&mut self) {
        if self.armed {
            let _ = fs::remove_file(&self.path);
        }
    }
}
