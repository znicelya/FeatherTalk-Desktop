use crate::{
    MediaError, MediaProbe, MediaToolchain, ProcessRunner, SystemProcessRunner,
    commands::probe_command,
    probe::{StreamExpectation, parse_probe_json_for},
    process::MAX_CAPTURE_BYTES,
};

/// Probe an input the pipeline has not normalized yet: one video stream and one
/// audio stream are both required.
pub fn probe_media_with_runner<R: ProcessRunner + ?Sized>(
    input: &crate::ValidatedInput,
    toolchain: &MediaToolchain,
    runner: &R,
) -> Result<MediaProbe, MediaError> {
    probe_with_expectation(input, toolchain, runner, StreamExpectation::AudioVideo)
}

/// Probe a normalized video: one video stream, no audio.
///
/// This is the shape `normalize_media` writes to `video_25fps.mp4` and the shape
/// it verifies that file against, so a consumer of that artifact -- frame
/// extraction -- has to ask for the same thing. `probe_media_with_runner` would
/// reject it for the audio stream normalization deliberately moved out.
pub fn probe_video_with_runner<R: ProcessRunner + ?Sized>(
    input: &crate::ValidatedInput,
    toolchain: &MediaToolchain,
    runner: &R,
) -> Result<MediaProbe, MediaError> {
    probe_with_expectation(input, toolchain, runner, StreamExpectation::VideoOnly)
}

fn probe_with_expectation<R: ProcessRunner + ?Sized>(
    input: &crate::ValidatedInput,
    toolchain: &MediaToolchain,
    runner: &R,
    expectation: StreamExpectation,
) -> Result<MediaProbe, MediaError> {
    let command = probe_command(toolchain, input.source());
    let output = runner.run(&command, toolchain.timeout())?;
    if output.stdout().len() > MAX_CAPTURE_BYTES {
        return Err(MediaError::ToolOutputTooLarge {
            operation: "probe",
            stream: "stdout",
            limit: MAX_CAPTURE_BYTES,
            actual: output.stdout().len(),
        });
    }
    if output.stderr().len() > MAX_CAPTURE_BYTES {
        return Err(MediaError::ToolOutputTooLarge {
            operation: "probe",
            stream: "stderr",
            limit: MAX_CAPTURE_BYTES,
            actual: output.stderr().len(),
        });
    }
    if output.exit_code() != Some(0) {
        return Err(MediaError::ToolFailed {
            operation: "probe",
            exit_code: output.exit_code(),
            stderr: String::from_utf8_lossy(output.stderr()).into_owned(),
        });
    }
    parse_probe_json_for(output.stdout(), expectation)
}

pub fn probe_media(
    input: &crate::ValidatedInput,
    toolchain: &MediaToolchain,
) -> Result<MediaProbe, MediaError> {
    probe_media_with_runner(input, toolchain, &SystemProcessRunner)
}
