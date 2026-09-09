use std::{collections::VecDeque, sync::Mutex, time::Duration};

use feathertalk_media::{
    CommandSpec, MediaError, MediaInput, MediaToolchain, ProcessOutput, ProcessRunner,
    probe_media_with_runner, probe_video_with_runner, validate_input,
};

struct FakeRunner {
    outputs: Mutex<VecDeque<Result<ProcessOutput, MediaError>>>,
    commands: Mutex<Vec<CommandSpec>>,
}

impl FakeRunner {
    fn new(outputs: Vec<Result<ProcessOutput, MediaError>>) -> Self {
        Self {
            outputs: Mutex::new(outputs.into()),
            commands: Mutex::new(Vec::new()),
        }
    }
}

impl ProcessRunner for FakeRunner {
    fn run(&self, command: &CommandSpec, _timeout: Duration) -> Result<ProcessOutput, MediaError> {
        self.commands.lock().unwrap().push(command.clone());
        self.outputs.lock().unwrap().pop_front().unwrap()
    }
}

fn toolchain() -> MediaToolchain {
    let root = std::env::current_dir().unwrap();
    MediaToolchain::new(
        root.join("ffmpeg-test"),
        root.join("ffprobe-test"),
        Duration::from_secs(10),
    )
    .unwrap()
}

fn input() -> (tempfile::TempDir, feathertalk_media::ValidatedInput) {
    let temp = tempfile::tempdir().unwrap();
    let source = temp.path().join("input.mov");
    std::fs::write(&source, b"media").unwrap();
    let input = validate_input(&MediaInput { source }).unwrap();
    (temp, input)
}

fn valid_probe() -> Vec<u8> {
    br#"{
      "format":{"format_name":"mov,mp4","duration":"2.0"},
      "streams":[
        {"codec_type":"video","codec_name":"h264","pix_fmt":"yuv420p","width":640,"height":480,"avg_frame_rate":"25/1","nb_read_frames":"50","duration":"2.0"},
        {"codec_type":"audio","codec_name":"aac","sample_fmt":"fltp","sample_rate":"48000","channels":2,"duration":"2.0"}
      ]
    }"#
    .to_vec()
}

/// What `normalize_media` writes as `video_25fps.mp4`: one video stream and no
/// audio, which is also the shape `normalize_media` verifies its own output
/// against. This is the report `extract_frames` has to be able to read.
fn video_only_probe() -> Vec<u8> {
    br#"{
      "format":{"format_name":"mov,mp4","duration":"1.0"},
      "streams":[
        {"codec_type":"video","codec_name":"h264","pix_fmt":"yuv420p","width":1280,"height":720,"avg_frame_rate":"25/1","nb_read_frames":"25","duration":"1.0"}
      ]
    }"#
    .to_vec()
}

#[test]
fn successful_probe_runs_the_fixed_command_once() {
    let (_temp, input) = input();
    let runner = FakeRunner::new(vec![Ok(ProcessOutput::new(
        Some(0),
        valid_probe(),
        Vec::new(),
    ))]);
    let probe = probe_media_with_runner(&input, &toolchain(), &runner).unwrap();
    assert_eq!(probe.video().unwrap().frame_count(), 50);
    let commands = runner.commands.lock().unwrap();
    assert_eq!(commands.len(), 1);
    assert_eq!(commands[0].operation(), "probe");
    assert_eq!(commands[0].arguments().last().unwrap(), input.source());
}

#[test]
fn nonzero_exit_maps_to_bounded_tool_failure() {
    let (_temp, input) = input();
    let runner = FakeRunner::new(vec![Ok(ProcessOutput::new(
        Some(7),
        Vec::new(),
        b"invalid container".to_vec(),
    ))]);
    assert!(matches!(
        probe_media_with_runner(&input, &toolchain(), &runner),
        Err(MediaError::ToolFailed {
            operation: "probe",
            exit_code: Some(7),
            ..
        })
    ));
}

#[test]
fn injected_timeout_is_preserved() {
    let (_temp, input) = input();
    let runner = FakeRunner::new(vec![Err(MediaError::ToolTimedOut {
        operation: "probe",
        timeout_ms: 10_000,
    })]);
    assert!(matches!(
        probe_media_with_runner(&input, &toolchain(), &runner),
        Err(MediaError::ToolTimedOut {
            operation: "probe",
            ..
        })
    ));
}

#[test]
fn injected_output_cannot_bypass_capture_limit() {
    let (_temp, input) = input();
    let runner = FakeRunner::new(vec![Ok(ProcessOutput::new(
        Some(0),
        vec![b'x'; 1_048_577],
        Vec::new(),
    ))]);
    assert!(matches!(
        probe_media_with_runner(&input, &toolchain(), &runner),
        Err(MediaError::ToolOutputTooLarge {
            operation: "probe",
            stream: "stdout",
            limit: 1_048_576,
            ..
        })
    ));
}

#[test]
fn a_cancelled_probe_surfaces_as_tool_cancelled() {
    let toolchain = toolchain();
    let (_temp, input) = input();
    let runner = FakeRunner::new(vec![Err(MediaError::ToolCancelled { operation: "probe" })]);
    let error = probe_media_with_runner(&input, &toolchain, &runner)
        .expect_err("a cancelled probe must not report success");
    assert!(
        matches!(error, MediaError::ToolCancelled { operation: "probe" }),
        "{error:?}"
    );
}

#[test]
fn the_video_probe_accepts_a_report_without_audio() {
    let (_temp, input) = input();
    let runner = FakeRunner::new(vec![Ok(ProcessOutput::new(
        Some(0),
        video_only_probe(),
        Vec::new(),
    ))]);
    let probe = probe_video_with_runner(&input, &toolchain(), &runner).unwrap();
    assert_eq!(probe.video().unwrap().frame_count(), 25);
    assert!(probe.audio().is_none());
    // The audio/video probe cannot read that same report at all, which is the
    // whole reason this second entry point exists.
    let runner = FakeRunner::new(vec![Ok(ProcessOutput::new(
        Some(0),
        video_only_probe(),
        Vec::new(),
    ))]);
    assert!(matches!(
        probe_media_with_runner(&input, &toolchain(), &runner),
        Err(MediaError::MissingStream { stream: "audio" })
    ));
}

#[test]
fn the_video_probe_refuses_a_report_carrying_audio() {
    let (_temp, input) = input();
    let runner = FakeRunner::new(vec![Ok(ProcessOutput::new(
        Some(0),
        valid_probe(),
        Vec::new(),
    ))]);
    let error = probe_video_with_runner(&input, &toolchain(), &runner)
        .expect_err("a normalized video carries no audio stream");
    assert!(
        matches!(error, MediaError::ProbeContract { ref field, .. } if field == "streams.audio"),
        "{error:?}"
    );
}
