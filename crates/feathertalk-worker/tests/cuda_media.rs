//! Real FFmpeg decoding, including the libraries' software retry paths.
use std::{path::PathBuf, process::Command, sync::Mutex, time::Duration};

use feathertalk_frame_pipeline as frames;
use feathertalk_media as media;

#[derive(Default)]
struct RecordingRunner(Mutex<Vec<(&'static str, bool, Option<i32>)>>);

impl media::ProcessRunner for RecordingRunner {
    fn run(
        &self,
        command: &media::CommandSpec,
        timeout: Duration,
    ) -> Result<media::ProcessOutput, media::MediaError> {
        let output = media::ProcessRunner::run(&media::SystemProcessRunner, command, timeout)?;
        self.0.lock().unwrap().push((
            command.operation(),
            command.arguments().iter().any(|arg| arg == "-hwaccel"),
            output.exit_code(),
        ));
        Ok(output)
    }
}

impl frames::ProcessRunner for RecordingRunner {
    fn run(
        &self,
        command: &frames::CommandSpec,
        timeout: Duration,
    ) -> Result<frames::ProcessOutput, frames::PipelineError> {
        let output = frames::ProcessRunner::run(&frames::SystemProcessRunner, command, timeout)?;
        self.0.lock().unwrap().push((
            command.operation(),
            command.arguments().iter().any(|arg| arg == "-hwaccel"),
            output.exit_code(),
        ));
        Ok(output)
    }
}

impl RecordingRunner {
    fn assert_decoding(&self, operation: &str, fallback: bool) {
        let calls = self.0.lock().unwrap();
        let calls: Vec<_> = calls.iter().filter(|call| call.0 == operation).collect();
        assert_eq!(calls.len(), if fallback { 2 } else { 1 }, "{calls:?}");
        assert!(calls[0].1, "CUDA must be attempted first");
        if fallback {
            assert_ne!(calls[0].2, Some(0));
            assert!(!calls[1].1, "retry must use software decoding");
        }
        assert_eq!(calls.last().unwrap().2, Some(0));
    }
}

fn tool(variable: &str) -> PathBuf {
    let path = PathBuf::from(
        std::env::var_os(variable)
            .unwrap_or_else(|| panic!("set {variable} to the tool's absolute path")),
    );
    assert!(path.is_absolute() && path.is_file(), "invalid {variable}");
    path
}

#[test]
#[ignore = "requires NVIDIA CUDA decoding, FFmpeg with libx264, and FEATHERTALK_WORKER_FFMPEG/FFPROBE"]
fn cuda_decoding_and_software_retry_preserve_normalized_media_and_extracted_frames() {
    let root = tempfile::tempdir().unwrap();
    let ffmpeg = tool("FEATHERTALK_WORKER_FFMPEG");
    let ffprobe = tool("FEATHERTALK_WORKER_FFPROBE");
    let video = root.path().join("input.mp4");
    let generated = Command::new(&ffmpeg)
        .args([
            "-hide_banner",
            "-nostdin",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=160x160:rate=25:duration=1",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=16000:duration=1",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
        ])
        .arg(&video)
        .output()
        .unwrap();
    assert!(
        generated.status.success(),
        "{}",
        String::from_utf8_lossy(&generated.stderr)
    );
    let device = std::env::var("FEATHERTALK_TEST_CUDA_DEVICE")
        .map(|value| value.parse::<usize>().unwrap())
        .unwrap_or(0);
    for (ordinal, fallback) in [(device, false), (i32::MAX as usize, true)] {
        let directory = root.path().join(ordinal.to_string());
        std::fs::create_dir(&directory).unwrap();
        let tools =
            media::MediaToolchain::new(ffmpeg.clone(), ffprobe.clone(), Duration::from_secs(30))
                .unwrap()
                .with_cuda_device(Some(ordinal));
        let runner = RecordingRunner::default();
        let normalized = media::normalize_media_with_runner(
            &media::validate_input(&media::MediaInput {
                source: video.clone()})
            .unwrap(),
            &media::NormalizationSpec {
                target_video_fps: 25,
                target_audio_sample_rate: 16_000,
                target_audio_channels: 1,
                output_dir: directory.join("normalized")},
            &tools,
            &runner,
        )
        .unwrap();
        assert_eq!(normalized.video().unwrap().codec_name(), "mpeg4");
        assert_eq!(normalized.audio().unwrap().sample_rate(), 16_000);
        assert!(normalized.layout().video_path().is_file());
        assert!(normalized.layout().audio_path().is_file());
        runner.assert_decoding("normalize_video", fallback);

        let extractor = frames::FrameExtractor::new(ffmpeg.clone(), Duration::from_secs(30))
            .unwrap()
            .with_cuda_device(Some(ordinal));
        let batch = frames::extract_frames_with_runner(
            &frames::FramePipelineSpec::new(video.clone(), directory.join("frames"), 25, 160, 160)
                .unwrap(),
            &extractor,
            &runner,
        )
        .unwrap();
        assert_eq!(batch.frames().len(), 25);
        assert!(
            batch
                .frames()
                .iter()
                .all(|frame| frame.bytes() > 0 && frame.sha256().len() == 64)
        );
        runner.assert_decoding("extract_frames", fallback);
    }
}
