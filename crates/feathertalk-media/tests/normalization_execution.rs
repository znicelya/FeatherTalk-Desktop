use std::{
    collections::VecDeque,
    fs::{self, OpenOptions},
    path::{Path, PathBuf},
    sync::Mutex,
    time::Duration,
};

use feathertalk_media::{
    CommandSpec, MediaError, MediaInput, MediaToolchain, NormalizationSpec, NormalizePhase,
    ProcessOutput, ProcessRunner, normalize_media_observed, normalize_media_with_runner,
    validate_input, validate_normalization,
};

struct FakeRunner {
    outputs: Mutex<VecDeque<Result<ProcessOutput, MediaError>>>,
    commands: Mutex<Vec<CommandSpec>>,
    staged_writes: Mutex<VecDeque<StagedWrite>>,
}

impl FakeRunner {
    fn new(outputs: Vec<Result<ProcessOutput, MediaError>>) -> Self {
        Self::with_staged_writes(
            outputs,
            vec![
                StagedWrite::Bytes(b"normalized-video"),
                StagedWrite::Bytes(b"normalized-audio"),
            ],
        )
    }

    fn with_staged_writes(
        outputs: Vec<Result<ProcessOutput, MediaError>>,
        staged_writes: Vec<StagedWrite>,
    ) -> Self {
        Self {
            outputs: Mutex::new(outputs.into()),
            commands: Mutex::new(Vec::new()),
            staged_writes: Mutex::new(staged_writes.into()),
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum StagedWrite {
    Bytes(&'static [u8]),
    Missing,
    Sparse(u64),
}

impl ProcessRunner for FakeRunner {
    fn run(&self, command: &CommandSpec, _timeout: Duration) -> Result<ProcessOutput, MediaError> {
        let mut commands = self.commands.lock().unwrap();
        commands.push(command.clone());
        let output = self.outputs.lock().unwrap().pop_front().unwrap()?;
        if matches!(command.operation(), "normalize_video" | "normalize_audio") {
            let path = Path::new(command.arguments().last().unwrap());
            match self.staged_writes.lock().unwrap().pop_front().unwrap() {
                StagedWrite::Bytes(bytes) => fs::write(path, bytes).unwrap(),
                StagedWrite::Missing => fs::remove_file(path).unwrap(),
                StagedWrite::Sparse(bytes) => OpenOptions::new()
                    .write(true)
                    .open(path)
                    .unwrap()
                    .set_len(bytes)
                    .unwrap(),
            }
        }
        Ok(output)
    }
}

fn tools(root: &Path) -> MediaToolchain {
    MediaToolchain::new(
        root.join("ffmpeg"),
        root.join("ffprobe"),
        Duration::from_secs(10),
    )
    .unwrap()
}

fn source_probe() -> Vec<u8> {
    br#"{"format":{"format_name":"mov,mp4","duration":"2.0"},"streams":[{"codec_type":"video","codec_name":"h264","pix_fmt":"yuv420p","width":640,"height":480,"avg_frame_rate":"25/1","nb_read_frames":"50","duration":"2.0"},{"codec_type":"audio","codec_name":"aac","sample_fmt":"fltp","sample_rate":"48000","channels":2,"duration":"2.0"}]}"#.to_vec()
}

fn video_probe() -> Vec<u8> {
    video_probe_with("mp4", "mpeg4", "yuv420p", "25/1", "2.0")
}

fn audio_probe() -> Vec<u8> {
    audio_probe_with("wav", "pcm_s16le", "s16", "16000", 1, "2.0")
}

fn video_probe_with(
    format_name: &str,
    codec_name: &str,
    pixel_format: &str,
    frame_rate: &str,
    duration: &str,
) -> Vec<u8> {
    format!(
        r#"{{"format":{{"format_name":"{format_name}","duration":"{duration}"}},"streams":[{{"codec_type":"video","codec_name":"{codec_name}","pix_fmt":"{pixel_format}","width":640,"height":480,"avg_frame_rate":"{frame_rate}","nb_read_frames":"50","duration":"{duration}"}}]}}"#
    )
    .into_bytes()
}

fn audio_probe_with(
    format_name: &str,
    codec_name: &str,
    sample_format: &str,
    sample_rate: &str,
    channels: u16,
    duration: &str,
) -> Vec<u8> {
    format!(
        r#"{{"format":{{"format_name":"{format_name}","duration":"{duration}"}},"streams":[{{"codec_type":"audio","codec_name":"{codec_name}","sample_fmt":"{sample_format}","sample_rate":"{sample_rate}","channels":{channels},"duration":"{duration}"}}]}}"#
    )
    .into_bytes()
}

fn setup() -> (
    tempfile::TempDir,
    feathertalk_media::ValidatedInput,
    NormalizationSpec,
) {
    let root = tempfile::tempdir().unwrap();
    let source = root.path().join("input.mov");
    fs::write(&source, b"source").unwrap();
    let input = validate_input(&MediaInput { source }).unwrap();
    let spec = NormalizationSpec {
        target_video_fps: 25,
        target_audio_sample_rate: 16_000,
        target_audio_channels: 1,
        output_dir: root.path().join("assets"),
    };
    (root, input, spec)
}

fn seed_old_outputs(layout: &feathertalk_media::NormalizedMediaLayout) {
    fs::write(layout.video_path(), b"old-video").unwrap();
    fs::write(layout.audio_path(), b"old-audio").unwrap();
}

fn assert_old_outputs_and_no_staging(
    layout: &feathertalk_media::NormalizedMediaLayout,
    root: &Path,
) {
    assert_eq!(fs::read(layout.video_path()).unwrap(), b"old-video");
    assert_eq!(fs::read(layout.audio_path()).unwrap(), b"old-audio");
    assert_eq!(staging_paths(root), Vec::<PathBuf>::new());
}

fn staging_paths(root: &Path) -> Vec<PathBuf> {
    fs::read_dir(root)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with(".feathertalk-"))
        })
        .collect()
}

#[test]
fn successful_normalization_verifies_outputs_and_hashes() {
    let (root, input, spec) = setup();
    let layout = validate_normalization(&input, &spec).unwrap();
    let runner = FakeRunner::new(vec![
        Ok(ProcessOutput::new(Some(0), source_probe(), Vec::new())),
        Ok(ProcessOutput::new(Some(0), Vec::new(), Vec::new())),
        Ok(ProcessOutput::new(Some(0), Vec::new(), Vec::new())),
        Ok(ProcessOutput::new(Some(0), video_probe(), Vec::new())),
        Ok(ProcessOutput::new(Some(0), audio_probe(), Vec::new())),
    ]);

    let result = normalize_media_with_runner(&input, &spec, &tools(root.path()), &runner).unwrap();
    assert_eq!(result.layout().video_path(), layout.video_path());
    assert_eq!(result.video().unwrap().codec_name(), "mpeg4");
    assert_eq!(result.audio().unwrap().sample_rate(), 16_000);
    assert_eq!(
        result.video_artifact().bytes(),
        b"normalized-video".len() as u64
    );
    assert_eq!(
        result.audio_artifact().bytes(),
        b"normalized-audio".len() as u64
    );
    assert_eq!(fs::read(layout.video_path()).unwrap(), b"normalized-video");
    assert_eq!(fs::read(layout.audio_path()).unwrap(), b"normalized-audio");
    assert_eq!(runner.commands.lock().unwrap().len(), 5);
}

#[test]
fn first_ffmpeg_failure_preserves_existing_outputs_and_cleans_staging() {
    let (root, input, spec) = setup();
    let layout = validate_normalization(&input, &spec).unwrap();
    seed_old_outputs(&layout);
    let runner = FakeRunner::new(vec![
        Ok(ProcessOutput::new(Some(0), source_probe(), Vec::new())),
        Ok(ProcessOutput::new(
            Some(1),
            Vec::new(),
            b"encode failed".to_vec(),
        )),
    ]);
    assert!(matches!(
        normalize_media_with_runner(&input, &spec, &tools(root.path()), &runner),
        Err(MediaError::ToolFailed {
            operation: "normalize_video",
            ..
        })
    ));
    assert_old_outputs_and_no_staging(&layout, layout.output_dir());
}

#[test]
fn second_ffmpeg_failure_preserves_existing_outputs_and_cleans_staging() {
    let (root, input, spec) = setup();
    let layout = validate_normalization(&input, &spec).unwrap();
    seed_old_outputs(&layout);
    let runner = FakeRunner::new(vec![
        Ok(ProcessOutput::new(Some(0), source_probe(), Vec::new())),
        Ok(ProcessOutput::new(Some(0), Vec::new(), Vec::new())),
        Ok(ProcessOutput::new(
            Some(1),
            Vec::new(),
            b"audio encode failed".to_vec(),
        )),
    ]);
    assert!(matches!(
        normalize_media_with_runner(&input, &spec, &tools(root.path()), &runner),
        Err(MediaError::ToolFailed {
            operation: "normalize_audio",
            ..
        })
    ));
    assert_old_outputs_and_no_staging(&layout, layout.output_dir());
}

#[test]
fn rejects_invalid_normalized_video_contracts_without_touching_destinations() {
    for (probe, expected_field) in [
        (
            video_probe_with("mp4", "h264", "yuv420p", "25/1", "2.0"),
            "video.codec_name",
        ),
        (
            video_probe_with("mp4", "mpeg4", "yuv444p", "25/1", "2.0"),
            "video.pixel_format",
        ),
        (
            video_probe_with("mp4", "mpeg4", "yuv420p", "30/1", "2.0"),
            "video.frame_rate",
        ),
        (
            video_probe_with("matroska", "mpeg4", "yuv420p", "25/1", "2.0"),
            "video.format_name",
        ),
    ] {
        let (root, input, spec) = setup();
        let layout = validate_normalization(&input, &spec).unwrap();
        seed_old_outputs(&layout);
        let runner = FakeRunner::new(vec![
            Ok(ProcessOutput::new(Some(0), source_probe(), Vec::new())),
            Ok(ProcessOutput::new(Some(0), Vec::new(), Vec::new())),
            Ok(ProcessOutput::new(Some(0), Vec::new(), Vec::new())),
            Ok(ProcessOutput::new(Some(0), probe, Vec::new())),
            Ok(ProcessOutput::new(Some(0), audio_probe(), Vec::new())),
        ]);

        assert!(matches!(
            normalize_media_with_runner(&input, &spec, &tools(root.path()), &runner),
            Err(MediaError::NormalizationVerificationFailed { field, .. })
                if field == expected_field
        ));
        assert_old_outputs_and_no_staging(&layout, layout.output_dir());
    }
}

#[test]
fn rejects_invalid_normalized_audio_contracts_without_touching_destinations() {
    for (probe, expected_field) in [
        (
            audio_probe_with("wav", "aac", "s16", "16000", 1, "2.0"),
            "audio.codec_name",
        ),
        (
            audio_probe_with("wav", "pcm_s16le", "fltp", "16000", 1, "2.0"),
            "audio.sample_fmt",
        ),
        (
            audio_probe_with("wav", "pcm_s16le", "s16", "48000", 1, "2.0"),
            "audio.sample_rate",
        ),
        (
            audio_probe_with("wav", "pcm_s16le", "s16", "16000", 2, "2.0"),
            "audio.channels",
        ),
        (
            audio_probe_with("aiff", "pcm_s16le", "s16", "16000", 1, "2.0"),
            "audio.format_name",
        ),
    ] {
        let (root, input, spec) = setup();
        let layout = validate_normalization(&input, &spec).unwrap();
        seed_old_outputs(&layout);
        let runner = FakeRunner::new(vec![
            Ok(ProcessOutput::new(Some(0), source_probe(), Vec::new())),
            Ok(ProcessOutput::new(Some(0), Vec::new(), Vec::new())),
            Ok(ProcessOutput::new(Some(0), Vec::new(), Vec::new())),
            Ok(ProcessOutput::new(Some(0), video_probe(), Vec::new())),
            Ok(ProcessOutput::new(Some(0), probe, Vec::new())),
        ]);

        assert!(matches!(
            normalize_media_with_runner(&input, &spec, &tools(root.path()), &runner),
            Err(MediaError::NormalizationVerificationFailed { field, .. })
                if field == expected_field
        ));
        assert_old_outputs_and_no_staging(&layout, layout.output_dir());
    }
}

#[test]
fn rejects_a_duration_delta_no_intact_container_would_carry() {
    let (root, input, spec) = setup();
    let layout = validate_normalization(&input, &spec).unwrap();
    seed_old_outputs(&layout);
    let runner = FakeRunner::new(vec![
        Ok(ProcessOutput::new(Some(0), source_probe(), Vec::new())),
        Ok(ProcessOutput::new(Some(0), Vec::new(), Vec::new())),
        Ok(ProcessOutput::new(Some(0), Vec::new(), Vec::new())),
        Ok(ProcessOutput::new(Some(0), video_probe(), Vec::new())),
        Ok(ProcessOutput::new(
            Some(0),
            audio_probe_with("wav", "pcm_s16le", "s16", "16000", 1, "2.3"),
            Vec::new(),
        )),
    ]);

    assert!(matches!(
        normalize_media_with_runner(&input, &spec, &tools(root.path()), &runner),
        Err(MediaError::NormalizationVerificationFailed {
            field: "duration_delta",
            ..
        })
    ));
    assert_old_outputs_and_no_staging(&layout, layout.output_dir());
}

/// The tolerance has to admit the alignment intact containers carry. The demo
/// clip this repository ships probes 60.44 s of video against 60.48 s of audio,
/// and normalization preserves each stream's duration, so the earlier 20 ms
/// bound rejected the project's own input.
#[test]
fn accepts_the_duration_delta_an_intact_container_carries() {
    let (root, input, spec) = setup();
    let layout = validate_normalization(&input, &spec).unwrap();
    let runner = FakeRunner::new(vec![
        Ok(ProcessOutput::new(Some(0), source_probe(), Vec::new())),
        Ok(ProcessOutput::new(Some(0), Vec::new(), Vec::new())),
        Ok(ProcessOutput::new(Some(0), Vec::new(), Vec::new())),
        Ok(ProcessOutput::new(Some(0), video_probe(), Vec::new())),
        Ok(ProcessOutput::new(
            Some(0),
            audio_probe_with("wav", "pcm_s16le", "s16", "16000", 1, "2.04"),
            Vec::new(),
        )),
    ]);

    let result = normalize_media_with_runner(&input, &spec, &tools(root.path()), &runner).unwrap();
    assert_eq!(result.layout().audio_path(), layout.audio_path());
    assert_eq!(fs::read(layout.audio_path()).unwrap(), b"normalized-audio");
}

#[test]
fn successful_ffmpeg_without_a_temp_output_is_a_verification_failure() {
    let (root, input, spec) = setup();
    let layout = validate_normalization(&input, &spec).unwrap();
    seed_old_outputs(&layout);
    let runner = FakeRunner::with_staged_writes(
        vec![
            Ok(ProcessOutput::new(Some(0), source_probe(), Vec::new())),
            Ok(ProcessOutput::new(Some(0), Vec::new(), Vec::new())),
            Ok(ProcessOutput::new(Some(0), Vec::new(), Vec::new())),
        ],
        vec![
            StagedWrite::Missing,
            StagedWrite::Bytes(b"normalized-audio"),
        ],
    );

    assert!(matches!(
        normalize_media_with_runner(&input, &spec, &tools(root.path()), &runner),
        Err(MediaError::NormalizationVerificationFailed {
            field: "output_file",
            ..
        })
    ));
    assert_old_outputs_and_no_staging(&layout, layout.output_dir());
}

#[test]
fn rejects_normalized_output_larger_than_two_gibibytes() {
    let (root, input, spec) = setup();
    let layout = validate_normalization(&input, &spec).unwrap();
    seed_old_outputs(&layout);
    let runner = FakeRunner::with_staged_writes(
        vec![
            Ok(ProcessOutput::new(Some(0), source_probe(), Vec::new())),
            Ok(ProcessOutput::new(Some(0), Vec::new(), Vec::new())),
            Ok(ProcessOutput::new(Some(0), Vec::new(), Vec::new())),
        ],
        vec![
            StagedWrite::Sparse(2 * 1024 * 1024 * 1024 + 1),
            StagedWrite::Bytes(b"normalized-audio"),
        ],
    );

    assert!(matches!(
        normalize_media_with_runner(&input, &spec, &tools(root.path()), &runner),
        Err(MediaError::NormalizationVerificationFailed {
            field: "output_file_size",
            ..
        })
    ));
    assert_old_outputs_and_no_staging(&layout, layout.output_dir());
}

#[test]
fn a_successful_run_reports_every_phase_in_order() {
    let (root, input, spec) = setup();
    let runner = FakeRunner::new(vec![
        Ok(ProcessOutput::new(Some(0), source_probe(), Vec::new())),
        Ok(ProcessOutput::new(Some(0), Vec::new(), Vec::new())),
        Ok(ProcessOutput::new(Some(0), Vec::new(), Vec::new())),
        Ok(ProcessOutput::new(Some(0), video_probe(), Vec::new())),
        Ok(ProcessOutput::new(Some(0), audio_probe(), Vec::new())),
    ]);
    let phases = Mutex::new(Vec::new());

    normalize_media_observed(&input, &spec, &tools(root.path()), &runner, &|phase| {
        phases.lock().unwrap().push(phase)
    })
    .unwrap();

    assert_eq!(
        *phases.lock().unwrap(),
        vec![
            NormalizePhase::Probing,
            NormalizePhase::NormalizingVideo,
            NormalizePhase::NormalizingAudio,
            NormalizePhase::Verifying,
            NormalizePhase::Committing,
        ]
    );
}

#[test]
fn a_failing_video_pass_reports_no_phase_after_the_one_that_failed() {
    let (root, input, spec) = setup();
    let runner = FakeRunner::new(vec![
        Ok(ProcessOutput::new(Some(0), source_probe(), Vec::new())),
        Ok(ProcessOutput::new(
            Some(1),
            Vec::new(),
            b"encode failed".to_vec(),
        )),
    ]);
    let phases = Mutex::new(Vec::new());

    let error = normalize_media_observed(&input, &spec, &tools(root.path()), &runner, &|phase| {
        phases.lock().unwrap().push(phase)
    })
    .expect_err("the video pass fails");

    assert!(matches!(
        error,
        MediaError::ToolFailed {
            operation: "normalize_video",
            ..
        }
    ));
    assert_eq!(
        *phases.lock().unwrap(),
        vec![NormalizePhase::Probing, NormalizePhase::NormalizingVideo]
    );
}

#[test]
fn an_unusable_output_directory_reports_no_phase_at_all() {
    // A file where the output directory should be: layout validation fails
    // before any work is announced.
    let (root, input, mut spec) = setup();
    let blocked = root.path().join("blocked");
    fs::write(&blocked, b"not-a-directory").unwrap();
    spec.output_dir = blocked;
    let runner = FakeRunner::new(vec![]);
    let phases = Mutex::new(Vec::new());

    let error = normalize_media_observed(&input, &spec, &tools(root.path()), &runner, &|phase| {
        phases.lock().unwrap().push(phase)
    })
    .expect_err("an output directory that is a file is refused");

    assert!(matches!(error, MediaError::OutputDirectoryInvalid { .. }));
    assert!(phases.lock().unwrap().is_empty());
}
