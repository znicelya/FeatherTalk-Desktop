mod support;

use std::{
    collections::VecDeque,
    fs,
    path::{Path, PathBuf},
    sync::Mutex,
    time::Duration,
};

use feathertalk_frame_pipeline::{
    CommandSpec, FRAME_CHUNK, FrameExtractor, FramePipelineSpec, PipelineError, PipelinePhase,
    ProcessOutput, ProcessRunner, extract_frames_observed, extract_frames_with_runner,
};

use support::{Recorder, chunk_outputs, flag_number, flag_value};

struct FakeRunner {
    outputs: Mutex<VecDeque<Result<ProcessOutput, PipelineError>>>,
    commands: Mutex<Vec<CommandSpec>>,
    writes: WriteMode,
}

#[derive(Clone, Copy)]
enum WriteMode {
    Bytes,
    Missing,
    Empty,
    Oversized,
}

impl FakeRunner {
    fn new(outputs: Vec<Result<ProcessOutput, PipelineError>>, writes: WriteMode) -> Self {
        Self {
            outputs: Mutex::new(outputs.into()),
            commands: Mutex::new(Vec::new()),
            writes,
        }
    }
}

impl ProcessRunner for FakeRunner {
    fn run(
        &self,
        command: &CommandSpec,
        _timeout: Duration,
    ) -> Result<ProcessOutput, PipelineError> {
        self.commands.lock().unwrap().push(command.clone());
        let output = self.outputs.lock().unwrap().pop_front().unwrap()?;
        for (index, path) in chunk_outputs(command) {
            match self.writes {
                WriteMode::Bytes => fs::write(&path, format!("frame:{index}")).unwrap(),
                WriteMode::Missing => {}
                WriteMode::Empty => fs::write(&path, []).unwrap(),
                WriteMode::Oversized => {
                    fs::write(&path, b"x").unwrap();
                    let file = fs::OpenOptions::new().write(true).open(&path).unwrap();
                    file.set_len(16 * 1024 * 1024 + 1).unwrap();
                }
            }
        }
        Ok(output)
    }
}

fn setup(frame_count: u64) -> (tempfile::TempDir, FramePipelineSpec, FrameExtractor) {
    let root = tempfile::tempdir().unwrap();
    let source = root.path().join("video_25fps.mp4");
    fs::write(&source, b"video").unwrap();
    let output = root.path().join("assets");
    let spec = FramePipelineSpec::new(source, output, frame_count, 640, 480).unwrap();
    let extractor =
        FrameExtractor::new(root.path().join("ffmpeg"), Duration::from_secs(10)).unwrap();
    (root, spec, extractor)
}

fn ok_outputs(count: usize) -> Vec<Result<ProcessOutput, PipelineError>> {
    (0..count)
        .map(|_| Ok(ProcessOutput::new(Some(0), vec![], vec![])))
        .collect()
}

fn staging_dirs(root: &Path) -> Vec<PathBuf> {
    fs::read_dir(root)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with(".feathertalk-frame-build-"))
        })
        .collect()
}

#[test]
fn extracts_exact_frame_count_and_records_hashes() {
    let (root, spec, extractor) = setup(3);
    let runner = FakeRunner::new(ok_outputs(1), WriteMode::Bytes);
    let batch = extract_frames_with_runner(&spec, &extractor, &runner).unwrap();
    assert_eq!(batch.frames().len(), 3);
    assert_eq!(batch.frames()[0].index(), 0);
    assert_eq!(batch.frames()[2].path().file_name().unwrap(), "000002.jpg");
    assert!(
        batch
            .frames()
            .iter()
            .all(|frame| frame.bytes() > 0 && frame.sha256().len() == 64)
    );
    assert_eq!(runner.commands.lock().unwrap().len(), 1);
    assert!(batch.staging_dir().starts_with(root.path()));
}

#[test]
fn nonzero_tool_exit_cleans_staging_and_preserves_old_outputs() {
    let (root, spec, extractor) = setup(2);
    fs::create_dir_all(spec.output_root().join("frames")).unwrap();
    fs::write(spec.frame_path(0), b"old").unwrap();
    let runner = FakeRunner::new(
        vec![
            Ok(ProcessOutput::new(Some(0), vec![], vec![])),
            Ok(ProcessOutput::new(Some(2), vec![], b"bad input".to_vec())),
        ],
        WriteMode::Bytes,
    );
    assert!(matches!(
        extract_frames_with_runner(&spec, &extractor, &runner),
        Err(PipelineError::OutputDestinationExists { .. })
    ));
    assert_eq!(fs::read(spec.frame_path(0)).unwrap(), b"old");
    assert!(staging_dirs(root.path()).is_empty());
}

#[test]
fn missing_output_is_rejected_and_staging_is_removed() {
    let (root, spec, extractor) = setup(1);
    let runner = FakeRunner::new(ok_outputs(1), WriteMode::Missing);
    assert!(matches!(
        extract_frames_with_runner(&spec, &extractor, &runner),
        Err(PipelineError::FrameMissing { .. })
    ));
    assert!(staging_dirs(root.path()).is_empty());
}

#[test]
fn empty_and_oversized_outputs_are_rejected() {
    for (mode, expected) in [(WriteMode::Empty, "empty"), (WriteMode::Oversized, "large")] {
        let (root, spec, extractor) = setup(1);
        let runner = FakeRunner::new(ok_outputs(1), mode);
        let result = extract_frames_with_runner(&spec, &extractor, &runner);
        match (expected, result) {
            ("empty", Err(PipelineError::FrameEmpty { .. })) => {}
            ("large", Err(PipelineError::FrameTooLarge { .. })) => {}
            other => panic!("unexpected result: {other:?}"),
        }
        assert!(staging_dirs(root.path()).is_empty());
    }
}

#[test]
fn injected_timeout_is_preserved() {
    let (_root, spec, extractor) = setup(1);
    let runner = FakeRunner::new(
        vec![Err(PipelineError::ToolTimedOut {
            operation: "extract_frames",
            timeout_ms: 10,
        })],
        WriteMode::Missing,
    );
    assert!(matches!(
        extract_frames_with_runner(&spec, &extractor, &runner),
        Err(PipelineError::ToolTimedOut {
            operation: "extract_frames",
            ..
        })
    ));
}

#[test]
fn frames_are_extracted_in_chunks_with_a_short_tail() {
    let (_root, spec, extractor) = setup(FRAME_CHUNK * 6 + 11);
    let runner = FakeRunner::new(ok_outputs(7), WriteMode::Bytes);
    let batch = extract_frames_with_runner(&spec, &extractor, &runner).unwrap();
    assert_eq!(batch.frames().len() as u64, FRAME_CHUNK * 6 + 11);
    let commands = runner.commands.lock().unwrap();
    assert_eq!(commands.len(), 7);
    for (position, command) in commands.iter().enumerate() {
        let first = FRAME_CHUNK * position as u64;
        assert_eq!(flag_number(command, "-start_number"), first);
        assert_eq!(
            flag_number(command, "-frames:v"),
            if position == 6 { 11 } else { FRAME_CHUNK }
        );
        // 250 frames are exactly 10 s, so every chunk starts on a whole second.
        assert_eq!(flag_value(command, "-ss"), format!("{}.000", position * 10));
    }
}

#[test]
fn extraction_reports_one_phase_per_finished_chunk() {
    let total = FRAME_CHUNK * 2 + 5;
    let (_root, spec, extractor) = setup(total);
    let runner = FakeRunner::new(ok_outputs(3), WriteMode::Bytes);
    let observer = Recorder::new(None);

    let batch = extract_frames_observed(&spec, &extractor, &runner, &observer).unwrap();

    assert_eq!(batch.frames().len() as u64, total);
    // The last phase carries the exact frame count, so the worker never has
    // to invent a final 100 % event.
    assert_eq!(
        observer.phases(),
        vec![
            PipelinePhase::Extracting {
                completed: FRAME_CHUNK,
                total
            },
            PipelinePhase::Extracting {
                completed: FRAME_CHUNK * 2,
                total
            },
            PipelinePhase::Extracting {
                completed: total,
                total
            },
        ]
    );
}

#[test]
fn cancellation_stops_extraction_at_the_next_chunk_boundary() {
    let (_root, spec, extractor) = setup(FRAME_CHUNK * 2);
    let runner = FakeRunner::new(ok_outputs(2), WriteMode::Bytes);
    // Cancelled as soon as the first chunk has been reported.
    let observer = Recorder::new(Some(1));

    let error = extract_frames_observed(&spec, &extractor, &runner, &observer).unwrap_err();

    assert!(
        matches!(
            error,
            PipelineError::Cancelled {
                operation: "extract_frames"
            }
        ),
        "{error:?}"
    );
    assert_eq!(runner.commands.lock().unwrap().len(), 1);
    assert!(staging_dirs(spec.output_root()).is_empty());
}

#[cfg(windows)]
#[test]
fn symlink_output_is_rejected() {
    let (root, spec, extractor) = setup(1);
    let target = root.path().join("target.jpg");
    fs::write(&target, b"target").unwrap();
    fs::create_dir_all(spec.output_root()).unwrap();
    let link = spec
        .output_root()
        .join(format!(".feathertalk-symlink-probe-{}", std::process::id()));
    match std::os::windows::fs::symlink_file(&target, &link) {
        Ok(()) => {
            let _ = fs::remove_file(&link);
        }
        Err(error) if error.raw_os_error() == Some(1314) => {
            eprintln!("skipping symlink test: Windows symlink privilege unavailable");
            return;
        }
        Err(error) => panic!("unable to create symlink fixture: {error}"),
    }
    let runner = SymlinkRunner { target };
    assert!(matches!(
        extract_frames_with_runner(&spec, &extractor, &runner),
        Err(PipelineError::FrameNotRegular { .. })
    ));
}

#[cfg(windows)]
struct SymlinkRunner {
    target: PathBuf,
}

#[cfg(windows)]
impl ProcessRunner for SymlinkRunner {
    fn run(
        &self,
        command: &CommandSpec,
        _timeout: Duration,
    ) -> Result<ProcessOutput, PipelineError> {
        for (_, path) in chunk_outputs(command) {
            std::os::windows::fs::symlink_file(&self.target, &path).unwrap();
        }
        Ok(ProcessOutput::new(Some(0), vec![], vec![]))
    }
}
