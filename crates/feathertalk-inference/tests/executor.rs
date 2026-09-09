use std::{
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};

use burn::tensor::{Tensor, TensorData};
use feathertalk_audio::{FeatureMatrix, write_feature_file};
use feathertalk_inference::{
    BgrFrame, FrameReader, InferenceError, OfflineRenderRequest, RawVideoSink, RawVideoSinkFactory,
    execute_offline_render,
};
use feathertalk_models::{backend::CpuBackend, unet::TalkingHeadModel};

#[path = "support/mod.rs"]
mod support;

fn request(root: &std::path::Path) -> OfflineRenderRequest {
    OfflineRenderRequest::new(
        root.join("frames"),
        root.join("landmarks"),
        root.join("features.f32"),
        root.join("audio.wav"),
        root.join("ffmpeg.exe"),
        root.join("result.mp4"),
        "task-01",
        2,
        None,
    )
    .unwrap()
}

#[test]
fn request_keeps_native_paths_and_limits() {
    let root = tempfile::tempdir().unwrap();
    let request = request(root.path());

    assert_eq!(request.frame_dir(), root.path().join("frames"));
    assert_eq!(request.landmark_dir(), root.path().join("landmarks"));
    assert_eq!(request.feature_path(), root.path().join("features.f32"));
    assert_eq!(request.audio_path(), root.path().join("audio.wav"));
    assert_eq!(request.ffmpeg_path(), root.path().join("ffmpeg.exe"));
    assert_eq!(request.output_path(), root.path().join("result.mp4"));
    assert_eq!(request.task_id(), "task-01");
    assert_eq!(request.source_frame_count(), 2);
    assert_eq!(request.max_output_frames(), None);
}

#[test]
fn request_rejects_relative_paths_and_invalid_counts() {
    let root = tempfile::tempdir().unwrap();

    assert!(matches!(
        OfflineRenderRequest::new(
            PathBuf::from("frames"),
            root.path().join("landmarks"),
            root.path().join("features.f32"),
            root.path().join("audio.wav"),
            root.path().join("ffmpeg.exe"),
            root.path().join("result.mp4"),
            "task-01",
            2,
            None,
        ),
        Err(InferenceError::InvalidField {
            field: "frame_dir",
            ..
        })
    ));
    assert!(matches!(
        OfflineRenderRequest::new(
            root.path().join("frames"),
            root.path().join("landmarks"),
            root.path().join("features.f32"),
            root.path().join("audio.wav"),
            root.path().join("ffmpeg.exe"),
            root.path().join("result.mp4"),
            "task-01",
            1,
            None,
        ),
        Err(InferenceError::FrameCountTooSmall {
            actual: 1,
            minimum: 2
        })
    ));
    assert!(matches!(
        OfflineRenderRequest::new(
            root.path().join("frames"),
            root.path().join("landmarks"),
            root.path().join("features.f32"),
            root.path().join("audio.wav"),
            root.path().join("ffmpeg.exe"),
            root.path().join("result.mp4"),
            "task-01",
            2,
            Some(0),
        ),
        Err(InferenceError::InvalidField {
            field: "max_output_frames",
            ..
        })
    ));
}

#[test]
fn request_reuses_output_and_task_id_validation() {
    let root = tempfile::tempdir().unwrap();
    let output = root.path().join("result.mp4");
    std::fs::write(&output, b"sentinel").unwrap();

    assert!(matches!(
        OfflineRenderRequest::new(
            root.path().join("frames"),
            root.path().join("landmarks"),
            root.path().join("features.f32"),
            root.path().join("audio.wav"),
            root.path().join("ffmpeg.exe"),
            output.clone(),
            "task-01",
            2,
            None,
        ),
        Err(InferenceError::OutputExists { path }) if path == output
    ));
    assert_eq!(std::fs::read(&output).unwrap(), b"sentinel");

    std::fs::remove_file(&output).unwrap();
    assert!(matches!(
        OfflineRenderRequest::new(
            root.path().join("frames"),
            root.path().join("landmarks"),
            root.path().join("features.f32"),
            root.path().join("audio.wav"),
            root.path().join("ffmpeg.exe"),
            output,
            "bad/task",
            2,
            None,
        ),
        Err(InferenceError::InvalidTaskId { .. })
    ));
}

struct OutputModel {
    value: f32,
}

impl TalkingHeadModel<CpuBackend> for OutputModel {
    fn forward_talking_head(
        &self,
        image: Tensor<CpuBackend, 4>,
        _audio: Tensor<CpuBackend, 4>,
    ) -> Tensor<CpuBackend, 4> {
        let device = image.device();
        Tensor::from_data(
            TensorData::new(vec![self.value; 3 * 160 * 160], [1, 3, 160, 160]),
            &device,
        )
    }
}

#[derive(Clone)]
struct RecordingReader {
    frames: Arc<Mutex<Vec<usize>>>,
    fail_at: Option<usize>,
    alternate_dimensions_at: Option<usize>,
}

impl FrameReader for RecordingReader {
    fn read(&self, index: usize, path: &Path) -> Result<BgrFrame, InferenceError> {
        self.frames.lock().unwrap().push(index);
        if self.fail_at == Some(index) {
            return Err(InferenceError::FrameReader {
                index,
                path: path.to_owned(),
                message: "injected reader failure".into(),
            });
        }
        let (width, height) = if self.alternate_dimensions_at == Some(index) {
            (160, 168)
        } else {
            (168, 168)
        };
        BgrFrame::new(
            width,
            height,
            vec![(index as u8) + 1; (width * height * 3) as usize],
        )
    }
}

#[derive(Clone, Default)]
struct RecordingSinkState {
    frames: Vec<Vec<u8>>,
    staging: Option<PathBuf>,
    fail_write: bool,
    fail_finish: bool,
}

struct RecordingSink {
    state: Arc<Mutex<RecordingSinkState>>,
}

impl RawVideoSink for RecordingSink {
    fn write_frame(&mut self, frame: &BgrFrame) -> Result<(), InferenceError> {
        let mut state = self.state.lock().unwrap();
        if state.fail_write {
            return Err(InferenceError::SinkWrite {
                message: "injected sink write failure".into(),
            });
        }
        state.frames.push(frame.as_bytes().to_vec());
        Ok(())
    }

    fn finish(self: Box<Self>) -> Result<(), InferenceError> {
        let state = self.state.lock().unwrap();
        if state.fail_finish {
            return Err(InferenceError::SinkFinish {
                message: "injected sink finish failure".into(),
            });
        }
        std::fs::write(state.staging.as_ref().unwrap(), b"rendered-video").unwrap();
        Ok(())
    }
}

struct RecordingSinkFactory {
    state: Arc<Mutex<RecordingSinkState>>,
}

impl RawVideoSinkFactory for RecordingSinkFactory {
    fn start(
        &self,
        command: &feathertalk_inference::CommandSpec,
    ) -> Result<Box<dyn RawVideoSink + '_>, InferenceError> {
        self.state.lock().unwrap().staging = Some(
            command
                .arguments()
                .last()
                .unwrap()
                .to_string_lossy()
                .into_owned()
                .into(),
        );
        Ok(Box::new(RecordingSink {
            state: Arc::clone(&self.state),
        }))
    }
}

fn artifact_tree(
    frame_count: usize,
    feature_frame_count: usize,
    output_count: Option<usize>,
) -> (tempfile::TempDir, OfflineRenderRequest) {
    let root = tempfile::tempdir().unwrap();
    let frames = root.path().join("frames");
    let landmarks = root.path().join("landmarks");
    std::fs::create_dir(&frames).unwrap();
    std::fs::create_dir(&landmarks).unwrap();
    for index in 0..frame_count {
        std::fs::write(frames.join(format!("{index:06}.jpg")), b"fixture").unwrap();
        let mut lms = String::new();
        for point in 0..110 {
            let x = if point == 31 { 168 } else { 0 };
            let y = 0;
            lms.push_str(&format!("{x} {y}\n"));
        }
        std::fs::write(landmarks.join(format!("{index:06}.lms")), lms).unwrap();
    }
    let feature_tokens = feature_frame_count * 2;
    let features =
        FeatureMatrix::new(feature_tokens, 1024, vec![0.0; feature_tokens * 1024]).unwrap();
    let feature_path = root.path().join("features.f32");
    write_feature_file(&feature_path, &features).unwrap();
    let audio_path = root.path().join("audio.wav");
    std::fs::write(&audio_path, b"audio").unwrap();
    let request = OfflineRenderRequest::new(
        frames,
        landmarks,
        feature_path,
        audio_path,
        std::env::current_exe().unwrap(),
        root.path().join("result.mp4"),
        "task-executor",
        frame_count,
        output_count,
    )
    .unwrap();
    (root, request)
}

#[test]
fn executor_reads_artifacts_renders_plan_and_publishes_result() {
    let (_root, request) = artifact_tree(3, 5, Some(5));
    let reader_calls = Arc::new(Mutex::new(Vec::new()));
    let sink_state = Arc::new(Mutex::new(RecordingSinkState::default()));
    let reader = RecordingReader {
        frames: Arc::clone(&reader_calls),
        fail_at: None,
        alternate_dimensions_at: None,
    };
    let sink_factory = RecordingSinkFactory {
        state: Arc::clone(&sink_state),
    };
    let device = Default::default();
    let result = execute_offline_render::<CpuBackend, _, _, _>(
        &OutputModel { value: 1.0 },
        &device,
        &request,
        &reader,
        &sink_factory,
    )
    .unwrap();

    assert_eq!(result.frame_count(), 5);
    assert_eq!(result.output_path(), request.output_path());
    assert_eq!(*reader_calls.lock().unwrap(), vec![0, 1, 2, 1, 0]);
    assert_eq!(sink_state.lock().unwrap().frames.len(), 5);
    assert!(request.output_path().is_file());
    assert_eq!(
        std::fs::read(request.output_path()).unwrap(),
        b"rendered-video"
    );
}

#[test]
fn executor_cleans_staging_and_preserves_existing_destination_on_failure() {
    let (_root, request) = artifact_tree(2, 2, None);
    std::fs::write(request.output_path(), b"sentinel").unwrap();
    let reader = RecordingReader {
        frames: Arc::new(Mutex::new(Vec::new())),
        fail_at: None,
        alternate_dimensions_at: None,
    };
    let sink_state = Arc::new(Mutex::new(RecordingSinkState::default()));
    let sink_factory = RecordingSinkFactory {
        state: Arc::clone(&sink_state),
    };
    let device = Default::default();
    assert!(matches!(
        execute_offline_render::<CpuBackend, _, _, _>(
            &OutputModel { value: 1.0 },
            &device,
            &request,
            &reader,
            &sink_factory,
        ),
        Err(InferenceError::OutputExists { .. })
    ));
    assert_eq!(std::fs::read(request.output_path()).unwrap(), b"sentinel");
}

#[test]
fn executor_propagates_reader_failure_without_publishing() {
    let (_root, request) = artifact_tree(2, 2, None);
    let reader_calls = Arc::new(Mutex::new(Vec::new()));
    let reader = RecordingReader {
        frames: Arc::clone(&reader_calls),
        fail_at: Some(1),
        alternate_dimensions_at: None,
    };
    let sink_state = Arc::new(Mutex::new(RecordingSinkState::default()));
    let sink_factory = RecordingSinkFactory {
        state: Arc::clone(&sink_state),
    };
    let device = Default::default();
    assert!(matches!(
        execute_offline_render::<CpuBackend, _, _, _>(
            &OutputModel { value: 1.0 },
            &device,
            &request,
            &reader,
            &sink_factory,
        ),
        Err(InferenceError::FrameReader { index: 1, .. })
    ));
    assert!(!request.output_path().exists());
    assert_eq!(*reader_calls.lock().unwrap(), vec![0, 1]);
}

#[test]
fn executor_rejects_model_output_before_writing_the_failed_frame() {
    let (_root, request) = artifact_tree(2, 2, None);
    let reader = RecordingReader {
        frames: Arc::new(Mutex::new(Vec::new())),
        fail_at: None,
        alternate_dimensions_at: None,
    };
    let sink_state = Arc::new(Mutex::new(RecordingSinkState::default()));
    let sink_factory = RecordingSinkFactory {
        state: Arc::clone(&sink_state),
    };
    let device = Default::default();

    assert!(matches!(
        execute_offline_render::<CpuBackend, _, _, _>(
            &OutputModel { value: f32::NAN },
            &device,
            &request,
            &reader,
            &sink_factory,
        ),
        Err(InferenceError::NonFiniteModelOutput { .. })
    ));
    assert!(sink_state.lock().unwrap().frames.is_empty());
    assert!(!request.output_path().exists());
    let staging = sink_state.lock().unwrap().staging.clone().unwrap();
    assert!(!staging.exists());
}

#[test]
fn executor_cleans_staging_after_sink_write_or_finish_failure() {
    for (fail_write, fail_finish) in [(true, false), (false, true)] {
        let (_root, request) = artifact_tree(2, 2, None);
        let reader = RecordingReader {
            frames: Arc::new(Mutex::new(Vec::new())),
            fail_at: None,
            alternate_dimensions_at: None,
        };
        let sink_state = Arc::new(Mutex::new(RecordingSinkState {
            fail_write,
            fail_finish,
            ..RecordingSinkState::default()
        }));
        let sink_factory = RecordingSinkFactory {
            state: Arc::clone(&sink_state),
        };
        let device = Default::default();
        let error = execute_offline_render::<CpuBackend, _, _, _>(
            &OutputModel { value: 1.0 },
            &device,
            &request,
            &reader,
            &sink_factory,
        )
        .unwrap_err();
        assert!(matches!(
            error,
            InferenceError::SinkWrite { .. } | InferenceError::SinkFinish { .. }
        ));
        assert!(!request.output_path().exists());
        let staging = sink_state.lock().unwrap().staging.clone().unwrap();
        assert!(!staging.exists());
    }
}

#[test]
fn executor_rejects_missing_input_artifacts_before_starting_sink() {
    let (_root, request) = artifact_tree(2, 2, None);
    std::fs::remove_file(request.feature_path()).unwrap();
    let reader = RecordingReader {
        frames: Arc::new(Mutex::new(Vec::new())),
        fail_at: None,
        alternate_dimensions_at: None,
    };
    let sink_state = Arc::new(Mutex::new(RecordingSinkState::default()));
    let sink_factory = RecordingSinkFactory {
        state: Arc::clone(&sink_state),
    };
    let device = Default::default();
    assert!(matches!(
        execute_offline_render::<CpuBackend, _, _, _>(
            &OutputModel { value: 1.0 },
            &device,
            &request,
            &reader,
            &sink_factory,
        ),
        Err(InferenceError::InvalidInputArtifact {
            field: "feature_path",
            ..
        })
    ));
    assert!(sink_state.lock().unwrap().staging.is_none());
}

#[test]
fn executor_rejects_frame_dimension_changes_without_publishing() {
    let (_root, request) = artifact_tree(2, 2, None);
    let reader = RecordingReader {
        frames: Arc::new(Mutex::new(Vec::new())),
        fail_at: None,
        alternate_dimensions_at: Some(1),
    };
    let sink_state = Arc::new(Mutex::new(RecordingSinkState::default()));
    let sink_factory = RecordingSinkFactory {
        state: Arc::clone(&sink_state),
    };
    let device = Default::default();
    assert!(matches!(
        execute_offline_render::<CpuBackend, _, _, _>(
            &OutputModel { value: 1.0 },
            &device,
            &request,
            &reader,
            &sink_factory,
        ),
        Err(InferenceError::FrameDimensionsMismatch { index: 1, .. })
    ));
    assert!(!request.output_path().exists());
    let staging = sink_state.lock().unwrap().staging.clone().unwrap();
    assert!(!staging.exists());
}

#[test]
fn executor_rejects_landmark_symlinks_without_following_them() {
    let (_root, request) = artifact_tree(2, 2, None);
    let landmark = request.landmark_dir().join("000000.lms");
    let target = request.landmark_dir().join("landmark-target.lms");
    std::fs::rename(&landmark, &target).unwrap();
    #[cfg(windows)]
    let link_result = std::os::windows::fs::symlink_file(&target, &landmark);
    #[cfg(unix)]
    let link_result = std::os::unix::fs::symlink(&target, &landmark);
    if link_result.is_err() {
        return;
    }

    let reader = RecordingReader {
        frames: Arc::new(Mutex::new(Vec::new())),
        fail_at: None,
        alternate_dimensions_at: None,
    };
    let sink_state = Arc::new(Mutex::new(RecordingSinkState::default()));
    let sink_factory = RecordingSinkFactory {
        state: Arc::clone(&sink_state),
    };
    let device = Default::default();
    assert!(matches!(
        execute_offline_render::<CpuBackend, _, _, _>(
            &OutputModel { value: 1.0 },
            &device,
            &request,
            &reader,
            &sink_factory,
        ),
        Err(InferenceError::InvalidInputArtifact {
            field: "landmark_path",
            ..
        })
    ));
    assert!(!request.output_path().exists());
    assert!(sink_state.lock().unwrap().staging.is_none());
}

#[test]
fn executor_rejects_symlinked_input_path_components_before_starting_sink() {
    let (root, request) = artifact_tree(2, 2, None);
    let real_parent = root.path().join("real-parent");
    let real_frames = real_parent.join("frames");
    std::fs::create_dir(&real_parent).unwrap();
    std::fs::rename(request.frame_dir(), &real_frames).unwrap();
    let linked_parent = root.path().join("linked-parent");
    if support::create_dir_symlink(&real_parent, &linked_parent).is_err() {
        return;
    }
    let linked_request = OfflineRenderRequest::new(
        linked_parent.join("frames"),
        request.landmark_dir().to_owned(),
        request.feature_path().to_owned(),
        request.audio_path().to_owned(),
        request.ffmpeg_path().to_owned(),
        request.output_path().to_owned(),
        "task-symlink-component",
        request.source_frame_count(),
        request.max_output_frames(),
    )
    .unwrap();
    let reader = RecordingReader {
        frames: Arc::new(Mutex::new(Vec::new())),
        fail_at: None,
        alternate_dimensions_at: None,
    };
    let sink_state = Arc::new(Mutex::new(RecordingSinkState::default()));
    let sink_factory = RecordingSinkFactory {
        state: Arc::clone(&sink_state),
    };
    let device = Default::default();

    assert!(matches!(
        execute_offline_render::<CpuBackend, _, _, _>(
            &OutputModel { value: 1.0 },
            &device,
            &linked_request,
            &reader,
            &sink_factory,
        ),
        Err(InferenceError::InvalidInputDirectory {
            field: "frame_dir",
            ..
        })
    ));
    assert!(sink_state.lock().unwrap().staging.is_none());
    assert!(!linked_request.output_path().exists());
}

/// A sink that writes `cancel_at` frames and then reports the render cancelled,
/// which is what the worker's observing sink does once its token is set.
struct CancellingSink {
    state: Arc<Mutex<RecordingSinkState>>,
    cancel_at: usize,
}

impl RawVideoSink for CancellingSink {
    fn write_frame(&mut self, frame: &BgrFrame) -> Result<(), InferenceError> {
        let mut state = self.state.lock().unwrap();
        if state.frames.len() >= self.cancel_at {
            return Err(InferenceError::Cancelled {
                operation: "render",
            });
        }
        state.frames.push(frame.as_bytes().to_vec());
        Ok(())
    }

    fn finish(self: Box<Self>) -> Result<(), InferenceError> {
        Ok(())
    }
}

struct CancellingSinkFactory {
    state: Arc<Mutex<RecordingSinkState>>,
    cancel_at: usize,
}

impl RawVideoSinkFactory for CancellingSinkFactory {
    fn start(
        &self,
        command: &feathertalk_inference::CommandSpec,
    ) -> Result<Box<dyn RawVideoSink + '_>, InferenceError> {
        self.state.lock().unwrap().staging = Some(
            command
                .arguments()
                .last()
                .unwrap()
                .to_string_lossy()
                .into_owned()
                .into(),
        );
        Ok(Box::new(CancellingSink {
            state: Arc::clone(&self.state),
            cancel_at: self.cancel_at,
        }))
    }
}

#[test]
fn a_cancelled_sink_stops_the_render_and_leaves_no_output() {
    let (_root, request) = artifact_tree(2, 4, None);
    let reader = RecordingReader {
        frames: Arc::new(Mutex::new(Vec::new())),
        fail_at: None,
        alternate_dimensions_at: None,
    };
    let sink_state = Arc::new(Mutex::new(RecordingSinkState::default()));
    let sink_factory = CancellingSinkFactory {
        state: Arc::clone(&sink_state),
        cancel_at: 2,
    };
    let device = Default::default();

    let error = execute_offline_render::<CpuBackend, _, _, _>(
        &OutputModel { value: 1.0 },
        &device,
        &request,
        &reader,
        &sink_factory,
    )
    .expect_err("a cancelled sink fails the render");

    assert!(
        matches!(error, InferenceError::Cancelled { operation } if operation == "render"),
        "{error:?}"
    );
    assert_eq!(error.to_string(), "cancelled during render");
    // The two frames before the cancellation were written, the staging file the
    // guard reserved is gone, and the destination was never created.
    let staging = {
        let state = sink_state.lock().unwrap();
        assert_eq!(state.frames.len(), 2);
        state.staging.clone().unwrap()
    };
    assert!(!staging.exists(), "{}", staging.display());
    assert!(!request.output_path().exists());
}

/// A factory whose sink borrows the factory. This is the shape the worker needs
/// for progress reporting, and it compiles only because `start` ties the boxed
/// sink to the `&self` borrow.
struct BorrowingSinkFactory {
    written: Mutex<usize>,
}

struct BorrowingSink<'a> {
    factory: &'a BorrowingSinkFactory,
    staging: PathBuf,
}

impl RawVideoSink for BorrowingSink<'_> {
    fn write_frame(&mut self, _frame: &BgrFrame) -> Result<(), InferenceError> {
        let mut written = self.factory.written.lock().unwrap();
        *written += 1;
        Ok(())
    }

    fn finish(self: Box<Self>) -> Result<(), InferenceError> {
        std::fs::write(&self.staging, b"rendered-video").unwrap();
        Ok(())
    }
}

impl RawVideoSinkFactory for BorrowingSinkFactory {
    fn start(
        &self,
        command: &feathertalk_inference::CommandSpec,
    ) -> Result<Box<dyn RawVideoSink + '_>, InferenceError> {
        Ok(Box::new(BorrowingSink {
            factory: self,
            staging: PathBuf::from(
                command
                    .arguments()
                    .last()
                    .unwrap()
                    .to_string_lossy()
                    .into_owned(),
            ),
        }))
    }
}

#[test]
fn a_sink_may_borrow_the_factory_that_started_it() {
    let (_root, request) = artifact_tree(2, 3, None);
    let reader = RecordingReader {
        frames: Arc::new(Mutex::new(Vec::new())),
        fail_at: None,
        alternate_dimensions_at: None,
    };
    let sink_factory = BorrowingSinkFactory {
        written: Mutex::new(0),
    };
    let device = Default::default();

    let result = execute_offline_render::<CpuBackend, _, _, _>(
        &OutputModel { value: 1.0 },
        &device,
        &request,
        &reader,
        &sink_factory,
    )
    .expect("a borrowing sink renders like any other");

    assert_eq!(result.frame_count(), 3);
    // Counted through the borrow, which is the whole point of the lifetime.
    assert_eq!(*sink_factory.written.lock().unwrap(), 3);
    assert!(request.output_path().is_file());
}
