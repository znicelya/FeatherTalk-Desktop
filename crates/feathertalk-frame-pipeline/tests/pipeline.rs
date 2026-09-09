mod support;

use std::{collections::VecDeque, fs, path::Path, sync::Mutex, time::Duration};

use feathertalk_frame_pipeline::{
    CommandSpec, DecodedFrame, FaceDetection, FaceDetector, FrameDecoder, FrameExtractor,
    FramePipelineSpec, LandmarkPredictor, PipelineError, ProcessOutput, ProcessRunner,
    run_frame_pipeline_with_runner,
};
use feathertalk_pfld::{CropGeometry, PFLDLandmarks, decode_landmarks};

use support::chunk_outputs;

struct Runner;

impl ProcessRunner for Runner {
    fn run(
        &self,
        command: &CommandSpec,
        _timeout: Duration,
    ) -> Result<ProcessOutput, PipelineError> {
        for (_, path) in chunk_outputs(command) {
            fs::write(path, b"jpeg-frame").unwrap();
        }
        Ok(ProcessOutput::new(Some(0), vec![], vec![]))
    }
}

struct Decoder;

impl FrameDecoder for Decoder {
    fn decode(&self, _index: u64, path: &Path) -> Result<DecodedFrame, PipelineError> {
        DecodedFrame::new(path.to_owned(), 640, 480, 30.0)
    }
}

struct Detector;

impl FaceDetector for Detector {
    fn detect(&self, _frame: &DecodedFrame) -> Result<Vec<FaceDetection>, PipelineError> {
        Ok(vec![FaceDetection {
            bbox: [0.0, 0.0, 400.0, 350.0],
            score: 0.9,
            keypoints: [[1.0, 1.0]; 5],
        }])
    }
}

struct Predictor {
    values: Mutex<VecDeque<PFLDLandmarks>>,
}

impl LandmarkPredictor for Predictor {
    fn predict(
        &self,
        _frame: &DecodedFrame,
        _face: &FaceDetection,
    ) -> Result<PFLDLandmarks, PipelineError> {
        self.values
            .lock()
            .unwrap()
            .pop_front()
            .ok_or_else(|| PipelineError::Adapter {
                component: "pfld",
                message: "no fixture landmark output".into(),
            })
    }
}

fn landmarks() -> PFLDLandmarks {
    decode_landmarks(
        &vec![0.5; 220],
        &vec![0.0; 220],
        CropGeometry {
            width: 640,
            height: 480,
            offset_x: 0,
            offset_y: 0,
        },
    )
    .unwrap()
}

#[test]
fn runs_extraction_evaluation_landmarks_report_and_publish_as_one_pipeline() {
    let root = tempfile::tempdir().unwrap();
    let video = root.path().join("video_25fps.mp4");
    fs::write(&video, b"video").unwrap();
    let spec = FramePipelineSpec::new(video, root.path().join("assets"), 2, 640, 480).unwrap();
    let extractor =
        FrameExtractor::new(root.path().join("ffmpeg"), Duration::from_secs(1)).unwrap();
    let predictor = Predictor {
        values: Mutex::new(VecDeque::from([landmarks(), landmarks()])),
    };

    let report =
        run_frame_pipeline_with_runner(&spec, &extractor, &Runner, &Decoder, &Detector, &predictor)
            .unwrap();

    assert_eq!(report.frame_count(), 2);
    assert_eq!(report.accepted_count(), 2);
    assert_eq!(fs::read(spec.frame_path(1)).unwrap(), b"jpeg-frame");
    assert_eq!(
        fs::read_to_string(spec.landmark_path(1))
            .unwrap()
            .lines()
            .count(),
        110
    );
    assert!(spec.quality_path().is_file());
}

#[test]
fn model_failure_aborts_pipeline_before_publishing_any_outputs() {
    let root = tempfile::tempdir().unwrap();
    let video = root.path().join("video_25fps.mp4");
    fs::write(&video, b"video").unwrap();
    let spec = FramePipelineSpec::new(video, root.path().join("assets"), 1, 640, 480).unwrap();
    let extractor =
        FrameExtractor::new(root.path().join("ffmpeg"), Duration::from_secs(1)).unwrap();
    let error = run_frame_pipeline_with_runner(
        &spec,
        &extractor,
        &Runner,
        &Decoder,
        &Detector,
        &Predictor {
            values: Mutex::new(VecDeque::new()),
        },
    )
    .unwrap_err();
    assert!(matches!(error, PipelineError::QualityRejected { count: 1 }));
    assert!(!spec.frame_path(0).exists());
    assert!(!spec.landmark_path(0).exists());
    assert!(!spec.quality_path().exists());
}
