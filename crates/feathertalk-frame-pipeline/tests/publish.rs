mod support;

use std::{fs, path::Path, sync::Mutex, time::Duration};

use feathertalk_frame_pipeline::{
    CommandSpec, DecodedFrame, FaceDetection, FaceDetector, FrameBatch, FrameDecoder,
    FrameEvaluation, FrameExtractor, FramePipelineSpec, LandmarkPredictor, PipelineError,
    ProcessOutput, ProcessRunner, RecoveryAction, evaluate_frames_with_models,
    extract_frames_with_runner, publish_frame_artifacts, read_quality_report,
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
    value: Mutex<Option<PFLDLandmarks>>,
}

impl LandmarkPredictor for Predictor {
    fn predict(
        &self,
        _frame: &DecodedFrame,
        _face: &FaceDetection,
    ) -> Result<PFLDLandmarks, PipelineError> {
        Ok(self.value.lock().unwrap().take().unwrap())
    }
}

fn setup(
    frame_count: u64,
) -> (
    tempfile::TempDir,
    FramePipelineSpec,
    FrameExtractor,
    FrameBatch,
    FrameEvaluation,
) {
    let root = tempfile::tempdir().unwrap();
    let video = root.path().join("video_25fps.mp4");
    fs::write(&video, b"video").unwrap();
    let output = root.path().join("assets");
    let spec = FramePipelineSpec::new(video, output, frame_count, 640, 480).unwrap();
    let extractor =
        FrameExtractor::new(root.path().join("ffmpeg"), Duration::from_secs(1)).unwrap();
    let batch = extract_frames_with_runner(&spec, &extractor, &Runner).unwrap();
    let landmarks = decode_landmarks(
        &vec![0.5; 220],
        &vec![0.0; 220],
        CropGeometry {
            width: 640,
            height: 480,
            offset_x: 0,
            offset_y: 0,
        },
    )
    .unwrap();
    let evaluation = evaluate_frames_with_models(
        &batch,
        &Decoder,
        &Detector,
        &Predictor {
            value: Mutex::new(Some(landmarks)),
        },
    )
    .unwrap();
    (root, spec, extractor, batch, evaluation)
}

#[test]
fn publishes_frames_landmarks_and_report_with_hashes() {
    let (root, spec, _extractor, mut batch, evaluation) = setup(1);
    let staging = batch.staging_dir().to_owned();
    let report = publish_frame_artifacts(&spec, &mut batch, &evaluation).unwrap();
    assert_eq!(report.frame_count(), 1);
    assert_eq!(report.accepted_count(), 1);
    assert_eq!(fs::read(spec.frame_path(0)).unwrap(), b"jpeg-frame");
    assert_eq!(
        fs::read_to_string(spec.landmark_path(0))
            .unwrap()
            .lines()
            .count(),
        110
    );
    assert_eq!(read_quality_report(&spec.quality_path()).unwrap(), report);
    assert_eq!(report.frames()[0].frame_bytes(), b"jpeg-frame".len() as u64);
    assert_eq!(report.frames()[0].frame_sha256().len(), 64);
    assert_eq!(report.frames()[0].landmark_sha256().len(), 64);
    assert!(
        !staging.exists(),
        "successful publish must remove owned staging"
    );
    assert!(fs::read_dir(root.path()).unwrap().all(|entry| {
        !entry
            .unwrap()
            .file_name()
            .to_string_lossy()
            .starts_with(".feathertalk-frame-backup-")
    }));
}

#[test]
fn anomalies_reject_publish_and_leave_existing_outputs_untouched() {
    let (_root, spec, _extractor, mut batch, _evaluation) = setup(1);
    fs::create_dir_all(spec.output_root().join("frames")).unwrap();
    fs::create_dir_all(spec.output_root().join("landmarks")).unwrap();
    fs::write(spec.frame_path(0), b"old-frame").unwrap();
    fs::write(spec.landmark_path(0), b"old-landmark").unwrap();
    fs::write(spec.quality_path(), b"old-report").unwrap();
    let anomaly = feathertalk_frame_pipeline::FrameAnomaly::new(
        0,
        feathertalk_frame_pipeline::AnomalyCode::BlurredFrame,
        "blurred",
        "variance=1",
        RecoveryAction::ExcludeFrame,
    )
    .unwrap();
    let evaluation = FrameEvaluation::from_parts(Vec::new(), vec![anomaly]);
    assert!(matches!(
        publish_frame_artifacts(&spec, &mut batch, &evaluation),
        Err(PipelineError::QualityRejected { count: 1 })
    ));
    assert_eq!(fs::read(spec.frame_path(0)).unwrap(), b"old-frame");
    assert_eq!(fs::read(spec.landmark_path(0)).unwrap(), b"old-landmark");
    assert_eq!(fs::read(spec.quality_path()).unwrap(), b"old-report");
}

#[test]
fn successful_publish_replaces_complete_existing_output_set() {
    let (_root, spec, _extractor, mut batch, evaluation) = setup(1);
    fs::create_dir_all(spec.output_root().join("frames")).unwrap();
    fs::create_dir_all(spec.output_root().join("landmarks")).unwrap();
    fs::write(spec.frame_path(0), b"old-frame").unwrap();
    fs::write(spec.landmark_path(0), b"old-landmark").unwrap();
    fs::write(spec.quality_path(), b"old-report").unwrap();

    publish_frame_artifacts(&spec, &mut batch, &evaluation).unwrap();
    assert_eq!(fs::read(spec.frame_path(0)).unwrap(), b"jpeg-frame");
    assert_ne!(fs::read(spec.landmark_path(0)).unwrap(), b"old-landmark");
    assert_ne!(fs::read(spec.quality_path()).unwrap(), b"old-report");
}

#[test]
fn changed_extracted_frame_rejects_publish_without_touching_outputs() {
    let (_root, spec, _extractor, mut batch, evaluation) = setup(1);
    fs::write(batch.frames()[0].path(), b"mutated-frame").unwrap();
    assert!(matches!(
        publish_frame_artifacts(&spec, &mut batch, &evaluation),
        Err(PipelineError::PublishFailed {
            operation: "validate_frame_integrity",
            ..
        })
    ));
    assert!(!spec.frame_path(0).exists());
    assert!(!spec.landmark_path(0).exists());
    assert!(!spec.quality_path().exists());
}

#[test]
fn invalid_existing_destination_is_rejected_without_replacing_it() {
    let (_root, spec, _extractor, mut batch, evaluation) = setup(1);
    fs::write(spec.output_root().join("frames"), b"not-a-directory").unwrap();
    assert!(matches!(
        publish_frame_artifacts(&spec, &mut batch, &evaluation),
        Err(PipelineError::PublishFailed {
            operation: "validate_destination",
            ..
        })
    ));
    assert_eq!(
        fs::read(spec.output_root().join("frames")).unwrap(),
        b"not-a-directory"
    );
}
