mod support;

use std::{
    fs,
    path::{Path, PathBuf},
    sync::Mutex,
    time::Duration,
};

use feathertalk_frame_pipeline::{
    AnomalyCode, CommandSpec, DecodedFrame, FaceDetection, FaceDetector, FrameBatch, FrameDecoder,
    FrameExtractor, FramePipelineSpec, LandmarkPredictor, PipelineError, PipelinePhase,
    ProcessOutput, ProcessRunner, RecoveryAction, evaluate_frames_observed,
    evaluate_frames_with_models, extract_frames_with_runner,
};
use feathertalk_pfld::{CropGeometry, PFLDLandmarks, decode_landmarks};

use support::{Recorder, chunk_outputs};

struct OneFrameRunner;

impl ProcessRunner for OneFrameRunner {
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

fn batch() -> (tempfile::TempDir, FrameBatch) {
    batch_of(1)
}

fn batch_of(frame_count: u64) -> (tempfile::TempDir, FrameBatch) {
    let root = tempfile::tempdir().unwrap();
    let video = root.path().join("video_25fps.mp4");
    fs::write(&video, b"video").unwrap();
    let spec =
        FramePipelineSpec::new(video, root.path().join("assets"), frame_count, 640, 480).unwrap();
    let extractor =
        FrameExtractor::new(root.path().join("ffmpeg"), Duration::from_secs(1)).unwrap();
    let batch = extract_frames_with_runner(&spec, &extractor, &OneFrameRunner).unwrap();
    (root, batch)
}

#[derive(Clone, Copy)]
struct Decoder {
    blur: f64,
    fail: bool,
}

impl FrameDecoder for Decoder {
    fn decode(&self, _index: u64, path: &Path) -> Result<DecodedFrame, PipelineError> {
        if self.fail {
            return Err(PipelineError::Adapter {
                component: "decoder",
                message: "invalid jpeg".into(),
            });
        }
        Ok(DecodedFrame::new(path.to_owned(), 640, 480, self.blur).unwrap())
    }
}

struct Detector {
    detections: Vec<FaceDetection>,
    fail: bool,
}

impl FaceDetector for Detector {
    fn detect(&self, _frame: &DecodedFrame) -> Result<Vec<FaceDetection>, PipelineError> {
        if self.fail {
            Err(PipelineError::Adapter {
                component: "scrfd",
                message: "device lost".into(),
            })
        } else {
            Ok(self.detections.clone())
        }
    }
}

struct Predictor {
    landmarks: Mutex<Option<PFLDLandmarks>>,
    seen_score: Mutex<Option<f32>>,
    fail: bool,
}

impl LandmarkPredictor for Predictor {
    fn predict(
        &self,
        _frame: &DecodedFrame,
        face: &FaceDetection,
    ) -> Result<PFLDLandmarks, PipelineError> {
        *self.seen_score.lock().unwrap() = Some(face.score);
        if self.fail {
            return Err(PipelineError::Adapter {
                component: "pfld",
                message: "bad tensor".into(),
            });
        }
        Ok(self.landmarks.lock().unwrap().take().unwrap())
    }
}

fn detection(score: f32, bbox: [f32; 4]) -> FaceDetection {
    FaceDetection {
        bbox,
        score,
        keypoints: [[0.0, 0.0]; 5],
    }
}

fn landmarks(normalized: f32) -> PFLDLandmarks {
    decode_landmarks(
        &vec![normalized; 220],
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

fn predictor(normalized: f32) -> Predictor {
    Predictor {
        landmarks: Mutex::new(Some(landmarks(normalized))),
        seen_score: Mutex::new(None),
        fail: false,
    }
}

fn anomaly_for(
    decoder: Decoder,
    detector: Detector,
    predictor: Predictor,
) -> feathertalk_frame_pipeline::FrameAnomaly {
    let (_root, batch) = batch();
    let evaluation = evaluate_frames_with_models(&batch, &decoder, &detector, &predictor).unwrap();
    assert!(evaluation.accepted().is_empty());
    evaluation.anomalies()[0].clone()
}

#[test]
fn classifies_no_face_and_multiple_faces() {
    let no_face = anomaly_for(
        Decoder {
            blur: 30.0,
            fail: false,
        },
        Detector {
            detections: vec![],
            fail: false,
        },
        predictor(0.5),
    );
    assert_eq!(no_face.code(), AnomalyCode::FaceNotFound);
    assert_eq!(no_face.recovery_action(), RecoveryAction::ExcludeFrame);

    let multiple = anomaly_for(
        Decoder {
            blur: 30.0,
            fail: false,
        },
        Detector {
            detections: vec![
                detection(0.9, [0.0, 0.0, 320.0, 300.0]),
                detection(0.8, [330.0, 0.0, 300.0, 300.0]),
            ],
            fail: false,
        },
        predictor(0.5),
    );
    assert_eq!(multiple.code(), AnomalyCode::MultipleFaces);
}

#[test]
fn nms_selects_highest_score_with_stable_order() {
    let (_root, batch) = batch();
    let predictor = predictor(0.5);
    let evaluation = evaluate_frames_with_models(
        &batch,
        &Decoder {
            blur: 30.0,
            fail: false,
        },
        &Detector {
            detections: vec![
                detection(0.8, [10.0, 10.0, 400.0, 350.0]),
                detection(0.9, [12.0, 12.0, 400.0, 350.0]),
            ],
            fail: false,
        },
        &predictor,
    )
    .unwrap();
    assert!(evaluation.anomalies().is_empty());
    assert_eq!(evaluation.accepted().len(), 1);
    assert_eq!(*predictor.seen_score.lock().unwrap(), Some(0.9));
    assert!(evaluation.accepted()[0].landmark_bytes().ends_with(b"\n"));
    assert_eq!(
        String::from_utf8(evaluation.accepted()[0].landmark_bytes().to_vec())
            .unwrap()
            .lines()
            .count(),
        110
    );
}

#[test]
fn classifies_bbox_landmark_and_blur_contract_failures() {
    let bbox = anomaly_for(
        Decoder {
            blur: 30.0,
            fail: false,
        },
        Detector {
            detections: vec![detection(0.9, [-95.0, 10.0, 100.0, 100.0])],
            fail: false,
        },
        predictor(0.5),
    );
    assert_eq!(bbox.code(), AnomalyCode::BboxOutOfBounds);

    let landmark = anomaly_for(
        Decoder {
            blur: 30.0,
            fail: false,
        },
        Detector {
            detections: vec![detection(0.9, [0.0, 0.0, 400.0, 350.0])],
            fail: false,
        },
        predictor(2.0),
    );
    assert_eq!(landmark.code(), AnomalyCode::LandmarkInvalid);

    let blurred = anomaly_for(
        Decoder {
            blur: 19.999,
            fail: false,
        },
        Detector {
            detections: vec![detection(0.9, [0.0, 0.0, 400.0, 350.0])],
            fail: false,
        },
        predictor(0.5),
    );
    assert_eq!(blurred.code(), AnomalyCode::BlurredFrame);
}

#[test]
fn adapter_failures_become_stable_frame_anomalies() {
    let decode = anomaly_for(
        Decoder {
            blur: 30.0,
            fail: true,
        },
        Detector {
            detections: vec![],
            fail: false,
        },
        predictor(0.5),
    );
    assert_eq!(decode.code(), AnomalyCode::FrameDecodeFailed);
    assert_eq!(decode.recovery_action(), RecoveryAction::RerunFrame);

    let model = anomaly_for(
        Decoder {
            blur: 30.0,
            fail: false,
        },
        Detector {
            detections: vec![],
            fail: true,
        },
        predictor(0.5),
    );
    assert_eq!(model.code(), AnomalyCode::ModelFailed);

    let pfld = anomaly_for(
        Decoder {
            blur: 30.0,
            fail: false,
        },
        Detector {
            detections: vec![detection(0.9, [0.0, 0.0, 400.0, 350.0])],
            fail: false,
        },
        Predictor {
            landmarks: Mutex::new(Some(landmarks(0.5))),
            seen_score: Mutex::new(None),
            fail: true,
        },
    );
    assert_eq!(pfld.code(), AnomalyCode::ModelFailed);
}

#[test]
fn rejected_low_confidence_detection_counts_as_no_face() {
    let anomaly = anomaly_for(
        Decoder {
            blur: 30.0,
            fail: false,
        },
        Detector {
            detections: vec![detection(0.499, [0.0, 0.0, 400.0, 350.0])],
            fail: false,
        },
        predictor(0.5),
    );
    assert_eq!(anomaly.code(), AnomalyCode::FaceNotFound);
}

#[test]
fn small_but_fully_inside_bbox_reaches_the_accepted_path() {
    let (_root, batch) = batch();
    let predictor = predictor(0.5);
    let evaluation = evaluate_frames_with_models(
        &batch,
        &Decoder {
            blur: 30.0,
            fail: false,
        },
        &Detector {
            detections: vec![detection(0.9, [240.0, 130.0, 100.0, 110.0])],
            fail: false,
        },
        &predictor,
    )
    .unwrap();
    assert!(
        evaluation.anomalies().is_empty(),
        "{:?}",
        evaluation.anomalies()
    );
    assert_eq!(evaluation.accepted().len(), 1);
    assert_eq!(
        evaluation.accepted()[0].bbox(),
        [240.0, 130.0, 100.0, 110.0]
    );
}

#[test]
fn evaluation_reports_one_phase_before_each_frame() {
    let (_root, batch) = batch_of(3);
    // A failing decoder turns every frame into an anomaly, which keeps the
    // test focused on the counter instead of on model plumbing.
    let decoder = Decoder {
        blur: 0.0,
        fail: true,
    };
    let detector = Detector {
        detections: Vec::new(),
        fail: false,
    };
    let observer = Recorder::new(None);

    let evaluation =
        evaluate_frames_observed(&batch, &decoder, &detector, &predictor(0.5), &observer).unwrap();

    assert_eq!(evaluation.anomalies().len(), 3);
    assert_eq!(
        observer.phases(),
        vec![
            PipelinePhase::Evaluating {
                completed: 0,
                total: 3
            },
            PipelinePhase::Evaluating {
                completed: 1,
                total: 3
            },
            PipelinePhase::Evaluating {
                completed: 2,
                total: 3
            },
        ]
    );
}

#[test]
fn cancellation_stops_evaluation_at_the_next_frame() {
    let (_root, batch) = batch_of(4);
    let decoder = Decoder {
        blur: 0.0,
        fail: true,
    };
    let detector = Detector {
        detections: Vec::new(),
        fail: false,
    };
    // Cancelled once two frames have been reported, so the third frame is
    // where the run stops.
    let observer = Recorder::new(Some(2));

    let error = evaluate_frames_observed(&batch, &decoder, &detector, &predictor(0.5), &observer)
        .unwrap_err();

    assert!(
        matches!(
            error,
            PipelineError::Cancelled {
                operation: "evaluate_frames"
            }
        ),
        "{error:?}"
    );
    assert_eq!(observer.phases().len(), 2);
}

#[allow(dead_code)]
fn _path(_path: PathBuf) {}
