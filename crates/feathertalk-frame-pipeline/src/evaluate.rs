use std::path::{Path, PathBuf};

use feathertalk_face::{Detection, DetectionConfig, non_max_suppression};
use feathertalk_pfld::PFLDLandmarks;

use crate::{
    AnomalyCode, FaceDetection, FrameAnomaly, FrameBatch, LANDMARK_POINTS, NoObserver,
    PipelineError, PipelineObserver, PipelinePhase, RecoveryAction,
};

pub const FACE_CONFIDENCE_THRESHOLD: f32 = 0.50;
pub const NMS_IOU_THRESHOLD: f32 = 0.40;
/// Minimum fraction of the detection box that must fall inside the frame.
pub const MIN_BBOX_INTERSECTION_RATIO: f32 = 0.10;
pub const BLUR_VARIANCE_THRESHOLD: f64 = 20.0;

#[derive(Debug, Clone, PartialEq)]
pub struct DecodedFrame {
    path: PathBuf,
    width: u32,
    height: u32,
    laplacian_variance: f64,
}

impl DecodedFrame {
    pub fn new(
        path: PathBuf,
        width: u32,
        height: u32,
        laplacian_variance: f64,
    ) -> Result<Self, PipelineError> {
        if width == 0 || height == 0 {
            return Err(PipelineError::Adapter {
                component: "decoder",
                message: "decoded dimensions must be non-zero".into(),
            });
        }
        if !laplacian_variance.is_finite() || laplacian_variance < 0.0 {
            return Err(PipelineError::Adapter {
                component: "decoder",
                message: "laplacian variance must be finite and non-negative".into(),
            });
        }
        Ok(Self {
            path,
            width,
            height,
            laplacian_variance,
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
    pub fn width(&self) -> u32 {
        self.width
    }
    pub fn height(&self) -> u32 {
        self.height
    }
    pub fn laplacian_variance(&self) -> f64 {
        self.laplacian_variance
    }
}

pub trait FrameDecoder: Send + Sync {
    fn decode(&self, index: u64, path: &Path) -> Result<DecodedFrame, PipelineError>;
}

pub trait FaceDetector: Send + Sync {
    fn detect(&self, frame: &DecodedFrame) -> Result<Vec<FaceDetection>, PipelineError>;
}

pub trait LandmarkPredictor: Send + Sync {
    fn predict(
        &self,
        frame: &DecodedFrame,
        face: &FaceDetection,
    ) -> Result<PFLDLandmarks, PipelineError>;
}

#[derive(Debug, Clone, PartialEq)]
pub struct AcceptedFrame {
    index: u64,
    frame_path: PathBuf,
    landmark_bytes: Vec<u8>,
    face_score: f32,
    bbox: [f32; 4],
    blur_variance: f64,
}

impl AcceptedFrame {
    pub fn index(&self) -> u64 {
        self.index
    }
    pub fn frame_path(&self) -> &Path {
        &self.frame_path
    }
    pub fn landmark_bytes(&self) -> &[u8] {
        &self.landmark_bytes
    }
    pub fn landmark_bytes_mut(&mut self) -> &mut Vec<u8> {
        &mut self.landmark_bytes
    }
    pub fn face_score(&self) -> f32 {
        self.face_score
    }
    pub fn bbox(&self) -> [f32; 4] {
        self.bbox
    }
    pub fn blur_variance(&self) -> f64 {
        self.blur_variance
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct FrameEvaluation {
    accepted: Vec<AcceptedFrame>,
    anomalies: Vec<FrameAnomaly>,
}

impl FrameEvaluation {
    pub fn from_parts(accepted: Vec<AcceptedFrame>, anomalies: Vec<FrameAnomaly>) -> Self {
        Self {
            accepted,
            anomalies,
        }
    }
    pub fn accepted(&self) -> &[AcceptedFrame] {
        &self.accepted
    }
    pub fn anomalies(&self) -> &[FrameAnomaly] {
        &self.anomalies
    }
    pub fn is_success(&self) -> bool {
        self.anomalies.is_empty()
    }
}

/// Evaluates every extracted frame with no observer.
pub fn evaluate_frames_with_models<D, F, L>(
    batch: &FrameBatch,
    decoder: &D,
    detector: &F,
    predictor: &L,
) -> Result<FrameEvaluation, PipelineError>
where
    D: FrameDecoder + ?Sized,
    F: FaceDetector + ?Sized,
    L: LandmarkPredictor + ?Sized,
{
    evaluate_frames_observed(batch, decoder, detector, predictor, &NoObserver)
}

/// Evaluates every extracted frame, reporting one phase before each frame and
/// stopping before the next frame once the observer reports cancellation.
pub fn evaluate_frames_observed<D, F, L>(
    batch: &FrameBatch,
    decoder: &D,
    detector: &F,
    predictor: &L,
    observer: &dyn PipelineObserver,
) -> Result<FrameEvaluation, PipelineError>
where
    D: FrameDecoder + ?Sized,
    F: FaceDetector + ?Sized,
    L: LandmarkPredictor + ?Sized,
{
    let mut accepted = Vec::new();
    let mut anomalies = Vec::new();
    let total = batch.frames().len() as u64;
    for (completed, extracted) in batch.frames().iter().enumerate() {
        if observer.is_cancelled() {
            return Err(PipelineError::Cancelled {
                operation: "evaluate_frames",
            });
        }
        observer.phase(PipelinePhase::Evaluating {
            completed: completed as u64,
            total,
        });
        let index = extracted.index();
        let frame = match decoder.decode(index, extracted.path()) {
            Ok(frame) => frame,
            Err(error) => {
                anomalies.push(FrameAnomaly::new(
                    index,
                    AnomalyCode::FrameDecodeFailed,
                    "Frame decoding failed",
                    error.to_string(),
                    RecoveryAction::RerunFrame,
                )?);
                continue;
            }
        };
        let detections = match detector.detect(&frame) {
            Ok(detections) => detections,
            Err(error) => {
                anomalies.push(FrameAnomaly::new(
                    index,
                    AnomalyCode::ModelFailed,
                    "Face detection failed",
                    error.to_string(),
                    RecoveryAction::RerunFrame,
                )?);
                continue;
            }
        };
        let primary = match choose_primary(&detections, frame.width(), frame.height())? {
            PrimaryDecision::NoFace => {
                anomalies.push(FrameAnomaly::new(
                    index,
                    AnomalyCode::FaceNotFound,
                    "No face was detected",
                    "no detection met confidence threshold",
                    RecoveryAction::ExcludeFrame,
                )?);
                continue;
            }
            PrimaryDecision::Multiple => {
                anomalies.push(FrameAnomaly::new(
                    index,
                    AnomalyCode::MultipleFaces,
                    "Multiple faces were detected",
                    "more than one face remained after NMS",
                    RecoveryAction::ExcludeFrame,
                )?);
                continue;
            }
            PrimaryDecision::One(face) => face,
            PrimaryDecision::InvalidBbox(detail) => {
                anomalies.push(FrameAnomaly::new(
                    index,
                    AnomalyCode::BboxOutOfBounds,
                    "Face bounding box is outside the frame",
                    detail,
                    RecoveryAction::RerunFrame,
                )?);
                continue;
            }
        };
        let bbox_ratio = intersection_ratio(primary.bbox, frame.width(), frame.height());
        if bbox_ratio < MIN_BBOX_INTERSECTION_RATIO {
            anomalies.push(FrameAnomaly::new(
                index,
                AnomalyCode::BboxOutOfBounds,
                "Face bounding box is outside the frame",
                format!("intersection_ratio={bbox_ratio:.6}"),
                RecoveryAction::RerunFrame,
            )?);
            continue;
        }
        let landmarks = match predictor.predict(&frame, &primary) {
            Ok(landmarks) => landmarks,
            Err(error) => {
                anomalies.push(FrameAnomaly::new(
                    index,
                    AnomalyCode::ModelFailed,
                    "Landmark model failed",
                    error.to_string(),
                    RecoveryAction::RerunFrame,
                )?);
                continue;
            }
        };
        let landmark_bytes = match serialize_landmarks(&landmarks, frame.width(), frame.height()) {
            Ok(bytes) => bytes,
            Err(detail) => {
                anomalies.push(FrameAnomaly::new(
                    index,
                    AnomalyCode::LandmarkInvalid,
                    "Landmarks are invalid",
                    detail,
                    RecoveryAction::RerunFrame,
                )?);
                continue;
            }
        };
        if frame.laplacian_variance() < BLUR_VARIANCE_THRESHOLD {
            anomalies.push(FrameAnomaly::new(
                index,
                AnomalyCode::BlurredFrame,
                "Frame is too blurry",
                format!("laplacian_variance={:.6}", frame.laplacian_variance()),
                RecoveryAction::ExcludeFrame,
            )?);
            continue;
        }
        accepted.push(AcceptedFrame {
            index,
            frame_path: extracted.path().to_owned(),
            landmark_bytes,
            face_score: primary.score,
            bbox: primary.bbox,
            blur_variance: frame.laplacian_variance(),
        });
    }
    Ok(FrameEvaluation {
        accepted,
        anomalies,
    })
}

enum PrimaryDecision {
    NoFace,
    Multiple,
    One(FaceDetection),
    InvalidBbox(String),
}

fn choose_primary(
    detections: &[FaceDetection],
    width: u32,
    height: u32,
) -> Result<PrimaryDecision, PipelineError> {
    let mut candidates = Vec::new();
    for (index, detection) in detections.iter().enumerate() {
        if !detection.score.is_finite() || detection.bbox.iter().any(|value| !value.is_finite()) {
            return Ok(PrimaryDecision::InvalidBbox(format!(
                "non-finite detection at index {index}"
            )));
        }
        if detection.bbox[2] <= 0.0 || detection.bbox[3] <= 0.0 {
            return Ok(PrimaryDecision::InvalidBbox(format!(
                "non-positive detection at index {index}"
            )));
        }
        if detection
            .keypoints
            .iter()
            .flatten()
            .any(|value| !value.is_finite())
        {
            return Err(PipelineError::Adapter {
                component: "scrfd",
                message: format!("non-finite keypoint at index {index}"),
            });
        }
        if detection.score >= FACE_CONFIDENCE_THRESHOLD {
            candidates.push((index, *detection));
        }
    }
    if candidates.is_empty() {
        return Ok(PrimaryDecision::NoFace);
    }
    candidates.sort_by(|(left_index, left), (right_index, right)| {
        right
            .score
            .total_cmp(&left.score)
            .then_with(|| left_index.cmp(right_index))
    });
    let raw: Vec<Detection> = candidates
        .iter()
        .map(|(_, detection)| Detection {
            bbox: detection.bbox,
            score: detection.score,
            keypoints: detection.keypoints,
        })
        .collect();
    let kept = non_max_suppression(
        &raw,
        &DetectionConfig {
            confidence_threshold: FACE_CONFIDENCE_THRESHOLD,
            nms_iou_threshold: NMS_IOU_THRESHOLD,
        },
    )
    .map_err(|error| PipelineError::Adapter {
        component: "scrfd",
        message: error.to_string(),
    })?;
    if kept.len() != 1 {
        return Ok(if kept.is_empty() {
            PrimaryDecision::NoFace
        } else {
            PrimaryDecision::Multiple
        });
    }
    let selected = candidates[kept[0]].1;
    if selected.bbox[0] + selected.bbox[2] <= -f32::EPSILON
        || selected.bbox[1] + selected.bbox[3] <= -f32::EPSILON
        || selected.bbox[0] >= width as f32
        || selected.bbox[1] >= height as f32
    {
        return Ok(PrimaryDecision::InvalidBbox(
            "bounding box does not intersect image".into(),
        ));
    }
    Ok(PrimaryDecision::One(selected))
}

/// Fraction of the detection box that falls inside the frame.
///
/// The denominator is the box area, not the frame area, so the result answers
/// "how much of this face did we actually capture?" independently of frame size.
fn intersection_ratio(bbox: [f32; 4], width: u32, height: u32) -> f32 {
    let x1 = bbox[0].max(0.0);
    let y1 = bbox[1].max(0.0);
    let x2 = (bbox[0] + bbox[2]).min(width as f32);
    let y2 = (bbox[1] + bbox[3]).min(height as f32);
    if x2 <= x1 || y2 <= y1 {
        return 0.0;
    }
    let area = bbox[2] * bbox[3];
    if area <= 0.0 {
        return 0.0;
    }
    ((x2 - x1) * (y2 - y1)) / area
}

fn serialize_landmarks(
    landmarks: &PFLDLandmarks,
    width: u32,
    height: u32,
) -> Result<Vec<u8>, String> {
    if landmarks.points().len() != LANDMARK_POINTS {
        return Err(format!(
            "expected {LANDMARK_POINTS} points, got {}",
            landmarks.points().len()
        ));
    }
    let mut bytes = Vec::with_capacity(LANDMARK_POINTS * 16);
    for point in landmarks.points() {
        if point.x < 0 || point.y < 0 || point.x >= width as i32 || point.y >= height as i32 {
            return Err(format!(
                "point ({}, {}) outside {width}x{height}",
                point.x, point.y
            ));
        }
        bytes.extend_from_slice(format!("{} {}\n", point.x, point.y).as_bytes());
    }
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::intersection_ratio;

    const WIDTH: u32 = 640;
    const HEIGHT: u32 = 480;

    #[test]
    fn a_box_fully_inside_the_frame_is_entirely_visible() {
        assert_eq!(
            intersection_ratio([10.0, 10.0, 100.0, 100.0], WIDTH, HEIGHT),
            1.0
        );
    }

    #[test]
    fn a_box_half_past_the_right_edge_is_half_visible() {
        assert_eq!(
            intersection_ratio([590.0, 10.0, 100.0, 100.0], WIDTH, HEIGHT),
            0.5
        );
    }

    #[test]
    fn a_box_past_the_right_edge_is_invisible() {
        assert_eq!(
            intersection_ratio([700.0, 10.0, 100.0, 100.0], WIDTH, HEIGHT),
            0.0
        );
    }

    #[test]
    fn a_box_past_the_left_edge_is_invisible() {
        assert_eq!(
            intersection_ratio([-100.0, 10.0, 100.0, 100.0], WIDTH, HEIGHT),
            0.0
        );
    }

    #[test]
    fn a_degenerate_box_is_invisible() {
        assert_eq!(
            intersection_ratio([10.0, 10.0, 0.0, 100.0], WIDTH, HEIGHT),
            0.0
        );
        assert_eq!(
            intersection_ratio([10.0, 10.0, 100.0, 0.0], WIDTH, HEIGHT),
            0.0
        );
    }
}
