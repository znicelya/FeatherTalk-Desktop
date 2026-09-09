use crate::{Detection, FaceError};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DetectionConfig {
    pub confidence_threshold: f32,
    pub nms_iou_threshold: f32,
}

impl Default for DetectionConfig {
    fn default() -> Self {
        Self {
            confidence_threshold: 0.1,
            nms_iou_threshold: 0.5,
        }
    }
}

pub fn non_max_suppression(
    detections: &[Detection],
    config: &DetectionConfig,
) -> Result<Vec<usize>, FaceError> {
    validate_threshold("confidence_threshold", config.confidence_threshold)?;
    validate_threshold("nms_iou_threshold", config.nms_iou_threshold)?;

    let mut candidates = Vec::with_capacity(detections.len());
    for (index, detection) in detections.iter().enumerate() {
        if !detection.score.is_finite() {
            return Err(FaceError::NonFiniteValue {
                level: 0,
                field: "scores",
                index,
            });
        }
        if detection.bbox.iter().any(|value| !value.is_finite()) {
            return Err(FaceError::NonFiniteValue {
                level: 0,
                field: "bbox",
                index,
            });
        }
        if detection.bbox[2] <= 0.0 || detection.bbox[3] <= 0.0 {
            return Err(FaceError::InvalidDetectionGeometry { index });
        }
        if detection.score >= config.confidence_threshold {
            candidates.push((index, detection));
        }
    }

    candidates.sort_by(|(left_index, left), (right_index, right)| {
        right
            .score
            .total_cmp(&left.score)
            .then_with(|| left_index.cmp(right_index))
    });

    let mut kept: Vec<usize> = Vec::with_capacity(candidates.len());
    for &(index, candidate) in &candidates {
        if kept.iter().all(|&kept_index| {
            let kept_detection = &detections[kept_index];
            iou(candidate.bbox, kept_detection.bbox) <= config.nms_iou_threshold
        }) {
            kept.push(index);
        }
    }
    Ok(kept)
}

fn validate_threshold(field: &'static str, value: f32) -> Result<(), FaceError> {
    if !value.is_finite() || !(0.0..=1.0).contains(&value) {
        return Err(FaceError::InvalidConfiguration {
            field,
            message: "must be finite and within [0, 1]".into(),
        });
    }
    Ok(())
}

fn iou(left: [f32; 4], right: [f32; 4]) -> f32 {
    let left_x2 = left[0] + left[2];
    let left_y2 = left[1] + left[3];
    let right_x2 = right[0] + right[2];
    let right_y2 = right[1] + right[3];
    let intersection_width = (left_x2.min(right_x2) - left[0].max(right[0])).max(0.0);
    let intersection_height = (left_y2.min(right_y2) - left[1].max(right[1])).max(0.0);
    let intersection = intersection_width * intersection_height;
    let union = left[2] * left[3] + right[2] * right[3] - intersection;
    if union > 0.0 {
        intersection / union
    } else {
        0.0
    }
}
