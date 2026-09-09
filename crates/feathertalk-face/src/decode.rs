use crate::{FaceError, ResizeTransform};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Detection {
    pub bbox: [f32; 4],
    pub score: f32,
    pub keypoints: [[f32; 2]; 5],
}

pub fn decode_level(
    level: usize,
    stride: u32,
    anchors: &[[f32; 2]],
    scores: &[f32],
    bbox_distances: &[[f32; 4]],
    keypoint_distances: &[[f32; 10]],
    transform: &ResizeTransform,
) -> Result<Vec<Detection>, FaceError> {
    let expected = anchors.len();
    for (field, actual) in [
        ("scores", scores.len()),
        ("bbox_distances", bbox_distances.len()),
        ("keypoint_distances", keypoint_distances.len()),
    ] {
        if actual != expected {
            return Err(FaceError::InvalidTensorLength {
                level,
                field,
                expected,
                actual,
            });
        }
    }
    let mut output = Vec::with_capacity(expected);
    for index in 0..expected {
        if !scores[index].is_finite() {
            return Err(FaceError::NonFiniteValue {
                level,
                field: "scores",
                index,
            });
        }
        if bbox_distances[index].iter().any(|value| !value.is_finite()) {
            return Err(FaceError::NonFiniteValue {
                level,
                field: "bbox_distances",
                index,
            });
        }
        if keypoint_distances[index]
            .iter()
            .any(|value| !value.is_finite())
        {
            return Err(FaceError::NonFiniteValue {
                level,
                field: "keypoint_distances",
                index,
            });
        }
        let [cx, cy] = anchors[index];
        if !cx.is_finite() || !cy.is_finite() {
            return Err(FaceError::NonFiniteValue {
                level,
                field: "anchors",
                index,
            });
        }
        let [left, top, right, bottom] = bbox_distances[index];
        let mut x1 = map_x(cx - left * stride as f32, transform);
        let mut y1 = map_y(cy - top * stride as f32, transform);
        let mut x2 = map_x(cx + right * stride as f32, transform);
        let mut y2 = map_y(cy + bottom * stride as f32, transform);
        x1 = x1.clamp(0.0, transform.input.width as f32);
        y1 = y1.clamp(0.0, transform.input.height as f32);
        x2 = x2.clamp(0.0, transform.input.width as f32);
        y2 = y2.clamp(0.0, transform.input.height as f32);
        if !(x2 > x1 && y2 > y1) {
            return Err(FaceError::InvalidDetectionGeometry { index });
        }
        let mut keypoints = [[0.0; 2]; 5];
        for point in 0..5 {
            let kx = map_x(
                cx + keypoint_distances[index][point * 2] * stride as f32,
                transform,
            )
            .clamp(0.0, transform.input.width as f32);
            let ky = map_y(
                cy + keypoint_distances[index][point * 2 + 1] * stride as f32,
                transform,
            )
            .clamp(0.0, transform.input.height as f32);
            keypoints[point] = [kx, ky];
        }
        output.push(Detection {
            bbox: [x1, y1, x2 - x1, y2 - y1],
            score: scores[index],
            keypoints,
        });
    }
    Ok(output)
}

fn map_x(value: f32, transform: &ResizeTransform) -> f32 {
    (value - transform.pad_x as f32) * transform.scale_x
}
fn map_y(value: f32, transform: &ResizeTransform) -> f32 {
    (value - transform.pad_y as f32) * transform.scale_y
}
