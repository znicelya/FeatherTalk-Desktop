use crate::{MEAN_FACE, MeanFace, PfldError};

pub const PFLD_OUTPUT_VALUE_COUNT: usize = 220;
pub const PFLD_LANDMARK_COUNT: usize = 110;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CropGeometry {
    pub width: u32,
    pub height: u32,
    pub offset_x: i32,
    pub offset_y: i32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LandmarkPoint {
    pub x: i32,
    pub y: i32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PFLDLandmarks {
    points: Vec<LandmarkPoint>,
}

impl PFLDLandmarks {
    pub fn points(&self) -> &[LandmarkPoint] {
        &self.points
    }
}

pub fn decode_landmarks(
    model_output: &[f32],
    mean_face: &[f32],
    crop: CropGeometry,
) -> Result<PFLDLandmarks, PfldError> {
    validate_length("model_output", model_output.len())?;
    validate_length("mean_face", mean_face.len())?;
    validate_values("model_output", model_output)?;
    validate_values("mean_face", mean_face)?;
    if crop.width == 0 || crop.height == 0 {
        return Err(PfldError::InvalidCropGeometry);
    }

    let mut points = Vec::with_capacity(PFLD_LANDMARK_COUNT);
    for index in 0..PFLD_LANDMARK_COUNT {
        let value_index = index * 2;
        let normalized_x = model_output[value_index] + mean_face[value_index];
        let normalized_y = model_output[value_index + 1] + mean_face[value_index + 1];
        let x = map_coordinate(normalized_x, crop.width, crop.offset_x, index, "x")?;
        let y = map_coordinate(normalized_y, crop.height, crop.offset_y, index, "y")?;
        points.push(LandmarkPoint { x, y });
    }
    Ok(PFLDLandmarks { points })
}

pub fn decode_landmarks_with_mean_face(
    model_output: &[f32],
    mean_face: &MeanFace,
    crop: CropGeometry,
) -> Result<PFLDLandmarks, PfldError> {
    decode_landmarks(model_output, mean_face.values(), crop)
}

pub fn decode_landmarks_with_default_mean_face(
    model_output: &[f32],
    crop: CropGeometry,
) -> Result<PFLDLandmarks, PfldError> {
    decode_landmarks_with_mean_face(model_output, &MEAN_FACE, crop)
}

fn validate_length(field: &'static str, actual: usize) -> Result<(), PfldError> {
    if actual != PFLD_OUTPUT_VALUE_COUNT {
        return Err(PfldError::InvalidVectorLength {
            field,
            expected: PFLD_OUTPUT_VALUE_COUNT,
            actual,
        });
    }
    Ok(())
}

fn validate_values(field: &'static str, values: &[f32]) -> Result<(), PfldError> {
    if let Some(index) = values.iter().position(|value| !value.is_finite()) {
        return Err(PfldError::NonFiniteValue { field, index });
    }
    Ok(())
}

fn map_coordinate(
    normalized: f32,
    dimension: u32,
    offset: i32,
    index: usize,
    axis: &'static str,
) -> Result<i32, PfldError> {
    let scaled = f64::from(normalized) * f64::from(dimension);
    let truncated = scaled.trunc();
    if !truncated.is_finite() || truncated < f64::from(i32::MIN) || truncated > f64::from(i32::MAX)
    {
        return Err(PfldError::CoordinateOutOfRange { index, axis });
    }
    let coordinate = truncated as i64 + i64::from(offset);
    if coordinate < i64::from(i32::MIN) || coordinate > i64::from(i32::MAX) {
        return Err(PfldError::CoordinateOutOfRange { index, axis });
    }
    Ok(coordinate as i32)
}
