use crate::{FaceError, ImageSize};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RectI {
    pub x: i32,
    pub y: i32,
    pub width: u32,
    pub height: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Padding {
    pub left: u32,
    pub top: u32,
    pub right: u32,
    pub bottom: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FaceCropGeometry {
    pub requested: RectI,
    pub source: RectI,
    pub padding: Padding,
    pub size: u32,
    pub origin_x: i32,
    pub origin_y: i32,
}

pub fn compute_face_crop_geometry(
    image: ImageSize,
    bbox: [f32; 4],
) -> Result<FaceCropGeometry, FaceError> {
    if image.width == 0 || image.height == 0 {
        return Err(FaceError::InvalidImageSize);
    }
    for (index, value) in bbox.iter().enumerate() {
        if !value.is_finite() {
            return Err(FaceError::NonFiniteValue {
                level: 0,
                field: "bbox",
                index,
            });
        }
    }
    if bbox[2] <= 0.0 || bbox[3] <= 0.0 {
        return Err(FaceError::InvalidCropGeometry {
            field: "bbox",
            message: "width and height must be positive".into(),
        });
    }

    let x1 = trunc_i32(f64::from(bbox[0]), "x1")?;
    let y1 = trunc_i32(f64::from(bbox[1]), "y1")?;
    let x2 = trunc_i32(f64::from(bbox[0] + bbox[2]), "x2")?;
    let y2 = trunc_i32(f64::from(bbox[1] + bbox[3]), "y2")?;
    let width = i64::from(x2) - i64::from(x1);
    let height = i64::from(y2) - i64::from(y1);
    if width <= 0 || height <= 0 {
        return Err(FaceError::InvalidCropGeometry {
            field: "bbox",
            message: "integer edges must define a positive rectangle".into(),
        });
    }

    let max_dimension = width.max(height);
    let size_i64 = (max_dimension as f64 * 1.05).trunc();
    if !size_i64.is_finite() || size_i64 < 1.0 || size_i64 > f64::from(u32::MAX) {
        return Err(FaceError::InvalidCropGeometry {
            field: "size",
            message: "expanded crop size is outside u32 range".into(),
        });
    }
    let size = size_i64 as u32;
    let center_x = (i64::from(x1) + i64::from(x2)).div_euclid(2);
    let center_y = (i64::from(y1) + i64::from(y2)).div_euclid(2);
    let origin_x = center_x - i64::from(size / 2);
    let origin_y = center_y - i64::from(size / 2);
    let requested = RectI {
        x: checked_i32(origin_x, "origin_x")?,
        y: checked_i32(origin_y, "origin_y")?,
        width: size,
        height: size,
    };

    let requested_right = origin_x + i64::from(size);
    let requested_bottom = origin_y + i64::from(size);
    let image_right = i64::from(image.width);
    let image_bottom = i64::from(image.height);
    let source_x = origin_x.max(0).min(image_right);
    let source_y = origin_y.max(0).min(image_bottom);
    let source_right = requested_right.max(0).min(image_right);
    let source_bottom = requested_bottom.max(0).min(image_bottom);
    let source_width = source_right - source_x;
    let source_height = source_bottom - source_y;
    if source_width <= 0 || source_height <= 0 {
        return Err(FaceError::InvalidCropGeometry {
            field: "source",
            message: "requested crop does not intersect image".into(),
        });
    }
    let padding = Padding {
        left: u32::try_from((0_i64).max(-origin_x)).map_err(|_| crop_overflow("left"))?,
        top: u32::try_from((0_i64).max(-origin_y)).map_err(|_| crop_overflow("top"))?,
        right: u32::try_from((0_i64).max(requested_right - image_right))
            .map_err(|_| crop_overflow("right"))?,
        bottom: u32::try_from((0_i64).max(requested_bottom - image_bottom))
            .map_err(|_| crop_overflow("bottom"))?,
    };
    let source = RectI {
        x: checked_i32(source_x, "source_x")?,
        y: checked_i32(source_y, "source_y")?,
        width: u32::try_from(source_width).map_err(|_| crop_overflow("source_width"))?,
        height: u32::try_from(source_height).map_err(|_| crop_overflow("source_height"))?,
    };
    Ok(FaceCropGeometry {
        requested,
        source,
        padding,
        size,
        origin_x: requested.x,
        origin_y: requested.y,
    })
}

fn trunc_i32(value: f64, field: &'static str) -> Result<i32, FaceError> {
    let truncated = value.trunc();
    if !truncated.is_finite() || truncated < f64::from(i32::MIN) || truncated > f64::from(i32::MAX)
    {
        return Err(FaceError::InvalidCropGeometry {
            field,
            message: "coordinate is outside i32 range".into(),
        });
    }
    Ok(truncated as i32)
}

fn checked_i32(value: i64, field: &'static str) -> Result<i32, FaceError> {
    i32::try_from(value).map_err(|_| crop_overflow(field))
}

fn crop_overflow(field: &'static str) -> FaceError {
    FaceError::InvalidCropGeometry {
        field,
        message: "value is outside supported range".into(),
    }
}
