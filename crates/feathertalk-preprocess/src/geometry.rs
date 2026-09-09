use crate::{Landmarks, PreprocessError};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FaceBoundingBox {
    pub xmin: i32,
    pub ymin: i32,
    pub xmax: i32,
    pub ymax: i32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MaskRect {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CropSpec {
    pub crop_size: u32,
    pub inner_size: u32,
    pub border: u32,
    pub mouth_mask: MaskRect,
}

pub fn compute_face_bbox(landmarks: &Landmarks) -> Result<FaceBoundingBox, PreprocessError> {
    let points = landmarks.points();
    let xmin = points[1].x as i32;
    let ymin = points[52].y as i32;
    let xmax = points[31].x as i32;
    let width = xmax - xmin;
    if width <= 0 {
        return Err(PreprocessError::InvalidGeometry {
            field: "face_bbox",
            message: "xmax must be greater than xmin".into(),
        });
    }
    let ymax = ymin
        .checked_add(width)
        .ok_or_else(|| PreprocessError::InvalidGeometry {
            field: "face_bbox",
            message: "ymax overflow".into(),
        })?;
    Ok(FaceBoundingBox {
        xmin,
        ymin,
        xmax,
        ymax,
    })
}

pub fn default_crop_spec() -> CropSpec {
    CropSpec {
        crop_size: 168,
        inner_size: 160,
        border: 4,
        mouth_mask: MaskRect {
            x: 5,
            y: 5,
            width: 150,
            height: 145,
        },
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MouthRoiSpec {
    pub start: usize,
    pub end: usize,
    pub expand_x: f32,
    pub expand_y: f32,
    pub min_w: u32,
    pub min_h: u32,
    pub pad: u32,
}

pub fn default_mouth_roi_spec() -> MouthRoiSpec {
    MouthRoiSpec {
        start: 90,
        end: 110,
        expand_x: 1.45,
        expand_y: 1.75,
        min_w: 52,
        min_h: 36,
        pad: 2,
    }
}

pub fn mouth_roi_rect(
    landmarks: &Landmarks,
    crop: &CropSpec,
    spec: &MouthRoiSpec,
) -> Result<MaskRect, PreprocessError> {
    if spec.start >= spec.end {
        return Err(invalid_geometry(
            "mouth_roi_range",
            "start must be smaller than end",
        ));
    }
    let points = landmarks.points();
    if spec.end > points.len() {
        return Err(invalid_geometry(
            "mouth_roi_range",
            "end exceeds the landmark count",
        ));
    }
    if !spec.expand_x.is_finite()
        || spec.expand_x <= 0.0
        || !spec.expand_y.is_finite()
        || spec.expand_y <= 0.0
    {
        return Err(invalid_geometry(
            "mouth_roi_expand",
            "expansion factors must be finite and positive",
        ));
    }
    if crop.inner_size == 0 {
        return Err(invalid_geometry(
            "inner_size",
            "inner crop size must be positive",
        ));
    }
    if spec.min_w == 0
        || spec.min_h == 0
        || spec.min_w > crop.inner_size
        || spec.min_h > crop.inner_size
    {
        return Err(invalid_geometry(
            "mouth_roi_min_size",
            "minimum extents must fit inside the inner crop",
        ));
    }
    let bbox = compute_face_bbox(landmarks)?;
    let scale = crop.crop_size as f32 / (bbox.xmax - bbox.xmin) as f32;
    if !scale.is_finite() || scale <= 0.0 {
        return Err(invalid_geometry(
            "mouth_roi_projection",
            "crop scale must be finite and positive",
        ));
    }
    let border = crop.border as f32;
    let mut min_x = f32::INFINITY;
    let mut max_x = f32::NEG_INFINITY;
    let mut min_y = f32::INFINITY;
    let mut max_y = f32::NEG_INFINITY;
    for point in &points[spec.start..spec.end] {
        let x = (point.x.trunc() - bbox.xmin as f32) * scale - border;
        let y = (point.y.trunc() - bbox.ymin as f32) * scale - border;
        if !x.is_finite() || !y.is_finite() {
            return Err(invalid_geometry(
                "mouth_roi_projection",
                "projected landmark is not finite",
            ));
        }
        min_x = min_x.min(x);
        max_x = max_x.max(x);
        min_y = min_y.min(y);
        max_y = max_y.max(y);
    }
    let (x, width) = extent(
        min_x,
        max_x,
        spec.pad,
        spec.expand_x,
        spec.min_w,
        crop.inner_size,
    );
    let (y, height) = extent(
        min_y,
        max_y,
        spec.pad,
        spec.expand_y,
        spec.min_h,
        crop.inner_size,
    );
    Ok(MaskRect {
        x,
        y,
        width,
        height,
    })
}

fn extent(
    low: f32,
    high: f32,
    pad: u32,
    expand: f32,
    min_extent: u32,
    inner_size: u32,
) -> (u32, u32) {
    let center = (low + high) / 2.0;
    let size = ((high - low + 2.0 * pad as f32) * expand).max(min_extent as f32);
    let start = (center - size / 2.0).round_ties_even() as i64;
    let end = (center + size / 2.0).round_ties_even() as i64;
    let inner = i64::from(inner_size);
    let start = start.clamp(0, inner - 1);
    let end = (start + 1).max(end.min(inner));
    (start as u32, (end - start) as u32)
}

fn invalid_geometry(field: &'static str, message: impl Into<String>) -> PreprocessError {
    PreprocessError::InvalidGeometry {
        field,
        message: message.into(),
    }
}
