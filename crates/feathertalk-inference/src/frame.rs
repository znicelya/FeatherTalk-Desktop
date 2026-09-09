use crate::InferenceError;
use feathertalk_preprocess::FaceBoundingBox;

const UNET_INPUT_CHANNELS: usize = 6;
const UNET_OUTPUT_CHANNELS: usize = 3;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BgrFrame {
    width: u32,
    height: u32,
    bgr: Vec<u8>,
}

impl BgrFrame {
    pub fn new(width: u32, height: u32, bgr: Vec<u8>) -> Result<Self, InferenceError> {
        if width == 0 || height == 0 {
            return Err(InferenceError::InvalidFrameDimensions { width, height });
        }
        let expected = checked_byte_len(width, height)?;
        if bgr.len() != expected {
            return Err(InferenceError::FrameBufferLengthMismatch {
                expected,
                actual: bgr.len(),
            });
        }
        Ok(Self { width, height, bgr })
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    pub fn as_bytes(&self) -> &[u8] {
        &self.bgr
    }

    pub fn into_bytes(self) -> Vec<u8> {
        self.bgr
    }

    pub fn pixel(&self, x: u32, y: u32) -> Result<[u8; 3], InferenceError> {
        if x >= self.width || y >= self.height {
            return Err(InferenceError::PixelOutOfRange {
                x,
                y,
                width: self.width,
                height: self.height,
            });
        }
        let offset = pixel_offset(self.width, x, y)?;
        let pixel = self
            .bgr
            .get(offset..offset + 3)
            .ok_or(InferenceError::ArithmeticOverflow)?;
        Ok([pixel[0], pixel[1], pixel[2]])
    }

    fn zeroed(width: u32, height: u32) -> Result<Self, InferenceError> {
        if width == 0 || height == 0 {
            return Err(InferenceError::InvalidFrameDimensions { width, height });
        }
        let expected = checked_byte_len(width, height)?;
        let mut bgr = Vec::new();
        bgr.try_reserve_exact(expected)
            .map_err(|_| InferenceError::AllocationFailure { bytes: expected })?;
        bgr.resize(expected, 0);
        Ok(Self { width, height, bgr })
    }

    fn pixel_offset_checked(&self, x: u32, y: u32) -> Result<usize, InferenceError> {
        pixel_offset(self.width, x, y)
    }
}

pub fn crop_bgr(frame: &BgrFrame, bbox: &FaceBoundingBox) -> Result<BgrFrame, InferenceError> {
    if bbox.xmin < 0
        || bbox.ymin < 0
        || bbox.xmax <= bbox.xmin
        || bbox.ymax <= bbox.ymin
        || i64::from(bbox.xmax) > i64::from(frame.width)
        || i64::from(bbox.ymax) > i64::from(frame.height)
    {
        return Err(InferenceError::InvalidBbox {
            xmin: bbox.xmin,
            ymin: bbox.ymin,
            xmax: bbox.xmax,
            ymax: bbox.ymax,
            frame_width: frame.width,
            frame_height: frame.height,
        });
    }
    let width =
        u32::try_from(bbox.xmax - bbox.xmin).map_err(|_| InferenceError::ArithmeticOverflow)?;
    let height =
        u32::try_from(bbox.ymax - bbox.ymin).map_err(|_| InferenceError::ArithmeticOverflow)?;
    let mut crop = BgrFrame::zeroed(width, height)?;
    let source_x = u32::try_from(bbox.xmin).map_err(|_| InferenceError::ArithmeticOverflow)?;
    let source_y = u32::try_from(bbox.ymin).map_err(|_| InferenceError::ArithmeticOverflow)?;
    let row_bytes = checked_byte_len(width, 1)?;
    for y in 0..height {
        let source_row = source_y
            .checked_add(y)
            .ok_or(InferenceError::ArithmeticOverflow)?;
        let source_offset = frame.pixel_offset_checked(source_x, source_row)?;
        let destination_offset = crop.pixel_offset_checked(0, y)?;
        crop.bgr[destination_offset..destination_offset + row_bytes]
            .copy_from_slice(&frame.bgr[source_offset..source_offset + row_bytes]);
    }
    Ok(crop)
}

pub fn resize_bilinear(
    frame: &BgrFrame,
    width: u32,
    height: u32,
) -> Result<BgrFrame, InferenceError> {
    if width == 0 || height == 0 {
        return Err(InferenceError::InvalidResizeTarget { width, height });
    }
    let mut result = BgrFrame::zeroed(width, height)?;
    let scale_x = frame.width as f32 / width as f32;
    let scale_y = frame.height as f32 / height as f32;
    let max_x = frame.width - 1;
    let max_y = frame.height - 1;
    for y in 0..height {
        let source_y = (y as f32 + 0.5) * scale_y - 0.5;
        let floor_y = source_y.floor();
        let y0 = floor_y.clamp(0.0, max_y as f32) as u32;
        let y1 = y0.saturating_add(1).min(max_y);
        let wy = (source_y - floor_y).clamp(0.0, 1.0);
        for x in 0..width {
            let source_x = (x as f32 + 0.5) * scale_x - 0.5;
            let floor_x = source_x.floor();
            let x0 = floor_x.clamp(0.0, max_x as f32) as u32;
            let x1 = x0.saturating_add(1).min(max_x);
            let wx = (source_x - floor_x).clamp(0.0, 1.0);
            let p00 = frame.pixel(x0, y0)?;
            let p01 = frame.pixel(x1, y0)?;
            let p10 = frame.pixel(x0, y1)?;
            let p11 = frame.pixel(x1, y1)?;
            let destination_offset = result.pixel_offset_checked(x, y)?;
            for channel in 0..3 {
                let top = p00[channel] as f32 * (1.0 - wx) + p01[channel] as f32 * wx;
                let bottom = p10[channel] as f32 * (1.0 - wx) + p11[channel] as f32 * wx;
                let value = (top * (1.0 - wy) + bottom * wy).clamp(0.0, 255.0).round();
                result.bgr[destination_offset + channel] = value as u8;
            }
        }
    }
    Ok(result)
}

fn checked_byte_len(width: u32, height: u32) -> Result<usize, InferenceError> {
    let width = usize::try_from(width).map_err(|_| InferenceError::ArithmeticOverflow)?;
    let height = usize::try_from(height).map_err(|_| InferenceError::ArithmeticOverflow)?;
    width
        .checked_mul(height)
        .and_then(|pixels| pixels.checked_mul(3))
        .ok_or(InferenceError::ArithmeticOverflow)
}

fn pixel_offset(width: u32, x: u32, y: u32) -> Result<usize, InferenceError> {
    let width = usize::try_from(width).map_err(|_| InferenceError::ArithmeticOverflow)?;
    let x = usize::try_from(x).map_err(|_| InferenceError::ArithmeticOverflow)?;
    let y = usize::try_from(y).map_err(|_| InferenceError::ArithmeticOverflow)?;
    y.checked_mul(width)
        .and_then(|row| row.checked_add(x))
        .and_then(|pixel| pixel.checked_mul(3))
        .ok_or(InferenceError::ArithmeticOverflow)
}

#[derive(Debug, Clone, PartialEq)]
pub struct UnetImageInput {
    values: Vec<f32>,
}

impl UnetImageInput {
    pub fn shape(&self) -> [usize; 4] {
        [1, UNET_INPUT_CHANNELS, 160, 160]
    }

    pub fn as_slice(&self) -> &[f32] {
        &self.values
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MouthMasking {
    Keep,
    Blackout,
}

#[derive(Debug, Clone, PartialEq)]
pub struct InnerImagePlanes {
    values: Vec<f32>,
}

impl InnerImagePlanes {
    pub fn shape(&self) -> [usize; 4] {
        [1, UNET_OUTPUT_CHANNELS, 160, 160]
    }

    pub fn as_slice(&self) -> &[f32] {
        &self.values
    }

    pub fn into_values(self) -> Vec<f32> {
        self.values
    }
}

pub fn build_inner_image_planes(
    face_crop: &BgrFrame,
    geometry: &crate::RenderGeometry,
    masking: MouthMasking,
) -> Result<InnerImagePlanes, InferenceError> {
    let (crop_size, inner_size, border) = validate_geometry_and_crop(face_crop, geometry)?;
    let crop_spec = feathertalk_preprocess::default_crop_spec();
    let mask_right = crop_spec
        .mouth_mask
        .x
        .checked_add(crop_spec.mouth_mask.width)
        .ok_or(InferenceError::ArithmeticOverflow)?;
    let mask_bottom = crop_spec
        .mouth_mask
        .y
        .checked_add(crop_spec.mouth_mask.height)
        .ok_or(InferenceError::ArithmeticOverflow)?;
    if mask_right > inner_size || mask_bottom > inner_size {
        return Err(InferenceError::InvalidField {
            field: "mouth_mask",
            message: "mask rectangle exceeds the inner crop".into(),
        });
    }
    let plane = checked_elements(inner_size, inner_size)?;
    let elements = plane
        .checked_mul(UNET_OUTPUT_CHANNELS)
        .ok_or(InferenceError::ArithmeticOverflow)?;
    let mut values = allocate_f32(elements)?;
    for y in 0..inner_size {
        for x in 0..inner_size {
            let source_x = x
                .checked_add(border)
                .ok_or(InferenceError::ArithmeticOverflow)?;
            let source_y = y
                .checked_add(border)
                .ok_or(InferenceError::ArithmeticOverflow)?;
            let source_offset = face_crop.pixel_offset_checked(source_x, source_y)?;
            let offset = linear_offset(inner_size, x, y)?;
            let masked = masking == MouthMasking::Blackout
                && x >= crop_spec.mouth_mask.x
                && x < mask_right
                && y >= crop_spec.mouth_mask.y
                && y < mask_bottom;
            for channel in 0..UNET_OUTPUT_CHANNELS {
                let normalized = f32::from(face_crop.bgr[source_offset + channel]) / 255.0;
                values[channel * plane + offset] = if masked { 0.0 } else { normalized };
            }
        }
    }
    debug_assert_eq!(crop_size, inner_size + 2 * border);
    Ok(InnerImagePlanes { values })
}

pub fn build_unet_image_input(
    face_crop: &BgrFrame,
    geometry: &crate::RenderGeometry,
) -> Result<UnetImageInput, InferenceError> {
    let keep = build_inner_image_planes(face_crop, geometry, MouthMasking::Keep)?;
    let blackout = build_inner_image_planes(face_crop, geometry, MouthMasking::Blackout)?;
    let mut values = keep.into_values();
    let mut tail = blackout.into_values();
    let total = values
        .len()
        .checked_add(tail.len())
        .ok_or(InferenceError::ArithmeticOverflow)?;
    values
        .try_reserve_exact(tail.len())
        .map_err(|_| InferenceError::AllocationFailure {
            bytes: total.saturating_mul(std::mem::size_of::<f32>()),
        })?;
    values.append(&mut tail);
    Ok(UnetImageInput { values })
}

pub fn apply_unet_prediction(
    face_crop: &mut BgrFrame,
    prediction: &[f32],
    geometry: &crate::RenderGeometry,
) -> Result<(), InferenceError> {
    let (_crop_size, inner_size, border) = validate_geometry_and_crop(face_crop, geometry)?;
    let plane = checked_elements(inner_size, inner_size)?;
    let expected = UNET_OUTPUT_CHANNELS
        .checked_mul(plane)
        .ok_or(InferenceError::ArithmeticOverflow)?;
    if prediction.len() != expected {
        return Err(InferenceError::TensorShapeMismatch {
            context: "unet_prediction",
            expected: vec![
                1,
                UNET_OUTPUT_CHANNELS,
                inner_size as usize,
                inner_size as usize,
            ],
            actual: vec![prediction.len()],
        });
    }
    if let Some((index, _)) = prediction
        .iter()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(InferenceError::NonFinitePrediction { index });
    }
    for y in 0..inner_size {
        for x in 0..inner_size {
            let crop_x = x
                .checked_add(border)
                .ok_or(InferenceError::ArithmeticOverflow)?;
            let crop_y = y
                .checked_add(border)
                .ok_or(InferenceError::ArithmeticOverflow)?;
            let destination_offset = face_crop.pixel_offset_checked(crop_x, crop_y)?;
            let offset = linear_offset(inner_size, x, y)?;
            for channel in 0..3 {
                let value = (prediction[channel * plane + offset] * 255.0)
                    .clamp(0.0, 255.0)
                    .round();
                face_crop.bgr[destination_offset + channel] = value as u8;
            }
        }
    }
    Ok(())
}

pub fn paste_bgr(
    destination: &mut BgrFrame,
    source: &BgrFrame,
    x: i32,
    y: i32,
) -> Result<(), InferenceError> {
    if x < 0 || y < 0 {
        return Err(InferenceError::PasteOutOfBounds {
            x,
            y,
            source_width: source.width,
            source_height: source.height,
            destination_width: destination.width,
            destination_height: destination.height,
        });
    }
    let x = u32::try_from(x).map_err(|_| InferenceError::ArithmeticOverflow)?;
    let y = u32::try_from(y).map_err(|_| InferenceError::ArithmeticOverflow)?;
    if x > destination.width
        || y > destination.height
        || source.width > destination.width.saturating_sub(x)
        || source.height > destination.height.saturating_sub(y)
    {
        return Err(InferenceError::PasteOutOfBounds {
            x: i32::try_from(x).unwrap_or(i32::MAX),
            y: i32::try_from(y).unwrap_or(i32::MAX),
            source_width: source.width,
            source_height: source.height,
            destination_width: destination.width,
            destination_height: destination.height,
        });
    }
    let row_bytes = checked_byte_len(source.width, 1)?;
    for row in 0..source.height {
        let source_offset = source.pixel_offset_checked(0, row)?;
        let destination_y = y
            .checked_add(row)
            .ok_or(InferenceError::ArithmeticOverflow)?;
        let destination_offset = destination.pixel_offset_checked(x, destination_y)?;
        destination.bgr[destination_offset..destination_offset + row_bytes]
            .copy_from_slice(&source.bgr[source_offset..source_offset + row_bytes]);
    }
    Ok(())
}

pub fn build_face_crop(
    frame: &BgrFrame,
    bbox: &FaceBoundingBox,
    geometry: &crate::RenderGeometry,
) -> Result<BgrFrame, InferenceError> {
    let source_crop = crop_bgr(frame, bbox)?;
    resize_bilinear(&source_crop, geometry.crop_size(), geometry.crop_size())
}

pub fn render_frame(
    frame: &BgrFrame,
    bbox: &FaceBoundingBox,
    prediction: &[f32],
    geometry: &crate::RenderGeometry,
) -> Result<BgrFrame, InferenceError> {
    let mut face_crop = build_face_crop(frame, bbox, geometry)?;
    apply_unet_prediction(&mut face_crop, prediction, geometry)?;
    let bbox_width =
        u32::try_from(bbox.xmax - bbox.xmin).map_err(|_| InferenceError::ArithmeticOverflow)?;
    let bbox_height =
        u32::try_from(bbox.ymax - bbox.ymin).map_err(|_| InferenceError::ArithmeticOverflow)?;
    let resized = resize_bilinear(&face_crop, bbox_width, bbox_height)?;
    let mut rendered = frame.clone();
    paste_bgr(&mut rendered, &resized, bbox.xmin, bbox.ymin)?;
    Ok(rendered)
}

fn validate_geometry_and_crop(
    face_crop: &BgrFrame,
    geometry: &crate::RenderGeometry,
) -> Result<(u32, u32, u32), InferenceError> {
    let standard = crate::RenderGeometry::standard();
    let expected_geometry = vec![
        standard.crop_size() as usize,
        standard.inner_size() as usize,
        standard.border() as usize,
    ];
    let actual_geometry = vec![
        geometry.crop_size() as usize,
        geometry.inner_size() as usize,
        geometry.border() as usize,
    ];
    if geometry != &standard {
        return Err(InferenceError::TensorShapeMismatch {
            context: "render_geometry",
            expected: expected_geometry,
            actual: actual_geometry,
        });
    }
    if face_crop.width() != geometry.crop_size() || face_crop.height() != geometry.crop_size() {
        return Err(InferenceError::TensorShapeMismatch {
            context: "face_crop",
            expected: vec![
                1,
                geometry.crop_size() as usize,
                geometry.crop_size() as usize,
            ],
            actual: vec![1, face_crop.height() as usize, face_crop.width() as usize],
        });
    }
    Ok((
        geometry.crop_size(),
        geometry.inner_size(),
        geometry.border(),
    ))
}

fn checked_elements(width: u32, height: u32) -> Result<usize, InferenceError> {
    usize::try_from(width)
        .map_err(|_| InferenceError::ArithmeticOverflow)?
        .checked_mul(usize::try_from(height).map_err(|_| InferenceError::ArithmeticOverflow)?)
        .ok_or(InferenceError::ArithmeticOverflow)
}

fn linear_offset(width: u32, x: u32, y: u32) -> Result<usize, InferenceError> {
    let row = checked_elements(width, y)?;
    row.checked_add(usize::try_from(x).map_err(|_| InferenceError::ArithmeticOverflow)?)
        .ok_or(InferenceError::ArithmeticOverflow)
}

fn allocate_f32(elements: usize) -> Result<Vec<f32>, InferenceError> {
    let bytes = elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or(InferenceError::ArithmeticOverflow)?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(elements)
        .map_err(|_| InferenceError::AllocationFailure { bytes })?;
    values.resize(elements, 0.0);
    Ok(values)
}
