use crate::FaceError;

const MODEL_SIZE: ImageSize = ImageSize {
    width: 640,
    height: 640,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImageSize {
    pub width: u32,
    pub height: u32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ResizeTransform {
    pub input: ImageSize,
    pub model: ImageSize,
    pub new_width: u32,
    pub new_height: u32,
    pub pad_x: u32,
    pub pad_y: u32,
    pub scale_x: f32,
    pub scale_y: f32,
}

pub fn resize_with_padding(input: ImageSize) -> Result<ResizeTransform, FaceError> {
    if input.width == 0 || input.height == 0 {
        return Err(FaceError::InvalidImageSize);
    }
    let (new_width, new_height) = if input.width == input.height {
        (640, 640)
    } else if input.height > input.width {
        let ratio = input.height as f64 / input.width as f64;
        ((640.0 / ratio).floor() as u32, 640)
    } else {
        let ratio = input.height as f64 / input.width as f64;
        (640, (640.0 * ratio).floor() as u32 + 1)
    };
    if new_width == 0 || new_height == 0 {
        return Err(FaceError::InvalidImageSize);
    }
    let pad_x = (640 - new_width) / 2;
    let pad_y = (640 - new_height) / 2;
    Ok(ResizeTransform {
        input,
        model: MODEL_SIZE,
        new_width,
        new_height,
        pad_x,
        pad_y,
        scale_x: input.width as f32 / new_width as f32,
        scale_y: input.height as f32 / new_height as f32,
    })
}

pub fn generate_anchor_centers(
    model: ImageSize,
    stride: u32,
    anchors_per_location: u32,
) -> Result<Vec<[f32; 2]>, FaceError> {
    if model != MODEL_SIZE {
        return Err(config("model", "must be 640x640"));
    }
    if !matches!(stride, 8 | 16 | 32) {
        return Err(config("stride", "must be 8, 16, or 32"));
    }
    if anchors_per_location != 2 {
        return Err(config("anchors_per_location", "must be 2"));
    }
    let width = model.width / stride;
    let height = model.height / stride;
    let mut anchors = Vec::with_capacity((width * height * anchors_per_location) as usize);
    for y in 0..height {
        for x in 0..width {
            let center = [(x * stride) as f32, (y * stride) as f32];
            anchors.push(center);
            anchors.push(center);
        }
    }
    Ok(anchors)
}

fn config(field: &'static str, message: &str) -> FaceError {
    FaceError::InvalidConfiguration {
        field,
        message: message.into(),
    }
}
