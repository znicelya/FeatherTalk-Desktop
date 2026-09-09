use crate::error::{ImageError, expected_len};

/// Row-major, pixel-interleaved BGR24 image.
///
/// The same memory layout as `feathertalk_inference::BgrFrame`. The two are kept
/// separate on purpose: this crate must not depend on the inference crate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BgrImage {
    width: u32,
    height: u32,
    bgr: Vec<u8>,
}

impl BgrImage {
    /// Take ownership of a buffer that is exactly `width * height * 3` bytes.
    pub fn new(width: u32, height: u32, bgr: Vec<u8>) -> Result<Self, ImageError> {
        let expected = expected_len(width, height, 3)?;
        if bgr.len() != expected {
            return Err(ImageError::BufferLengthMismatch {
                width,
                height,
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

    /// The three bytes at `(x, y)` in B, G, R order.
    pub fn pixel(&self, x: u32, y: u32) -> Result<[u8; 3], ImageError> {
        if x >= self.width || y >= self.height {
            return Err(ImageError::PixelOutOfBounds {
                x,
                y,
                width: self.width,
                height: self.height,
            });
        }
        let base = (y as usize * self.width as usize + x as usize) * 3;
        Ok([self.bgr[base], self.bgr[base + 1], self.bgr[base + 2]])
    }

    /// Row `y`, exactly `width * 3` bytes, which the resize kernels index directly.
    pub(crate) fn row(&self, y: u32) -> &[u8] {
        let stride = self.width as usize * 3;
        let start = y as usize * stride;
        &self.bgr[start..start + stride]
    }
}

/// Row-major, single-channel 8-bit image.
///
/// It exists so call sites cannot mismatch a `(&[u8], width, height)` triple.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GrayImage {
    width: u32,
    height: u32,
    pixels: Vec<u8>,
}

impl GrayImage {
    /// Width in pixels, always nonzero.
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Height in pixels, always nonzero.
    pub fn height(&self) -> u32 {
        self.height
    }

    /// Row-major pixels, exactly `width * height` bytes.
    pub fn as_bytes(&self) -> &[u8] {
        &self.pixels
    }

    /// Row `y`, which the kernels in this crate index directly.
    pub(crate) fn row(&self, y: u32) -> &[u8] {
        let width = self.width as usize;
        let start = y as usize * width;
        &self.pixels[start..start + width]
    }
}

/// Fixed-point BGR to gray, identical to `cv2.cvtColor(.., COLOR_BGR2GRAY)`.
///
/// `gray = (B * 3735 + G * 19235 + R * 9798 + 16384) >> 15`, evaluated in `u32`.
/// The weights sum to exactly 32768, so the largest possible numerator is
/// `255 * 32768 + 16384` and the result is always within `0..=255`.
pub fn to_gray(image: &BgrImage) -> GrayImage {
    const BLUE: u32 = 3_735;
    const GREEN: u32 = 19_235;
    const RED: u32 = 9_798;

    let bytes = image.as_bytes();
    let mut pixels = Vec::with_capacity(bytes.len() / 3);
    for pixel in bytes.chunks_exact(3) {
        let value = u32::from(pixel[0]) * BLUE
            + u32::from(pixel[1]) * GREEN
            + u32::from(pixel[2]) * RED
            + 16_384;
        pixels.push((value >> 15) as u8);
    }
    GrayImage {
        width: image.width(),
        height: image.height(),
        pixels,
    }
}
