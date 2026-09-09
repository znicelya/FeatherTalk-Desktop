use thiserror::Error;

/// Largest edge length this crate is willing to allocate for.
///
/// Well above any video frame, low enough that `width * height * 3` cannot
/// overflow a 64-bit `usize`.
pub(crate) const MAX_EDGE: u32 = 32_768;

/// Every failure the pixel kernels can produce.
#[derive(Debug, Error)]
pub enum ImageError {
    /// A dimension was zero or beyond `MAX_EDGE`.
    #[error("invalid image dimensions: {width}x{height}")]
    InvalidDimensions { width: u32, height: u32 },

    /// A buffer did not match the geometry it was declared with.
    #[error("buffer length mismatch for {width}x{height}: expected {expected} bytes, got {actual}")]
    BufferLengthMismatch {
        width: u32,
        height: u32,
        expected: usize,
        actual: usize,
    },

    /// A pixel accessor was called outside the image.
    #[error("pixel ({x}, {y}) is outside a {width}x{height} image")]
    PixelOutOfBounds {
        x: u32,
        y: u32,
        width: u32,
        height: u32,
    },

    /// The JPEG header declares more pixels than the caller allows.
    #[error("JPEG declares {pixels} pixels, exceeding the {max_pixels} pixel budget")]
    FrameTooLarge { pixels: u64, max_pixels: u64 },

    /// The JPEG is neither 8-bit RGB nor 8-bit grayscale.
    #[error("unsupported JPEG pixel format: {format}")]
    UnsupportedPixelFormat { format: String },

    /// `jpeg-decoder` rejected the input.
    #[error("JPEG decode failed: {message}")]
    JpegDecode { message: String },

    /// A resize target was zero or beyond `MAX_EDGE`.
    #[error("invalid resize target {width}x{height}, each edge must be within 1..={max_dimension}")]
    InvalidTargetSize {
        width: u32,
        height: u32,
        max_dimension: u32,
    },
}

/// Byte length of a validated `width x height x channels` buffer.
pub(crate) fn expected_len(width: u32, height: u32, channels: usize) -> Result<usize, ImageError> {
    if width == 0 || height == 0 || width > MAX_EDGE || height > MAX_EDGE {
        return Err(ImageError::InvalidDimensions { width, height });
    }
    (width as usize)
        .checked_mul(height as usize)
        .and_then(|pixels| pixels.checked_mul(channels))
        .ok_or(ImageError::InvalidDimensions { width, height })
}

/// Byte length of a validated resize target.
///
/// The bounds are the same as `expected_len`, but a bad target is a caller
/// mistake rather than a corrupt buffer, so it reports `InvalidTargetSize`.
pub(crate) fn check_target(width: u32, height: u32, channels: usize) -> Result<usize, ImageError> {
    if width == 0 || height == 0 || width > MAX_EDGE || height > MAX_EDGE {
        return Err(ImageError::InvalidTargetSize {
            width,
            height,
            max_dimension: MAX_EDGE,
        });
    }
    expected_len(width, height, channels)
}
