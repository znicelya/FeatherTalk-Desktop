use std::{
    fs,
    io::Cursor,
    path::{Path, PathBuf},
};

use jpeg_decoder::{Decoder, PixelFormat};

use crate::{BgrFrame, InferenceError};

const MAX_COMPRESSED_BYTES: u64 = 16 * 1024 * 1024;
pub const DEFAULT_MAX_FRAME_PIXELS: u64 = 64 * 1024 * 1024;

pub trait FrameReader: Send + Sync {
    fn read(&self, index: usize, path: &Path) -> Result<BgrFrame, InferenceError>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct JpegFrameReader {
    max_pixels: u64,
}

impl JpegFrameReader {
    pub const fn new(max_pixels: u64) -> Self {
        Self { max_pixels }
    }

    pub const fn max_pixels(&self) -> u64 {
        self.max_pixels
    }
}

impl Default for JpegFrameReader {
    fn default() -> Self {
        Self::new(DEFAULT_MAX_FRAME_PIXELS)
    }
}

impl FrameReader for JpegFrameReader {
    fn read(&self, index: usize, path: &Path) -> Result<BgrFrame, InferenceError> {
        if self.max_pixels == 0 {
            return Err(reader_error(
                index,
                path,
                "maximum decoded pixel count must be greater than zero",
            ));
        }
        let metadata = fs::symlink_metadata(path).map_err(|error| {
            reader_error(index, path, format!("unable to inspect input: {error}"))
        })?;
        if metadata.file_type().is_symlink() || !metadata.is_file() {
            return Err(reader_error(
                index,
                path,
                "input must be a regular non-symlink file",
            ));
        }
        if metadata.len() > MAX_COMPRESSED_BYTES {
            return Err(reader_error(
                index,
                path,
                format!("compressed JPEG exceeds {MAX_COMPRESSED_BYTES} byte limit"),
            ));
        }
        let bytes = fs::read(path)
            .map_err(|error| reader_error(index, path, format!("unable to read input: {error}")))?;
        let max_decode_bytes = self
            .max_pixels
            .checked_mul(3)
            .and_then(|bytes| usize::try_from(bytes).ok())
            .ok_or_else(|| reader_error(index, path, "decoded buffer size overflows usize"))?;
        let mut decoder = Decoder::new(Cursor::new(bytes));
        decoder.set_max_decoding_buffer_size(max_decode_bytes);
        let pixels = decoder
            .decode()
            .map_err(|error| reader_error(index, path, format!("JPEG decode failed: {error}")))?;
        let info = decoder.info().ok_or_else(|| {
            reader_error(index, path, "JPEG decoder returned no image information")
        })?;
        if info.pixel_format != PixelFormat::RGB24 {
            return Err(reader_error(
                index,
                path,
                format!("unsupported JPEG pixel format: {:?}", info.pixel_format),
            ));
        }
        let width = u32::from(info.width);
        let height = u32::from(info.height);
        if width == 0 || height == 0 {
            return Err(reader_error(
                index,
                path,
                "decoded dimensions must be non-zero",
            ));
        }
        let pixel_count = u64::from(width)
            .checked_mul(u64::from(height))
            .ok_or_else(|| reader_error(index, path, "decoded pixel count overflows u64"))?;
        if pixel_count > self.max_pixels {
            return Err(reader_error(
                index,
                path,
                format!(
                    "decoded pixel count {pixel_count} exceeds limit {}",
                    self.max_pixels
                ),
            ));
        }
        let expected = pixel_count
            .checked_mul(3)
            .and_then(|bytes| usize::try_from(bytes).ok())
            .ok_or_else(|| reader_error(index, path, "decoded buffer length overflows usize"))?;
        if pixels.len() != expected {
            return Err(reader_error(
                index,
                path,
                format!(
                    "decoded RGB buffer length mismatch: expected {expected}, got {}",
                    pixels.len()
                ),
            ));
        }
        let mut bgr = Vec::with_capacity(expected);
        for rgb in pixels.chunks_exact(3) {
            bgr.extend_from_slice(&[rgb[2], rgb[1], rgb[0]]);
        }
        BgrFrame::new(width, height, bgr)
            .map_err(|error| reader_error(index, path, format!("BGR frame rejected: {error}")))
    }
}

fn reader_error(index: usize, path: &Path, message: impl Into<String>) -> InferenceError {
    InferenceError::FrameReader {
        index,
        path: PathBuf::from(path),
        message: message.into(),
    }
}
