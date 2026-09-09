use std::io::{Cursor, Read};

use jpeg_decoder::{Decoder, PixelFormat};

use crate::{BgrImage, error::ImageError};

/// Read the SOF header and reject the degenerate sizes.
///
/// Split out of `decode_jpeg` so that `jpeg_dimensions` cannot drift from it.
/// The decoder is borrowed rather than owned because `decode_jpeg` keeps using
/// it afterwards, and the pixel format comes back for the same reason.
fn read_header<R: Read>(decoder: &mut Decoder<R>) -> Result<(u32, u32, PixelFormat), ImageError> {
    decoder
        .read_info()
        .map_err(|error| ImageError::JpegDecode {
            message: error.to_string(),
        })?;
    let info = decoder.info().ok_or_else(|| ImageError::JpegDecode {
        message: "JPEG header carried no image information".to_owned(),
    })?;
    let width = u32::from(info.width);
    let height = u32::from(info.height);
    if width == 0 || height == 0 {
        return Err(ImageError::InvalidDimensions { width, height });
    }
    Ok((width, height, info.pixel_format))
}

/// Read a JPEG's pixel dimensions without decoding a single scan.
///
/// Allocates nothing beyond the decoder's header state, so it takes no pixel
/// budget: a caller that cares about size limits owns that policy. The asset
/// lock uses it to learn the frame geometry a quality report does not record.
pub fn jpeg_dimensions(bytes: &[u8]) -> Result<(u32, u32), ImageError> {
    let mut decoder = Decoder::new(Cursor::new(bytes));
    let (width, height, _) = read_header(&mut decoder)?;
    Ok((width, height))
}

/// Decode a JPEG into BGR24.
///
/// The header is parsed first, so an image larger than `max_pixels` is rejected
/// before any pixel buffer is allocated. `max_pixels` counts pixels, not bytes;
/// the decoder's own buffer limit is derived from it.
pub fn decode_jpeg(bytes: &[u8], max_pixels: u64) -> Result<BgrImage, ImageError> {
    let mut decoder = Decoder::new(Cursor::new(bytes));
    let (width, height, pixel_format) = read_header(&mut decoder)?;
    let pixels = u64::from(width) * u64::from(height);
    if pixels > max_pixels {
        return Err(ImageError::FrameTooLarge { pixels, max_pixels });
    }
    let bgr_len = pixels
        .checked_mul(3)
        .and_then(|bytes| usize::try_from(bytes).ok())
        .ok_or(ImageError::FrameTooLarge { pixels, max_pixels })?;
    decoder.set_max_decoding_buffer_size(bgr_len);
    let data = decoder.decode().map_err(|error| ImageError::JpegDecode {
        message: error.to_string(),
    })?;
    let pixel_count = bgr_len / 3;
    let mut bgr = vec![0u8; bgr_len];
    match pixel_format {
        PixelFormat::RGB24 => {
            if data.len() != bgr_len {
                return Err(ImageError::JpegDecode {
                    message: format!(
                        "decoder returned {} bytes for a {width}x{height} RGB image, expected {bgr_len}",
                        data.len()
                    ),
                });
            }
            for (destination, source) in bgr.chunks_exact_mut(3).zip(data.chunks_exact(3)) {
                destination[0] = source[2];
                destination[1] = source[1];
                destination[2] = source[0];
            }
        }
        PixelFormat::L8 => {
            if data.len() != pixel_count {
                return Err(ImageError::JpegDecode {
                    message: format!(
                        "decoder returned {} bytes for a {width}x{height} grayscale image, expected {pixel_count}",
                        data.len()
                    ),
                });
            }
            for (destination, &luma) in bgr.chunks_exact_mut(3).zip(data.iter()) {
                destination.fill(luma);
            }
        }
        other => {
            return Err(ImageError::UnsupportedPixelFormat {
                format: format!("{other:?}"),
            });
        }
    }
    BgrImage::new(width, height, bgr)
}
