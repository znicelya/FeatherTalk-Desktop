use crate::{
    error::{ImageError, check_target},
    image::BgrImage,
};

/// OpenCV's `INTER_RESIZE_COEF_SCALE`: two-tap weights are 11-bit fixed point.
const INTER_RESIZE_COEF_SCALE: i32 = 2048;

/// BGR24 is the only layout this crate carries.
const CHANNELS: usize = 3;

/// `cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)`.
///
/// OpenCV picks one of four kernels here, and so does this function:
///
/// - identical size: a byte copy;
/// - both axes shrinking by an integer ratio: exact block averaging;
/// - both axes shrinking otherwise: the decimation tables;
/// - either axis growing: the two-tap resampler with the area tap mapping.
///
/// Every branch is byte-identical to OpenCV; see the parity tests.
pub fn resize_area(image: &BgrImage, width: u32, height: u32) -> Result<BgrImage, ImageError> {
    check_target(width, height, CHANNELS)?;
    if width == image.width() && height == image.height() {
        return BgrImage::new(width, height, image.as_bytes().to_vec());
    }

    let scale_x = f64::from(image.width()) / f64::from(width);
    let scale_y = f64::from(image.height()) / f64::from(height);
    let pixels = if scale_x >= 1.0 && scale_y >= 1.0 {
        if image.width().is_multiple_of(width) && image.height().is_multiple_of(height) {
            area_integer(image, width, height)
        } else {
            area_fractional(image, width, height, scale_x, scale_y)
        }
    } else {
        two_tap(image, width, height, scale_x, scale_y, TapMode::Area)
    };
    BgrImage::new(width, height, pixels)
}

/// `cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)`.
///
/// Always the two-tap resampler, in either direction. Shrinking and identical
/// sizes are byte-identical to OpenCV; upscaling keeps a plus or minus one
/// residual on a small fraction of samples, which the parity test bounds.
pub fn resize_linear(image: &BgrImage, width: u32, height: u32) -> Result<BgrImage, ImageError> {
    check_target(width, height, CHANNELS)?;
    let scale_x = f64::from(image.width()) / f64::from(width);
    let scale_y = f64::from(image.height()) / f64::from(height);
    let pixels = two_tap(image, width, height, scale_x, scale_y, TapMode::Linear);
    BgrImage::new(width, height, pixels)
}

/// OpenCV's `saturate_cast<uchar>`: round half to even, then clamp.
pub(crate) fn saturate_u8(value: f32) -> u8 {
    let rounded = value.round_ties_even();
    if rounded <= 0.0 {
        0
    } else if rounded >= 255.0 {
        255
    } else {
        rounded as u8
    }
}

/// Exact block averaging, OpenCV's `resizeAreaFast_` path.
///
/// Both ratios divide evenly, so every destination pixel is the mean of a
/// disjoint `scale_x` by `scale_y` block. The largest block this crate can see
/// is bounded by `MAX_EDGE`, and `255 * 32768 * 32768` still fits in `u64`;
/// `u32` is enough here because a block that large cannot also be a frame.
fn area_integer(image: &BgrImage, width: u32, height: u32) -> Vec<u8> {
    let scale_x = image.width() / width;
    let scale_y = image.height() / height;
    let area = scale_x * scale_y;
    let inverse_area = 1.0 / area as f32;
    let mut pixels = Vec::with_capacity(width as usize * height as usize * CHANNELS);
    for dy in 0..height {
        for dx in 0..width {
            let mut sums = [0_u32; CHANNELS];
            for offset_y in 0..scale_y {
                let row = image.row(dy * scale_y + offset_y);
                for offset_x in 0..scale_x {
                    let base = (dx * scale_x + offset_x) as usize * CHANNELS;
                    for (sum, value) in sums.iter_mut().zip(&row[base..base + CHANNELS]) {
                        *sum += u32::from(*value);
                    }
                }
            }
            if scale_x == 2 && scale_y == 2 {
                // OpenCV's 2x2 specialisation rounds in integer arithmetic.
                pixels.extend(sums.map(|sum| ((sum + 2) >> 2) as u8));
            } else {
                pixels.extend(sums.map(|sum| saturate_u8(sum as f32 * inverse_area)));
            }
        }
    }
    pixels
}

/// One source-to-destination weight, OpenCV's `DecimateAlpha`.
#[derive(Clone, Copy)]
struct DecimateAlpha {
    si: usize,
    di: usize,
    alpha: f32,
}

/// Port of OpenCV's `computeResizeAreaTab`.
///
/// `channels` scales both indices, so an x table addresses bytes inside a row
/// while a y table built with `channels = 1` addresses row numbers. The `1e-3`
/// guards and the `cell_width` clamp are OpenCV's, not tolerances of ours:
/// dropping them changes which taps exist for ratios such as 13/7.
fn decimate_alpha(
    source_size: u32,
    destination_size: u32,
    scale: f64,
    channels: usize,
) -> Vec<DecimateAlpha> {
    let source_size = i64::from(source_size);
    let mut table = Vec::new();
    for d in 0..i64::from(destination_size) {
        let fs1 = d as f64 * scale;
        let fs2 = fs1 + scale;
        let cell_width = scale.min(source_size as f64 - fs1);
        let s2 = (fs2.floor() as i64).min(source_size - 1);
        let s1 = (fs1.ceil() as i64).min(s2);
        let di = d as usize * channels;

        if s1 as f64 - fs1 > 1e-3 {
            table.push(DecimateAlpha {
                si: (s1 - 1) as usize * channels,
                di,
                alpha: ((s1 as f64 - fs1) / cell_width) as f32,
            });
        }
        for s in s1..s2 {
            table.push(DecimateAlpha {
                si: s as usize * channels,
                di,
                alpha: (1.0 / cell_width) as f32,
            });
        }
        if fs2 - s2 as f64 > 1e-3 {
            table.push(DecimateAlpha {
                si: s2 as usize * channels,
                di,
                alpha: ((fs2 - s2 as f64).min(1.0).min(cell_width) / cell_width) as f32,
            });
        }
    }
    table
}

/// Port of OpenCV's `ResizeArea_Invoker` for fractional shrink ratios.
///
/// The y table is walked in order and its entries for one destination row are
/// contiguous, so a single `sums` row is enough: a change in `di` flushes the
/// previous destination row and restarts the accumulator.
fn area_fractional(
    image: &BgrImage,
    width: u32,
    height: u32,
    scale_x: f64,
    scale_y: f64,
) -> Vec<u8> {
    let xtab = decimate_alpha(image.width(), width, scale_x, CHANNELS);
    let ytab = decimate_alpha(image.height(), height, scale_y, 1);
    let row_len = width as usize * CHANNELS;
    let mut pixels = vec![0_u8; row_len * height as usize];
    let mut buffer = vec![0.0_f32; row_len];
    let mut sums = vec![0.0_f32; row_len];
    let mut previous_di = ytab[0].di;

    for entry in &ytab {
        buffer.fill(0.0);
        let row = image.row(entry.si as u32);
        for tap in &xtab {
            let source = &row[tap.si..tap.si + CHANNELS];
            let target = &mut buffer[tap.di..tap.di + CHANNELS];
            for (accumulated, value) in target.iter_mut().zip(source) {
                *accumulated += f32::from(*value) * tap.alpha;
            }
        }
        if entry.di == previous_di {
            for (accumulated, value) in sums.iter_mut().zip(&buffer) {
                *accumulated += entry.alpha * *value;
            }
        } else {
            flush_row(&mut pixels, previous_di * row_len, &sums);
            for (accumulated, value) in sums.iter_mut().zip(&buffer) {
                *accumulated = entry.alpha * *value;
            }
            previous_di = entry.di;
        }
    }
    flush_row(&mut pixels, previous_di * row_len, &sums);
    pixels
}

/// Round one accumulated destination row into the output buffer.
fn flush_row(pixels: &mut [u8], start: usize, sums: &[f32]) {
    let target = &mut pixels[start..start + sums.len()];
    for (slot, value) in target.iter_mut().zip(sums) {
        *slot = saturate_u8(*value);
    }
}

/// One destination sample's two source taps with fixed-point weights.
#[derive(Clone, Copy)]
pub(crate) struct Tap {
    s: usize,
    a0: i32,
    a1: i32,
}

/// Which mapping OpenCV uses to place the two taps.
#[derive(Clone, Copy)]
pub(crate) enum TapMode {
    Area,
    Linear,
}

/// Port of the tap table OpenCV builds at the top of `resize`.
///
/// Both mappings clamp to `0..=source_size - 1` and zero the fractional part at
/// the borders, which is how OpenCV replicates edge pixels without a padded
/// buffer.
pub(crate) fn compute_taps(
    source_size: u32,
    destination_size: u32,
    scale: f64,
    mode: TapMode,
) -> Vec<Tap> {
    let source_size = i64::from(source_size);
    let mut taps = Vec::with_capacity(destination_size as usize);
    for d in 0..i64::from(destination_size) {
        let (mut s, mut fraction) = match mode {
            TapMode::Linear => {
                let value = ((d as f64 + 0.5) * scale - 0.5) as f32;
                let floor = value.floor();
                (floor as i64, value - floor)
            }
            TapMode::Area => {
                let s = (d as f64 * scale).floor() as i64;
                let value = ((d as f64 + 1.0) - (s as f64 + 1.0) / scale) as f32;
                (
                    s,
                    if value <= 0.0 {
                        0.0
                    } else {
                        value - value.floor()
                    },
                )
            }
        };
        if s < 0 {
            s = 0;
            fraction = 0.0;
        }
        if s >= source_size - 1 {
            s = source_size - 1;
            fraction = 0.0;
        }
        let a1 = (fraction * INTER_RESIZE_COEF_SCALE as f32).round_ties_even() as i32;
        taps.push(Tap {
            s: s as usize,
            a0: INTER_RESIZE_COEF_SCALE - a1,
            a1,
        });
    }
    taps
}

/// OpenCV's separable two-tap resampler for 8-bit input.
///
/// The horizontal pass keeps its results in `i32` at full 11-bit weight scale
/// (at most `255 * 2048 * 2 = 1_044_480`), and the vertical pass reproduces
/// OpenCV's shift sequence exactly, including the `>> 4` that discards the low
/// bits before the second multiply.
pub(crate) fn two_tap(
    image: &BgrImage,
    width: u32,
    height: u32,
    scale_x: f64,
    scale_y: f64,
    mode: TapMode,
) -> Vec<u8> {
    let xtaps = compute_taps(image.width(), width, scale_x, mode);
    let ytaps = compute_taps(image.height(), height, scale_y, mode);
    let row_len = width as usize * CHANNELS;
    let last_column = (image.width() - 1) as usize;
    let last_row = (image.height() - 1) as usize;

    let mut horizontal = vec![0_i32; row_len * image.height() as usize];
    for (y, target) in horizontal.chunks_exact_mut(row_len).enumerate() {
        let row = image.row(y as u32);
        for (destination, tap) in target.chunks_exact_mut(CHANNELS).zip(&xtaps) {
            let left_start = tap.s * CHANNELS;
            let right_start = (tap.s + 1).min(last_column) * CHANNELS;
            let left = &row[left_start..left_start + CHANNELS];
            let right = &row[right_start..right_start + CHANNELS];
            for (slot, (left, right)) in destination.iter_mut().zip(left.iter().zip(right)) {
                *slot = i32::from(*left) * tap.a0 + i32::from(*right) * tap.a1;
            }
        }
    }

    let mut pixels = Vec::with_capacity(row_len * height as usize);
    for tap in &ytaps {
        let top_start = tap.s * row_len;
        let bottom_start = (tap.s + 1).min(last_row) * row_len;
        let top = &horizontal[top_start..top_start + row_len];
        let bottom = &horizontal[bottom_start..bottom_start + row_len];
        for (above, below) in top.iter().zip(bottom) {
            let accumulated = ((tap.a0 * (above >> 4)) >> 16) + ((tap.a1 * (below >> 4)) >> 16);
            pixels.push(((accumulated + 2) >> 2).clamp(0, 255) as u8);
        }
    }
    pixels
}
