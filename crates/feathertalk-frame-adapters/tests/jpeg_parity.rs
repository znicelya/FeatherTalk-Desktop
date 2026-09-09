//! `jpeg-decoder` against libjpeg-turbo, as design section 9 requires.
//!
//! The two decoders do not agree bit for bit. Measured on this fixture
//! (640x640 pattern, quality 90, 4:2:0) the largest per-sample difference is 3
//! and 9.4316 % of the 1 228 800 samples differ, with the delta histogram
//! 0 => 1 112 905, 1 => 87 618, 2 => 27 755, 3 => 522. Re-encoding the same
//! frame at 4:4:4 still differs (6.4696 % of samples), so chroma upsampling is
//! not the only cause: the IDCT rounding and the YCbCr to RGB matrix contribute
//! too. Nothing downstream is sensitive at that scale - the blur variance of
//! the demo frame moves by 0.008 % and the SCRFD score by 1.24e-4 - so this
//! test pins a tolerance instead of equality.

mod support;

use feathertalk_frame_adapters::DEFAULT_MAX_FRAME_PIXELS;
use feathertalk_image::decode_jpeg;

/// Largest per-sample difference this crate tolerates.
const MAX_ABSOLUTE_DELTA: i16 = 3;

/// Largest fraction of samples allowed to differ at all.
const MAX_MISMATCH_RATIO: f64 = 0.10;

#[test]
fn the_committed_frame_decodes_within_the_opencv_tolerance() {
    let fixture = support::load_and_verify_fixture().unwrap();
    let encoded = std::fs::read(fixture.root.join("frame.jpg")).unwrap();
    let decoded = decode_jpeg(&encoded, DEFAULT_MAX_FRAME_PIXELS).unwrap();
    let reference = support::read_u8_array(&fixture.root.join("frame_decode.npy")).unwrap();
    let expected = support::flatten_u8(&reference);

    assert_eq!(decoded.width(), 640);
    assert_eq!(decoded.height(), 640);
    assert_eq!(decoded.as_bytes().len(), expected.len());

    let mut histogram = [0_u64; MAX_ABSOLUTE_DELTA as usize + 1];
    let mut mismatches = 0_u64;
    let mut over_tolerance = Vec::new();
    for (index, (actual, wanted)) in decoded.as_bytes().iter().zip(&expected).enumerate() {
        let delta = (i16::from(*actual) - i16::from(*wanted)).abs();
        if delta > 0 {
            mismatches += 1;
        }
        if delta <= MAX_ABSOLUTE_DELTA {
            histogram[delta as usize] += 1;
        } else if over_tolerance.len() < 8 {
            over_tolerance.push((index, *actual, *wanted));
        }
    }

    assert!(
        over_tolerance.is_empty(),
        "samples differ by more than {MAX_ABSOLUTE_DELTA}, first offenders as \
         (index, ours, opencv): {over_tolerance:?}"
    );
    let ratio = mismatches as f64 / expected.len() as f64;
    assert!(
        ratio <= MAX_MISMATCH_RATIO,
        "{mismatches} of {} samples differ, ratio {ratio:.6} over {MAX_MISMATCH_RATIO}, \
         histogram {histogram:?}",
        expected.len()
    );
}
