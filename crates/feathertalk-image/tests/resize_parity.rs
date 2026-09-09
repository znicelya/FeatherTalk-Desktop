mod support;

use feathertalk_image::{BgrImage, ImageError, resize_area, resize_linear};

/// Assert that `resize_area` reproduces one fixture case byte for byte.
fn assert_area_case(case: &str, width: u32, height: u32) {
    let fixture = support::load_and_verify_fixture().unwrap();
    let source = support::read_u8_array(&fixture.root.join(format!("{case}_src.npy"))).unwrap();
    let expected = support::read_u8_array(&fixture.root.join(format!("{case}_dst.npy"))).unwrap();
    let resized = resize_area(&support::bgr_from_array(&source), width, height).unwrap();
    assert_eq!(resized.width(), width, "{case}");
    assert_eq!(resized.height(), height, "{case}");
    assert_eq!(
        resized.as_bytes(),
        support::flatten_u8(&expected).as_slice(),
        "{case}"
    );
}

#[test]
fn area_halves_an_integer_ratio_exactly() {
    // 8x8 -> 4x4 takes OpenCV's dedicated 2x2 block shortcut.
    assert_area_case("area_int_2x2", 4, 4);
}

#[test]
fn area_averages_a_four_by_four_block_exactly() {
    // 8x8 -> 2x2 takes the general integer block path.
    assert_area_case("area_int_4x4", 2, 2);
}

#[test]
fn area_matches_opencv_on_a_fractional_ratio() {
    // 13x9 -> 7x5 has no integer ratio on either axis.
    assert_area_case("area_shrink", 7, 5);
}

#[test]
fn area_matches_opencv_when_upscaling() {
    // 5x5 -> 8x8 falls back to the two-tap resampler.
    assert_area_case("area_upscale", 8, 8);
}

#[test]
fn area_copies_an_identically_sized_image() {
    let image = BgrImage::new(3, 2, (0..18_u8).collect()).unwrap();
    assert_eq!(resize_area(&image, 3, 2).unwrap(), image);
}

#[test]
fn area_rejects_a_degenerate_target() {
    let image = BgrImage::new(4, 4, vec![0; 48]).unwrap();
    for (width, height) in [(0, 4), (4, 0)] {
        let error = resize_area(&image, width, height).unwrap_err();
        assert!(
            matches!(
                error,
                ImageError::InvalidTargetSize {
                    max_dimension: 32_768,
                    ..
                }
            ),
            "{width}x{height}: {error}"
        );
    }
}

#[test]
fn area_rejects_a_target_beyond_the_edge_limit() {
    let image = BgrImage::new(4, 4, vec![0; 48]).unwrap();
    let error = resize_area(&image, 32_769, 4).unwrap_err();
    assert!(
        matches!(
            error,
            ImageError::InvalidTargetSize {
                width: 32_769,
                height: 4,
                max_dimension: 32_768,
            }
        ),
        "{error}"
    );
}

/// Largest absolute difference and mismatch ratio between two byte planes.
fn byte_differences(actual: &[u8], expected: &[u8]) -> (u8, f64) {
    assert_eq!(actual.len(), expected.len(), "plane lengths differ");
    let mut max_abs = 0_u8;
    let mut mismatches = 0_usize;
    for (actual, expected) in actual.iter().zip(expected) {
        let difference = actual.abs_diff(*expected);
        max_abs = max_abs.max(difference);
        if difference != 0 {
            mismatches += 1;
        }
    }
    (max_abs, mismatches as f64 / actual.len() as f64)
}

#[test]
fn linear_shrink_matches_opencv_byte_for_byte() {
    let fixture = support::load_and_verify_fixture().unwrap();
    let source = support::read_u8_array(&fixture.root.join("linear_shrink_src.npy")).unwrap();
    let expected = support::read_u8_array(&fixture.root.join("linear_shrink_dst.npy")).unwrap();
    let resized = resize_linear(&support::bgr_from_array(&source), 192, 192).unwrap();
    assert_eq!(resized.width(), 192);
    assert_eq!(resized.height(), 192);
    assert_eq!(
        resized.as_bytes(),
        support::flatten_u8(&expected).as_slice()
    );
}

#[test]
fn linear_upscale_stays_within_one_level_of_opencv() {
    let fixture = support::load_and_verify_fixture().unwrap();
    let source = support::read_u8_array(&fixture.root.join("linear_upscale_src.npy")).unwrap();
    let expected = support::read_u8_array(&fixture.root.join("linear_upscale_dst.npy")).unwrap();
    let resized = resize_linear(&support::bgr_from_array(&source), 192, 192).unwrap();
    let (max_abs, mismatch_ratio) =
        byte_differences(resized.as_bytes(), &support::flatten_u8(&expected));
    // Design section 3.7. Measured at 289 of 110592 samples, all off by one.
    assert!(max_abs <= 1, "max_abs {max_abs}");
    assert!(mismatch_ratio <= 0.01, "mismatch ratio {mismatch_ratio}");
}

#[test]
fn linear_reproduces_an_identically_sized_image() {
    // Unlike resize_area this is not a short circuit: the taps collapse to
    // (2048, 0) at every position, so the resampler itself is lossless here.
    let image = BgrImage::new(3, 2, (0..18_u8).collect()).unwrap();
    assert_eq!(resize_linear(&image, 3, 2).unwrap(), image);
}

#[test]
fn linear_rejects_a_degenerate_target() {
    let image = BgrImage::new(4, 4, vec![0; 48]).unwrap();
    let error = resize_linear(&image, 4, 0).unwrap_err();
    assert!(
        matches!(
            error,
            ImageError::InvalidTargetSize {
                width: 4,
                height: 0,
                max_dimension: 32_768,
            }
        ),
        "{error}"
    );
}
