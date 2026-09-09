mod support;

use feathertalk_image::{BgrImage, laplacian_response, laplacian_variance, to_gray};

#[test]
fn to_gray_matches_the_opencv_fixture_byte_for_byte() {
    let fixture = support::load_and_verify_fixture().unwrap();
    let source = support::read_u8_array(&fixture.root.join("gray_src.npy")).unwrap();
    let expected = support::read_u8_array(&fixture.root.join("gray_dst.npy")).unwrap();
    let gray = to_gray(&support::bgr_from_array(&source));
    assert_eq!(gray.width(), 64);
    assert_eq!(gray.height(), 64);
    assert_eq!(gray.as_bytes(), support::flatten_u8(&expected).as_slice());
}

#[test]
fn the_laplacian_response_matches_the_opencv_fixture_exactly() {
    let fixture = support::load_and_verify_fixture().unwrap();
    let source = support::read_u8_array(&fixture.root.join("gray_src.npy")).unwrap();
    let expected = support::read_f64_array(&fixture.root.join("laplacian_response.npy")).unwrap();
    let response = laplacian_response(&to_gray(&support::bgr_from_array(&source)));
    assert_eq!(response.len(), expected.len());
    for (index, (actual, expected)) in response.iter().zip(expected.iter()).enumerate() {
        assert_eq!(actual, expected, "flattened index {index}");
    }
}

#[test]
fn the_laplacian_variance_matches_the_recorded_scalar() {
    let fixture = support::load_and_verify_fixture().unwrap();
    let source = support::read_u8_array(&fixture.root.join("gray_src.npy")).unwrap();
    let expected = support::scalar(&fixture, "laplacian_variance");
    let actual = laplacian_variance(&to_gray(&support::bgr_from_array(&source)));
    // NumPy sums pairwise and this crate sums linearly, so the last bits of the
    // f64 accumulation may differ even though every input value is identical.
    let relative = (actual - expected).abs() / expected.abs();
    assert!(relative <= 1e-12, "expected {expected}, got {actual}");
}

#[test]
fn a_uniform_image_has_no_laplacian_response() {
    let gray = to_gray(&BgrImage::new(3, 2, vec![7; 18]).unwrap());
    assert!(gray.as_bytes().iter().all(|value| *value == 7));
    assert_eq!(laplacian_response(&gray), vec![0.0; 6]);
    assert_eq!(laplacian_variance(&gray), 0.0);
}

#[test]
fn a_single_column_image_mirrors_within_its_own_width() {
    let image = BgrImage::new(1, 3, vec![0, 0, 0, 255, 255, 255, 0, 0, 0]).unwrap();
    let gray = to_gray(&image);
    assert_eq!(gray.as_bytes(), [0, 255, 0]);
    // With a single column both horizontal taps collapse onto the pixel itself,
    // so the kernel degenerates to `up + down - 2 * self`.
    assert_eq!(laplacian_response(&gray), vec![510.0, -510.0, 510.0]);
}
