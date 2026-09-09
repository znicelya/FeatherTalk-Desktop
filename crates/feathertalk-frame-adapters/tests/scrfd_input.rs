mod support;

use feathertalk_face::ImageSize;
use feathertalk_frame_adapters::scrfd_input;

const EDGE: usize = 640;
const PLANE: usize = EDGE * EDGE;

#[test]
fn the_square_pattern_reproduces_the_committed_scrfd_blob() {
    let image = support::pattern_bgr(640, 640);
    let input = scrfd_input(&image).unwrap();

    assert_eq!(
        input.transform.input,
        ImageSize {
            width: 640,
            height: 640,
        }
    );
    assert_eq!(input.transform.new_width, 640);
    assert_eq!(input.transform.new_height, 640);
    assert_eq!(input.transform.pad_x, 0);
    assert_eq!(input.transform.pad_y, 0);
    assert_eq!(input.transform.scale_x, 1.0);
    assert_eq!(input.transform.scale_y, 1.0);

    let reference = support::flatten_f32(&support::read_reference_array("input.npy"));
    assert_eq!(input.data.len(), 3 * PLANE);
    let metrics = support::compare_f32(&input.data, &reference);
    assert_eq!(
        metrics.max_abs, 0.0,
        "the square path must be bit exact, got {metrics:?}"
    );
}

#[test]
fn the_landscape_letterbox_matches_the_hash_pinned_blob() {
    let fixture = support::load_and_verify_fixture().unwrap();
    let pin = support::letterbox_pin(&fixture);
    assert_eq!(pin.source_width, 1280);
    assert_eq!(pin.source_height, 720);
    assert_eq!(pin.shape, vec![1, 3, EDGE, EDGE]);

    let image = support::pattern_bgr(pin.source_width, pin.source_height);
    let input = scrfd_input(&image).unwrap();

    assert_eq!(input.transform.new_width, pin.new_width);
    assert_eq!(input.transform.new_height, pin.new_height);
    assert_eq!(input.transform.pad_x, pin.pad_x);
    assert_eq!(input.transform.pad_y, pin.pad_y);
    // 1280 / 640 and 720 / 361 are the mappings Task 11 inverts.
    assert_eq!(input.transform.scale_x, 2.0);
    assert_eq!(input.transform.scale_y, 720.0 / 361.0);

    assert_eq!(input.data.len(), 3 * PLANE);
    assert_eq!(
        support::sha256_f32_le(&input.data),
        pin.sha256,
        "the 1280x720 blob must hash to the pinned digest"
    );

    assert_eq!(pin.samples.len(), 8, "the manifest pins eight samples");
    for (index, expected) in &pin.samples {
        let [channel, row, column] = *index;
        let offset = channel * PLANE + row * EDGE + column;
        assert_eq!(
            input.data[offset], *expected,
            "sample [{channel}, {row}, {column}]"
        );
    }
}

#[test]
fn the_letterbox_padding_holds_the_normalized_zero() {
    let image = support::pattern_bgr(1280, 720);
    let input = scrfd_input(&image).unwrap();
    let pad_y = input.transform.pad_y as usize;
    let filled = pad_y + input.transform.new_height as usize;
    assert_eq!(pad_y, 139);
    assert_eq!(filled, 500);

    let mut padded = 0_usize;
    for channel in 0..3 {
        for row in (0..pad_y).chain(filled..EDGE) {
            for column in 0..EDGE {
                let offset = channel * PLANE + row * EDGE + column;
                assert_eq!(
                    input.data[offset],
                    support::PADDED_BLOB_VALUE,
                    "padding at [{channel}, {row}, {column}]"
                );
                padded += 1;
            }
        }
    }
    assert_eq!(
        padded,
        3 * 279 * EDGE,
        "279 padded rows on each of three channels"
    );
}

#[test]
fn a_portrait_frame_pads_horizontally() {
    // 1280 / 640 is exactly 2, so the floor in `resize_with_padding` has no
    // rounding to argue about and the expected geometry is unambiguous.
    let image = support::pattern_bgr(640, 1280);
    let input = scrfd_input(&image).unwrap();

    assert_eq!(input.transform.new_width, 320);
    assert_eq!(input.transform.new_height, 640);
    assert_eq!(input.transform.pad_x, 160);
    assert_eq!(input.transform.pad_y, 0);
    assert_eq!(input.transform.scale_x, 2.0);
    assert_eq!(input.transform.scale_y, 2.0);

    let mut padded = 0_usize;
    for channel in 0..3 {
        for row in 0..EDGE {
            for column in (0..160).chain(480..EDGE) {
                let offset = channel * PLANE + row * EDGE + column;
                assert_eq!(
                    input.data[offset],
                    support::PADDED_BLOB_VALUE,
                    "padding at [{channel}, {row}, {column}]"
                );
                padded += 1;
            }
        }
    }
    assert_eq!(
        padded,
        3 * EDGE * 320,
        "320 padded columns on every row of three channels"
    );
}

#[test]
fn the_channel_order_is_rgb() {
    let fixture = support::load_and_verify_fixture().unwrap();
    let decoded = support::read_u8_array(&fixture.root.join("frame_decode.npy")).unwrap();
    let flat = support::flatten_u8(&decoded);
    let image = support::bgr_from_array(&decoded);
    let input = scrfd_input(&image).unwrap();
    assert_eq!(input.transform.new_width, 640);
    assert_eq!(input.transform.new_height, 640);

    let mut compared = 0_usize;
    for row in (0..EDGE).step_by(37) {
        for column in (0..EDGE).step_by(37) {
            let pixel = (row * EDGE + column) * 3;
            for channel in 0..3 {
                // Channel 0 is red, which is the third byte of a BGR pixel.
                let expected = (f32::from(flat[pixel + 2 - channel]) - 127.5) / 128.0;
                assert_eq!(
                    input.data[channel * PLANE + row * EDGE + column],
                    expected,
                    "[{channel}, {row}, {column}]"
                );
                compared += 1;
            }
        }
    }
    assert_eq!(compared, 3 * 18 * 18, "18 sampled rows and 18 columns");
}
