mod support;

use feathertalk_face::{FaceCropGeometry, ImageSize, Padding, RectI, compute_face_crop_geometry};
use feathertalk_frame_adapters::pfld_input;
use feathertalk_frame_pipeline::PipelineError;
use feathertalk_image::BgrImage;

/// The 640x640 decode of the committed JPEG. The generator cropped both
/// reference blobs out of exactly these pixels.
fn source_image(fixture: &support::VerifiedFixture) -> BgrImage {
    let decoded = support::read_u8_array(&fixture.root.join("frame_decode.npy")).unwrap();
    support::bgr_from_array(&decoded)
}

fn expected_blob(fixture: &support::VerifiedFixture, name: &str) -> Vec<f32> {
    support::flatten_f32(&support::read_f32_array(&fixture.root.join(name)).unwrap())
}

/// A geometry that `compute_face_crop_geometry` can never return, used to reach
/// the guards.
fn handmade_geometry(source: RectI, size: u32) -> FaceCropGeometry {
    FaceCropGeometry {
        requested: source,
        source,
        padding: Padding {
            left: 0,
            top: 0,
            right: 0,
            bottom: 0,
        },
        size,
        origin_x: source.x,
        origin_y: source.y,
    }
}

#[test]
fn both_crop_cases_are_byte_exact_against_opencv() {
    let fixture = support::load_and_verify_fixture().unwrap();
    let image = source_image(&fixture);

    for name in support::CROP_CASES {
        let case = support::crop_case(&fixture, name);
        let geometry = compute_face_crop_geometry(
            ImageSize {
                width: image.width(),
                height: image.height(),
            },
            case.bbox,
        )
        .unwrap();

        let actual = pfld_input(&image, &geometry).unwrap();
        assert_eq!(actual.len(), 3 * 192 * 192, "{name} length");

        // Both cases shrink, and Task 6 pinned the shrink path as
        // byte-identical to OpenCV, so nothing here is toleranced.
        let metrics = support::compare_f32(&actual, &expected_blob(&fixture, &case.array));
        assert_eq!(metrics.max_abs, 0.0, "{name}: {metrics:?}");
    }
}

#[test]
fn the_crop_geometry_matches_the_recorded_cases() {
    let fixture = support::load_and_verify_fixture().unwrap();
    let size = ImageSize {
        width: 640,
        height: 640,
    };

    for name in support::CROP_CASES {
        let case = support::crop_case(&fixture, name);
        let geometry = compute_face_crop_geometry(size, case.bbox).unwrap();

        assert_eq!(geometry.size, case.size, "{name} size");
        assert_eq!(
            (geometry.origin_x, geometry.origin_y),
            (case.origin_x, case.origin_y),
            "{name} origin"
        );
        assert_eq!(
            geometry.padding,
            Padding {
                left: case.padding[0],
                top: case.padding[1],
                right: case.padding[2],
                bottom: case.padding[3],
            },
            "{name} padding"
        );
        assert_eq!(
            geometry.source,
            RectI {
                x: i32::try_from(case.source[0]).unwrap(),
                y: i32::try_from(case.source[1]).unwrap(),
                width: u32::try_from(case.source[2]).unwrap(),
                height: u32::try_from(case.source[3]).unwrap(),
            },
            "{name} source"
        );
    }
}

#[test]
fn a_zero_sized_crop_is_rejected() {
    let fixture = support::load_and_verify_fixture().unwrap();
    let image = source_image(&fixture);
    let geometry = handmade_geometry(
        RectI {
            x: 0,
            y: 0,
            width: 0,
            height: 0,
        },
        0,
    );

    match pfld_input(&image, &geometry).unwrap_err() {
        PipelineError::Adapter { component, message } => {
            assert_eq!(component, "pfld");
            assert_eq!(message, "crop size must be non-zero");
        }
        other => panic!("expected an adapter error, got {other}"),
    }
}

#[test]
fn a_source_rectangle_past_the_frame_is_rejected() {
    let fixture = support::load_and_verify_fixture().unwrap();
    let image = source_image(&fixture);
    let geometry = handmade_geometry(
        RectI {
            x: 600,
            y: 600,
            width: 100,
            height: 100,
        },
        100,
    );

    match pfld_input(&image, &geometry).unwrap_err() {
        PipelineError::Adapter { component, message } => {
            assert_eq!(component, "pfld");
            assert_eq!(
                message,
                "source rectangle 100x100 at (600, 600) exceeds the 640x640 frame"
            );
        }
        other => panic!("expected an adapter error, got {other}"),
    }
}
