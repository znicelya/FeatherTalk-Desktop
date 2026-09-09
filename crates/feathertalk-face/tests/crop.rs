use feathertalk_face::{
    FaceCropGeometry, FaceError, ImageSize, Padding, RectI, compute_face_crop_geometry,
};

fn image() -> ImageSize {
    ImageSize {
        width: 100,
        height: 80,
    }
}

#[test]
fn computes_centered_square_for_normal_box() {
    let geometry = compute_face_crop_geometry(image(), [20.0, 10.0, 30.0, 20.0]).unwrap();
    assert_eq!(geometry.size, 31);
    assert_eq!(
        geometry.requested,
        RectI {
            x: 20,
            y: 5,
            width: 31,
            height: 31
        }
    );
    assert_eq!(
        geometry.source,
        RectI {
            x: 20,
            y: 5,
            width: 31,
            height: 31
        }
    );
    assert_eq!(
        geometry.padding,
        Padding {
            left: 0,
            top: 0,
            right: 0,
            bottom: 0
        }
    );
    assert_eq!((geometry.origin_x, geometry.origin_y), (20, 5));
}

#[test]
fn expands_wide_and_tall_boxes_using_larger_dimension() {
    let wide = compute_face_crop_geometry(image(), [20.0, 10.0, 40.0, 10.0]).unwrap();
    assert_eq!(wide.size, 42);
    assert_eq!(
        wide.requested,
        RectI {
            x: 19,
            y: -6,
            width: 42,
            height: 42
        }
    );
    let tall = compute_face_crop_geometry(image(), [20.0, 10.0, 10.0, 40.0]).unwrap();
    assert_eq!(tall.size, 42);
    assert_eq!(
        tall.requested,
        RectI {
            x: 4,
            y: 9,
            width: 42,
            height: 42
        }
    );
}

#[test]
fn computes_padding_for_all_image_boundaries() {
    let geometry = compute_face_crop_geometry(image(), [0.0, 0.0, 100.0, 80.0]).unwrap();
    assert_eq!(geometry.size, 105);
    assert_eq!(
        geometry.requested,
        RectI {
            x: -2,
            y: -12,
            width: 105,
            height: 105
        }
    );
    assert_eq!(
        geometry.source,
        RectI {
            x: 0,
            y: 0,
            width: 100,
            height: 80
        }
    );
    assert_eq!(
        geometry.padding,
        Padding {
            left: 2,
            top: 12,
            right: 3,
            bottom: 13
        }
    );
}

#[test]
fn rejects_invalid_image_and_bbox_values() {
    assert!(matches!(
        compute_face_crop_geometry(
            ImageSize {
                width: 0,
                height: 80
            },
            [0.0, 0.0, 1.0, 1.0]
        ),
        Err(FaceError::InvalidImageSize)
    ));
    assert!(matches!(
        compute_face_crop_geometry(image(), [0.0, 0.0, 0.0, 1.0]),
        Err(FaceError::InvalidCropGeometry { .. })
    ));
    assert!(matches!(
        compute_face_crop_geometry(image(), [f32::NAN, 0.0, 1.0, 1.0]),
        Err(FaceError::NonFiniteValue { .. })
    ));
}

#[test]
fn preserves_float32_edge_addition_before_integer_conversion() {
    assert!(matches!(
        compute_face_crop_geometry(
            ImageSize {
                width: 20_000_000,
                height: 10,
            },
            [16_777_216.0, 1.0, 1.0, 2.0],
        ),
        Err(FaceError::InvalidCropGeometry { .. })
    ));
}

#[test]
fn public_geometry_is_a_value_type() {
    let _: Option<FaceCropGeometry> = None;
}
