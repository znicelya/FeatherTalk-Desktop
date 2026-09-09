use feathertalk_face::{Detection, FaceError, ImageSize, decode_level, resize_with_padding};

fn transform() -> feathertalk_face::ResizeTransform {
    resize_with_padding(ImageSize {
        width: 320,
        height: 640,
    })
    .unwrap()
}

#[test]
fn decodes_bbox_and_keypoints_and_maps_padding() {
    let anchors = vec![[320.0, 320.0]];
    let detections = decode_level(
        0,
        8,
        &anchors,
        &[0.9],
        &[[1.0, 2.0, 3.0, 4.0]],
        &[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]],
        &transform(),
    )
    .unwrap();
    let detection = &detections[0];
    assert_eq!(detection.bbox, [152.0, 304.0, 32.0, 48.0]);
    assert_eq!(detection.keypoints[0], [168.0, 336.0]);
    assert_eq!(detection.score, 0.9);
}

#[test]
fn rejects_slice_length_mismatch_and_non_finite_values() {
    let anchors = vec![[10.0, 10.0]];
    assert!(matches!(
        decode_level(
            2,
            8,
            &anchors,
            &[],
            &[[1.0, 1.0, 1.0, 1.0]],
            &[[0.0; 10]],
            &transform()
        ),
        Err(FaceError::InvalidTensorLength { .. })
    ));
    assert!(matches!(
        decode_level(
            2,
            8,
            &anchors,
            &[f32::NAN],
            &[[1.0; 4]],
            &[[0.0; 10]],
            &transform()
        ),
        Err(FaceError::NonFiniteValue { .. })
    ));
}

#[test]
fn rejects_non_positive_clipped_area() {
    let anchors = vec![[0.0, 0.0]];
    assert!(matches!(
        decode_level(
            1,
            8,
            &anchors,
            &[0.9],
            &[[0.0, 0.0, 0.0, 0.0]],
            &[[0.0; 10]],
            &resize_with_padding(ImageSize {
                width: 640,
                height: 640
            })
            .unwrap()
        ),
        Err(FaceError::InvalidDetectionGeometry { index: 0 })
    ));
}

#[test]
fn public_detection_is_a_value_type() {
    let detection = Detection {
        bbox: [0.0; 4],
        score: 0.1,
        keypoints: [[0.0; 2]; 5],
    };
    assert_eq!(detection.bbox, [0.0; 4]);
}
