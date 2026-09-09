use feathertalk_pfld::{
    CropGeometry, PFLD_LANDMARK_COUNT, PFLD_OUTPUT_VALUE_COUNT, PfldError, decode_landmarks,
};

fn crop() -> CropGeometry {
    CropGeometry {
        width: 100,
        height: 80,
        offset_x: 10,
        offset_y: -5,
    }
}

fn vectors() -> (Vec<f32>, Vec<f32>) {
    let mut model = vec![0.0; PFLD_OUTPUT_VALUE_COUNT];
    let mut mean = vec![0.0; PFLD_OUTPUT_VALUE_COUNT];
    model[0] = 0.123;
    model[1] = 0.456;
    mean[0] = 0.1;
    mean[1] = 0.2;
    (model, mean)
}

#[test]
fn maps_mean_face_scale_truncation_and_offsets() {
    let (model, mean) = vectors();
    let landmarks = decode_landmarks(&model, &mean, crop()).unwrap();
    assert_eq!(landmarks.points().len(), PFLD_LANDMARK_COUNT);
    assert_eq!(landmarks.points()[0].x, 32);
    assert_eq!(landmarks.points()[0].y, 47);
}

#[test]
fn truncates_negative_values_toward_zero() {
    let mut model = vec![0.0; PFLD_OUTPUT_VALUE_COUNT];
    model[0] = -0.019;
    model[1] = -0.026;
    let landmarks = decode_landmarks(
        &model,
        &vec![0.0; PFLD_OUTPUT_VALUE_COUNT],
        CropGeometry {
            width: 100,
            height: 100,
            offset_x: 0,
            offset_y: 0,
        },
    )
    .unwrap();
    assert_eq!(landmarks.points()[0].x, -1);
    assert_eq!(landmarks.points()[0].y, -2);
}

#[test]
fn rejects_lengths_non_finite_values_and_zero_crop_dimensions() {
    let valid = vec![0.0; PFLD_OUTPUT_VALUE_COUNT];
    assert!(matches!(
        decode_landmarks(&valid[..PFLD_OUTPUT_VALUE_COUNT - 1], &valid, crop()),
        Err(PfldError::InvalidVectorLength { .. })
    ));
    let mut non_finite = valid.clone();
    non_finite[3] = f32::NAN;
    assert!(matches!(
        decode_landmarks(&non_finite, &valid, crop()),
        Err(PfldError::NonFiniteValue { .. })
    ));
    assert!(matches!(
        decode_landmarks(
            &valid,
            &valid,
            CropGeometry {
                width: 0,
                height: 80,
                offset_x: 0,
                offset_y: 0,
            }
        ),
        Err(PfldError::InvalidCropGeometry)
    ));
}

#[test]
fn rejects_coordinates_outside_i32_range() {
    let mut model = vec![0.0; PFLD_OUTPUT_VALUE_COUNT];
    model[0] = f32::MAX;
    assert!(matches!(
        decode_landmarks(&model, &vec![0.0; PFLD_OUTPUT_VALUE_COUNT], crop()),
        Err(PfldError::CoordinateOutOfRange { .. })
    ));
}
