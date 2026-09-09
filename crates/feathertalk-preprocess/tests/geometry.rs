use feathertalk_preprocess::{
    CropSpec, FaceBoundingBox, Landmarks, MaskRect, MouthRoiSpec, PFLD_LANDMARK_COUNT,
    PreprocessError, compute_face_bbox, default_crop_spec, default_mouth_roi_spec, mouth_roi_rect,
    read_landmarks,
};

fn landmarks_file(x1: f32, x31: f32, y52: f32) -> (tempfile::TempDir, std::path::PathBuf) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("face.lms");
    let mut lines = (0..PFLD_LANDMARK_COUNT)
        .map(|_| "0 0".to_owned())
        .collect::<Vec<_>>();
    lines[1] = format!("{x1} 0");
    lines[31] = format!("{x31} 0");
    lines[52] = format!("0 {y52}");
    std::fs::write(&path, lines.join("\n")).unwrap();
    (dir, path)
}

#[test]
fn computes_square_bbox_with_python_truncation() {
    let (_dir, path) = landmarks_file(10.9, 50.8, 20.7);
    let landmarks = read_landmarks(&path).unwrap();
    let bbox = compute_face_bbox(&landmarks).unwrap();
    assert_eq!(
        bbox,
        FaceBoundingBox {
            xmin: 10,
            ymin: 20,
            xmax: 50,
            ymax: 60
        }
    );
}

#[test]
fn rejects_non_positive_bbox_width() {
    let (_dir, path) = landmarks_file(50.0, 10.0, 20.0);
    let landmarks = read_landmarks(&path).unwrap();
    assert!(matches!(
        compute_face_bbox(&landmarks),
        Err(PreprocessError::InvalidGeometry { .. })
    ));
}

#[test]
fn default_crop_spec_matches_python_constants() {
    let spec = default_crop_spec();
    assert_eq!(spec.crop_size, 168);
    assert_eq!(spec.inner_size, 160);
    assert_eq!(spec.border, 4);
    assert_eq!(
        spec.mouth_mask,
        MaskRect {
            x: 5,
            y: 5,
            width: 150,
            height: 145
        }
    );
    assert_eq!(spec.crop_size, spec.inner_size + 2 * spec.border);
}

fn mouth_landmarks_file(
    x1: f32,
    x31: f32,
    y52: f32,
    mouth: &[(f32, f32)],
) -> (tempfile::TempDir, std::path::PathBuf) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("face.lms");
    let mut lines = (0..PFLD_LANDMARK_COUNT)
        .map(|_| "0 0".to_owned())
        .collect::<Vec<_>>();
    lines[1] = format!("{x1} 0");
    lines[31] = format!("{x31} 0");
    lines[52] = format!("0 {y52}");
    for (offset, (x, y)) in mouth.iter().enumerate() {
        lines[90 + offset] = format!("{x} {y}");
    }
    std::fs::write(&path, lines.join("\n")).unwrap();
    (dir, path)
}

fn roi(path: &std::path::Path) -> MaskRect {
    let landmarks = read_landmarks(path).unwrap();
    mouth_roi_rect(&landmarks, &default_crop_spec(), &default_mouth_roi_spec()).unwrap()
}

fn rejection_field(landmarks: &Landmarks, crop: &CropSpec, spec: &MouthRoiSpec) -> &'static str {
    match mouth_roi_rect(landmarks, crop, spec) {
        Err(PreprocessError::InvalidGeometry { field, .. }) => field,
        other => panic!("expected a geometry rejection, got {other:?}"),
    }
}

#[test]
fn default_mouth_roi_spec_matches_python_constants() {
    let spec = default_mouth_roi_spec();
    assert_eq!((spec.start, spec.end), (90, 110));
    assert_eq!((spec.expand_x, spec.expand_y), (1.45, 1.75));
    assert_eq!((spec.min_w, spec.min_h, spec.pad), (52, 36, 2));
}

#[test]
fn mouth_roi_projects_landmarks_into_the_inner_crop() {
    let mut mouth = vec![(120.0, 170.0); 20];
    mouth[0] = (100.0, 160.0);
    mouth[19] = (140.0, 180.0);
    let (_dir, path) = mouth_landmarks_file(40.0, 200.0, 60.0, &mouth);
    assert_eq!(
        roi(&path),
        MaskRect {
            x: 47,
            y: 90,
            width: 66,
            height: 43
        }
    );
}

#[test]
fn mouth_roi_truncates_landmark_coordinates_like_python() {
    let mut mouth = vec![(120.9, 170.9); 20];
    mouth[0] = (100.9, 160.9);
    mouth[19] = (140.9, 180.9);
    let (_dir, path) = mouth_landmarks_file(40.9, 200.9, 60.9, &mouth);
    assert_eq!(
        roi(&path),
        MaskRect {
            x: 47,
            y: 90,
            width: 66,
            height: 43
        }
    );
}

#[test]
fn mouth_roi_rounds_half_to_even_instead_of_half_away_from_zero() {
    let mut mouth = vec![(120.0, 160.0); 20];
    mouth[0] = (114.0, 154.0);
    mouth[19] = (135.0, 174.0);
    let (_dir, path) = mouth_landmarks_file(40.0, 208.0, 60.0, &mouth);
    assert_eq!(
        roi(&path),
        MaskRect {
            x: 54,
            y: 79,
            width: 52,
            height: 42
        }
    );
}

#[test]
fn mouth_roi_grows_a_degenerate_span_to_the_minimum_extents() {
    let (_dir, path) = mouth_landmarks_file(40.0, 208.0, 60.0, &[(114.0, 154.0); 20]);
    assert_eq!(
        roi(&path),
        MaskRect {
            x: 44,
            y: 72,
            width: 52,
            height: 36
        }
    );
}

#[test]
fn mouth_roi_clamps_to_the_left_and_right_edges() {
    let mut left = vec![(49.0, 164.0); 20];
    left[0] = (44.0, 164.0);
    let (_left_dir, left_path) = mouth_landmarks_file(40.0, 208.0, 60.0, &left);
    assert_eq!(
        roi(&left_path),
        MaskRect {
            x: 0,
            y: 82,
            width: 28,
            height: 36
        }
    );
    let mut right = vec![(199.0, 164.0); 20];
    right[19] = (203.0, 164.0);
    let (_right_dir, right_path) = mouth_landmarks_file(40.0, 208.0, 60.0, &right);
    assert_eq!(
        roi(&right_path),
        MaskRect {
            x: 131,
            y: 82,
            width: 29,
            height: 36
        }
    );
}

#[test]
fn mouth_roi_keeps_a_one_pixel_rectangle_for_landmarks_outside_the_crop() {
    let (_dir, path) = mouth_landmarks_file(40.0, 208.0, 60.0, &[(344.0, 344.0); 20]);
    assert_eq!(
        roi(&path),
        MaskRect {
            x: 159,
            y: 159,
            width: 1,
            height: 1
        }
    );
}

#[test]
fn mouth_roi_rejects_inconsistent_specs() {
    let (_dir, path) = mouth_landmarks_file(40.0, 208.0, 60.0, &[(120.0, 160.0); 20]);
    let landmarks = read_landmarks(&path).unwrap();
    let crop = default_crop_spec();
    let spec = default_mouth_roi_spec();
    assert_eq!(
        rejection_field(&landmarks, &crop, &MouthRoiSpec { start: 110, ..spec }),
        "mouth_roi_range"
    );
    assert_eq!(
        rejection_field(&landmarks, &crop, &MouthRoiSpec { end: 111, ..spec }),
        "mouth_roi_range"
    );
    assert_eq!(
        rejection_field(
            &landmarks,
            &crop,
            &MouthRoiSpec {
                expand_x: 0.0,
                ..spec
            }
        ),
        "mouth_roi_expand"
    );
    assert_eq!(
        rejection_field(
            &landmarks,
            &crop,
            &MouthRoiSpec {
                expand_y: f32::NAN,
                ..spec
            }
        ),
        "mouth_roi_expand"
    );
    assert_eq!(
        rejection_field(&landmarks, &crop, &MouthRoiSpec { min_w: 0, ..spec }),
        "mouth_roi_min_size"
    );
    assert_eq!(
        rejection_field(&landmarks, &crop, &MouthRoiSpec { min_h: 161, ..spec }),
        "mouth_roi_min_size"
    );
    assert_eq!(
        rejection_field(
            &landmarks,
            &CropSpec {
                inner_size: 0,
                ..crop
            },
            &spec
        ),
        "inner_size"
    );
    assert_eq!(
        rejection_field(
            &landmarks,
            &CropSpec {
                crop_size: 0,
                ..crop
            },
            &spec
        ),
        "mouth_roi_projection"
    );
}
