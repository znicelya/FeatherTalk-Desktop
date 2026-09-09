use std::path::Path;

use feathertalk_preprocess::{
    PFLD_LANDMARK_COUNT, Point, audio_window_indices, compute_face_bbox, default_crop_spec,
    read_landmarks,
};

#[test]
fn crate_root_exposes_read_only_preprocess_contract() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("face.lms");
    let contents = (0..PFLD_LANDMARK_COUNT)
        .map(|i| format!("{} {}\n", i + 1, i + 2))
        .collect::<String>();
    std::fs::write(&path, contents).unwrap();
    let landmarks = read_landmarks(&path).unwrap();
    let _: &[Point] = landmarks.points();
    let _ = compute_face_bbox(&landmarks).unwrap();
    let crop = default_crop_spec();
    assert_eq!(crop.crop_size, 168);
    assert_eq!(audio_window_indices(0, 2).unwrap()[0], None);
    let _: &Path = path.as_path();
}
