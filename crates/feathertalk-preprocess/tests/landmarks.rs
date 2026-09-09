use std::path::Path;

use feathertalk_preprocess::{PFLD_LANDMARK_COUNT, Point, PreprocessError, read_landmarks};

fn content() -> String {
    (0..PFLD_LANDMARK_COUNT)
        .map(|i| format!("{} {}\n", i + 1, i + 2))
        .collect()
}

#[test]
fn parses_exactly_110_points_and_ignores_blank_lines() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("face.lms");
    std::fs::write(&path, format!("\n{}\n", content())).unwrap();
    let landmarks = read_landmarks(&path).unwrap();
    assert_eq!(landmarks.points().len(), PFLD_LANDMARK_COUNT);
    assert_eq!(landmarks.points()[0], Point { x: 1.0, y: 2.0 });
}

#[test]
fn rejects_wrong_count_and_invalid_lines() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("face.lms");
    std::fs::write(&path, "1 2 3\n").unwrap();
    assert!(matches!(
        read_landmarks(&path),
        Err(PreprocessError::InvalidLine { .. })
    ));
    std::fs::write(
        &path,
        content().lines().take(109).collect::<Vec<_>>().join("\n"),
    )
    .unwrap();
    assert!(matches!(
        read_landmarks(&path),
        Err(PreprocessError::WrongLandmarkCount {
            expected: 110,
            actual: 109,
            ..
        })
    ));
    std::fs::write(&path, format!("{}0 0\n", content())).unwrap();
    assert!(matches!(
        read_landmarks(&path),
        Err(PreprocessError::WrongLandmarkCount {
            expected: 110,
            actual: 111,
            ..
        })
    ));
}

#[test]
fn rejects_non_finite_negative_invalid_utf8_and_missing_input() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("face.lms");
    std::fs::write(&path, "NaN 1\n").unwrap();
    assert!(matches!(
        read_landmarks(&path),
        Err(PreprocessError::NonFiniteCoordinate { .. })
    ));
    std::fs::write(&path, "-1 1\n").unwrap();
    assert!(matches!(
        read_landmarks(&path),
        Err(PreprocessError::NegativeCoordinate { .. })
    ));
    std::fs::write(&path, [0xff, 0xfe]).unwrap();
    assert!(matches!(
        read_landmarks(&path),
        Err(PreprocessError::InvalidUtf8 { .. })
    ));
    assert!(matches!(
        read_landmarks(&dir.path().join("missing.lms")),
        Err(PreprocessError::Io { .. })
    ));
}

#[test]
fn parser_api_accepts_path_references() {
    let _: fn(&Path) -> Result<_, _> = read_landmarks;
}
