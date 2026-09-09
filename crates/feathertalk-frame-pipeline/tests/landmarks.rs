//! Reading landmark files back out of an asset package.

use std::fs;
use std::path::PathBuf;

use feathertalk_frame_pipeline::{
    LANDMARK_POINTS, MAX_LANDMARK_FILE_BYTES, PipelineError, read_landmark_file,
};
use tempfile::TempDir;

const FRAME_WIDTH: u32 = 512;
const FRAME_HEIGHT: u32 = 512;

/// The exact shape `serialize_landmarks` writes: one `"{x} {y}"` per line,
/// every line terminated. The largest point is (109, 218), well inside the
/// 512x512 frame these tests declare.
fn valid_text() -> String {
    let mut text = String::new();
    for index in 0..LANDMARK_POINTS {
        text.push_str(&format!("{index} {}\n", index * 2));
    }
    text
}

fn write(dir: &TempDir, name: &str, text: &str) -> PathBuf {
    let path = dir.path().join(name);
    fs::write(&path, text).expect("the fixture must be writable");
    path
}

#[test]
fn a_well_formed_file_reads_back_every_point() {
    let dir = TempDir::new().unwrap();
    let path = write(&dir, "000000.lms", &valid_text());
    let points = read_landmark_file(&path, FRAME_WIDTH, FRAME_HEIGHT).unwrap();
    assert_eq!(points.len(), LANDMARK_POINTS);
    assert_eq!(points[0], (0, 0));
    assert_eq!(points[109], (109, 218));
}

#[test]
fn malformed_bodies_are_refused_one_by_one() {
    let valid = valid_text();
    let one_line_short: String = valid
        .lines()
        .take(LANDMARK_POINTS - 1)
        .map(|line| format!("{line}\n"))
        .collect();
    let cases = vec![
        ("one line short", one_line_short),
        ("one line long", format!("{valid}110 220\n")),
        ("no trailing newline", valid.trim_end().to_owned()),
        ("windows line endings", valid.replace('\n', "\r\n")),
        ("two separators", valid.replacen("0 0", "0  0", 1)),
        ("fractional coordinate", valid.replacen("0 0", "0.5 0", 1)),
        ("negative coordinate", valid.replacen("0 0", "-1 0", 1)),
        ("point outside the frame", valid.replacen("0 0", "512 0", 1)),
        ("empty file", String::new()),
    ];
    for (label, text) in cases {
        let dir = TempDir::new().unwrap();
        let path = write(&dir, "000000.lms", &text);
        let error = read_landmark_file(&path, FRAME_WIDTH, FRAME_HEIGHT)
            .expect_err(&format!("{label} must be refused"));
        assert!(
            matches!(error, PipelineError::InvalidLandmark { .. }),
            "{label}: {error:?}"
        );
    }
}

#[test]
fn a_non_utf8_file_is_not_a_landmark_file() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("000000.lms");
    fs::write(&path, [0xFF, 0xFE, 0x00, b'\n']).unwrap();
    let error = read_landmark_file(&path, FRAME_WIDTH, FRAME_HEIGHT).unwrap_err();
    let PipelineError::InvalidLandmark { message, .. } = error else {
        panic!("non-UTF-8 bytes must be a landmark problem: {error:?}");
    };
    assert!(message.contains("not UTF-8"), "{message}");
}

#[test]
fn an_oversized_file_is_refused_with_its_limit() {
    let dir = TempDir::new().unwrap();
    let text = "0 0\n".repeat(4096);
    assert!(text.len() as u64 > MAX_LANDMARK_FILE_BYTES);
    let path = write(&dir, "000000.lms", &text);
    let error = read_landmark_file(&path, FRAME_WIDTH, FRAME_HEIGHT).unwrap_err();
    let PipelineError::InvalidLandmark { message, .. } = error else {
        panic!("an oversized file must be a landmark problem: {error:?}");
    };
    assert!(
        message.contains(&MAX_LANDMARK_FILE_BYTES.to_string()),
        "{message}"
    );
}

#[test]
fn a_directory_is_not_a_regular_landmark_file() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("000000.lms");
    fs::create_dir(&path).unwrap();
    let error = read_landmark_file(&path, FRAME_WIDTH, FRAME_HEIGHT).unwrap_err();
    assert!(
        matches!(error, PipelineError::LandmarkNotRegular { .. }),
        "{error:?}"
    );
}

#[test]
fn a_symlink_is_not_a_regular_landmark_file() {
    let dir = TempDir::new().unwrap();
    let target = write(&dir, "target.lms", &valid_text());
    let link = dir.path().join("000000.lms");
    #[cfg(windows)]
    let result = std::os::windows::fs::symlink_file(&target, &link);
    #[cfg(unix)]
    let result = std::os::unix::fs::symlink(&target, &link);
    if let Err(error) = result {
        // 1314 is ERROR_PRIVILEGE_NOT_HELD: an unprivileged Windows account
        // cannot create symlinks, so the case is skipped rather than failed.
        if error.raw_os_error() == Some(1314) {
            eprintln!("skipping: this account may not create symlinks");
            return;
        }
        panic!("the symlink must be creatable: {error:?}");
    }
    let error = read_landmark_file(&link, FRAME_WIDTH, FRAME_HEIGHT).unwrap_err();
    assert!(
        matches!(error, PipelineError::LandmarkNotRegular { .. }),
        "{error:?}"
    );
}

#[test]
fn a_missing_file_names_the_failed_operation() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("000000.lms");
    let error = read_landmark_file(&path, FRAME_WIDTH, FRAME_HEIGHT).unwrap_err();
    let PipelineError::Io {
        operation,
        path: reported,
        ..
    } = error
    else {
        panic!("a missing file must be an IO failure: {error:?}");
    };
    assert_eq!(operation, "stat_landmarks");
    assert_eq!(reported, path);
}
