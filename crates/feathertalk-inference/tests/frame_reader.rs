use std::path::Path;

use base64::Engine;
use feathertalk_inference::{FrameReader, InferenceError, JpegFrameReader};

// Deterministic 1x1 red JPEG, embedded so tests do not depend on repo assets.
const RED_1X1_JPEG_BASE64: &str = "/9j/4AAQSkZJRgABAQEAYABgAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/2wBDAQkJCQwLDBgNDRgyIRwhMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjL/wAARCAABAAEDASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwDi6KKK+ZP3E//Z";

fn red_jpeg() -> Vec<u8> {
    base64::engine::general_purpose::STANDARD
        .decode(RED_1X1_JPEG_BASE64)
        .unwrap()
}

#[test]
fn jpeg_reader_decodes_rgb_as_bgr() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("000000.jpg");
    std::fs::write(&path, red_jpeg()).unwrap();

    let frame = JpegFrameReader::default().read(0, &path).unwrap();
    assert_eq!((frame.width(), frame.height()), (1, 1));
    let pixel = frame.as_bytes();
    assert_eq!(pixel.len(), 3);
    assert!(pixel[2] > pixel[0]);
    assert!(pixel[2] > pixel[1]);
}

#[test]
fn reader_rejects_corrupt_and_zero_limit_inputs() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("bad.jpg");
    std::fs::write(&path, b"not jpeg").unwrap();
    assert!(matches!(
        JpegFrameReader::default().read(0, &path),
        Err(InferenceError::FrameReader { index: 0, .. })
    ));
    assert!(matches!(
        JpegFrameReader::new(0).read(0, &path),
        Err(InferenceError::FrameReader { index: 0, .. })
    ));
}

#[test]
fn reader_rejects_non_regular_paths_without_following_symlinks() {
    let dir = tempfile::tempdir().unwrap();
    let directory = dir.path().join("directory");
    std::fs::create_dir(&directory).unwrap();
    assert!(matches!(
        JpegFrameReader::default().read(3, &directory),
        Err(InferenceError::FrameReader { index: 3, .. })
    ));

    let target = dir.path().join("target.jpg");
    std::fs::write(&target, red_jpeg()).unwrap();
    let link = dir.path().join("link.jpg");
    #[cfg(windows)]
    let result = std::os::windows::fs::symlink_file(&target, &link);
    #[cfg(unix)]
    let result = std::os::unix::fs::symlink(&target, &link);
    if result.is_ok() {
        assert!(matches!(
            JpegFrameReader::default().read(4, Path::new(&link)),
            Err(InferenceError::FrameReader { index: 4, .. })
        ));
    }
}
