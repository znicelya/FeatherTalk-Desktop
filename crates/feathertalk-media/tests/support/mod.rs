#![allow(dead_code)]

use std::path::Path;

use feathertalk_media::{MediaInput, NormalizationSpec, ValidatedInput, validate_input};

pub fn validated_source() -> (tempfile::TempDir, ValidatedInput) {
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("input.mp4");
    std::fs::write(&source, b"media").unwrap();
    let input = validate_input(&MediaInput { source }).unwrap();
    (dir, input)
}

pub fn normalization_spec(output_dir: std::path::PathBuf) -> NormalizationSpec {
    NormalizationSpec {
        target_video_fps: 25,
        target_audio_sample_rate: 16_000,
        target_audio_channels: 1,
        output_dir,
    }
}

#[cfg(unix)]
pub fn create_file_symlink(target: &Path, link: &Path) -> std::io::Result<()> {
    std::os::unix::fs::symlink(target, link)
}

#[cfg(windows)]
pub fn create_file_symlink(target: &Path, link: &Path) -> std::io::Result<()> {
    std::os::windows::fs::symlink_file(target, link)
}

#[cfg(unix)]
pub fn create_dir_symlink(target: &Path, link: &Path) -> std::io::Result<()> {
    std::os::unix::fs::symlink(target, link)
}

#[cfg(windows)]
pub fn create_dir_symlink(target: &Path, link: &Path) -> std::io::Result<()> {
    std::os::windows::fs::symlink_dir(target, link)
}
