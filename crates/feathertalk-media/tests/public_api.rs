use std::path::Path;

use feathertalk_media::{MediaInput, NormalizationSpec, validate_input, validate_normalization};

#[test]
fn crate_root_exposes_read_only_validated_paths() {
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("input.mp4");
    std::fs::write(&source, b"media").unwrap();
    let input = validate_input(&MediaInput { source }).unwrap();
    let spec = NormalizationSpec {
        target_video_fps: 25,
        target_audio_sample_rate: 16_000,
        target_audio_channels: 1,
        output_dir: dir.path().join("assets"),
    };
    let layout = validate_normalization(&input, &spec).unwrap();
    let _: &Path = input.source();
    let _: &Path = layout.output_dir();
    let _: &Path = layout.video_path();
    let _: &Path = layout.audio_path();
    assert!(layout.video_path().ends_with("video_25fps.mp4"));
}
