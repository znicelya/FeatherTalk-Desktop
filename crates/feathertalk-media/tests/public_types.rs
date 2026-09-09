use std::path::{Path, PathBuf};

use feathertalk_media::{MediaInput, NormalizationSpec};

#[test]
fn request_types_hold_native_paths_and_fixed_target_values() {
    let input = MediaInput {
        source: PathBuf::from("source/input.mp4"),
    };
    let spec = NormalizationSpec {
        target_video_fps: 25,
        target_audio_sample_rate: 16_000,
        target_audio_channels: 1,
        output_dir: PathBuf::from("assets"),
    };
    assert_eq!(input.source, Path::new("source/input.mp4"));
    assert_eq!(spec.target_video_fps, 25);
    assert_eq!(spec.target_audio_sample_rate, 16_000);
    assert_eq!(spec.target_audio_channels, 1);
    assert_eq!(spec.output_dir, Path::new("assets"));
}
