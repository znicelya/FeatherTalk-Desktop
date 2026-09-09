use std::{
    path::{Path, PathBuf},
    time::Duration,
};

use feathertalk_media::{FrameRate, MediaError, MediaToolchain};

fn absolute(name: &str) -> PathBuf {
    std::env::current_dir().unwrap().join(name)
}

#[test]
fn toolchain_and_metadata_are_read_only_value_types() {
    let tools = MediaToolchain::new(
        absolute("ffmpeg.exe"),
        absolute("ffprobe.exe"),
        Duration::from_secs(30),
    )
    .unwrap();

    assert_eq!(tools.timeout(), Duration::from_secs(30));
    assert_eq!(tools.ffmpeg(), Path::new(&absolute("ffmpeg.exe")));
    assert_eq!(tools.ffprobe(), Path::new(&absolute("ffprobe.exe")));
    assert_eq!(FrameRate::new(25, 1).unwrap().frames_per_second(), 25.0);
    assert_eq!(FrameRate::new(25, 1).unwrap().numerator(), 25);
    assert_eq!(FrameRate::new(25, 1).unwrap().denominator(), 1);
}

#[test]
fn toolchain_rejects_relative_paths_and_invalid_timeout() {
    let absolute_ffprobe = absolute("ffprobe.exe");
    assert!(matches!(
        MediaToolchain::new(
            "ffmpeg.exe".into(),
            absolute_ffprobe.clone(),
            Duration::from_secs(1)
        ),
        Err(MediaError::InvalidToolchain {
            field: "ffmpeg",
            ..
        })
    ));
    assert!(matches!(
        MediaToolchain::new(
            absolute("ffmpeg.exe"),
            absolute_ffprobe.clone(),
            Duration::ZERO
        ),
        Err(MediaError::InvalidToolchain {
            field: "timeout",
            ..
        })
    ));
    assert!(matches!(
        MediaToolchain::new(
            absolute("ffmpeg.exe"),
            absolute_ffprobe,
            Duration::from_secs(24 * 60 * 60 + 1),
        ),
        Err(MediaError::InvalidToolchain {
            field: "timeout",
            ..
        })
    ));
}

#[test]
fn frame_rate_rejects_zero_components() {
    assert!(FrameRate::new(0, 1).is_err());
    assert!(FrameRate::new(25, 0).is_err());
}
