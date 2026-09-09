mod support;

use std::{path::PathBuf, time::Duration};

use feathertalk_frame_pipeline::{FrameExtractor, FramePipelineSpec, frame_command};

use support::{flag_number, flag_value};

fn extractor() -> FrameExtractor {
    FrameExtractor::new(
        PathBuf::from(r"C:\bundle\ffmpeg.exe"),
        Duration::from_secs(10),
    )
    .unwrap()
}

fn spec() -> FramePipelineSpec {
    FramePipelineSpec::new(
        PathBuf::from(r"C:\media\hostile name & $() ;.mp4"),
        PathBuf::from(r"C:\project\assets"),
        2,
        640,
        480,
    )
    .unwrap()
}

fn pattern() -> PathBuf {
    PathBuf::from(r"C:\project\assets\.feathertalk-frame-build-1-1\frames\%06d.jpg")
}

#[test]
fn frame_command_uses_fixed_flags_and_native_path_arguments() {
    let value = spec();
    let command = frame_command(&extractor(), value.video_path(), 26, 250, &pattern());
    assert_eq!(command.operation(), "extract_frames");
    assert_eq!(command.executable(), PathBuf::from(r"C:\bundle\ffmpeg.exe"));
    assert!(
        command
            .arguments()
            .windows(2)
            .any(|pair| pair == ["-vf", "fps=25"])
    );
    assert_eq!(flag_number(&command, "-frames:v"), 250);
    assert_eq!(flag_number(&command, "-start_number"), 26);
    assert_eq!(flag_value(&command, "-ss"), "1.040");
    assert_eq!(
        command.arguments().last(),
        Some(&pattern().into_os_string())
    );
    assert!(
        command
            .arguments()
            .contains(&value.video_path().as_os_str().to_owned())
    );
}

#[test]
fn frame_timestamps_keep_millisecond_precision() {
    let value = spec();
    let tool = extractor();
    // 25 frames per second divide 1000 ms exactly, so every index lands on a
    // whole millisecond and no case needs rounding.
    for (index, expected) in [
        (0_u64, "0.000"),
        (24, "0.960"),
        (25, "1.000"),
        (26, "1.040"),
        (49, "1.960"),
        (50, "2.000"),
        (1510, "60.400"),
    ] {
        let command = frame_command(&tool, value.video_path(), index, 1, &pattern());
        assert_eq!(flag_value(&command, "-ss"), expected, "frame {index}");
    }
}
