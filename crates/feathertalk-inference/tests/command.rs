use std::{ffi::OsString, path::Path};

use feathertalk_inference::{RawFrameRenderSpec, raw_video_command};

#[test]
fn raw_video_command_has_stable_argument_order_and_native_paths() {
    let spec = RawFrameRenderSpec::new(
        640,
        480,
        Path::new("audio file.wav"),
        Path::new("result file.mp4"),
    )
    .unwrap();
    let command = raw_video_command(Path::new("C:/tools/ffmpeg.exe"), &spec).unwrap();
    assert_eq!(command.executable(), Path::new("C:/tools/ffmpeg.exe"));
    assert_eq!(command.operation(), "render_raw_video");
    assert_eq!(
        command.arguments(),
        &[
            OsString::from("-hide_banner"),
            OsString::from("-nostdin"),
            OsString::from("-y"),
            OsString::from("-v"),
            OsString::from("error"),
            OsString::from("-f"),
            OsString::from("rawvideo"),
            OsString::from("-pix_fmt"),
            OsString::from("bgr24"),
            OsString::from("-video_size"),
            OsString::from("640x480"),
            OsString::from("-framerate"),
            OsString::from("25"),
            OsString::from("-i"),
            OsString::from("-"),
            OsString::from("-i"),
            OsString::from("audio file.wav"),
            OsString::from("-c:v"),
            OsString::from("libx264"),
            OsString::from("-pix_fmt"),
            OsString::from("yuv420p"),
            OsString::from("-c:a"),
            OsString::from("aac"),
            OsString::from("-shortest"),
            OsString::from("result file.mp4"),
        ]
    );
}

#[test]
fn command_rejects_empty_or_relative_ffmpeg_paths() {
    let spec = RawFrameRenderSpec::new(1, 1, Path::new("a.wav"), Path::new("o.mp4")).unwrap();
    assert!(raw_video_command(Path::new(""), &spec).is_err());
    assert!(raw_video_command(Path::new("ffmpeg"), &spec).is_err());
}
