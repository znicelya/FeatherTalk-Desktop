use std::path::PathBuf;

use feathertalk_media::{
    MediaToolchain, audio_normalization_command, probe_command, video_normalization_command,
};

fn toolchain() -> MediaToolchain {
    MediaToolchain::new(
        PathBuf::from(r"C:\bundle\ffmpeg.exe"),
        PathBuf::from(r"C:\bundle\ffprobe.exe"),
        std::time::Duration::from_secs(30),
    )
    .unwrap()
}

#[test]
fn probe_command_has_fixed_entries_and_single_path_argument() {
    let source = PathBuf::from(r"C:\media\name with 'quotes' & $() ;.mov");
    let command = probe_command(&toolchain(), &source);
    assert_eq!(
        command.executable(),
        PathBuf::from(r"C:\bundle\ffprobe.exe")
    );
    assert_eq!(command.arguments()[0], "-v");
    assert!(
        command
            .arguments()
            .windows(2)
            .any(|pair| pair == ["-count_frames", "-show_entries"])
    );
    assert!(command.arguments().contains(&"-of".into()));
    assert_eq!(command.arguments().last(), Some(&source.into_os_string()));
    assert_eq!(command.operation(), "probe");
}

#[test]
fn normalization_commands_use_fixed_codecs_filters_and_outputs() {
    let source = PathBuf::from(r"C:\media\input.mov");
    let video = PathBuf::from(r"C:\assets\.video.tmp.mp4");
    let audio = PathBuf::from(r"C:\assets\.audio.tmp.wav");
    let video_command = video_normalization_command(&toolchain(), &source, &video);
    let audio_command = audio_normalization_command(&toolchain(), &source, &audio);

    assert!(
        video_command
            .arguments()
            .windows(2)
            .any(|pair| pair == ["-vf", "fps=25"])
    );
    assert!(
        video_command
            .arguments()
            .windows(2)
            .any(|pair| pair == ["-c:v", "mpeg4"])
    );
    assert!(
        video_command
            .arguments()
            .windows(2)
            .any(|pair| pair == ["-pix_fmt", "yuv420p"])
    );
    assert_eq!(
        video_command.arguments().last(),
        Some(&video.into_os_string())
    );
    assert!(
        audio_command
            .arguments()
            .windows(2)
            .any(|pair| pair == ["-ac", "1"])
    );
    assert!(
        audio_command
            .arguments()
            .windows(2)
            .any(|pair| pair == ["-ar", "16000"])
    );
    assert!(
        audio_command
            .arguments()
            .windows(2)
            .any(|pair| pair == ["-c:a", "pcm_s16le"])
    );
    assert_eq!(
        audio_command.arguments().last(),
        Some(&audio.into_os_string())
    );
}
