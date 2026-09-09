use std::{
    ffi::OsString,
    path::{Path, PathBuf},
};

use crate::MediaToolchain;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommandSpec {
    executable: PathBuf,
    arguments: Vec<OsString>,
    operation: &'static str,
}

impl CommandSpec {
    pub(crate) fn new(
        executable: PathBuf,
        arguments: Vec<OsString>,
        operation: &'static str,
    ) -> Self {
        Self {
            executable,
            arguments,
            operation,
        }
    }

    pub fn executable(&self) -> &Path {
        &self.executable
    }
    pub fn arguments(&self) -> &[OsString] {
        &self.arguments
    }
    pub fn operation(&self) -> &'static str {
        self.operation
    }
}

pub fn probe_command(toolchain: &MediaToolchain, source: &Path) -> CommandSpec {
    let entries = "format=format_name,duration:stream=codec_type,codec_name,width,height,pix_fmt,avg_frame_rate,r_frame_rate,nb_read_frames,duration,sample_fmt,sample_rate,channels,duration_ts,time_base";
    CommandSpec::new(
        toolchain.ffprobe().to_owned(),
        args([
            "-v",
            "error",
            "-count_frames",
            "-show_entries",
            entries,
            "-of",
            "json",
        ])
        .into_iter()
        .chain([source.as_os_str().to_owned()])
        .collect(),
        "probe",
    )
}

pub fn video_normalization_command(
    toolchain: &MediaToolchain,
    source: &Path,
    output: &Path,
) -> CommandSpec {
    let mut arguments = args(["-hide_banner", "-nostdin", "-y", "-v", "error", "-i"]);
    arguments.extend([source.as_os_str().to_owned()]);
    arguments.extend(args([
        "-map",
        "0:v:0",
        "-an",
        "-sn",
        "-dn",
        "-map_metadata",
        "-1",
        "-map_chapters",
        "-1",
        "-vf",
        "fps=25",
        "-fps_mode",
        "cfr",
        "-c:v",
        "mpeg4",
        "-q:v",
        "2",
        "-pix_fmt",
        "yuv420p",
        "-f",
        "mp4",
    ]));
    arguments.push(output.as_os_str().to_owned());
    CommandSpec::new(toolchain.ffmpeg().to_owned(), arguments, "normalize_video")
}

pub fn audio_normalization_command(
    toolchain: &MediaToolchain,
    source: &Path,
    output: &Path,
) -> CommandSpec {
    let mut arguments = args(["-hide_banner", "-nostdin", "-y", "-v", "error", "-i"]);
    arguments.extend([source.as_os_str().to_owned()]);
    arguments.extend(args([
        "-map",
        "0:a:0",
        "-vn",
        "-sn",
        "-dn",
        "-map_metadata",
        "-1",
        "-map_chapters",
        "-1",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        "-f",
        "wav",
    ]));
    arguments.push(output.as_os_str().to_owned());
    CommandSpec::new(toolchain.ffmpeg().to_owned(), arguments, "normalize_audio")
}

fn args<const N: usize>(values: [&str; N]) -> Vec<OsString> {
    values.into_iter().map(OsString::from).collect()
}
