use std::{
    ffi::OsString,
    path::{Path, PathBuf},
};

use crate::{InferenceError, RawFrameRenderSpec};

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

pub fn raw_video_command(
    ffmpeg: &Path,
    spec: &RawFrameRenderSpec,
) -> Result<CommandSpec, InferenceError> {
    if ffmpeg.as_os_str().is_empty() {
        return Err(InferenceError::EmptyFfmpegPath);
    }
    if !ffmpeg.is_absolute() {
        return Err(InferenceError::FfmpegPathNotAbsolute {
            path: ffmpeg.to_owned(),
        });
    }

    let mut arguments = Vec::with_capacity(26);
    push_args(
        &mut arguments,
        [
            "-hide_banner",
            "-nostdin",
            "-y",
            "-v",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-video_size",
        ],
    );
    arguments.push(format!("{}x{}", spec.width(), spec.height()).into());
    push_args(&mut arguments, ["-framerate", "25", "-i", "-"]);
    arguments.push("-i".into());
    arguments.push(spec.audio_path().as_os_str().to_owned());
    push_args(
        &mut arguments,
        [
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
        ],
    );
    arguments.push(spec.output_path().as_os_str().to_owned());

    Ok(CommandSpec::new(
        ffmpeg.to_owned(),
        arguments,
        "render_raw_video",
    ))
}

fn push_args<const N: usize>(arguments: &mut Vec<OsString>, values: [&str; N]) {
    arguments.extend(values.into_iter().map(OsString::from));
}
