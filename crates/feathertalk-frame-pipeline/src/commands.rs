use std::{
    ffi::OsString,
    path::{Path, PathBuf},
};

use crate::process::FrameExtractor;

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

/// The frame rate the extraction pipeline pins ffmpeg to.
///
/// `-vf fps=25` stays a literal in the argument list: it is filter syntax, and
/// spelling it out keeps the command readable next to ffmpeg documentation.
const FRAME_RATE: u64 = 25;

/// Milliseconds one frame occupies. 25 divides 1000 exactly, so every frame
/// index maps onto a whole number of milliseconds and `-ss` never rounds.
const MILLIS_PER_FRAME: u64 = 1_000 / FRAME_RATE;

/// One ffmpeg invocation that writes `count` frames starting at `first_index`.
///
/// `output_pattern` must be an `image2` pattern such as `frames/%06d.jpg`;
/// ffmpeg expands it with `-start_number`, so the file names match
/// `FramePipelineSpec::frame_path`.
pub fn frame_command(
    extractor: &FrameExtractor,
    source: &Path,
    first_index: u64,
    count: u64,
    output_pattern: &Path,
) -> CommandSpec {
    let timestamp = format_timestamp(first_index);
    let mut arguments = args(["-hide_banner", "-nostdin", "-y", "-v", "error", "-ss"]);
    arguments.push(timestamp);
    arguments.push("-i".into());
    arguments.push(source.as_os_str().to_owned());
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
    ]));
    arguments.push("-frames:v".into());
    arguments.push(count.to_string().into());
    arguments.push("-start_number".into());
    arguments.push(first_index.to_string().into());
    arguments.extend(args(["-q:v", "2", "-f", "image2"]));
    arguments.push(output_pattern.as_os_str().to_owned());
    CommandSpec::new(extractor.ffmpeg().to_owned(), arguments, "extract_frames")
}

fn format_timestamp(index: u64) -> OsString {
    let seconds = index / FRAME_RATE;
    let millis = (index % FRAME_RATE) * MILLIS_PER_FRAME;
    format!("{seconds}.{millis:03}").into()
}

fn args<const N: usize>(values: [&str; N]) -> Vec<OsString> {
    values.into_iter().map(OsString::from).collect()
}
