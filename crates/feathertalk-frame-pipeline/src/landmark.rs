//! Landmark files read back out of a finished asset package.

use std::fs::{self, File};
use std::io::Read;
use std::path::Path;

use crate::error::PipelineError;

/// The number of landmark points PFLD produces for one frame.
///
/// `serialize_landmarks` writes exactly this many lines and the reader below
/// demands exactly this many, so the writer and the reader cannot drift.
pub const LANDMARK_POINTS: usize = 110;

/// The largest landmark file this reader will accept.
///
/// The longest line the writer can emit is `"32767 32767\n"`, twelve bytes,
/// so a complete file is at most 1 320 bytes. Eight KiB leaves six times that
/// headroom while still refusing a file that has been replaced by something
/// else entirely, before any of it is read into memory.
pub const MAX_LANDMARK_FILE_BYTES: u64 = 8 * 1024;

/// Read one landmark file and validate it against the frame it belongs to.
///
/// Accepts only what `serialize_landmarks` writes: `LANDMARK_POINTS` lines of
/// `"{x} {y}"`, each terminated by a single `\n`, every point inside the
/// frame. The geometry is passed in because the file does not record it.
pub fn read_landmark_file(
    path: &Path,
    frame_width: u32,
    frame_height: u32,
) -> Result<Vec<(i32, i32)>, PipelineError> {
    let metadata =
        fs::symlink_metadata(path).map_err(|source| io("stat_landmarks", path, source))?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(PipelineError::LandmarkNotRegular {
            path: path.to_owned(),
        });
    }
    let size = metadata.len();
    if size > MAX_LANDMARK_FILE_BYTES {
        return Err(invalid_landmark(
            path,
            format!("file is {size} bytes, over the {MAX_LANDMARK_FILE_BYTES} byte limit"),
        ));
    }
    let mut file = File::open(path).map_err(|source| io("open_landmarks", path, source))?;
    let mut bytes = Vec::with_capacity(size as usize);
    file.read_to_end(&mut bytes)
        .map_err(|source| io("read_landmarks", path, source))?;
    parse_landmarks(path, &bytes, frame_width, frame_height)
}

/// Parse the file body, kept separate from the IO so the shape rules read in
/// one place.
fn parse_landmarks(
    path: &Path,
    bytes: &[u8],
    frame_width: u32,
    frame_height: u32,
) -> Result<Vec<(i32, i32)>, PipelineError> {
    let text = std::str::from_utf8(bytes)
        .map_err(|error| invalid_landmark(path, format!("file is not UTF-8: {error}")))?;
    let body = text
        .strip_suffix('\n')
        .ok_or_else(|| invalid_landmark(path, "file does not end with a newline".to_owned()))?;
    // `split` rather than `lines`: `lines` tolerates a missing final
    // terminator and silently strips a trailing `\r`, and both of those are
    // files this reader must refuse rather than quietly repair.
    let lines: Vec<&str> = body.split('\n').collect();
    if lines.len() != LANDMARK_POINTS {
        return Err(invalid_landmark(
            path,
            format!("expected {LANDMARK_POINTS} lines, found {}", lines.len()),
        ));
    }
    let mut points = Vec::with_capacity(LANDMARK_POINTS);
    for (index, line) in lines.iter().enumerate() {
        points.push(parse_point(path, index, line, frame_width, frame_height)?);
    }
    Ok(points)
}

fn parse_point(
    path: &Path,
    index: usize,
    line: &str,
    frame_width: u32,
    frame_height: u32,
) -> Result<(i32, i32), PipelineError> {
    let (x_text, y_text) = line.split_once(' ').ok_or_else(|| {
        invalid_landmark(
            path,
            format!("line {index} is not two integers separated by one space: {line:?}"),
        )
    })?;
    let x = parse_coordinate(path, index, "x", x_text)?;
    let y = parse_coordinate(path, index, "y", y_text)?;
    if x < 0 || y < 0 || x >= frame_width as i32 || y >= frame_height as i32 {
        return Err(invalid_landmark(
            path,
            format!("line {index} point ({x}, {y}) is outside {frame_width}x{frame_height}"),
        ));
    }
    Ok((x, y))
}

fn parse_coordinate(
    path: &Path,
    index: usize,
    axis: &'static str,
    text: &str,
) -> Result<i32, PipelineError> {
    text.parse::<i32>().map_err(|error| {
        invalid_landmark(
            path,
            format!("line {index} has a bad {axis} coordinate {text:?}: {error}"),
        )
    })
}

/// `publish.rs` keeps its own private copy of this helper rather than sharing
/// one, so this module follows the same local pattern.
fn io(operation: &'static str, path: &Path, source: std::io::Error) -> PipelineError {
    PipelineError::Io {
        operation,
        path: path.to_owned(),
        source,
    }
}

fn invalid_landmark(path: &Path, message: String) -> PipelineError {
    PipelineError::InvalidLandmark {
        path: path.to_owned(),
        message,
    }
}
