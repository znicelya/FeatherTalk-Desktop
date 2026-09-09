use std::{
    fs::{self, OpenOptions},
    io::Read,
    path::{Path, PathBuf},
};

use sha2::{Digest, Sha256};

use crate::{
    CommandSpec, FramePipelineSpec, NoObserver, PipelineError, PipelineObserver, PipelinePhase,
    ProcessOutput, ProcessRunner, SystemProcessRunner,
    commands::frame_command,
    process::{FrameExtractor, MAX_CAPTURE_BYTES, MAX_FRAME_BYTES, next_id},
};

/// How many frames one ffmpeg invocation writes.
///
/// Measured against `demo/feathertalk_demo_latest_188.mp4` (1511 frames,
/// 1280x720, 25 fps): one process per frame costs 129-193 ms of process and
/// decoder start-up, roughly 255 s for the clip, while chunks of 250 finish in
/// 3.2 s with byte-identical JPEG output. The chunk also bounds how long a
/// cancellation waits, measured at about 1.1 s for a chunk of 250 frames at
/// this resolution.
pub const FRAME_CHUNK: u64 = 250;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExtractedFrame {
    index: u64,
    path: PathBuf,
    bytes: u64,
    sha256: String,
}

impl ExtractedFrame {
    pub fn index(&self) -> u64 {
        self.index
    }
    pub fn path(&self) -> &Path {
        &self.path
    }
    pub fn bytes(&self) -> u64 {
        self.bytes
    }
    pub fn sha256(&self) -> &str {
        &self.sha256
    }
}

#[derive(Debug)]
pub struct FrameBatch {
    staging_dir: PathBuf,
    frames: Vec<ExtractedFrame>,
    armed: bool,
}

impl FrameBatch {
    pub fn staging_dir(&self) -> &Path {
        &self.staging_dir
    }
    pub fn frames(&self) -> &[ExtractedFrame] {
        &self.frames
    }
    pub fn disarm(&mut self) {
        self.armed = false;
    }

    #[cfg(test)]
    pub(crate) fn from_staging_dir_for_test(staging_dir: PathBuf) -> Self {
        Self {
            staging_dir,
            frames: Vec::new(),
            armed: true,
        }
    }
}

impl Drop for FrameBatch {
    fn drop(&mut self) {
        if self.armed {
            let _ = fs::remove_dir_all(&self.staging_dir);
        }
    }
}

pub fn extract_frames(
    spec: &FramePipelineSpec,
    extractor: &FrameExtractor,
) -> Result<FrameBatch, PipelineError> {
    extract_frames_with_runner(spec, extractor, &SystemProcessRunner)
}

/// Extracts every frame with the given runner and no observer.
pub fn extract_frames_with_runner<R: ProcessRunner + ?Sized>(
    spec: &FramePipelineSpec,
    extractor: &FrameExtractor,
    runner: &R,
) -> Result<FrameBatch, PipelineError> {
    extract_frames_observed(spec, extractor, runner, &NoObserver)
}

/// Extracts every frame, reporting one phase per finished chunk and stopping
/// at the next chunk boundary once the observer reports cancellation.
pub fn extract_frames_observed<R: ProcessRunner + ?Sized>(
    spec: &FramePipelineSpec,
    extractor: &FrameExtractor,
    runner: &R,
    observer: &dyn PipelineObserver,
) -> Result<FrameBatch, PipelineError> {
    reject_final_destinations(spec)?;
    fs::create_dir_all(spec.output_root())
        .map_err(|source| io("create_output_root", spec.output_root(), source))?;
    let staging = create_staging(spec.output_root())?;
    let frames_dir = staging.join("frames");
    if let Err(error) =
        fs::create_dir(&frames_dir).map_err(|source| io("create_frames_dir", &frames_dir, source))
    {
        let _ = fs::remove_dir_all(&staging);
        return Err(error);
    }

    let mut frames = Vec::with_capacity(spec.frame_count() as usize);
    let pattern = frames_dir.join("%06d.jpg");
    let mut first_index = 0;
    while first_index < spec.frame_count() {
        if observer.is_cancelled() {
            // Staging is disposable: a cancelled run leaves the previous
            // outputs, if any, exactly as they were.
            let _ = fs::remove_dir_all(&staging);
            return Err(PipelineError::Cancelled {
                operation: "extract_frames",
            });
        }
        let count = FRAME_CHUNK.min(spec.frame_count() - first_index);
        let command = frame_command(extractor, spec.video_path(), first_index, count, &pattern);
        if let Err(error) = run_frame(runner, &command, extractor.timeout()) {
            let _ = fs::remove_dir_all(&staging);
            return Err(error);
        }
        // ffmpeg writing fewer files than asked is a hard failure, not
        // something to compensate for: `inspect_frame` reports the first gap.
        for index in first_index..first_index + count {
            let output = frames_dir.join(format!("{index:06}.jpg"));
            match inspect_frame(index, output) {
                Ok(frame) => frames.push(frame),
                Err(error) => {
                    let _ = fs::remove_dir_all(&staging);
                    return Err(error);
                }
            }
        }
        first_index += count;
        observer.phase(PipelinePhase::Extracting {
            completed: first_index,
            total: spec.frame_count(),
        });
    }
    Ok(FrameBatch {
        staging_dir: staging,
        frames,
        armed: true,
    })
}

fn run_frame<R: ProcessRunner + ?Sized>(
    runner: &R,
    command: &CommandSpec,
    timeout: std::time::Duration,
) -> Result<(), PipelineError> {
    let output = runner.run(command, timeout)?;
    if output.stdout().len() > MAX_CAPTURE_BYTES {
        return Err(PipelineError::ToolOutputTooLarge {
            operation: command.operation(),
            stream: "stdout",
            limit: MAX_CAPTURE_BYTES,
            actual: output.stdout().len(),
        });
    }
    if output.stderr().len() > MAX_CAPTURE_BYTES {
        return Err(PipelineError::ToolOutputTooLarge {
            operation: command.operation(),
            stream: "stderr",
            limit: MAX_CAPTURE_BYTES,
            actual: output.stderr().len(),
        });
    }
    if output.exit_code() != Some(0) {
        return Err(PipelineError::ToolFailed {
            operation: command.operation(),
            exit_code: output.exit_code(),
            stderr: String::from_utf8_lossy(output.stderr()).into_owned(),
        });
    }
    Ok(())
}

fn inspect_frame(index: u64, path: PathBuf) -> Result<ExtractedFrame, PipelineError> {
    let metadata = match fs::symlink_metadata(&path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            return Err(PipelineError::FrameMissing { path });
        }
        Err(source) => return Err(io("stat_frame", &path, source)),
    };
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(PipelineError::FrameNotRegular { path });
    }
    if metadata.len() == 0 {
        return Err(PipelineError::FrameEmpty { path });
    }
    if metadata.len() > MAX_FRAME_BYTES {
        return Err(PipelineError::FrameTooLarge {
            path,
            limit: MAX_FRAME_BYTES,
            actual: metadata.len(),
        });
    }
    let mut file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(&path)
        .map_err(|source| io("open_frame", &path, source))?;
    file.sync_all()
        .map_err(|source| io("sync_frame", &path, source))?;
    let mut digest = Sha256::new();
    let mut bytes = 0_u64;
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let read = file
            .read(&mut buffer)
            .map_err(|source| io("hash_frame", &path, source))?;
        if read == 0 {
            break;
        }
        bytes = bytes
            .checked_add(read as u64)
            .ok_or_else(|| PipelineError::FrameTooLarge {
                path: path.clone(),
                limit: MAX_FRAME_BYTES,
                actual: u64::MAX,
            })?;
        digest.update(&buffer[..read]);
    }
    Ok(ExtractedFrame {
        index,
        path,
        bytes,
        sha256: hex::encode(digest.finalize()),
    })
}

fn reject_final_destinations(spec: &FramePipelineSpec) -> Result<(), PipelineError> {
    for path in [
        spec.output_root().join("frames"),
        spec.output_root().join("landmarks"),
        spec.quality_path(),
    ] {
        if fs::symlink_metadata(&path).is_ok() {
            return Err(PipelineError::OutputDestinationExists { path });
        }
    }
    Ok(())
}

fn create_staging(root: &Path) -> Result<PathBuf, PipelineError> {
    for _ in 0..32 {
        let path = root.join(format!(
            ".feathertalk-frame-build-{}-{}",
            std::process::id(),
            next_id()
        ));
        match OpenOptions::new().write(true).create_new(true).open(&path) {
            Ok(_) => {
                fs::remove_file(&path)
                    .map_err(|source| io("remove_staging_placeholder", &path, source))?;
                fs::create_dir(&path).map_err(|source| io("create_staging", &path, source))?;
                return Ok(path);
            }
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(source) => return Err(io("create_staging", &path, source)),
        }
    }
    Err(PipelineError::OutputDestinationExists {
        path: root.to_owned(),
    })
}

fn io(operation: &'static str, path: &Path, source: std::io::Error) -> PipelineError {
    PipelineError::Io {
        operation,
        path: path.to_owned(),
        source,
    }
}

#[allow(dead_code)]
fn _output(_output: &ProcessOutput) {}
