use std::{
    io::Read,
    path::{Path, PathBuf},
    process::{Command, Stdio},
    sync::atomic::{AtomicU64, Ordering},
    thread,
    time::{Duration, Instant},
};

use crate::{CommandSpec, PipelineError};

pub const MAX_CAPTURE_BYTES: usize = 1024 * 1024;
pub const MAX_FRAME_BYTES: u64 = 16 * 1024 * 1024;
static COUNTER: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProcessOutput {
    exit_code: Option<i32>,
    stdout: Vec<u8>,
    stderr: Vec<u8>,
}

impl ProcessOutput {
    pub fn new(exit_code: Option<i32>, stdout: Vec<u8>, stderr: Vec<u8>) -> Self {
        Self {
            exit_code,
            stdout,
            stderr,
        }
    }
    pub fn exit_code(&self) -> Option<i32> {
        self.exit_code
    }
    pub fn stdout(&self) -> &[u8] {
        &self.stdout
    }
    pub fn stderr(&self) -> &[u8] {
        &self.stderr
    }
}

pub trait ProcessRunner: Send + Sync {
    fn run(&self, command: &CommandSpec, timeout: Duration)
    -> Result<ProcessOutput, PipelineError>;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FrameExtractor {
    ffmpeg: PathBuf,
    timeout: Duration,
    cuda_device: Option<usize>,
}

impl FrameExtractor {
    pub fn new(ffmpeg: PathBuf, timeout: Duration) -> Result<Self, PipelineError> {
        if !ffmpeg.is_absolute() || ffmpeg.as_os_str().is_empty() {
            return Err(PipelineError::InvalidField {
                field: "ffmpeg",
                message: "must be a non-empty absolute path".into(),
            });
        }
        if timeout.is_zero() || timeout > Duration::from_secs(24 * 60 * 60) {
            return Err(PipelineError::InvalidField {
                field: "timeout",
                message: "must be within 1 second and 24 hours".into(),
            });
        }
        Ok(Self {
            ffmpeg,
            timeout,
            cuda_device: None,
        })
    }

    pub fn ffmpeg(&self) -> &Path {
        &self.ffmpeg
    }
    pub fn timeout(&self) -> Duration {
        self.timeout
    }

    pub fn with_cuda_device(mut self, device: Option<usize>) -> Self {
        self.cuda_device = device;
        self
    }

    pub fn cuda_device(&self) -> Option<usize> {
        self.cuda_device
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct SystemProcessRunner;

impl ProcessRunner for SystemProcessRunner {
    fn run(
        &self,
        command: &CommandSpec,
        timeout: Duration,
    ) -> Result<ProcessOutput, PipelineError> {
        validate_executable(command)?;
        let mut process = Command::new(command.executable());
        process
            .args(command.arguments())
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            process.creation_flags(0x08000000); // CREATE_NO_WINDOW: all output is piped.
        }
        let mut child = process.spawn().map_err(|source| PipelineError::ToolSpawn {
            operation: command.operation(),
            message: source.to_string(),
        })?;
        let mut stdout = child.stdout.take().expect("stdout was piped");
        let mut stderr = child.stderr.take().expect("stderr was piped");
        let stdout_thread = thread::spawn(move || read_limited(&mut stdout));
        let stderr_thread = thread::spawn(move || read_limited(&mut stderr));
        let started = Instant::now();
        let status = loop {
            match child.try_wait() {
                Ok(Some(status)) => break status,
                Ok(None) if started.elapsed() >= timeout => {
                    let _ = child.kill();
                    let _ = child.wait();
                    let _ = stdout_thread.join();
                    let _ = stderr_thread.join();
                    return Err(PipelineError::ToolTimedOut {
                        operation: command.operation(),
                        timeout_ms: timeout.as_millis().min(u128::from(u64::MAX)) as u64,
                    });
                }
                Ok(None) => thread::sleep(Duration::from_millis(5)),
                Err(source) => {
                    let _ = child.kill();
                    let _ = child.wait();
                    let _ = stdout_thread.join();
                    let _ = stderr_thread.join();
                    return Err(PipelineError::ToolSpawn {
                        operation: command.operation(),
                        message: source.to_string(),
                    });
                }
            }
        };
        let stdout = join_read(stdout_thread, command.operation(), "stdout")?;
        let stderr = join_read(stderr_thread, command.operation(), "stderr")?;
        Ok(ProcessOutput::new(status.code(), stdout, stderr))
    }
}

enum ReadResult {
    Bytes(Vec<u8>),
    TooLarge(usize),
    Error(String),
}

fn join_read(
    handle: thread::JoinHandle<ReadResult>,
    operation: &'static str,
    stream: &'static str,
) -> Result<Vec<u8>, PipelineError> {
    match handle
        .join()
        .unwrap_or_else(|_| ReadResult::Error("reader panicked".into()))
    {
        ReadResult::Bytes(bytes) => Ok(bytes),
        ReadResult::TooLarge(actual) => Err(PipelineError::ToolOutputTooLarge {
            operation,
            stream,
            limit: MAX_CAPTURE_BYTES,
            actual,
        }),
        ReadResult::Error(message) => Err(PipelineError::ToolSpawn { operation, message }),
    }
}

fn read_limited(reader: &mut impl Read) -> ReadResult {
    let mut output = Vec::new();
    let mut buffer = [0_u8; 16 * 1024];
    let mut actual = 0usize;
    let mut exceeded = false;
    loop {
        match reader.read(&mut buffer) {
            Ok(0) if exceeded => return ReadResult::TooLarge(actual),
            Ok(0) => return ReadResult::Bytes(output),
            Ok(read) => {
                actual = actual.saturating_add(read);
                if !exceeded {
                    let keep = (MAX_CAPTURE_BYTES - output.len()).min(read);
                    output.extend_from_slice(&buffer[..keep]);
                    exceeded = keep < read;
                }
            }
            Err(source) => return ReadResult::Error(source.to_string()),
        }
    }
}

fn validate_executable(command: &CommandSpec) -> Result<(), PipelineError> {
    if !command.executable().is_absolute() {
        return Err(PipelineError::ToolSpawn {
            operation: command.operation(),
            message: "executable path must be absolute".into(),
        });
    }
    let metadata = std::fs::symlink_metadata(command.executable()).map_err(|source| {
        PipelineError::ToolSpawn {
            operation: command.operation(),
            message: source.to_string(),
        }
    })?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(PipelineError::ToolSpawn {
            operation: command.operation(),
            message: "executable must be a regular non-symlink file".into(),
        });
    }
    Ok(())
}

pub(crate) fn next_id() -> u64 {
    COUNTER.fetch_add(1, Ordering::Relaxed)
}
