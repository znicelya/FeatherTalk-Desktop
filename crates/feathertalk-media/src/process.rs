use std::{
    io::Read,
    process::{Command, Stdio},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    thread,
    time::{Duration, Instant},
};

use crate::{CommandSpec, MediaError};

pub const MAX_CAPTURE_BYTES: usize = 1024 * 1024;

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
    fn run(&self, command: &CommandSpec, timeout: Duration) -> Result<ProcessOutput, MediaError>;
}

#[derive(Debug, Default, Clone, Copy)]
pub struct SystemProcessRunner;

impl ProcessRunner for SystemProcessRunner {
    fn run(&self, command: &CommandSpec, timeout: Duration) -> Result<ProcessOutput, MediaError> {
        run_child(command, timeout, None)
    }
}

/// Cooperative cancellation flag shared between the thread that requests a stop
/// and the thread waiting on a child process.
#[derive(Debug, Clone, Default)]
pub struct CancellationToken {
    cancelled: Arc<AtomicBool>,
}

impl CancellationToken {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::SeqCst);
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::SeqCst)
    }
}

/// A [`SystemProcessRunner`] that kills its child as soon as its token is
/// cancelled instead of waiting for the timeout.
#[derive(Debug, Clone)]
pub struct CancellableProcessRunner {
    token: CancellationToken,
}

impl CancellableProcessRunner {
    pub fn new(token: CancellationToken) -> Self {
        Self { token }
    }
}

impl ProcessRunner for CancellableProcessRunner {
    fn run(&self, command: &CommandSpec, timeout: Duration) -> Result<ProcessOutput, MediaError> {
        run_child(command, timeout, Some(&self.token))
    }
}

fn run_child(
    command: &CommandSpec,
    timeout: Duration,
    token: Option<&CancellationToken>,
) -> Result<ProcessOutput, MediaError> {
    validate_executable(command)?;
    if token.is_some_and(CancellationToken::is_cancelled) {
        return Err(MediaError::ToolCancelled {
            operation: command.operation(),
        });
    }
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
    let mut child = process.spawn().map_err(|error| MediaError::ToolSpawn {
        operation: command.operation(),
        message: error.to_string(),
    })?;
    let mut stdout = child.stdout.take().expect("stdout was piped");
    let mut stderr = child.stderr.take().expect("stderr was piped");
    let stdout_thread = thread::spawn(move || read_limited(&mut stdout));
    let stderr_thread = thread::spawn(move || read_limited(&mut stderr));
    let started = Instant::now();
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) if token.is_some_and(CancellationToken::is_cancelled) => {
                let _ = child.kill();
                let _ = child.wait();
                let _ = stdout_thread.join();
                let _ = stderr_thread.join();
                return Err(MediaError::ToolCancelled {
                    operation: command.operation(),
                });
            }
            Ok(None) if started.elapsed() >= timeout => {
                let _ = child.kill();
                let _ = child.wait();
                let _ = stdout_thread.join();
                let _ = stderr_thread.join();
                return Err(MediaError::ToolTimedOut {
                    operation: command.operation(),
                    timeout_ms: timeout.as_millis().min(u128::from(u64::MAX)) as u64,
                });
            }
            Ok(None) => thread::sleep(Duration::from_millis(5)),
            Err(error) => {
                let _ = child.kill();
                let _ = child.wait();
                let _ = stdout_thread.join();
                let _ = stderr_thread.join();
                return Err(MediaError::ToolSpawn {
                    operation: command.operation(),
                    message: error.to_string(),
                });
            }
        }
    };
    let stdout = stdout_thread
        .join()
        .unwrap_or_else(|_| ReadResult::Error("stdout reader panicked".to_owned()));
    let stderr = stderr_thread
        .join()
        .unwrap_or_else(|_| ReadResult::Error("stderr reader panicked".to_owned()));
    let stdout = stdout.into_result(command.operation(), "stdout")?;
    let stderr = stderr.into_result(command.operation(), "stderr")?;
    Ok(ProcessOutput::new(status.code(), stdout, stderr))
}

enum ReadResult {
    Bytes(Vec<u8>),
    TooLarge(usize),
    Error(String),
}

impl ReadResult {
    fn into_result(
        self,
        operation: &'static str,
        stream: &'static str,
    ) -> Result<Vec<u8>, MediaError> {
        match self {
            Self::Bytes(bytes) => Ok(bytes),
            Self::TooLarge(actual) => Err(MediaError::ToolOutputTooLarge {
                operation,
                stream,
                limit: MAX_CAPTURE_BYTES,
                actual,
            }),
            Self::Error(message) => Err(MediaError::ToolSpawn { operation, message }),
        }
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
                    let remaining = MAX_CAPTURE_BYTES.saturating_sub(output.len());
                    let retained = remaining.min(read);
                    output.extend_from_slice(&buffer[..retained]);
                    exceeded = retained < read;
                }
            }
            Err(error) => return ReadResult::Error(error.to_string()),
        }
    }
}

fn validate_executable(command: &CommandSpec) -> Result<(), MediaError> {
    if !command.executable().is_absolute() {
        return Err(MediaError::ToolSpawn {
            operation: command.operation(),
            message: "executable path must be absolute".to_owned(),
        });
    }
    let metadata =
        std::fs::symlink_metadata(command.executable()).map_err(|error| MediaError::ToolSpawn {
            operation: command.operation(),
            message: error.to_string(),
        })?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(MediaError::ToolSpawn {
            operation: command.operation(),
            message: "executable must be a regular non-symlink file".to_owned(),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::{ffi::OsString, io::Write, path::PathBuf, thread, time::Duration};

    use super::*;

    fn helper_command(name: &str) -> CommandSpec {
        CommandSpec::new(
            std::env::current_exe().unwrap(),
            [
                OsString::from("--ignored"),
                OsString::from("--exact"),
                OsString::from(format!("process::tests::{name}")),
                OsString::from("--nocapture"),
            ]
            .into_iter()
            .collect(),
            "test_process",
        )
    }

    #[test]
    fn system_runner_captures_successful_child_output() {
        let output = SystemProcessRunner
            .run(&helper_command("helper_success"), Duration::from_secs(10))
            .unwrap();
        assert_eq!(output.exit_code(), Some(0));
        assert!(String::from_utf8_lossy(output.stdout()).contains("helper-output"));
    }

    #[test]
    fn system_runner_returns_nonzero_child_status() {
        let output = SystemProcessRunner
            .run(&helper_command("helper_failure"), Duration::from_secs(10))
            .unwrap();
        assert_ne!(output.exit_code(), Some(0));
    }

    #[test]
    fn system_runner_kills_child_after_timeout() {
        assert!(matches!(
            SystemProcessRunner.run(&helper_command("helper_sleep"), Duration::from_millis(20),),
            Err(MediaError::ToolTimedOut {
                operation: "test_process",
                ..
            })
        ));
    }

    #[test]
    fn system_runner_drains_but_rejects_oversized_stdout() {
        assert!(matches!(
            SystemProcessRunner.run(
                &helper_command("helper_large_output"),
                Duration::from_secs(10),
            ),
            Err(MediaError::ToolOutputTooLarge {
                operation: "test_process",
                stream: "stdout",
                limit: MAX_CAPTURE_BYTES,
                ..
            })
        ));
    }

    #[test]
    fn cancelling_mid_run_kills_the_child() {
        let token = CancellationToken::new();
        let runner = CancellableProcessRunner::new(token.clone());
        thread::spawn(move || {
            thread::sleep(Duration::from_millis(50));
            token.cancel();
        });
        let started = Instant::now();
        let error = runner
            .run(&helper_command("helper_sleep"), Duration::from_secs(30))
            .expect_err("cancellation must surface as an error");
        assert!(
            matches!(
                error,
                MediaError::ToolCancelled {
                    operation: "test_process"
                }
            ),
            "{error:?}"
        );
        assert!(
            started.elapsed() < Duration::from_secs(3),
            "cancellation waited {:?} for a child that sleeps 5 s",
            started.elapsed()
        );
    }

    #[test]
    fn an_already_cancelled_token_never_spawns() {
        let token = CancellationToken::new();
        token.cancel();
        let runner = CancellableProcessRunner::new(token);
        let started = Instant::now();
        let error = runner
            .run(&helper_command("helper_sleep"), Duration::from_secs(30))
            .expect_err("a cancelled token must refuse to start work");
        assert!(
            matches!(error, MediaError::ToolCancelled { .. }),
            "{error:?}"
        );
        assert!(started.elapsed() < Duration::from_millis(500));
    }

    #[test]
    fn an_uncancelled_token_behaves_like_the_system_runner() {
        let runner = CancellableProcessRunner::new(CancellationToken::new());
        let output = runner
            .run(&helper_command("helper_success"), Duration::from_secs(10))
            .unwrap();
        assert_eq!(output.exit_code(), Some(0));
        assert!(String::from_utf8_lossy(output.stdout()).contains("helper-output"));
    }

    #[test]
    #[ignore]
    fn helper_success() {
        println!("helper-output");
    }

    #[test]
    #[ignore]
    fn helper_failure() {
        panic!("intentional child failure");
    }

    #[test]
    #[ignore]
    fn helper_sleep() {
        thread::sleep(Duration::from_secs(5));
    }

    #[test]
    #[ignore]
    fn helper_large_output() {
        let bytes = vec![b'x'; MAX_CAPTURE_BYTES + 64 * 1024];
        std::io::stdout().write_all(&bytes).unwrap();
    }

    #[allow(dead_code)]
    fn _path_buf(path: &str) -> PathBuf {
        PathBuf::from(path)
    }
}
