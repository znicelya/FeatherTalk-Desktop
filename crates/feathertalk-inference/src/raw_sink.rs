use std::{
    fs,
    io::{Read, Write},
    process::{Child, ChildStdin, Command, Stdio},
    thread::{self, JoinHandle}};

use crate::{BgrFrame, CommandSpec, InferenceError};

const MAX_ERROR_MESSAGE_BYTES: usize = 512;

/// The render loop writes every frame on the thread that started it, so neither
/// half of this pair carries an auto-trait bound. Requiring `Send` here would
/// rule out a sink that borrows a progress reporter, and a reporter that owns a
/// channel to the control loop is not `Sync`.
pub trait RawVideoSink {
    fn write_frame(&mut self, frame: &BgrFrame) -> Result<(), InferenceError>;
    fn finish(self: Box<Self>) -> Result<(), InferenceError>;
}

pub trait RawVideoSinkFactory {
    /// The returned sink may borrow the factory. `Box<dyn RawVideoSink>` would
    /// mean `+ 'static` and rule out a sink that observes something the caller
    /// owns -- a progress reporter, a cancellation token -- which is exactly
    /// what a worker wraps around a system sink.
    fn start(&self, command: &CommandSpec) -> Result<Box<dyn RawVideoSink + '_>, InferenceError>;
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct SystemRawVideoSinkFactory;

impl SystemRawVideoSinkFactory {
    pub const fn new() -> Self {
        Self
    }
}

impl RawVideoSinkFactory for SystemRawVideoSinkFactory {
    fn start(&self, command: &CommandSpec) -> Result<Box<dyn RawVideoSink + '_>, InferenceError> {
        validate_executable(command)?;
        let mut process = Command::new(command.executable());
        process
            .args(command.arguments())
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::piped());
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            process.creation_flags(0x08000000); // CREATE_NO_WINDOW: rendering uses pipes.
        }
        let mut child = process.spawn().map_err(|error| InferenceError::SinkStart {
            message: bounded(error.to_string())})?;
        let Some(stdin) = child.stdin.take() else {
            let _ = child.kill();
            let _ = child.wait();
            return Err(InferenceError::SinkStart {
                message: "child stdin was not piped".into()});
        };
        let Some(stderr) = child.stderr.take() else {
            let _ = child.kill();
            let _ = child.wait();
            return Err(InferenceError::SinkStart {
                message: "child stderr was not piped".into()});
        };
        let stderr_thread = thread::spawn(move || read_limited(stderr));
        Ok(Box::new(SystemRawVideoSink {
            child: Some(child),
            stdin: Some(stdin),
            stderr_thread: Some(stderr_thread),
            operation: command.operation()}))
    }
}

struct SystemRawVideoSink {
    child: Option<Child>,
    stdin: Option<ChildStdin>,
    stderr_thread: Option<JoinHandle<ReadResult>>,
    operation: &'static str}

impl RawVideoSink for SystemRawVideoSink {
    fn write_frame(&mut self, frame: &BgrFrame) -> Result<(), InferenceError> {
        let result = self
            .stdin
            .as_mut()
            .ok_or_else(|| std::io::Error::other("child stdin is closed"))
            .and_then(|stdin| stdin.write_all(frame.as_bytes()));
        if let Err(error) = result {
            self.abort();
            return Err(InferenceError::SinkWrite {
                message: bounded(error.to_string())});
        }
        Ok(())
    }

    fn finish(mut self: Box<Self>) -> Result<(), InferenceError> {
        self.stdin.take();
        let status = match self.child.as_mut() {
            Some(child) => match child.wait() {
                Ok(status) => status,
                Err(error) => {
                    self.abort();
                    return Err(InferenceError::SinkFinish {
                        message: bounded(error.to_string())});
                }
            },
            None => {
                return Err(InferenceError::SinkFinish {
                    message: "child process is unavailable".into()});
            }
        };
        self.child.take();
        let stderr = self.join_stderr()?;
        if status.success() {
            Ok(())
        } else {
            Err(InferenceError::ToolFailed {
                operation: self.operation,
                exit_code: status.code(),
                stderr: bounded(String::from_utf8_lossy(&stderr).into_owned())})
        }
    }
}

impl SystemRawVideoSink {
    fn join_stderr(&mut self) -> Result<Vec<u8>, InferenceError> {
        let thread = self
            .stderr_thread
            .take()
            .ok_or_else(|| InferenceError::SinkFinish {
                message: "stderr reader is unavailable".into()})?;
        match thread.join() {
            Ok(ReadResult::Bytes(bytes)) => Ok(bytes),
            Ok(ReadResult::TooLarge(actual)) => Err(InferenceError::SinkFinish {
                message: format!(
                    "child stderr exceeds {} bytes: {actual}",
                    feathertalk_media::MAX_CAPTURE_BYTES
                )}),
            Ok(ReadResult::Error(message)) => Err(InferenceError::SinkFinish {
                message: bounded(message)}),
            Err(_) => Err(InferenceError::SinkFinish {
                message: "stderr reader panicked".into()})}
    }

    fn abort(&mut self) {
        self.stdin.take();
        if let Some(mut child) = self.child.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
        if let Some(thread) = self.stderr_thread.take() {
            let _ = thread.join();
        }
    }
}

impl Drop for SystemRawVideoSink {
    fn drop(&mut self) {
        self.abort();
    }
}

enum ReadResult {
    Bytes(Vec<u8>),
    TooLarge(usize),
    Error(String)}

fn read_limited(mut stderr: impl Read) -> ReadResult {
    let mut retained = Vec::new();
    let mut buffer = [0_u8; 16 * 1024];
    let mut actual = 0usize;
    let mut exceeded = false;
    loop {
        match stderr.read(&mut buffer) {
            Ok(0) if exceeded => return ReadResult::TooLarge(actual),
            Ok(0) => return ReadResult::Bytes(retained),
            Ok(read) => {
                actual = actual.saturating_add(read);
                if !exceeded {
                    let remaining =
                        feathertalk_media::MAX_CAPTURE_BYTES.saturating_sub(retained.len());
                    let keep = remaining.min(read);
                    retained.extend_from_slice(&buffer[..keep]);
                    exceeded = keep < read;
                }
            }
            Err(error) => return ReadResult::Error(error.to_string())}
    }
}

fn validate_executable(command: &CommandSpec) -> Result<(), InferenceError> {
    if !command.executable().is_absolute() {
        return Err(InferenceError::SinkStart {
            message: "executable path must be absolute".into()});
    }
    let metadata =
        fs::symlink_metadata(command.executable()).map_err(|error| InferenceError::SinkStart {
            message: bounded(error.to_string())})?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(InferenceError::SinkStart {
            message: "executable must be a regular non-symlink file".into()});
    }
    Ok(())
}

fn bounded(mut message: String) -> String {
    if message.len() <= MAX_ERROR_MESSAGE_BYTES {
        return message;
    }
    let mut end = MAX_ERROR_MESSAGE_BYTES;
    while !message.is_char_boundary(end) {
        end -= 1;
    }
    message.truncate(end);
    message
}

#[cfg(test)]
mod tests {
    use std::{
        ffi::OsString,
        io::{Read, Write}};

    use crate::{BgrFrame, CommandSpec, InferenceError};

    use super::{RawVideoSinkFactory, SystemRawVideoSinkFactory};

    fn helper_command(name: &str) -> CommandSpec {
        CommandSpec::new(
            std::env::current_exe().unwrap(),
            [
                OsString::from("--ignored"),
                OsString::from("--exact"),
                OsString::from(format!("raw_sink::tests::{name}")),
                OsString::from("--nocapture"),
            ]
            .into_iter()
            .collect(),
            "test_raw_sink",
        )
    }

    #[test]
    fn system_sink_streams_exact_bgr_bytes_and_waits_for_success() {
        let factory = SystemRawVideoSinkFactory::new();
        let mut sink = factory.start(&helper_command("helper_success")).unwrap();
        let frame = BgrFrame::new(2, 1, vec![1, 2, 3, 4, 5, 6]).unwrap();

        sink.write_frame(&frame).unwrap();
        sink.finish().unwrap();
    }

    #[test]
    fn system_sink_reports_nonzero_child_status() {
        let factory = SystemRawVideoSinkFactory::new();
        let sink = factory.start(&helper_command("helper_failure")).unwrap();

        assert!(matches!(
            sink.finish(),
            Err(InferenceError::ToolFailed {
                operation: "test_raw_sink",
                exit_code: Some(code),
                ..
            }) if code != 0
        ));
    }

    #[test]
    #[ignore]
    fn helper_success() {
        let mut bytes = Vec::new();
        std::io::stdin().read_to_end(&mut bytes).unwrap();
        assert_eq!(bytes, [1, 2, 3, 4, 5, 6]);
    }

    #[test]
    #[ignore]
    fn helper_failure() {
        std::io::stderr()
            .write_all(b"intentional child failure")
            .unwrap();
        panic!("intentional child failure");
    }
}
