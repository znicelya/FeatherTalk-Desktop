use std::{
    io::{BufRead, BufReader, Write},
    process::{Command, Stdio}};

use feathertalk_domain::{
    ClientFrame, PROTOCOL_VERSION, ServerFrame, ShutdownFrame, TaskKind, decode_line, encode_line};
use feathertalk_worker::{ENV_FFMPEG, ENV_FFPROBE, ENV_MEDIA_TIMEOUT_MS};

/// The real binary with a cleared media environment, so the handshake it
/// reports does not depend on the developer's machine.
fn worker_command() -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_feathertalk-worker"));
    command
        .env_remove(ENV_FFPROBE)
        .env_remove(ENV_FFMPEG)
        .env_remove(ENV_MEDIA_TIMEOUT_MS)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    command
}

#[test]
fn the_binary_announces_itself_and_exits_zero_on_shutdown() {
    let mut child = worker_command().spawn().unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut lines = BufReader::new(child.stdout.take().unwrap()).lines();

    let ready_line = lines.next().expect("the worker must write ready").unwrap();
    let frame: ServerFrame = decode_line(&ready_line).unwrap();
    frame.validate().unwrap();
    let ServerFrame::Ready(ready) = frame else {
        panic!("the first frame must be ready: {ready_line}");
    };
    assert_eq!(ready.protocol_version, PROTOCOL_VERSION);
    // The two commands that need no toolchain, which is all a cleared
    // environment can offer.
    assert_eq!(
        ready.supported_commands,
        vec![
            TaskKind::ValidateProject,
            TaskKind::InspectModel,
            TaskKind::ImportLegacyModel,
            TaskKind::MigrateLegacyFeatures,
            TaskKind::ExportModelPackage,
            TaskKind::ExportOnnx,
        ]
    );

    let shutdown = ClientFrame::Shutdown(ShutdownFrame {
        protocol_version: PROTOCOL_VERSION});
    writeln!(stdin, "{}", encode_line(&shutdown).unwrap()).unwrap();
    stdin.flush().unwrap();

    // stdin stays open on purpose: shutdown alone must end the process.
    assert!(
        lines.next().is_none(),
        "the worker must write nothing after shutdown"
    );
    let status = child.wait().unwrap();
    assert!(status.success(), "{status:?}");
    drop(stdin);
}

#[test]
fn closing_stdin_exits_the_binary_cleanly() {
    let mut child = worker_command().spawn().unwrap();
    drop(child.stdin.take().unwrap());

    let output = child.wait_with_output().unwrap();
    assert!(
        output.status.success(),
        "{:?}: {}",
        output.status,
        String::from_utf8_lossy(&output.stderr)
    );
    let text = String::from_utf8(output.stdout).unwrap();
    let frames: Vec<ServerFrame> = text
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| decode_line(line).unwrap())
        .collect();
    assert_eq!(frames.len(), 1, "only the handshake is expected: {text}");
}
