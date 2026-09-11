// A scripted stand-in for `feathertalk-worker`.
//
// Compiled as a `[[bin]]` of `feathertalk-client` so integration tests spawn a
// real process and talk to it over real pipes. The behaviour is chosen by
// `FT_FAKE_WORKER_SCENARIO`; one scenario per test case, each one a straight
// line of writes with no branching, so a failing test names its own script.
//
// Only `feathertalk-domain`, `serde_json`, and std are available here, because
// a `[[bin]]` target cannot use dev-dependencies. Hence the fixed timestamp.
//
// Plain `//` comments rather than `//!`: `feathertalk-cli` `include!`s this file
// to get its own copy of the binary, and an inner doc comment is not permitted
// in a macro expansion.

use std::io::{BufReader, StdinLock, Write};
use std::time::Duration;

use feathertalk_domain::{
    AdapterInfo, AdapterKind, Backend, Capabilities, ClientFrame, ErrorCode, Event, FrameReader,
    MAX_FRAME_BYTES, PROTOCOL_VERSION, Progress, ReadyFrame, RejectedFrame, ServerFrame, TaskError,
    TaskId, TaskKind, TaskStage, encode_line,
};

/// The scenario selector. Tests set it; there is no command line.
const SCENARIO_ENV: &str = "FT_FAKE_WORKER_SCENARIO";

/// A fixed RFC 3339 instant. `Event::validate` only checks the format.
const EMITTED_AT: &str = "2026-09-01T00:00:00Z";

/// A well-formed task id that is never the one the client sends.
const FOREIGN_TASK_ID: &str = "1787900000000-0000beef";

type Reader = FrameReader<BufReader<StdinLock<'static>>>;

fn main() {
    let scenario = std::env::var(SCENARIO_ENV).unwrap_or_else(|_| "ready-complete".to_string());
    let mut reader = FrameReader::new(BufReader::new(std::io::stdin().lock()));
    match scenario.as_str() {
        // Never writes anything, so the handshake has to time out.
        "silent" => park(),
        // A syntactically valid frame that is not `ready`.
        "no-ready" => {
            let task_id = TaskId::parse(FOREIGN_TASK_ID).expect("the constant id is valid");
            write_frame(&ServerFrame::Event(stage_event(
                &task_id,
                TaskStage::Queued,
            )));
            park();
        }
        // Truncated JSON: decodable as a line, not as a frame.
        "invalid-line" => {
            write_line("{\"frame\":\"ready\"");
            park();
        }
        // A structurally valid `ready` frame from a future protocol.
        "bad-version" => {
            let mut value =
                serde_json::to_value(ready(default_commands())).expect("ready serializes");
            value["data"]["protocol_version"] = serde_json::json!(99);
            write_line(&serde_json::to_string(&value).expect("the patched value serializes"));
            park();
        }
        // A worker that advertises no commands at all: `ReadyFrame::validate` rejects it.
        "empty-commands" => {
            let mut value =
                serde_json::to_value(ready(default_commands())).expect("ready serializes");
            value["data"]["supported_commands"] = serde_json::json!([]);
            write_line(&serde_json::to_string(&value).expect("the patched value serializes"));
            park();
        }
        // Refuses the session outright instead of going ready.
        "rejected-handshake" => {
            write_frame(&ServerFrame::Rejected(RejectedFrame {
                protocol_version: PROTOCOL_VERSION,
                reason: "工作进程当前无法接受新会话".to_string(),
            }));
        }
        // Goes ready and then stops reading, so only a kill reaps it.
        "hang-after-ready" => {
            write_frame(&ready(default_commands()));
            park();
        }
        // Floods stderr before going ready, to exercise the tail bound.
        "noisy-stderr" => {
            for index in 0..200 {
                eprintln!("stderr line {index}");
            }
            write_frame(&ready(default_commands()));
            serve_one_task(&mut reader);
        }
        // The happy path: ready, one progress event, then completed.
        "ready-complete" => {
            write_frame(&ready(default_commands()));
            serve_one_task(&mut reader);
        }
        // Echo the actual child environment so CLI overrides are tested across
        // process creation, including inherited values and explicit clearing.
        "compute-environment" | "compute-cpu-only" => {
            let ServerFrame::Ready(mut ready) = compute_ready() else {
                unreachable!()
            };
            if scenario == "compute-cpu-only" {
                ready.backends = vec![Backend::Cpu];
                ready
                    .adapters
                    .retain(|adapter| adapter.backend == Backend::Cpu);
                ready.capabilities.wgpu_training = false;
            }
            write_frame(&ServerFrame::Ready(ready));
            if let Some(task_id) = wait_for_start(&mut reader) {
                let mut event = stage_event(&task_id, TaskStage::Completed);
                event.result = Some(serde_json::json!({
                    "backend": std::env::var("FEATHERTALK_WORKER_BACKEND").ok(),
                    "adapter": std::env::var("FEATHERTALK_WORKER_ADAPTER").ok(),
                }));
                write_frame(&ServerFrame::Event(event));
            }
        }
        // Three progress steps and a normalization result, like the real
        // worker's `normalize_media`.
        "normalize-progress" => {
            write_frame(&ready(default_commands()));
            serve_normalize_task(&mut reader);
        }
        // Advertises one command each, to exercise the capability gate both ways.
        "only-validate" => {
            write_frame(&ready(vec![TaskKind::ValidateProject]));
            serve_one_task(&mut reader);
        }
        "only-probe" => {
            write_frame(&ready(vec![TaskKind::ProbeMedia]));
            serve_one_task(&mut reader);
        }
        // Reports a task failure, which is exit 1 rather than a broken session.
        "fail" => {
            write_frame(&ready(default_commands()));
            if let Some(task_id) = wait_for_start(&mut reader) {
                write_frame(&ServerFrame::Event(failed(&task_id)));
            }
        }
        // Reports itself cancelled without being asked, so the terminal-stage
        // mapping can be tested without involving a signal.
        "self-cancel" => {
            write_frame(&ready(default_commands()));
            if let Some(task_id) = wait_for_start(&mut reader) {
                write_frame(&ServerFrame::Event(stage_event(
                    &task_id,
                    TaskStage::Cancelled,
                )));
            }
        }
        // Emits an event for a different task before the real one.
        "foreign-event" => {
            write_frame(&ready(default_commands()));
            if let Some(task_id) = wait_for_start(&mut reader) {
                let foreign = TaskId::parse(FOREIGN_TASK_ID).expect("the constant id is valid");
                write_frame(&ServerFrame::Event(stage_event(
                    &foreign,
                    TaskStage::Preparing,
                )));
                write_frame(&ServerFrame::Event(completed(&task_id)));
            }
        }
        // Exits immediately after the handshake: the client must diagnose a lost
        // worker whether the `start` write succeeds or fails.
        "die-after-ready" => {
            write_frame(&ready(default_commands()));
            std::process::exit(0);
        }
        // Writes a line past the protocol's frame bound.
        "oversized-line" => {
            write_frame(&ready(default_commands()));
            if wait_for_start(&mut reader).is_some() {
                write_line(&"x".repeat(MAX_FRAME_BYTES + 16));
                park();
            }
        }
        // The cooperative path: acknowledges the cancel with a terminal event.
        "cancel-acks" => {
            write_frame(&ready(default_commands()));
            if let Some(task_id) = wait_for_start(&mut reader)
                && wait_for_cancel(&mut reader).is_some()
            {
                write_frame(&ServerFrame::Event(stage_event(
                    &task_id,
                    TaskStage::Cancelled,
                )));
            }
        }
        // Finishes anyway: the completion must win over the pending cancel.
        "cancel-completes" => {
            write_frame(&ready(default_commands()));
            if let Some(task_id) = wait_for_start(&mut reader)
                && wait_for_cancel(&mut reader).is_some()
            {
                write_frame(&ServerFrame::Event(completed(&task_id)));
            }
        }
        // Reads nothing and answers nothing, so only the grace deadline ends it.
        "cancel-ignored" => {
            write_frame(&ready(default_commands()));
            if wait_for_start(&mut reader).is_some() {
                park();
            }
        }
        // Dies without acknowledging: EOF after a cancel is still a cancellation.
        "die-on-cancel" => {
            write_frame(&ready(default_commands()));
            if wait_for_start(&mut reader).is_some() && wait_for_cancel(&mut reader).is_some() {
                std::process::exit(0);
            }
        }
        other => {
            eprintln!("unknown fake worker scenario: {other}");
            std::process::exit(97);
        }
    }
}

/// Every task command the real worker offers when a media toolchain resolved.
fn default_commands() -> Vec<TaskKind> {
    vec![
        TaskKind::ValidateProject,
        TaskKind::ProbeMedia,
        TaskKind::NormalizeMedia,
    ]
}

fn ready(commands: Vec<TaskKind>) -> ServerFrame {
    ServerFrame::Ready(ReadyFrame {
        protocol_version: PROTOCOL_VERSION,
        worker_version: "fake-0".to_string(),
        backends: vec![Backend::Cpu],
        adapters: vec![AdapterInfo {
            id: "cpu-0".to_string(),
            name: "Fake CPU".to_string(),
            backend: Backend::Cpu,
            kind: AdapterKind::Cpu,
            certified: true,
            vram_bytes: None,
        }],
        supported_commands: commands,
        capabilities: Capabilities {
            training: false,
            wgpu_training: false,
            onnx_validation: false,
            ffmpeg: true,
        },
    })
}

fn compute_ready() -> ServerFrame {
    let ServerFrame::Ready(mut frame) = ready(vec![
        TaskKind::ValidateProject,
        TaskKind::NormalizeMedia,
        TaskKind::Train,
        TaskKind::Render,
        TaskKind::ExtractFrames,
        TaskKind::ExtractFeatures,
    ]) else {
        unreachable!()
    };
    frame.backends.push(Backend::Wgpu);
    frame.backends.push(Backend::Cuda);
    frame.adapters.push(AdapterInfo {
        id: "cuda-test-0".into(),
        name: "NVIDIA GPU (CUDA)".into(),
        backend: Backend::Cuda,
        kind: AdapterKind::Discrete,
        certified: true,
        vram_bytes: None,
    });
    frame.capabilities.training = true;
    frame.capabilities.wgpu_training = true;
    for (id, kind, certified) in [
        ("wgpu-test-0", AdapterKind::Discrete, true),
        ("wgpu-experimental", AdapterKind::Integrated, false),
        ("wgpu-software", AdapterKind::Cpu, true),
    ] {
        frame.adapters.push(AdapterInfo {
            id: id.into(),
            name: format!("{id} (Vulkan)"),
            backend: Backend::Wgpu,
            kind,
            certified,
            vram_bytes: None,
        });
    }
    ServerFrame::Ready(frame)
}

/// Read frames until a `start` arrives, then run the scripted happy path.
fn serve_one_task(reader: &mut Reader) {
    let Some(task_id) = wait_for_start(reader) else {
        return;
    };
    let mut preparing = stage_event(&task_id, TaskStage::Preparing);
    preparing.progress = Some(Progress {
        completed: 1,
        total: Some(2),
    });
    write_frame(&ServerFrame::Event(preparing));
    write_frame(&ServerFrame::Event(completed(&task_id)));
}

/// Read frames until a `start` arrives, then narrate three normalization steps
/// and finish with a payload shaped like `normalize_to_json`.
fn serve_normalize_task(reader: &mut Reader) {
    let Some(task_id) = wait_for_start(reader) else {
        return;
    };
    for (stage, completed) in [
        (TaskStage::Preparing, 1),
        (TaskStage::ExtractingFrames, 2),
        (TaskStage::ExtractingAudio, 3),
    ] {
        let mut event = stage_event(&task_id, stage);
        event.progress = Some(Progress {
            completed,
            total: Some(3),
        });
        write_frame(&ServerFrame::Event(event));
    }
    let mut event = stage_event(&task_id, TaskStage::Completed);
    event.result = Some(serde_json::json!({
        "output_dir": "C:/tmp/assets",
        "video": {
            "path": "C:/tmp/assets/video_25fps.mp4",
            "bytes": 16,
            "sha256": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
            "codec_name": "mpeg4",
            "pixel_format": "yuv420p",
            "width": 640,
            "height": 480,
            "frame_rate": { "numerator": 25, "denominator": 1 },
            "frame_count": 50,
            "duration_seconds": 2.0
        },
        "audio": {
            "path": "C:/tmp/assets/audio_16k_mono.wav",
            "bytes": 16,
            "sha256": "fedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210",
            "codec_name": "pcm_s16le",
            "sample_format": "s16",
            "sample_rate": 16000,
            "channels": 1,
            "sample_count": 32000,
            "duration_seconds": 2.0
        }
    }));
    write_frame(&ServerFrame::Event(event));
}

/// Block until the client sends `start`. Returns `None` on shutdown or EOF.
fn wait_for_start(reader: &mut Reader) -> Option<TaskId> {
    loop {
        match reader.read_frame::<ClientFrame>()? {
            Ok(ClientFrame::Start(start)) => {
                record_protocol("start");
                return Some(start.task_id);
            }
            Ok(ClientFrame::Cancel(_)) => continue,
            Ok(ClientFrame::Shutdown(_)) => {
                record_protocol("shutdown");
                return None;
            }
            Err(error) => {
                eprintln!("fake worker could not decode a client frame: {error}");
                return None;
            }
        }
    }
}

fn record_protocol(frame: &str) {
    if let Some(path) = std::env::var_os("FT_FAKE_PROTOCOL_LOG") {
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .unwrap();
        writeln!(file, "{frame}").unwrap();
    }
}

fn stage_event(task_id: &TaskId, stage: TaskStage) -> Event {
    Event::new(task_id.clone(), EMITTED_AT, stage)
}

/// Block until the client sends `cancel`. Returns `None` on shutdown or EOF.
fn wait_for_cancel(reader: &mut Reader) -> Option<TaskId> {
    loop {
        match reader.read_frame::<ClientFrame>()? {
            Ok(ClientFrame::Cancel(cancel)) => return Some(cancel.task_id),
            Ok(ClientFrame::Start(_)) => continue,
            Ok(ClientFrame::Shutdown(_)) => return None,
            Err(error) => {
                eprintln!("fake worker could not decode a client frame: {error}");
                return None;
            }
        }
    }
}

fn completed(task_id: &TaskId) -> Event {
    let mut event = stage_event(task_id, TaskStage::Completed);
    event.result = Some(serde_json::json!({ "checked": true }));
    event
}

fn failed(task_id: &TaskId) -> Event {
    // `TaskError::stage` records where the task was when it broke, so it must be
    // the non-terminal stage; the event's own stage is the terminal `Failed`.
    let error = TaskError::new(
        ErrorCode::MediaInvalid,
        "输入文件无法解析，请确认它是完整的视频",
        "ffprobe exited with status 1",
        TaskStage::Preparing,
    );
    let mut event = stage_event(
        task_id,
        TaskStage::Failed {
            code: error.code,
            message: error.summary.clone(),
        },
    );
    event.error = Some(error);
    event
}

fn write_frame(frame: &ServerFrame) {
    let line = encode_line(frame).expect("the scripted frame serializes");
    write_line(line.trim_end());
}

fn write_line(line: &str) {
    let mut stdout = std::io::stdout().lock();
    stdout
        .write_all(line.as_bytes())
        .expect("stdout accepts a line");
    stdout.write_all(b"\n").expect("stdout accepts a newline");
    stdout.flush().expect("stdout flushes");
}

/// Stay alive until the parent kills us. Several scenarios exist to prove the
/// client's deadlines and reaping actually work.
fn park() -> ! {
    loop {
        std::thread::sleep(Duration::from_millis(50));
    }
}
