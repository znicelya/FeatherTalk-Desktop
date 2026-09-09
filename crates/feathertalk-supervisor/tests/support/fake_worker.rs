// A scripted stand-in for `feathertalk-worker`, built for the supervisor's
// process tests.
//
// Compiled as a `[[bin]]` of `feathertalk-supervisor` so `tests/real_worker.rs`
// spawns a real child and talks to it over real pipes. A `[[bin]]` target cannot
// use dev-dependencies, so only `feathertalk-domain`, `serde_json`, and std are
// available here; hence the fixed timestamp.
//
// The script is named by files inside the request's `project_dir` rather than by
// an environment variable. The supervisor spawns the worker with the parent's
// environment, and two tests running in parallel in one test binary cannot share
// one variable; a directory each is race-free and needs no `unsafe`.

use std::io::{BufReader, StdinLock, Write};
use std::path::{Path, PathBuf};
use std::time::Duration;

use feathertalk_domain::{
    AdapterInfo, AdapterKind, Backend, Capabilities, ClientFrame, Event, FrameReader,
    PROTOCOL_VERSION, Progress, ReadyFrame, Request, ServerFrame, TaskId, TaskKind, TaskStage,
    encode_line,
};

/// Names the script to follow. Written by the test before the run.
const SCENARIO_FILE: &str = "scenario.txt";

/// Counts attempts across restarts. Owned by this binary.
const ATTEMPT_FILE: &str = "attempts.txt";

/// A fixed RFC 3339 instant. `Event::validate` only checks the format.
const EMITTED_AT: &str = "2026-09-05T13:00:00Z";

/// The two lines the crashing attempt leaves behind. `tests/real_worker.rs`
/// asserts on the same text, so the two copies have to agree.
const FIRST_STDERR_LINE: &str = "fake worker: about to abandon this attempt";
const SECOND_STDERR_LINE: &str = "fake worker: wgpu device lost while training";

type Reader = FrameReader<BufReader<StdinLock<'static>>>;

/// What the `start` frame told us to do.
struct Job {
    task_id: TaskId,
    project_dir: PathBuf,
    resume: bool,
}

fn main() {
    let handshake = std::env::var_os("FT_SUPERVISED_PROTOCOL_DIR")
        .map(|directory| count_file(&PathBuf::from(directory).join("handshakes.txt")))
        .unwrap_or(1);
    let mut reader = FrameReader::new(BufReader::new(std::io::stdin().lock()));
    let advertise_gpu = match std::env::var("FT_SUPERVISED_CAPABILITIES").as_deref() {
        Ok("cpu-only") => false,
        Ok("gpu-once") => handshake == 1,
        _ => true,
    };
    write_frame(&ready(advertise_gpu));
    let Some(job) = wait_for_start(&mut reader) else {
        return;
    };

    let attempt = count_attempt(&job.project_dir);
    let scenario = std::fs::read_to_string(job.project_dir.join(SCENARIO_FILE))
        .unwrap_or_else(|_| "complete".to_owned());
    if scenario.trim() == "compute-crash-once" {
        let received = serde_json::json!({
            "backend": std::env::var("FEATHERTALK_WORKER_BACKEND").ok(),
            "adapter": std::env::var("FEATHERTALK_WORKER_ADAPTER").ok(),
        });
        std::fs::write(
            job.project_dir
                .join(format!("compute-attempt-{attempt}.json")),
            serde_json::to_vec(&received).expect("the environment serializes"),
        )
        .expect("the received environment is recorded");
    }

    // Every attempt reports progress before it decides its fate, so the events
    // of a doomed attempt are observable too.
    write_frame(&ServerFrame::Event(preparing(&job.task_id)));

    match (scenario.trim(), attempt) {
        // Die without a terminal stage, which is what the client reads as a
        // vanished worker.
        ("crash-once" | "compute-crash-once", 1) => crash(),
        _ => {
            write_frame(&ServerFrame::Event(completed(
                &job.task_id,
                attempt,
                job.resume,
            )));
            wait_for_shutdown(&mut reader);
        }
    }
}

/// Increment the attempt counter in `project_dir` and return the new value.
fn count_attempt(project_dir: &Path) -> u32 {
    count_file(&project_dir.join(ATTEMPT_FILE))
}

fn count_file(path: &Path) -> u32 {
    let previous: u32 = std::fs::read_to_string(path)
        .ok()
        .and_then(|text| text.trim().parse().ok())
        .unwrap_or(0);
    let attempt = previous + 1;
    std::fs::write(path, attempt.to_string()).expect("the attempt counter is writable");
    attempt
}

/// Block until the client sends `start`. Returns `None` on shutdown or EOF.
fn wait_for_start(reader: &mut Reader) -> Option<Job> {
    loop {
        match reader.read_frame::<ClientFrame>()? {
            Ok(ClientFrame::Start(start)) => {
                record_protocol("start");
                let (project_dir, resume) = match start.request {
                    Request::Train(params) => (params.project_dir, params.resume),
                    Request::ValidateProject(params) => (params.project_dir, false),
                    _ => {
                        eprintln!("fake worker: only train and validate-project are scripted");
                        std::process::exit(2);
                    }
                };
                return Some(Job {
                    task_id: start.task_id,
                    project_dir,
                    resume,
                });
            }
            Ok(ClientFrame::Cancel(_)) => continue,
            Ok(ClientFrame::Shutdown(_)) => {
                record_protocol("shutdown");
                return None;
            }
            Err(error) => {
                eprintln!("fake worker: undecodable client frame: {error}");
                return None;
            }
        }
    }
}

/// Stay alive until the client says goodbye, so a clean attempt exits with 0
/// rather than being killed.
fn wait_for_shutdown(reader: &mut Reader) {
    while let Some(frame) = reader.read_frame::<ClientFrame>() {
        match frame {
            Ok(ClientFrame::Shutdown(_)) => {
                record_protocol("shutdown");
                return;
            }
            Ok(ClientFrame::Start(_)) | Ok(ClientFrame::Cancel(_)) => continue,
            Err(error) => {
                eprintln!("fake worker: undecodable client frame: {error}");
                return;
            }
        }
    }
}

/// Write two lines to stderr and vanish mid-task.
fn crash() -> ! {
    eprintln!("{FIRST_STDERR_LINE}");
    eprintln!("{SECOND_STDERR_LINE}");
    // Let the parent's stderr pump drain the pipe before the process
    // disappears, so the captured tail is deterministic rather than racy.
    std::thread::sleep(Duration::from_millis(200));
    std::process::exit(1);
}

fn ready(advertise_gpu: bool) -> ServerFrame {
    let mut ready = ReadyFrame {
        protocol_version: PROTOCOL_VERSION,
        worker_version: "supervised-fake-0".to_owned(),
        backends: vec![Backend::Cpu],
        adapters: vec![AdapterInfo {
            id: "cpu-0".to_owned(),
            name: "Fake CPU".to_owned(),
            backend: Backend::Cpu,
            kind: AdapterKind::Cpu,
            certified: true,
            vram_bytes: None,
        }],
        supported_commands: vec![TaskKind::Train, TaskKind::ValidateProject],
        capabilities: Capabilities {
            training: true,
            wgpu_training: false,
            onnx_validation: false,
            ffmpeg: true,
        },
    };
    if advertise_gpu {
        ready.backends.push(Backend::Wgpu);
        ready.adapters.push(AdapterInfo {
            id: "wgpu-selected".into(),
            name: "Supervised test GPU (Vulkan)".into(),
            backend: Backend::Wgpu,
            kind: AdapterKind::Discrete,
            certified: true,
            vram_bytes: None,
        });
        ready.capabilities.wgpu_training = true;
    }
    ServerFrame::Ready(ready)
}

fn record_protocol(frame: &str) {
    if let Some(directory) = std::env::var_os("FT_SUPERVISED_PROTOCOL_DIR") {
        let directory = PathBuf::from(directory);
        let handshake = std::fs::read_to_string(directory.join("handshakes.txt")).unwrap();
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(directory.join(format!("protocol-{}.txt", handshake.trim())))
            .unwrap();
        writeln!(file, "{frame}").unwrap();
    }
}

fn preparing(task_id: &TaskId) -> Event {
    let mut event = Event::new(task_id.clone(), EMITTED_AT, TaskStage::Preparing);
    event.progress = Some(Progress {
        completed: 1,
        total: Some(2),
    });
    event
}

/// Echo back which attempt this was and whether it was asked to resume, so the
/// test can prove the retry differed from the first try.
fn completed(task_id: &TaskId, attempt: u32, resume: bool) -> Event {
    let mut event = Event::new(task_id.clone(), EMITTED_AT, TaskStage::Completed);
    event.result = Some(serde_json::json!({ "attempt": attempt, "resume": resume }));
    event
}

fn write_frame(frame: &ServerFrame) {
    let line = encode_line(frame).expect("the scripted frame serializes");
    let mut stdout = std::io::stdout().lock();
    stdout
        .write_all(line.trim_end().as_bytes())
        .expect("stdout accepts a line");
    stdout.write_all(b"\n").expect("stdout accepts a newline");
    stdout.flush().expect("stdout flushes");
}
