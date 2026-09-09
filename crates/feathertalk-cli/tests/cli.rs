use std::process::{Command, Output};

const CLI: &str = env!("CARGO_BIN_EXE_feathertalk");
const FAKE_WORKER: &str = env!("CARGO_BIN_EXE_feathertalk-cli-fake-worker");

/// Run the CLI against the fake worker.
///
/// The worker path and the scenario are set on the child's environment rather
/// than this process's: `std::env::set_var` is `unsafe` in edition 2024 and
/// would race between tests that run in parallel.
fn run(scenario: &str, args: &[&str]) -> Output {
    Command::new(CLI)
        .args(args)
        .env("FEATHERTALK_WORKER_BIN", FAKE_WORKER)
        .env("FT_FAKE_WORKER_SCENARIO", scenario)
        .output()
        .expect("the CLI binary runs")
}

fn code(output: &Output) -> i32 {
    output.status.code().expect("the CLI exits normally")
}

fn stdout(output: &Output) -> String {
    String::from_utf8_lossy(&output.stdout).into_owned()
}

fn stderr(output: &Output) -> String {
    String::from_utf8_lossy(&output.stderr).into_owned()
}

#[test]
fn stdout_carries_only_the_result() {
    let output = run("ready-complete", &["validate-project", "some-project"]);
    assert_eq!(code(&output), 0, "stderr was: {}", stderr(&output));
    let value: serde_json::Value =
        serde_json::from_str(&stdout(&output)).expect("stdout is exactly one JSON document");
    assert_eq!(value, serde_json::json!({ "checked": true }));
    // Progress narration belongs on stderr so stdout stays redirectable.
    assert!(
        stderr(&output).contains("[preparing]"),
        "{}",
        stderr(&output)
    );
}

#[test]
fn json_mode_streams_the_workers_own_frames() {
    let output = run("ready-complete", &["--json", "validate-project", "p"]);
    assert_eq!(code(&output), 0, "stderr was: {}", stderr(&output));
    let text = stdout(&output);
    let lines: Vec<&str> = text.lines().filter(|line| !line.is_empty()).collect();
    assert_eq!(lines.len(), 3, "ready plus two events: {lines:?}");
    assert!(lines[0].contains("\"frame\":\"ready\""), "{:?}", lines[0]);
    for line in &lines {
        serde_json::from_str::<serde_json::Value>(line)
            .unwrap_or_else(|error| panic!("{line:?} is not a JSON frame: {error}"));
    }
    assert!(
        lines[2].contains("completed"),
        "the last frame is the terminal event: {:?}",
        lines[2]
    );
}

#[test]
fn json_and_quiet_are_refused() {
    let output = run("ready-complete", &["--json", "--quiet", "capabilities"]);
    assert_eq!(code(&output), 3, "a usage error is a session error, not 2");
}

#[test]
fn an_invalid_task_id_is_refused_before_the_task_starts() {
    let output = run(
        "ready-complete",
        &["--task-id", "nope", "validate-project", "p"],
    );
    assert_eq!(code(&output), 3);
    assert!(
        stderr(&output).contains("任务 ID 无效"),
        "{}",
        stderr(&output)
    );
}

#[test]
fn a_task_failure_exits_one_and_prints_the_wire_code() {
    let output = run("fail", &["validate-project", "p"]);
    assert_eq!(code(&output), 1);
    let text = stderr(&output);
    assert!(text.contains("MEDIA_INVALID"), "{text}");
    assert!(text.contains("输入文件无法解析"), "{text}");
    assert!(
        text.contains("详情: ffprobe exited with status 1"),
        "{text}"
    );
}

#[test]
fn a_cancelled_task_exits_two() {
    // The fake worker reports itself cancelled, so no signal is involved.
    let output = run("self-cancel", &["validate-project", "p"]);
    assert_eq!(code(&output), 2);
    assert!(
        stderr(&output).contains("任务已取消"),
        "{}",
        stderr(&output)
    );
}

#[test]
fn a_protocol_mismatch_exits_three() {
    let output = run("bad-version", &["validate-project", "p"]);
    assert_eq!(code(&output), 3);
    assert!(
        stderr(&output).contains("协议版本不匹配"),
        "{}",
        stderr(&output)
    );
}

#[test]
fn an_unsupported_command_names_the_variable_that_would_fix_it() {
    let output = run("only-validate", &["probe-media", "clip.mp4"]);
    assert_eq!(code(&output), 3);
    let text = stderr(&output);
    assert!(text.contains("FEATHERTALK_WORKER_FFPROBE"), "{text}");
    assert!(text.contains("validate_project"), "{text}");
}

#[test]
fn capabilities_reports_the_handshake() {
    let output = run("ready-complete", &["capabilities"]);
    assert_eq!(code(&output), 0, "stderr was: {}", stderr(&output));
    let text = stdout(&output);
    assert!(text.contains("cpu-0"), "{text}");
    assert!(text.contains("validate_project"), "{text}");
}

#[test]
fn normalize_media_prints_the_result_and_narrates_progress() {
    let output = run(
        "normalize-progress",
        &["normalize-media", "clip.mp4", "assets"],
    );
    assert_eq!(code(&output), 0, "stderr was: {}", stderr(&output));
    let text = stdout(&output);
    assert!(text.contains("video_25fps.mp4"), "{text}");
    assert!(text.contains("audio_16k_mono.wav"), "{text}");
    let narration = stderr(&output);
    assert!(narration.contains("正在提取视频帧"), "{narration}");
    assert!(narration.contains("正在提取音频"), "{narration}");
    assert!(narration.contains("进度 2/3 (66.7%)"), "{narration}");
}

#[test]
fn an_empty_path_argument_is_refused() {
    let output = run("ready-complete", &["validate-project", ""]);
    assert_eq!(code(&output), 3);
    // Clap refuses an empty value for a required positional before `run` sees
    // it, so the message is clap's own usage error rather than the CLI's
    // Chinese one; the exit code is the same. `build_request`'s own rejection is
    // covered by a unit test in `run.rs`.
    assert!(
        stderr(&output).contains("PROJECT_DIR"),
        "{}",
        stderr(&output)
    );
}

#[test]
fn an_unsupported_extract_frames_names_the_model_variables() {
    // The fake worker advertises `validate_project` alone, so the client's
    // capability gate answers before any task starts.
    let output = run("only-validate", &["extract-frames", "project", "clip.mp4"]);
    assert_eq!(code(&output), 3);
    let text = stderr(&output);
    assert!(text.contains("extract_frames"), "{text}");
    assert!(text.contains("FEATHERTALK_WORKER_SCRFD_DIR"), "{text}");
    assert!(text.contains("FEATHERTALK_WORKER_PFLD_DIR"), "{text}");
}

#[test]
fn an_unsupported_extract_features_names_the_hubert_variable() {
    // The fake worker advertises `validate_project` alone, so the client's
    // capability gate answers before any task starts.
    let output = run("only-validate", &["extract-features", "p", "audio.wav"]);
    assert_eq!(code(&output), 3);
    let text = stderr(&output);
    assert!(text.contains("extract_features"), "{text}");
    assert!(text.contains("FEATHERTALK_WORKER_HUBERT_DIR"), "{text}");
}

#[test]
fn an_unsupported_lock_asset_package_names_the_hubert_variable() {
    // The fake worker advertises `validate_project` alone, so the client's
    // capability gate answers before any task starts.
    let output = run("only-validate", &["lock-asset-package", "p"]);
    assert_eq!(code(&output), 3);
    let text = stderr(&output);
    assert!(text.contains("lock_asset_package"), "{text}");
    assert!(text.contains("FEATHERTALK_WORKER_HUBERT_DIR"), "{text}");
}

#[test]
fn an_unsupported_train_names_the_vgg19_variable() {
    // The fake worker advertises `validate_project` alone, so the client's
    // capability gate answers before any task starts.
    let output = run("only-validate", &["train", "p", "--epochs", "1"]);
    assert_eq!(code(&output), 3);
    let text = stderr(&output);
    assert!(text.contains("train"), "{text}");
    assert!(text.contains("FEATHERTALK_WORKER_VGG19_DIR"), "{text}");
}

#[test]
fn an_unsupported_render_names_the_media_variables() {
    // The fake worker advertises `validate_project` alone, so the client's
    // capability gate answers before any task starts.
    let output = run("only-validate", &["render", "p", "c", "a.wav", "o.mp4"]);
    assert_eq!(code(&output), 3);
    let text = stderr(&output);
    assert!(text.contains("render"), "{text}");
    assert!(text.contains("FEATHERTALK_WORKER_FFPROBE"), "{text}");
    assert!(text.contains("FEATHERTALK_WORKER_FFMPEG"), "{text}");
}

#[test]
fn an_unknown_training_mode_is_refused_with_the_choices() {
    let output = run(
        "ready-complete",
        &["train", "p", "--epochs", "1", "--mode", "mouth"],
    );
    assert_eq!(code(&output), 3, "a usage error is a session error");
    // Clap owns this message, which is exactly why the enums are mirrored.
    let text = stderr(&output);
    assert!(text.contains("mouth-roi"), "{text}");
    assert!(text.contains("temporal"), "{text}");
}

#[test]
fn train_needs_an_epoch_count() {
    let output = run("ready-complete", &["train", "p"]);
    assert_eq!(code(&output), 3);
    assert!(stderr(&output).contains("--epochs"), "{}", stderr(&output));
}
