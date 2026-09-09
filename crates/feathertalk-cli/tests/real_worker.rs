//! End-to-end coverage against the real worker binary.
//!
//! `cargo test -p feathertalk-cli` does not build `feathertalk-worker`, so there
//! is no `CARGO_BIN_EXE_feathertalk-worker`. The worker is found as a sibling of
//! this crate's own binary, which is where cargo puts every binary in the
//! workspace's shared target directory. When it is absent — a fresh clone that
//! only built this crate — each test prints why it skipped and passes, unless
//! `FEATHERTALK_REQUIRE_E2E=1` demands the real thing. CI sets that variable;
//! a developer running one crate's tests is not blocked by it.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use tempfile::TempDir;

const CLI: &str = env!("CARGO_BIN_EXE_feathertalk");

/// The variable that makes a missing worker a failure instead of a skip.
const REQUIRE_E2E: &str = "FEATHERTALK_REQUIRE_E2E";

/// How many frames the locked package holds. Two seconds of audio become 98
/// tokens (the extraction test above derives the number) and the lock demands
/// two tokens per frame, so 49 is the only frame count that makes the fit a
/// no-op.
const LOCKED_FRAME_COUNT: u64 = 49;

/// PFLD's digest, a compile-time constant in `feathertalk-pfld`, which this
/// crate does not depend on.
const PFLD_SHA256: &str = "e131dd764236fde54a27b2f7084906119f06c28b140bf127b459ec967e92915b";

/// The per-frame digests the report carries. The lock verifies structure and
/// never re-hashes a frame, so any 64-hex string does.
const SHA256: &str = "1111111111111111111111111111111111111111111111111111111111111111";

/// Locate `feathertalk-worker` next to the CLI binary under test.
fn worker_path() -> Option<PathBuf> {
    let cli = PathBuf::from(CLI);
    let path = cli.parent()?.join(format!(
        "feathertalk-worker{}",
        std::env::consts::EXE_SUFFIX
    ));
    path.is_file().then_some(path)
}

/// The worker, or `None` after explaining the skip.
fn worker_or_skip(test: &str) -> Option<PathBuf> {
    if let Some(path) = worker_path() {
        return Some(path);
    }
    let required = std::env::var(REQUIRE_E2E).as_deref() == Ok("1");
    assert!(
        !required,
        "{REQUIRE_E2E}=1 but feathertalk-worker was not found next to {CLI}; \
         build it with `cargo build -p feathertalk-worker`"
    );
    println!(
        "skipping {test}: feathertalk-worker is not built; run `cargo build -p feathertalk-worker`"
    );
    None
}

/// Run the CLI against the real worker. `env` is applied to the CLI process,
/// which the worker inherits — that is how the worker's own configuration is
/// reached without the CLI knowing anything about it.
fn run(worker: &Path, args: &[&str], env: &[(&str, &str)]) -> Output {
    let mut command = Command::new(CLI);
    command.arg("--worker").arg(worker).args(args);
    for (key, value) in env {
        command.env(key, value);
    }
    command.output().expect("the CLI binary runs")
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
fn capabilities_reports_the_real_handshake() {
    let Some(worker) = worker_or_skip("capabilities_reports_the_real_handshake") else {
        return;
    };
    let output = run(&worker, &["capabilities"], &[]);
    assert_eq!(code(&output), 0, "stderr was: {}", stderr(&output));
    let text = stdout(&output);
    // `CPU_ADAPTER_ID` and the six commands the worker always advertises.
    assert!(text.contains("cpu-0"), "{text}");
    assert!(text.contains("validate_project"), "{text}");
    assert!(text.contains("inspect_model"), "{text}");
    assert!(text.contains("import_legacy_model"), "{text}");
    assert!(text.contains("migrate_legacy_features"), "{text}");
    assert!(text.contains("export_model_package"), "{text}");
    assert!(text.contains("export_onnx"), "{text}");
}

#[test]
fn a_real_legacy_model_is_imported_end_to_end() {
    let Some(worker) = worker_or_skip("a_real_legacy_model_is_imported_end_to_end") else {
        return;
    };
    let Some(source) = std::env::var_os("FEATHERTALK_WORKER_LEGACY_MODEL")
        .map(PathBuf::from)
        .filter(|path| path.is_file())
    else {
        println!(
            "skipping a_real_legacy_model_is_imported_end_to_end: it needs an explicit \
             FEATHERTALK_WORKER_LEGACY_MODEL .pth file"
        );
        return;
    };
    let Some(licenses) = source
        .parent()
        .map(|parent| parent.join("LICENSES.json"))
        .filter(|path| path.is_file())
    else {
        println!(
            "skipping a_real_legacy_model_is_imported_end_to_end: source sibling LICENSES.json is missing"
        );
        return;
    };
    let destination_root = TempDir::new().expect("a temporary directory is available");
    let destination = destination_root.path().join("imported-model");
    let source_arg = source.to_string_lossy().into_owned();
    let destination_arg = destination.to_string_lossy().into_owned();
    let output = run(
        &worker,
        &[
            "import-legacy-model",
            &source_arg,
            "feather-hubert",
            &destination_arg,
        ],
        &[],
    );
    assert_eq!(code(&output), 0, "stderr was: {}", stderr(&output));
    assert!(licenses.is_file());
    let result: serde_json::Value =
        serde_json::from_str(&stdout(&output)).expect("stdout is exactly one JSON document");
    let manifest: serde_json::Value = serde_json::from_slice(
        &std::fs::read(destination.join("manifest.json")).expect("manifest is published"),
    )
    .expect("manifest is valid JSON");
    assert_eq!(result["kind"], "import_legacy_model");
    assert_eq!(result["model_kind"], "feather_hubert");
    assert_eq!(result["source"], source.display().to_string());
    assert_eq!(result["destination"], destination.display().to_string());
    assert_eq!(result["source_sha256"], manifest["source"]["sha256"]);
    assert_eq!(result["model_sha256"], manifest["model"]["sha256"]);
    assert!(
        result["tensor_count"]
            .as_u64()
            .expect("tensor count is numeric")
            > 0
    );
    assert!(
        result["total_elements"]
            .as_u64()
            .expect("element count is numeric")
            > 0
    );
    assert!(destination.join("model.safetensors").is_file());
    assert!(destination.join("LICENSES.json").is_file());
}

/// The legacy feature migration, through the real worker.
///
/// The fixture is written by this test rather than taken from the machine: the
/// NPY container is eight bytes of magic, a padded header line and raw
/// little-endian `f32`, so the test can produce a legitimate one without a new
/// dependency and without reading anything under `demo/`.
#[test]
fn real_legacy_features_are_migrated_end_to_end() {
    let Some(worker) = worker_or_skip("real_legacy_features_are_migrated_end_to_end") else {
        return;
    };
    let root = TempDir::new().expect("a temporary directory is available");
    let source = root.path().join("aud_hu.npy");
    let destination = root.path().join("features.f32");
    let frames = 3_usize;
    write_legacy_npy(&source, frames);
    let source_arg = source.to_string_lossy().into_owned();
    let destination_arg = destination.to_string_lossy().into_owned();

    let output = run(
        &worker,
        &["migrate-legacy-features", &source_arg, &destination_arg],
        &[],
    );

    assert_eq!(code(&output), 0, "stderr was: {}", stderr(&output));
    let result: serde_json::Value =
        serde_json::from_str(&stdout(&output)).expect("stdout is exactly one JSON document");
    assert_eq!(result["kind"], "migrate_legacy_features");
    assert_eq!(result["source"], source.display().to_string());
    assert_eq!(result["destination"], destination.display().to_string());
    assert_eq!(result["source_shape"], serde_json::json!([frames, 2, 1024]));
    assert_eq!(result["tokens"], frames * 2);
    assert_eq!(result["dims"], 1024);
    assert_eq!(
        result["sha256"]
            .as_str()
            .expect("the digest is a string")
            .len(),
        64
    );
    let published = std::fs::read(&destination).expect("the artifact is published");
    assert_eq!(
        result["bytes"].as_u64().expect("the size is numeric"),
        published.len() as u64
    );
    // The versioned container, not a copy of the NPY.
    assert_eq!(&published[..8], b"FTF32\0\0\0");

    // The destination is published once and never overwritten.
    let again = run(
        &worker,
        &["migrate-legacy-features", &source_arg, &destination_arg],
        &[],
    );
    assert_eq!(code(&again), 1, "stdout was: {}", stdout(&again));
    assert_eq!(
        std::fs::read(&destination).expect("the artifact survives"),
        published
    );
}

/// Write a rank-three `f32` NPY file shaped `[frames, 2, 1024]`.
fn write_legacy_npy(path: &Path, frames: usize) {
    let header =
        format!("{{'descr': '<f4', 'fortran_order': False, 'shape': ({frames}, 2, 1024), }}");
    // The format demands the data start on a 64-byte boundary, counting the six
    // magic bytes, the two version bytes and the two header-length bytes.
    let prefix = 10 + header.len() + 1;
    let padding = (64 - prefix % 64) % 64;
    let mut header_bytes = header.into_bytes();
    header_bytes.extend(std::iter::repeat_n(b' ', padding));
    header_bytes.push(b'\n');
    let mut bytes = Vec::with_capacity(10 + header_bytes.len() + frames * 2 * 1024 * 4);
    bytes.extend_from_slice(b"\x93NUMPY");
    bytes.push(1);
    bytes.push(0);
    bytes.extend_from_slice(
        &u16::try_from(header_bytes.len())
            .expect("the header fits in a v1.0 length")
            .to_le_bytes(),
    );
    bytes.extend_from_slice(&header_bytes);
    for index in 0..frames * 2 * 1024 {
        bytes.extend_from_slice(&(index as f32 / 32.0).to_le_bytes());
    }
    std::fs::write(path, bytes).expect("the NPY fixture is written");
}

/// The real FeatherHuBERT package, read through the real protocol. No ffmpeg and
/// no demo clip: inspection reads manifests and nothing else.
#[test]
fn a_real_model_package_is_inspected_end_to_end() {
    let Some(worker) = worker_or_skip("a_real_model_package_is_inspected_end_to_end") else {
        return;
    };
    let Some(hubert) = real_dir("HUBERT_DIR") else {
        println!(
            "skipping a_real_model_package_is_inspected_end_to_end: it needs \
             FEATHERTALK_WORKER_HUBERT_DIR"
        );
        return;
    };
    let source_arg = hubert.to_string_lossy().into_owned();
    let output = run(&worker, &["inspect-model", &source_arg], &[]);
    assert_eq!(code(&output), 0, "stderr was: {}", stderr(&output));
    let inspected: serde_json::Value =
        serde_json::from_str(&stdout(&output)).expect("stdout is exactly one JSON document");

    assert_eq!(inspected["source_kind"], "model_package");
    assert_eq!(inspected["source_path"], hubert.display().to_string());
    assert_eq!(inspected["schema_version"], 1);
    assert_eq!(inspected["model_kind"], "feather_hubert");
    assert_eq!(inspected["architecture_version"], "feather-hubert-burn-v1");
    // A published package is not a resume point and carries no step counter.
    assert!(inspected["model_config_sha256"].is_null());
    assert!(inspected["epoch"].is_null());
    assert_eq!(inspected["training_mode"], "inference");
    assert!(
        inspected["parameter_count"]
            .as_u64()
            .expect("parameter count is numeric")
            > 0
    );
    assert!(
        inspected["tensor_count"]
            .as_u64()
            .expect("tensor count is numeric")
            > 0
    );
    assert_eq!(inspected["inputs"][0]["name"], "waveform");
    assert_eq!(inspected["outputs"][0]["name"], "hidden");

    let files = inspected["files"].as_array().expect("files is an array");
    assert_eq!(files.len(), 2);
    for file in files {
        assert_eq!(file["sha256"].as_str().expect("sha256 is text").len(), 64);
        // The package on this machine is intact, so the manifest and the disk
        // agree -- which is also what makes `compatible` true below.
        assert_eq!(file["bytes"], file["bytes_on_disk"]);
    }

    assert_eq!(inspected["compatible"], true);
    assert_eq!(
        inspected["incompatibilities"]
            .as_array()
            .expect("incompatibilities is an array")
            .len(),
        0
    );
}

/// Model package export across the real process boundary.
///
/// A successful export needs a production-sized checkpoint: the architecture is
/// resolved from the kind the checkpoint recorded, so the weights on disk have
/// to match the released configuration, which no test can conjure. What is
/// pinned here is the gate every request passes first -- a published package is
/// not an export source -- and how that refusal reaches the client. No ffmpeg,
/// no model directories, nothing under `demo/`.
#[test]
fn exporting_a_published_package_is_a_task_failure() {
    let Some(worker) = worker_or_skip("exporting_a_published_package_is_a_task_failure") else {
        return;
    };
    let root = TempDir::new().expect("a temporary directory is available");
    let source = root.path().join("original_unet_v1");
    std::fs::create_dir(&source).expect("the temporary package directory is writable");
    // Which of the two layouts a directory is comes from the model file it
    // holds, so these two entries are enough to make it a package: the contents
    // are never read, because the request is refused before any reader runs.
    std::fs::write(source.join("manifest.json"), "{}").expect("the manifest is writable");
    std::fs::write(source.join("model.safetensors"), b"not weights")
        .expect("the model file is writable");
    let destination = root.path().join("exported");
    let source_arg = source.to_string_lossy().into_owned();
    let destination_arg = destination.to_string_lossy().into_owned();

    let output = run(
        &worker,
        &["export-model-package", &source_arg, &destination_arg],
        &[],
    );

    assert_eq!(code(&output), 1, "stdout was: {}", stdout(&output));
    let text = stderr(&output);
    assert!(text.contains("模型导出失败"), "{text}");
    assert!(text.contains("MODEL_INCOMPATIBLE"), "{text}");
    assert!(
        text.contains("export requires a training checkpoint"),
        "the refusal names the gate it failed: {text}"
    );
    // A refused request publishes nothing.
    assert!(
        !destination.exists(),
        "the destination stays absent: {}",
        destination.display()
    );
}

/// The ONNX export's own gate, across the real process boundary: a directory
/// that is neither layout is refused before any reader runs. No ffmpeg, no model
/// directories, nothing under `demo/`.
#[test]
fn exporting_a_directory_that_is_no_model_is_a_task_failure() {
    let Some(worker) = worker_or_skip("exporting_a_directory_that_is_no_model_is_a_task_failure")
    else {
        return;
    };
    let root = TempDir::new().expect("a temporary directory is available");
    let source = root.path().join("not-a-model");
    std::fs::create_dir(&source).expect("the temporary source directory is writable");
    std::fs::write(source.join("manifest.json"), "{}").expect("the manifest is writable");
    let destination = root.path().join("model.onnx");
    let source_arg = source.to_string_lossy().into_owned();
    let destination_arg = destination.to_string_lossy().into_owned();

    let output = run(
        &worker,
        &[
            "export-onnx",
            &source_arg,
            "feather-hubert",
            &destination_arg,
        ],
        &[],
    );

    assert_eq!(code(&output), 1, "stdout was: {}", stdout(&output));
    let text = stderr(&output);
    assert!(text.contains("ONNX 导出失败"), "{text}");
    assert!(text.contains("MODEL_INCOMPATIBLE"), "{text}");
    assert!(
        text.contains("holds neither"),
        "the refusal names the gate it failed: {text}"
    );
    // A refused request publishes nothing.
    assert!(
        !destination.exists(),
        "the destination stays absent: {}",
        destination.display()
    );
}

/// A released package becomes an opset 17 ONNX file, through the real protocol.
///
/// Gated on `FEATHERTALK_WORKER_HUBERT_DIR` for the same reason inspection is:
/// only a released package carries the configuration the public contract
/// describes, and no test can conjure one.
#[test]
fn a_real_model_package_is_exported_to_onnx_end_to_end() {
    let Some(worker) = worker_or_skip("a_real_model_package_is_exported_to_onnx_end_to_end") else {
        return;
    };
    let Some(hubert) = real_dir("HUBERT_DIR") else {
        println!(
            "skipping a_real_model_package_is_exported_to_onnx_end_to_end: it needs \
             FEATHERTALK_WORKER_HUBERT_DIR"
        );
        return;
    };
    let root = TempDir::new().expect("a temporary directory is available");
    let destination = root.path().join("feather_hubert.onnx");
    let source_arg = hubert.to_string_lossy().into_owned();
    let destination_arg = destination.to_string_lossy().into_owned();

    let output = run(
        &worker,
        &[
            "export-onnx",
            &source_arg,
            "feather-hubert",
            &destination_arg,
        ],
        &[],
    );

    assert_eq!(code(&output), 0, "stderr was: {}", stderr(&output));
    let exported: serde_json::Value =
        serde_json::from_str(&stdout(&output)).expect("stdout is exactly one JSON document");
    assert_eq!(exported["kind"], "export_onnx");
    assert_eq!(exported["model_kind"], "feather_hubert");
    assert_eq!(exported["opset"], 17);
    assert_eq!(exported["source"], hubert.display().to_string());
    assert_eq!(exported["destination"], destination.display().to_string());
    assert_eq!(
        exported["sha256"]
            .as_str()
            .expect("the digest is text")
            .len(),
        64
    );
    let published = std::fs::metadata(&destination).expect("the model is published");
    assert_eq!(
        exported["bytes"].as_u64().expect("the size is numeric"),
        published.len()
    );
}

#[test]
fn an_invalid_manifest_is_a_task_failure() {
    let Some(worker) = worker_or_skip("an_invalid_manifest_is_a_task_failure") else {
        return;
    };
    let project = TempDir::new().expect("a temporary directory is available");
    // A manifest that parses as JSON but is not a manifest. An *absent*
    // `project.json` would be `ProjectError::Io`, which `io_error_code` maps to
    // `WORKER_CRASHED`; only a manifest the worker managed to read and reject
    // reaches `MEDIA_INVALID`, which is the mapping worth pinning here.
    std::fs::write(project.path().join("project.json"), "{}")
        .expect("the temporary manifest is writable");
    let path = project.path().to_string_lossy().into_owned();
    let output = run(&worker, &["validate-project", &path], &[]);
    assert_eq!(code(&output), 1, "stdout was: {}", stdout(&output));
    assert!(
        stderr(&output).contains("MEDIA_INVALID"),
        "the wire error code is shown verbatim: {}",
        stderr(&output)
    );
}

#[test]
fn a_missing_ffprobe_makes_probe_media_unsupported() {
    let Some(worker) = worker_or_skip("a_missing_ffprobe_makes_probe_media_unsupported") else {
        return;
    };
    // With no resolvable ffprobe the worker drops `probe_media` from
    // `supported_commands`, so the client's capability gate refuses the request
    // before any task starts.
    let output = run(
        &worker,
        &["probe-media", "clip.mp4"],
        &[(
            "FEATHERTALK_WORKER_FFPROBE",
            "this-path-does-not-exist-ffprobe",
        )],
    );
    assert_eq!(code(&output), 3, "stdout was: {}", stdout(&output));
    let text = stderr(&output);
    assert!(
        text.contains("FEATHERTALK_WORKER_FFPROBE"),
        "the advice names the variable that would fix it: {text}"
    );
}

#[test]
fn a_missing_toolchain_makes_normalize_media_unsupported() {
    let Some(worker) = worker_or_skip("a_missing_toolchain_makes_normalize_media_unsupported")
    else {
        return;
    };
    // A relative tool path is refused by the worker's configuration, so
    // `normalize_media` never reaches `supported_commands` and the client's
    // capability gate answers instead of a task.
    let output = run(
        &worker,
        &["normalize-media", "clip.mp4", "assets"],
        &[("FEATHERTALK_WORKER_FFPROBE", "relative-ffprobe")],
    );
    assert_eq!(code(&output), 3, "stdout was: {}", stdout(&output));
    let text = stderr(&output);
    assert!(text.contains("normalize_media"), "{text}");
    assert!(text.contains("FEATHERTALK_WORKER_FFMPEG"), "{text}");
}

#[test]
fn a_missing_input_is_a_normalize_task_failure() {
    let Some(worker) = worker_or_skip("a_missing_input_is_a_normalize_task_failure") else {
        return;
    };
    // Absolute tool paths are all the worker's configuration requires, so the
    // command is accepted and fails where it should: on the input. This is the
    // one end-to-end normalization path that needs no real ffmpeg.
    let temp = TempDir::new().expect("a temporary directory is available");
    let missing = temp.path().join("absent.mp4");
    let assets = temp.path().join("assets");
    let fake_tool = temp
        .path()
        .join("not-a-real-ffmpeg")
        .to_string_lossy()
        .into_owned();
    let output = run(
        &worker,
        &[
            "normalize-media",
            &missing.to_string_lossy(),
            &assets.to_string_lossy(),
        ],
        &[
            ("FEATHERTALK_WORKER_FFPROBE", fake_tool.as_str()),
            ("FEATHERTALK_WORKER_FFMPEG", fake_tool.as_str()),
        ],
    );
    assert_eq!(code(&output), 1, "stdout was: {}", stdout(&output));
    let text = stderr(&output);
    assert!(text.contains("MEDIA_INVALID"), "{text}");
    // The task failed before the output directory was needed.
    assert!(!assets.join("video_25fps.mp4").exists());
}

/// A full normalization, only when real tools are pointed at by the
/// environment. Neither this repository nor CI ships ffmpeg, so an absent tool
/// is a skip, not a failure: the alternative is a test that fails for reasons
/// that have nothing to do with the code under test.
#[test]
fn a_real_clip_is_normalized_end_to_end() {
    let Some(worker) = worker_or_skip("a_real_clip_is_normalized_end_to_end") else {
        return;
    };
    let (Some(ffmpeg), Some(ffprobe)) = (real_tool("FFMPEG"), real_tool("FFPROBE")) else {
        println!(
            "skipping a_real_clip_is_normalized_end_to_end: set FEATHERTALK_WORKER_FFMPEG and \
             FEATHERTALK_WORKER_FFPROBE to real binaries to run it"
        );
        return;
    };
    let temp = TempDir::new().expect("a temporary directory is available");
    let clip = temp.path().join("clip.mp4");
    // One second of colour bars and a tone: the smallest input with both
    // streams the pipeline requires.
    let generated = Command::new(&ffmpeg)
        .args([
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=320x240:rate=30:duration=1",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=1",
            "-shortest",
        ])
        .arg(&clip)
        .output()
        .expect("ffmpeg runs");
    assert!(
        generated.status.success(),
        "ffmpeg could not generate the clip: {}",
        String::from_utf8_lossy(&generated.stderr)
    );
    let assets = temp.path().join("assets");
    let ffmpeg_arg = ffmpeg.to_string_lossy().into_owned();
    let ffprobe_arg = ffprobe.to_string_lossy().into_owned();
    let output = run(
        &worker,
        &[
            "normalize-media",
            &clip.to_string_lossy(),
            &assets.to_string_lossy(),
        ],
        &[
            ("FEATHERTALK_WORKER_FFMPEG", ffmpeg_arg.as_str()),
            ("FEATHERTALK_WORKER_FFPROBE", ffprobe_arg.as_str()),
        ],
    );
    assert_eq!(code(&output), 0, "stderr was: {}", stderr(&output));
    assert!(assets.join("video_25fps.mp4").is_file());
    assert!(assets.join("audio_16k_mono.wav").is_file());
    let text = stdout(&output);
    assert!(text.contains("pcm_s16le"), "{text}");
    let narration = stderr(&output);
    assert!(narration.contains("进度 3/3"), "{narration}");
}

/// A media tool from the environment, only if it is an existing file.
fn real_tool(suffix: &str) -> Option<PathBuf> {
    let path = PathBuf::from(std::env::var(format!("FEATHERTALK_WORKER_{suffix}")).ok()?);
    path.is_file().then_some(path)
}

/// The whole extraction, and only when the real toolchain, the imported models,
/// and the demo clip are all present. Neither this repository nor CI ships
/// ffmpeg or the model artifacts, so anything missing is a skip rather than a
/// failure: a test that fails for reasons unrelated to the code under test
/// teaches nothing.
#[test]
fn a_real_second_is_extracted_end_to_end() {
    let Some(worker) = worker_or_skip("a_real_second_is_extracted_end_to_end") else {
        return;
    };
    let (Some(ffmpeg), Some(ffprobe), Some(scrfd), Some(pfld), Some(demo)) = (
        real_tool("FFMPEG"),
        real_tool("FFPROBE"),
        real_dir("SCRFD_DIR"),
        real_dir("PFLD_DIR"),
        demo_clip(),
    ) else {
        println!(
            "skipping a_real_second_is_extracted_end_to_end: it needs \
             FEATHERTALK_WORKER_FFMPEG, FEATHERTALK_WORKER_FFPROBE, \
             FEATHERTALK_WORKER_SCRFD_DIR, FEATHERTALK_WORKER_PFLD_DIR, and \
             demo/feathertalk_demo_latest_188.mp4"
        );
        return;
    };
    let project = TempDir::new().expect("a temporary directory is available");
    let assets = project.path().join("assets");
    std::fs::create_dir_all(&assets).expect("the assets directory is writable");
    // Admission only asks that the manifest exists; reading it is
    // `validate-project`'s job, and this command runs before a project has the
    // assets that validation demands.
    std::fs::write(project.path().join("project.json"), "{}")
        .expect("the temporary manifest is writable");
    let video = assets.join("video_25fps.mp4");
    cut_one_second(&ffmpeg, &demo, &video);

    let project_arg = project.path().to_string_lossy().into_owned();
    let video_arg = video.to_string_lossy().into_owned();
    let ffmpeg_arg = ffmpeg.to_string_lossy().into_owned();
    let ffprobe_arg = ffprobe.to_string_lossy().into_owned();
    let scrfd_arg = scrfd.to_string_lossy().into_owned();
    let pfld_arg = pfld.to_string_lossy().into_owned();
    let output = run(
        &worker,
        &["extract-frames", &project_arg, &video_arg],
        &[
            ("FEATHERTALK_WORKER_FFMPEG", ffmpeg_arg.as_str()),
            ("FEATHERTALK_WORKER_FFPROBE", ffprobe_arg.as_str()),
            ("FEATHERTALK_WORKER_SCRFD_DIR", scrfd_arg.as_str()),
            ("FEATHERTALK_WORKER_PFLD_DIR", pfld_arg.as_str()),
        ],
    );
    assert_eq!(code(&output), 0, "stderr was: {}", stderr(&output));

    let result: serde_json::Value =
        serde_json::from_str(&stdout(&output)).expect("stdout is exactly one JSON document");
    assert_eq!(result["output_dir"], assets.display().to_string());
    assert_eq!(result["frame_count"], 25);
    assert_eq!(result["frame_width"], 1280);
    assert_eq!(result["frame_height"], 720);

    assert_eq!(file_count(&assets.join("frames")), 25);
    assert_eq!(file_count(&assets.join("landmarks")), 25);
    assert!(assets.join("frames").join("000000.jpg").is_file());
    assert!(assets.join("landmarks").join("000000.lms").is_file());

    let report =
        std::fs::read_to_string(assets.join("quality.json")).expect("the report is readable");
    let report: serde_json::Value = serde_json::from_str(&report).expect("the report is JSON");
    assert_eq!(report["frame_count"], 25);
    assert_eq!(report["accepted_count"], 25);
    assert!(
        report["anomalies"]
            .as_array()
            .expect("anomalies is an array")
            .is_empty(),
        "{report}"
    );

    let narration = stderr(&output);
    assert!(narration.contains("正在提取视频帧"), "{narration}");
    assert!(narration.contains("正在检测人脸"), "{narration}");
    assert!(narration.contains("进度 25/25"), "{narration}");
}

/// A model directory from the environment, only if it is an existing directory.
fn real_dir(suffix: &str) -> Option<PathBuf> {
    let path = PathBuf::from(std::env::var(format!("FEATHERTALK_WORKER_{suffix}")).ok()?);
    path.is_dir().then_some(path)
}

/// The demo video this repository ships, resolved from this crate's directory.
fn demo_clip() -> Option<PathBuf> {
    let path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../demo/feathertalk_demo_latest_188.mp4");
    path.is_file().then_some(path)
}

/// One second of the demo video, re-encoded at 25 fps into the project's
/// `assets/` under the name the pipeline expects. The offset is 30 s because the
/// committed `demo_frame_v1` fixture records a 0.8108 face score and a 776.03
/// blur variance for that frame, well clear of the 0.50 and 20.0 thresholds; if
/// a neighbouring frame is ever rejected, `quality.json` names its index and the
/// offset can move.
fn cut_one_second(ffmpeg: &Path, demo: &Path, video: &Path) {
    let output = Command::new(ffmpeg)
        .args(["-hide_banner", "-loglevel", "error", "-y", "-ss", "30.000"])
        .arg("-i")
        .arg(demo)
        .args([
            "-frames:v",
            "25",
            "-an",
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-r",
            "25",
        ])
        .arg(video)
        .output()
        .expect("ffmpeg runs");
    assert!(
        output.status.success(),
        "ffmpeg could not cut the clip: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

/// The number of regular files directly inside `dir`.
fn file_count(dir: &Path) -> usize {
    std::fs::read_dir(dir)
        .unwrap_or_else(|error| panic!("{} is readable: {error}", dir.display()))
        .filter_map(Result::ok)
        .filter(|entry| entry.path().is_file())
        .count()
}

/// The whole feature extraction, and only when the real ffmpeg, a built
/// FeatherHuBERT package, and the demo clip are all present. Neither this
/// repository nor CI ships ffmpeg or the model package, so anything missing is a
/// skip rather than a failure, for the reason the two tests above give.
#[test]
fn real_audio_becomes_features_end_to_end() {
    let Some(worker) = worker_or_skip("real_audio_becomes_features_end_to_end") else {
        return;
    };
    let (Some(ffmpeg), Some(hubert), Some(demo)) =
        (real_tool("FFMPEG"), real_dir("HUBERT_DIR"), demo_clip())
    else {
        println!(
            "skipping real_audio_becomes_features_end_to_end: it needs \
             FEATHERTALK_WORKER_FFMPEG, FEATHERTALK_WORKER_HUBERT_DIR, and \
             demo/feathertalk_demo_latest_188.mp4"
        );
        return;
    };
    let project = TempDir::new().expect("a temporary directory is available");
    let assets = project.path().join("assets");
    std::fs::create_dir_all(&assets).expect("the assets directory is writable");
    // Admission only asks that the manifest exists; reading it is
    // `validate-project`'s job, and this command runs before a project has the
    // assets that validation demands.
    std::fs::write(project.path().join("project.json"), "{}")
        .expect("the temporary manifest is writable");
    let audio = assets.join("audio_16k_mono.wav");
    cut_audio(&ffmpeg, &demo, &audio, "2");

    let project_arg = project.path().to_string_lossy().into_owned();
    let audio_arg = audio.to_string_lossy().into_owned();
    let hubert_arg = hubert.to_string_lossy().into_owned();
    let output = run(
        &worker,
        &["extract-features", &project_arg, &audio_arg],
        &[("FEATHERTALK_WORKER_HUBERT_DIR", hubert_arg.as_str())],
    );
    assert_eq!(code(&output), 0, "stderr was: {}", stderr(&output));

    // Two seconds at 16 kHz is 32 000 samples, which one chunk covers whole. The
    // 400-sample kernel and the 320-sample stride turn them into
    // `(32_000 - 80) / 320` = 99 frames, the odd-token trim drops one, and 98
    // tokens of 1024 dimensions behind a 44-byte header make
    // `44 + 98 * 1024 * 4` = 401_452 bytes. If a future resampler hands over a
    // different sample count, recompute the four numbers the same way rather
    // than adjusting them by hand.
    let features_dir = assets.join("features");
    let features = features_dir.join("feather_hubert.f32");
    let result: serde_json::Value =
        serde_json::from_str(&stdout(&output)).expect("stdout is exactly one JSON document");
    assert_eq!(result["output_dir"], features_dir.display().to_string());
    assert_eq!(result["feature_file"], features.display().to_string());
    assert_eq!(result["tokens"], 98);
    assert_eq!(result["dims"], 1024);
    assert_eq!(result["frame_count"], 49);
    assert_eq!(result["bytes"], 401_452);
    assert_eq!(
        std::fs::metadata(&features)
            .expect("the feature file is readable")
            .len(),
        401_452
    );
    // Exactly one file, and none of the bookkeeping this command stays out of.
    assert_eq!(file_count(&features_dir), 1);
    assert!(!assets.join("assets.json").exists());
    assert!(!assets.join("quality.json").exists());

    // The digest in the payload is the package's own, which is what lets a later
    // run decide whether these features still match the encoder.
    let manifest = std::fs::read_to_string(hubert.join("manifest.json"))
        .expect("the package manifest is readable");
    let manifest: serde_json::Value =
        serde_json::from_str(&manifest).expect("the manifest is JSON");
    assert_eq!(result["model_sha256"], manifest["model"]["sha256"]);

    let narration = stderr(&output);
    assert!(narration.contains("正在提取特征"), "{narration}");
    assert!(narration.contains("进度 1/1"), "{narration}");
}

/// `seconds` of the demo video's audio, decoded into the one shape the reader
/// admits -- 16 kHz, mono, 16-bit PCM -- and written under the name the pipeline
/// expects. No offset is needed: unlike a video frame, whose face score decides
/// whether it is usable, any cut of this clip's audio extracts alike.
fn cut_audio(ffmpeg: &Path, demo: &Path, audio: &Path, seconds: &str) {
    let output = Command::new(ffmpeg)
        .args(["-hide_banner", "-loglevel", "error", "-y"])
        .arg("-i")
        .arg(demo)
        .args(["-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", "-t"])
        .arg(seconds)
        .arg(audio)
        .output()
        .expect("ffmpeg runs");
    assert!(
        output.status.success(),
        "ffmpeg could not cut the audio: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn a_real_package_is_locked_end_to_end() {
    let Some(worker) = worker_or_skip("a_real_package_is_locked_end_to_end") else {
        return;
    };
    let (Some(ffmpeg), Some(hubert), Some(demo)) =
        (real_tool("FFMPEG"), real_dir("HUBERT_DIR"), demo_clip())
    else {
        println!(
            "skipping a_real_package_is_locked_end_to_end: it needs \
             FEATHERTALK_WORKER_FFMPEG, FEATHERTALK_WORKER_HUBERT_DIR, and \
             demo/feathertalk_demo_latest_188.mp4"
        );
        return;
    };
    let project = TempDir::new().expect("a temporary directory is available");
    let assets = project.path().join("assets");
    // `extract-features` creates `assets/features` itself, exactly as the
    // extraction test above relies on; only the frame directories are ours.
    for directory in ["frames", "landmarks"] {
        std::fs::create_dir_all(assets.join(directory)).expect("the assets tree is writable");
    }
    std::fs::write(project.path().join("project.json"), "{}")
        .expect("the temporary manifest is writable");
    let audio = assets.join("audio_16k_mono.wav");
    cut_audio(&ffmpeg, &demo, &audio, "2");
    // The lock only stats the video; nothing reads it. It is cut with the same
    // helper the extraction test uses so the package keeps its real shape.
    cut_one_second(&ffmpeg, &demo, &assets.join("video_25fps.mp4"));

    let project_arg = project.path().to_string_lossy().into_owned();
    let audio_arg = audio.to_string_lossy().into_owned();
    let hubert_arg = hubert.to_string_lossy().into_owned();
    let env = [("FEATHERTALK_WORKER_HUBERT_DIR", hubert_arg.as_str())];
    let output = run(
        &worker,
        &["extract-features", &project_arg, &audio_arg],
        &env,
    );
    assert_eq!(code(&output), 0, "stderr was: {}", stderr(&output));

    write_frame_fixtures(&assets, LOCKED_FRAME_COUNT);

    let output = run(&worker, &["lock-asset-package", &project_arg], &env);
    assert_eq!(code(&output), 0, "stderr was: {}", stderr(&output));

    let features_dir = assets.join("features");
    let features = features_dir.join("feather_hubert.f32");
    let manifest_file = assets.join("assets.json");
    let result: serde_json::Value =
        serde_json::from_str(&stdout(&output)).expect("stdout is exactly one JSON document");
    assert_eq!(result["project_dir"], project.path().display().to_string());
    assert_eq!(result["manifest_file"], manifest_file.display().to_string());
    assert_eq!(result["feature_file"], features.display().to_string());
    assert_eq!(result["frame_count"], 49);
    assert_eq!(result["frame_width"], 1280);
    assert_eq!(result["frame_height"], 720);
    assert_eq!(result["tokens"], 98);
    assert_eq!(result["dims"], 1024);
    // The same 44 + 98 * 1024 * 4 bytes the extraction wrote: 49 frames need
    // exactly the 98 tokens already in the file, so the fit changes nothing.
    assert_eq!(result["bytes"], 401_452);
    assert_eq!(result["token_adjustment"], 0);
    assert_eq!(result["landmark_model_sha256"], PFLD_SHA256);
    assert_eq!(result["sha256"].as_str().unwrap().len(), 64);
    let package_manifest = std::fs::read_to_string(hubert.join("manifest.json"))
        .expect("the package manifest is readable");
    let package_manifest: serde_json::Value =
        serde_json::from_str(&package_manifest).expect("the manifest is JSON");
    assert_eq!(
        result["feature_model_sha256"],
        package_manifest["model"]["sha256"]
    );

    let written = std::fs::read(&manifest_file).expect("the locked manifest is readable");
    let written: serde_json::Value =
        serde_json::from_slice(&written).expect("the locked manifest is JSON");
    assert_eq!(written["schema_version"], 1);
    assert_eq!(written["state"], "locked");
    assert_eq!(written["video_fps"], 25);
    assert_eq!(written["audio_sample_rate"], 16_000);
    assert_eq!(written["audio_channels"], 1);
    assert_eq!(written["frame_count"], 49);
    assert_eq!(written["frame_width"], 1280);
    assert_eq!(written["frame_height"], 720);
    assert_eq!(written["feature_type"], "feather_hubert");
    assert_eq!(written["feature_shape"], serde_json::json!([49, 2, 1024]));
    assert_eq!(written["landmark_model_sha256"], PFLD_SHA256);
    // The commit rewrote the file in place at its original size.
    assert_eq!(
        std::fs::metadata(&features)
            .expect("the feature file is readable")
            .len(),
        401_452
    );
    assert_eq!(file_count(&features_dir), 1);

    let narration = stderr(&output);
    assert!(narration.contains("准备中"), "{narration}");
    assert!(narration.contains("进度 49/49"), "{narration}");
}

/// Stand in for `extract-frames`, which needs SCRFD and PFLD this repository
/// does not ship. Copies the committed 1280x720 fixture `count` times, writes a
/// matching landmark file next to each frame, and hand-writes the quality report
/// the lock reads. The digests are placeholders: the lock verifies structure and
/// never re-hashes a frame.
fn write_frame_fixtures(assets: &Path, count: u64) {
    let fixture = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../feathertalk-frame-adapters/tests/fixtures/demo_frame_v1");
    let frame_bytes =
        std::fs::read(fixture.join("frame.jpg")).expect("the committed frame fixture is readable");
    let landmarks = fixture_landmarks(&fixture);
    let mut frames = Vec::new();
    for index in 0..count {
        let frame_file = format!("frames/{index:06}.jpg");
        let landmark_file = format!("landmarks/{index:06}.lms");
        std::fs::write(assets.join(&frame_file), &frame_bytes).expect("the frame is writable");
        std::fs::write(assets.join(&landmark_file), &landmarks)
            .expect("the landmark file is writable");
        frames.push(serde_json::json!({
            "index": index,
            "frame_file": frame_file,
            "landmark_file": landmark_file,
            "frame_bytes": frame_bytes.len(),
            "frame_sha256": SHA256,
            "landmark_sha256": SHA256,
            "face_score": 0.9,
            "bbox": [0.0, 0.0, 64.0, 64.0],
            "blur_variance": 120.0
        }));
    }
    let report = serde_json::json!({
        "schema_version": 1,
        "frame_count": count,
        "accepted_count": count,
        "frames": frames,
        "anomalies": []
    });
    let text = serde_json::to_string_pretty(&report).expect("the report serializes");
    std::fs::write(assets.join("quality.json"), text).expect("the report is writable");
}

/// The 110 landmarks the fixture records for `frame.jpg`, in the `.lms` shape the
/// frame pipeline writes: one `x y` line per point, in detection order. The
/// coordinates have to be the real ones -- points 1, 31 and 52 are the face box a
/// training run crops to, and 90..110 are the mouth it projects into the inner
/// crop.
fn fixture_landmarks(fixture: &Path) -> String {
    let manifest = std::fs::read_to_string(fixture.join("fixture.json"))
        .expect("the fixture manifest is readable");
    let manifest: serde_json::Value =
        serde_json::from_str(&manifest).expect("the fixture manifest is JSON");
    let points = manifest["frames"]["sharp"]["landmarks"]
        .as_array()
        .expect("the sharp frame records its landmarks");
    let mut text = String::new();
    for point in points {
        let x = point[0].as_f64().expect("a landmark x is a number");
        let y = point[1].as_f64().expect("a landmark y is a number");
        text.push_str(&format!("{x} {y}\n"));
    }
    text
}

/// The frames the training test locks, and half the tokens
/// `TRAINED_AUDIO_SECONDS` extracts: `2 * frame_count` exactly, so the lock fits
/// the feature file without trimming a token.
const TRAINED_FRAME_COUNT: u64 = 4;

/// 0.18 s at 16 kHz is 2 880 samples, which the 400-sample kernel and the
/// 320-sample stride turn into `(2 880 - 80) / 320` = 8 FeatherHuBERT frames and
/// so into 8 tokens. Every cut from 2 640 to 3 279 samples yields the same 8,
/// which is the slack this leaves a resampler that hands over a few samples more
/// or fewer than the arithmetic above.
const TRAINED_AUDIO_SECONDS: &str = "0.18";

/// A manifest `validate_project_dir` accepts. `extract-features` and
/// `lock-asset-package` only need the file to exist, which is why the tests
/// above write `{}`; training opens the project properly and reads every field.
const PROJECT_MANIFEST: &str = r#"{
  "schema_version": 1,
  "project_id": "demo",
  "display_name": "Demo",
  "asset_package": "assets/assets.json",
  "default_model": "original_unet",
  "task_history": []
}"#;

/// The whole training command, over a package this test locks itself, and only
/// when the real ffmpeg, a built FeatherHuBERT package, a built VGG19 package
/// and the demo clip are all present. Anything missing is a skip rather than a
/// failure, for the reason the tests above give.
#[test]
fn a_real_project_is_trained_end_to_end() {
    let Some(worker) = worker_or_skip("a_real_project_is_trained_end_to_end") else {
        return;
    };
    let (Some(ffmpeg), Some(hubert), Some(vgg19), Some(demo)) = (
        real_tool("FFMPEG"),
        real_dir("HUBERT_DIR"),
        real_dir("VGG19_DIR"),
        demo_clip(),
    ) else {
        println!(
            "skipping a_real_project_is_trained_end_to_end: it needs \
             FEATHERTALK_WORKER_FFMPEG, FEATHERTALK_WORKER_HUBERT_DIR, \
             FEATHERTALK_WORKER_VGG19_DIR, and \
             demo/feathertalk_demo_latest_188.mp4"
        );
        return;
    };
    let project = TempDir::new().expect("a temporary directory is available");
    let assets = project.path().join("assets");
    for directory in ["frames", "landmarks"] {
        std::fs::create_dir_all(assets.join(directory)).expect("the assets tree is writable");
    }
    std::fs::write(project.path().join("project.json"), PROJECT_MANIFEST)
        .expect("the temporary manifest is writable");
    let audio = assets.join("audio_16k_mono.wav");
    cut_audio(&ffmpeg, &demo, &audio, TRAINED_AUDIO_SECONDS);
    // Nothing reads the video: the lock stats it and so does the project
    // validation the dataset runs. It is cut with the helper the tests above use.
    cut_one_second(&ffmpeg, &demo, &assets.join("video_25fps.mp4"));

    let project_arg = project.path().to_string_lossy().into_owned();
    let audio_arg = audio.to_string_lossy().into_owned();
    let hubert_arg = hubert.to_string_lossy().into_owned();
    let vgg19_arg = vgg19.to_string_lossy().into_owned();
    let hubert_env = [("FEATHERTALK_WORKER_HUBERT_DIR", hubert_arg.as_str())];
    let output = run(
        &worker,
        &["extract-features", &project_arg, &audio_arg],
        &hubert_env,
    );
    assert_eq!(code(&output), 0, "stderr was: {}", stderr(&output));
    let extracted: serde_json::Value =
        serde_json::from_str(&stdout(&output)).expect("stdout is exactly one JSON document");
    // The audio cut is pinned here rather than left to a comment: if a resampler
    // ever moves the sample count out of the window above, this is the assertion
    // that names the number.
    assert_eq!(extracted["tokens"], 8);
    assert_eq!(extracted["frame_count"], TRAINED_FRAME_COUNT);

    write_frame_fixtures(&assets, TRAINED_FRAME_COUNT);
    let output = run(&worker, &["lock-asset-package", &project_arg], &hubert_env);
    assert_eq!(code(&output), 0, "stderr was: {}", stderr(&output));
    let locked: serde_json::Value =
        serde_json::from_str(&stdout(&output)).expect("stdout is exactly one JSON document");
    assert_eq!(locked["frame_count"], TRAINED_FRAME_COUNT);
    assert_eq!(locked["token_adjustment"], 0);

    let output = run(
        &worker,
        &["train", &project_arg, "--mode", "baseline", "--epochs", "1"],
        &[("FEATHERTALK_WORKER_VGG19_DIR", vgg19_arg.as_str())],
    );
    assert_eq!(code(&output), 0, "stderr was: {}", stderr(&output));
    let result: serde_json::Value =
        serde_json::from_str(&stdout(&output)).expect("stdout is exactly one JSON document");
    assert_eq!(result["mode"], "baseline");
    assert_eq!(result["variant"], "original_unet");
    assert_eq!(result["model_kind"], "original_unet");
    assert_eq!(result["backend"], "ndarray-cpu");
    assert_eq!(result["model_config_sha256"].as_str().unwrap().len(), 64);
    assert_eq!(result["frame_count"], TRAINED_FRAME_COUNT);
    assert_eq!(result["epochs_requested"], 1);
    assert_eq!(result["epochs_completed"], 1);
    // One step per frame, and the one epoch boundary is the only publish point.
    assert_eq!(result["global_step"], TRAINED_FRAME_COUNT);
    assert_eq!(result["samples_seen"], TRAINED_FRAME_COUNT);
    assert_eq!(result["resumed_from"], serde_json::Value::Null);
    assert_eq!(result["checkpoints_written"], 1);
    assert_eq!(result["metrics_written"], 1);
    assert_eq!(result["previews_written"], 1);
    let loss = result["total_loss"]
        .as_f64()
        .expect("a finished run reports the loss it last saw");
    assert!(loss.is_finite() && loss >= 0.0, "{loss}");

    // The three artifacts a publish writes, every one of them named by the same
    // global step. The checkpoint is joined one component at a time because the
    // worker reports a native path and `join` keeps a forward slash as written.
    let checkpoint = project
        .path()
        .join("models")
        .join("unet")
        .join("checkpoint-00000004");
    assert_eq!(result["checkpoint_dir"], checkpoint.display().to_string());
    assert!(checkpoint.join("manifest.json").is_file());
    let outputs = project.path().join("outputs");
    assert!(outputs.join("metrics/step-00000004.json").is_file());
    assert!(
        outputs
            .join("preview/step-00000004/manifest.json")
            .is_file()
    );

    let narration = stderr(&output);
    assert!(narration.contains("正在训练"), "{narration}");
    assert!(narration.contains("进度 4/4"), "{narration}");
}

/// The frames the render test locks, and half the tokens
/// `RENDERED_AUDIO_SECONDS` extracts, so the lock trims nothing.
const RENDERED_FRAME_COUNT: u64 = 2;

/// 0.09 s at 16 kHz is 1 440 samples, which the 400-sample kernel and the
/// 320-sample stride turn into `(1 440 - 80) / 320` = 4 FeatherHuBERT frames and
/// so into 4 tokens, which is two video frames. Every cut from 1 360 to 1 679
/// samples yields the same 4, which is the slack a resampler gets.
const RENDERED_AUDIO_SECONDS: &str = "0.09";

/// The whole slice, in the order a person would use it: extract features from
/// real audio, lock the package, train one epoch, then render the checkpoint
/// that training just wrote. This is the only test where a real ffmpeg, a real
/// JPEG decode, a real Burn record and a playable mp4 all meet. It skips for the
/// reasons the training test gives, plus `FEATHERTALK_WORKER_FFPROBE`, which the
/// render needs to probe the source video.
#[test]
fn a_real_project_is_rendered_end_to_end() {
    let Some(worker) = worker_or_skip("a_real_project_is_rendered_end_to_end") else {
        return;
    };
    let (Some(ffmpeg), Some(ffprobe), Some(hubert), Some(vgg19), Some(demo)) = (
        real_tool("FFMPEG"),
        real_tool("FFPROBE"),
        real_dir("HUBERT_DIR"),
        real_dir("VGG19_DIR"),
        demo_clip(),
    ) else {
        println!(
            "skipping a_real_project_is_rendered_end_to_end: it needs \
             FEATHERTALK_WORKER_FFMPEG, FEATHERTALK_WORKER_FFPROBE, \
             FEATHERTALK_WORKER_HUBERT_DIR, FEATHERTALK_WORKER_VGG19_DIR, and \
             demo/feathertalk_demo_latest_188.mp4"
        );
        return;
    };
    let project = TempDir::new().expect("a temporary directory is available");
    let assets = project.path().join("assets");
    for directory in ["frames", "landmarks"] {
        std::fs::create_dir_all(assets.join(directory)).expect("the assets tree is writable");
    }
    std::fs::write(project.path().join("project.json"), PROJECT_MANIFEST)
        .expect("the temporary manifest is writable");
    let audio = assets.join("audio_16k_mono.wav");
    cut_audio(&ffmpeg, &demo, &audio, RENDERED_AUDIO_SECONDS);
    // The render probes this video for its size and frame rate, so unlike the
    // training test it is read rather than only stated.
    cut_one_second(&ffmpeg, &demo, &assets.join("video_25fps.mp4"));

    let project_arg = project.path().to_string_lossy().into_owned();
    let audio_arg = audio.to_string_lossy().into_owned();
    let hubert_arg = hubert.to_string_lossy().into_owned();
    let vgg19_arg = vgg19.to_string_lossy().into_owned();
    let hubert_env = [("FEATHERTALK_WORKER_HUBERT_DIR", hubert_arg.as_str())];
    let output = run(
        &worker,
        &["extract-features", &project_arg, &audio_arg],
        &hubert_env,
    );
    assert_eq!(code(&output), 0, "stderr was: {}", stderr(&output));
    let extracted: serde_json::Value =
        serde_json::from_str(&stdout(&output)).expect("stdout is exactly one JSON document");
    assert_eq!(extracted["tokens"], 4);
    assert_eq!(extracted["frame_count"], RENDERED_FRAME_COUNT);

    write_frame_fixtures(&assets, RENDERED_FRAME_COUNT);
    let output = run(&worker, &["lock-asset-package", &project_arg], &hubert_env);
    assert_eq!(code(&output), 0, "stderr was: {}", stderr(&output));
    let locked: serde_json::Value =
        serde_json::from_str(&stdout(&output)).expect("stdout is exactly one JSON document");
    assert_eq!(locked["frame_count"], RENDERED_FRAME_COUNT);
    assert_eq!(locked["token_adjustment"], 0);

    // The training test pins what this prints; here it only has to leave a
    // checkpoint behind. One epoch over two frames is two steps.
    let output = run(
        &worker,
        &["train", &project_arg, "--mode", "baseline", "--epochs", "1"],
        &[("FEATHERTALK_WORKER_VGG19_DIR", vgg19_arg.as_str())],
    );
    assert_eq!(code(&output), 0, "stderr was: {}", stderr(&output));

    // The checkpoint is joined one component at a time because the worker
    // reports a native path and `join` keeps a forward slash as written.
    let checkpoint = project
        .path()
        .join("models")
        .join("unet")
        .join("checkpoint-00000002");
    assert!(checkpoint.join("manifest.json").is_file());

    let output_file = project.path().join("preview.mp4");
    let checkpoint_arg = checkpoint.to_string_lossy().into_owned();
    let output_arg = output_file.to_string_lossy().into_owned();
    let ffmpeg_arg = ffmpeg.to_string_lossy().into_owned();
    let ffprobe_arg = ffprobe.to_string_lossy().into_owned();
    let output = run(
        &worker,
        &[
            "render",
            &project_arg,
            &checkpoint_arg,
            &audio_arg,
            &output_arg,
        ],
        &[
            ("FEATHERTALK_WORKER_FFMPEG", ffmpeg_arg.as_str()),
            ("FEATHERTALK_WORKER_FFPROBE", ffprobe_arg.as_str()),
        ],
    );
    assert_eq!(code(&output), 0, "stderr was: {}", stderr(&output));
    let rendered: serde_json::Value =
        serde_json::from_str(&stdout(&output)).expect("stdout is exactly one JSON document");

    assert_eq!(rendered["output_path"], output_file.display().to_string());
    assert_eq!(rendered["frame_count"], RENDERED_FRAME_COUNT);
    // The committed frame fixture is 1280x720, and the render pastes the
    // generated mouth back into the full frame rather than cropping it.
    assert_eq!(rendered["width"], 1280);
    assert_eq!(rendered["height"], 720);
    assert_eq!(rendered["fps"], 25);
    assert_eq!(rendered["backend"], "ndarray-cpu");
    assert_eq!(rendered["checkpoint_dir"], checkpoint.display().to_string());
    assert_eq!(rendered["model_kind"], "original_unet");
    assert_eq!(rendered["model_config_sha256"].as_str().unwrap().len(), 64);
    assert_eq!(rendered["checkpoint_epoch"], 1);
    assert_eq!(rendered["checkpoint_global_step"], RENDERED_FRAME_COUNT);
    assert_eq!(rendered["source_frame_count"], RENDERED_FRAME_COUNT);
    assert_eq!(rendered["max_output_frames"], serde_json::Value::Null);

    // A file a person can play: the atomic publish renamed the staging file into
    // place, so this path is the mp4 and not a leftover fragment.
    let published = std::fs::metadata(&output_file).expect("the render published its output");
    assert!(published.is_file());
    assert!(published.len() > 0, "{}", published.len());

    let narration = stderr(&output);
    assert!(narration.contains("正在渲染"), "{narration}");
    assert!(narration.contains("进度 2/2"), "{narration}");
}
