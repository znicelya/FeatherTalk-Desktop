use std::process::{Command, Output};

const CLI: &str = env!("CARGO_BIN_EXE_feathertalk");
const FAKE_WORKER: &str = env!("CARGO_BIN_EXE_feathertalk-cli-fake-worker");

fn run(args: &[&str], backend: Option<&str>, adapter: Option<&str>) -> Output {
    let mut command = Command::new(CLI);
    command
        .args(args)
        .env("FEATHERTALK_WORKER_BIN", FAKE_WORKER)
        .env("FT_FAKE_WORKER_SCENARIO", "compute-environment")
        .env_remove("FEATHERTALK_WORKER_BACKEND")
        .env_remove("FEATHERTALK_WORKER_ADAPTER");
    if let Some(value) = backend {
        command.env("FEATHERTALK_WORKER_BACKEND", value);
    }
    if let Some(value) = adapter {
        command.env("FEATHERTALK_WORKER_ADAPTER", value);
    }
    command.output().expect("the CLI runs")
}

fn result(output: Output) -> serde_json::Value {
    assert_eq!(
        output.status.code(),
        Some(0),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).expect("one result document")
}

#[test]
fn compute_flags_are_global_and_reach_all_compute_commands() {
    for backend in ["wgpu", "cuda", "rocm", "auto"] {
        for task in [
            vec!["train", "p", "--epochs", "1"],
            vec!["render", "p", "checkpoint", "audio.wav", "out.mp4"],
            vec!["extract-frames", "p", "video.mp4"],
            vec!["extract-features", "p", "audio.wav"],
            vec!["normalize-media", "input.mov", "assets"],
        ] {
            let adapter = match backend {
                "cuda" => "cuda-test-0",
                "rocm" => "rocm-test-0",
                _ => "wgpu-test-0",
            };
            let mut args = vec!["--backend", backend];
            args.extend(task);
            args.extend(["--adapter", adapter]);
            assert_eq!(
                result(run(&args, None, None)),
                serde_json::json!({"backend": backend, "adapter": adapter})
            );
        }
    }
}

#[test]
fn choosing_a_backend_clears_an_inherited_adapter() {
    for (chosen, inherited_backend, inherited_adapter) in
        [("cpu", "wgpu", "wgpu-old"), ("wgpu", "cpu", "cpu-0")]
    {
        assert_eq!(
            result(run(
                &["extract-features", "p", "audio.wav", "--backend", chosen],
                Some(inherited_backend),
                Some(inherited_adapter),
            )),
            serde_json::json!({"backend": chosen, "adapter": ""})
        );
    }
}

#[test]
fn an_adapter_without_a_backend_infers_its_compute_backend() {
    for (adapter, backend) in [
        ("cpu-0", "cpu"),
        ("wgpu-test-0", "wgpu"),
        ("cuda-test-0", "cuda"),
        ("rocm-test-0", "rocm"),
    ] {
        assert_eq!(
            result(run(
                &["--adapter", adapter, "extract-features", "p", "audio.wav"],
                Some("inherited-invalid-backend"),
                Some("inherited-adapter"),
            )),
            serde_json::json!({"backend": backend, "adapter": adapter})
        );
    }
}

#[test]
fn no_flags_preserve_the_workers_inherited_configuration() {
    assert_eq!(
        result(run(
            &["extract-features", "p", "audio.wav"],
            Some("wgpu"),
            Some("wgpu-test-0"),
        )),
        serde_json::json!({"backend": "wgpu", "adapter": "wgpu-test-0"})
    );
    assert_eq!(
        result(run(&["extract-features", "p", "audio.wav"], None, None)),
        serde_json::json!({"backend": null, "adapter": null})
    );
}

#[test]
fn conflicting_choices_fail_before_worker_discovery() {
    let output = Command::new(CLI)
        .args([
            "--worker",
            "missing-worker",
            "--backend",
            "cpu",
            "--adapter",
            "wgpu-test-0",
            "extract-features",
            "p",
            "audio.wav",
        ])
        .output()
        .expect("the CLI runs");
    assert_eq!(output.status.code(), Some(3));
    let error = String::from_utf8_lossy(&output.stderr);
    assert!(error.contains("cpu-0"), "{error}");
    assert!(output.stdout.is_empty());
}

#[test]
fn unknown_uncertified_and_software_adapters_do_not_start_a_task() {
    for adapter in [
        "missing-device",
        "wgpu-experimental",
        "wgpu-software",
        "cpu-0",
    ] {
        let output = run(
            &[
                "--backend",
                "wgpu",
                "--adapter",
                adapter,
                "extract-features",
                "p",
                "a.wav",
            ],
            None,
            None,
        );
        assert_eq!(output.status.code(), Some(3), "adapter {adapter}");
        assert!(output.stdout.is_empty(), "adapter {adapter}");
        assert!(
            String::from_utf8_lossy(&output.stderr).contains(adapter),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
}

#[test]
fn clap_rejects_an_unknown_backend_with_the_session_error_exit_code() {
    let output = run(
        &["--backend", "unknown-backend", "capabilities"],
        None,
        None,
    );
    assert_eq!(output.status.code(), Some(3));
    assert!(String::from_utf8_lossy(&output.stderr).contains("unknown-backend"));
}

#[test]
fn capabilities_remain_available_to_diagnose_an_unusable_gpu_choice() {
    for args in [
        &[
            "--backend",
            "wgpu",
            "--adapter",
            "missing-device",
            "capabilities",
        ][..],
        &["capabilities"][..],
    ] {
        let output = run(
            args,
            Some("invalid-inherited-backend"),
            Some("missing-device"),
        );
        assert_eq!(
            output.status.code(),
            Some(0),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(String::from_utf8_lossy(&output.stdout).contains("wgpu-test-0"));
    }
}

#[test]
fn a_cpu_only_worker_cannot_silently_accept_an_inherited_gpu_request() {
    for args in [
        vec!["train", "p", "--epochs", "1"],
        vec!["render", "p", "checkpoint", "audio.wav", "out.mp4"],
        vec!["extract-frames", "p", "video.mp4"],
        vec!["extract-features", "p", "audio.wav"],
    ] {
        let directory = tempfile::tempdir().unwrap();
        let log = directory.path().join("protocol.txt");
        let output = Command::new(CLI)
            .args(args)
            .env("FEATHERTALK_WORKER_BIN", FAKE_WORKER)
            .env("FEATHERTALK_WORKER_BACKEND", "wgpu")
            .env("FEATHERTALK_WORKER_ADAPTER", "wgpu-test-0")
            .env("FT_FAKE_WORKER_SCENARIO", "compute-cpu-only")
            .env("FT_FAKE_PROTOCOL_LOG", &log)
            .output()
            .unwrap();
        assert_eq!(
            output.status.code(),
            Some(3),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert_eq!(std::fs::read_to_string(log).unwrap(), "shutdown\n");
    }
}

#[test]
fn inherited_invalid_or_unavailable_choices_are_rejected_before_start() {
    for (backend, adapter) in [
        ("invalid", "wgpu-test-0"),
        ("cpu", "wgpu-test-0"),
        ("wgpu", "missing-device"),
        ("wgpu", "wgpu-experimental"),
        ("wgpu", "wgpu-software"),
    ] {
        let output = run(
            &["extract-features", "p", "audio.wav"],
            Some(backend),
            Some(adapter),
        );
        assert_eq!(
            output.status.code(),
            Some(3),
            "backend={backend}, adapter={adapter}"
        );
        assert!(output.stdout.is_empty());
    }
}

#[test]
fn noncompute_commands_ignore_invalid_inherited_compute_configuration() {
    assert_eq!(
        result(run(
            &["validate-project", "p"],
            Some("invalid"),
            Some("missing-device")
        )),
        serde_json::json!({"backend": "invalid", "adapter": "missing-device"}),
    );
}

#[cfg(any(unix, windows))]
#[test]
fn nonunicode_inherited_compute_values_are_rejected_but_can_be_overridden() {
    #[cfg(windows)]
    let invalid = {
        use std::os::windows::ffi::OsStringExt;
        std::ffi::OsString::from_wide(&[0xd800])
    };
    #[cfg(unix)]
    let invalid = {
        use std::os::unix::ffi::OsStringExt;
        std::ffi::OsString::from_vec(vec![0xff])
    };

    for key in ["FEATHERTALK_WORKER_BACKEND", "FEATHERTALK_WORKER_ADAPTER"] {
        let command = || {
            let mut command = Command::new(CLI);
            command
                .env("FEATHERTALK_WORKER_BIN", FAKE_WORKER)
                .env("FT_FAKE_WORKER_SCENARIO", "compute-environment")
                .env("FEATHERTALK_WORKER_BACKEND", "cpu")
                .env("FEATHERTALK_WORKER_ADAPTER", "")
                .env(key, &invalid);
            command
        };
        let directory = tempfile::tempdir().unwrap();
        let log = directory.path().join("protocol.txt");
        let output = command()
            .args(["extract-features", "p", "audio.wav"])
            .env("FT_FAKE_PROTOCOL_LOG", &log)
            .output()
            .unwrap();
        assert_eq!(output.status.code(), Some(3), "{key}");
        let error = String::from_utf8_lossy(&output.stderr);
        assert!(error.contains(key), "{error}");
        assert!(error.contains("not valid Unicode"), "{error}");
        assert_eq!(std::fs::read_to_string(log).unwrap(), "shutdown\n");

        assert!(
            command()
                .arg("capabilities")
                .output()
                .unwrap()
                .status
                .success()
        );
        assert_eq!(
            result(
                command()
                    .args(["--backend", "cpu", "extract-features", "p", "audio.wav"])
                    .output()
                    .unwrap()
            ),
            serde_json::json!({"backend": "cpu", "adapter": ""})
        );
    }
}
