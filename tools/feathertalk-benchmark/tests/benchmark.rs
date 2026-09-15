use std::sync::{Arc, Mutex};
use std::time::Duration;
use std::time::Instant;

use feathertalk_benchmark::{
    BenchmarkConfig, BenchmarkError, BenchmarkExecutor, render_report, run_benchmark_with_executor,
};
use feathertalk_domain::{ErrorCode, Request, TaskError, TaskStage};
use feathertalk_media::CancellationToken;
use feathertalk_worker::{CommandOutcome, WorkerConfig};

#[test]
fn benchmarks_multiple_worker_commands_and_prints_a_report() {
    let root = std::env::temp_dir().join(format!("feathertalk-benchmark-{}", std::process::id()));
    std::fs::create_dir_all(&root).expect("create benchmark root");
    let request_file = root.join("requests.json");
    std::fs::write(
        &request_file,
        r#"[
            {
                "command": "validate_project",
                "params": { "project_dir": "project-{{repeat}}" }
            },
            {
                "command": "probe_media",
                "params": { "input": "input-{{repeat}}.mp4" }
            }
        ]"#,
    )
    .expect("write request file");

    let seen_projects = Arc::new(Mutex::new(Vec::new()));
    let captured_projects = seen_projects.clone();
    let executor: BenchmarkExecutor = Arc::new(move |request: &Request, _, _, _| {
        if let Request::ValidateProject(params) = request {
            captured_projects
                .lock()
                .expect("project capture lock")
                .push(params.project_dir.display().to_string());
        }
        CommandOutcome::Completed(None)
    });

    let report = run_benchmark_with_executor(
        BenchmarkConfig {
            request_file,
            repeats: 2,
            timeout: Duration::from_secs(10),
            label: "test".to_owned(),
        },
        &WorkerConfig::from_values(None, None, None),
        executor,
    )
    .expect("benchmark completes");

    assert_eq!(report.cases.len(), 2);
    assert_eq!(report.cases[0].command, "validate_project");
    assert_eq!(report.cases[0].first.repeat, 0);
    assert_eq!(report.cases[0].warm.len(), 2);
    assert!(report.cases[0].first.seconds >= 0.0);
    assert!(
        report.cases[0]
            .warm_median_seconds
            .expect("two warm samples produce a median")
            >= 0.0
    );
    assert_eq!(report.cases[1].command, "probe_media");
    assert_eq!(
        *seen_projects.lock().expect("project capture lock"),
        vec![
            "project-0".to_owned(),
            "project-1".to_owned(),
            "project-2".to_owned()
        ]
    );

    let text = render_report(&report);
    assert!(text.contains("FeatherTalk worker command benchmark"));
    assert!(text.contains("validate_project"));
    assert!(text.contains("probe_media"));
    assert!(text.contains("warm median"));
}

#[test]
fn a_timed_out_command_is_cancelled() {
    let root = std::env::temp_dir().join(format!(
        "feathertalk-benchmark-timeout-{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&root).expect("create benchmark root");
    let request_file = root.join("requests.json");
    std::fs::write(
        &request_file,
        r#"[{ "command": "validate_project", "params": { "project_dir": "project" } }]"#,
    )
    .expect("write request file");

    let executor: BenchmarkExecutor = Arc::new(|_, _, token: &CancellationToken, _| {
        while !token.is_cancelled() {
            std::thread::sleep(Duration::from_millis(1));
        }
        CommandOutcome::Cancelled
    });
    let started = Instant::now();
    let error = run_benchmark_with_executor(
        BenchmarkConfig {
            request_file,
            repeats: 0,
            timeout: Duration::from_millis(20),
            label: "timeout".to_owned(),
        },
        &WorkerConfig::from_values(None, None, None),
        executor,
    )
    .expect_err("the command exceeds the timeout");

    assert!(matches!(error, BenchmarkError::TaskTimeout { .. }));
    assert!(started.elapsed() < Duration::from_secs(2));
}

#[test]
fn a_failed_command_reports_the_worker_error() {
    let root = std::env::temp_dir().join(format!(
        "feathertalk-benchmark-failure-{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&root).expect("create benchmark root");
    let request_file = root.join("requests.json");
    std::fs::write(
        &request_file,
        r#"[{ "command": "validate_project", "params": { "project_dir": "project" } }]"#,
    )
    .expect("write request file");

    let executor: BenchmarkExecutor = Arc::new(|_, _, _, _| {
        CommandOutcome::Failed(TaskError::new(
            ErrorCode::MediaInvalid,
            "invalid media",
            "invalid media detail",
            TaskStage::Preparing,
        ))
    });
    let error = run_benchmark_with_executor(
        BenchmarkConfig {
            request_file,
            repeats: 0,
            timeout: Duration::from_secs(1),
            label: "failure".to_owned(),
        },
        &WorkerConfig::from_values(None, None, None),
        executor,
    )
    .expect_err("the worker command fails");

    let BenchmarkError::TaskFailed {
        command,
        code,
        summary,
    } = error
    else {
        panic!("expected a task failure, got {error:?}");
    };
    assert_eq!(command, "validate_project");
    assert_eq!(code, "MEDIA_INVALID");
    assert_eq!(summary, "invalid media");
}
