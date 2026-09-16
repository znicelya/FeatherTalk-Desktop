use std::sync::{Arc, Mutex};
use std::time::Duration;
use std::time::Instant;

use feathertalk_benchmark::{
    BenchmarkConfig, BenchmarkError, BenchmarkExecutor, PipelineConfig, render_pipeline_report,
    render_report, run_benchmark_with_executor, run_full_pipeline_with_executor,
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

#[test]
fn full_pipeline_runs_each_stage_in_order_on_a_fresh_project() {
    let root = std::env::temp_dir().join(format!(
        "feathertalk-benchmark-pipeline-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).expect("create pipeline benchmark root");
    let input = root.join("input.mp4");
    std::fs::write(&input, b"fixture").expect("write pipeline fixture");

    let seen = Arc::new(Mutex::new(Vec::new()));
    let captured = seen.clone();
    let executor: BenchmarkExecutor = Arc::new(move |request: &Request, _, _, _| {
        let command = request.kind().as_slug().to_owned();
        let project_dir = match request {
            Request::NormalizeMedia(params) => params
                .output_dir
                .parent()
                .expect("normalization output has a project parent")
                .to_path_buf(),
            Request::ExtractFrames(params) => params.project_dir.clone(),
            Request::ExtractFeatures(params) => params.project_dir.clone(),
            Request::LockAssetPackage(params) => params.project_dir.clone(),
            Request::Train(params) => params.project_dir.clone(),
            Request::Render(params) => params.project_dir.clone(),
            Request::ProbeMedia(_) => {
                captured
                    .lock()
                    .expect("pipeline capture lock")
                    .push((command.clone(), None, None));
                return CommandOutcome::Completed(None);
            }
            _ => panic!("unexpected full-pipeline command: {command}"),
        };
        let checkpoint = if let Request::Train(_) = request {
            Some(project_dir.join("models/unet/checkpoint-00000125"))
        } else if let Request::Render(params) = request {
            Some(params.checkpoint.clone())
        } else {
            None
        };
        captured.lock().expect("pipeline capture lock").push((
            command,
            Some(project_dir),
            checkpoint.clone(),
        ));
        if let Request::Train(_) = request {
            CommandOutcome::Completed(Some(serde_json::json!({
                "checkpoint_dir": checkpoint.expect("train captures a checkpoint")
            })))
        } else {
            CommandOutcome::Completed(None)
        }
    });

    let report = run_full_pipeline_with_executor(
        PipelineConfig {
            input,
            project_root: root.join("projects"),
            repeats: 1,
            timeout: Duration::from_secs(10),
            label: "full".to_owned(),
            epochs: 1,
            max_output_frames: Some(5),
            keep_projects: true,
        },
        &WorkerConfig::from_values(None, None, None),
        executor,
    )
    .expect("full pipeline completes");

    let expected_commands = [
        "probe_media",
        "normalize_media",
        "extract_frames",
        "extract_features",
        "lock_asset_package",
        "train",
        "render",
    ];
    assert_eq!(report.cases.len(), expected_commands.len());
    assert_eq!(report.repeats, 1);
    for (case, expected) in report.cases.iter().zip(expected_commands) {
        assert_eq!(case.command, expected);
        assert_eq!(case.first.repeat, 0);
        assert_eq!(case.warm.len(), 1);
    }

    let seen = seen.lock().expect("pipeline capture lock");
    assert_eq!(seen.len(), expected_commands.len() * 2);
    let first_pass = &seen[..expected_commands.len()];
    let second_pass = &seen[expected_commands.len()..];
    assert_eq!(
        first_pass
            .iter()
            .map(|(command, _, _)| command.as_str())
            .collect::<Vec<_>>(),
        expected_commands
    );
    assert_eq!(
        second_pass
            .iter()
            .map(|(command, _, _)| command.as_str())
            .collect::<Vec<_>>(),
        expected_commands
    );
    assert_ne!(
        first_pass[1].1.as_deref(),
        second_pass[1].1.as_deref(),
        "each repeat gets a fresh project"
    );
    assert_eq!(
        first_pass[6].2.as_deref(),
        first_pass[5].2.as_deref(),
        "render uses the checkpoint train published"
    );

    let text = render_pipeline_report(&report);
    assert!(text.contains("FeatherTalk full pipeline benchmark"));
    assert!(text.contains("probe_media"));
    assert!(text.contains("render"));
}
