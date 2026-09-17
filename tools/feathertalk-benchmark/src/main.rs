use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Duration;

use clap::{Parser, ValueEnum};
use feathertalk_benchmark::{
    BenchmarkConfig, PipelineConfig, render_pipeline_report, render_report, run_benchmark,
    run_full_pipeline,
};
use feathertalk_worker::WorkerConfig;

const DEFAULT_INPUT: &str = "tools/feathertalk-benchmark/fixtures/kanghui_5s.mp4";
const DEFAULT_PROJECT_ROOT: &str = "target/benchmark/full-pipeline";

#[derive(Debug, Parser)]
#[command(
    name = "feathertalk-benchmark",
    version,
    about = "Benchmark the FeatherTalk worker in-process"
)]
struct Arguments {
    /// Run a custom JSON request file instead of the full pipeline.
    #[arg(long, value_name = "PATH", conflicts_with = "input")]
    request_file: Option<PathBuf>,

    /// Input media for the full pipeline. Defaults to the bundled fixture.
    #[arg(long, value_name = "PATH")]
    input: Option<PathBuf>,

    /// Root directory for temporary full-pipeline projects.
    #[arg(long, value_name = "PATH", default_value = DEFAULT_PROJECT_ROOT)]
    project_root: PathBuf,

    /// Warm command executions after the first execution for each request.
    #[arg(long, default_value_t = 3)]
    repeats: usize,

    /// Command timeout in seconds.
    #[arg(long, default_value_t = 900.0)]
    timeout_secs: f64,

    /// Report label.
    #[arg(long, default_value = "main")]
    label: String,

    /// Compute backend used by worker commands.
    #[arg(long, value_enum)]
    backend: Option<BackendArg>,

    /// Compute adapter ID, when the backend supports selection.
    #[arg(long, value_name = "ID")]
    adapter: Option<String>,

    /// Print the report as JSON.
    #[arg(long)]
    json: bool,

    /// Training epochs in the full pipeline.
    #[arg(long, default_value_t = 1)]
    epochs: u32,

    /// Maximum frames rendered by the full pipeline. Omit to render everything.
    #[arg(long, value_name = "N")]
    max_output_frames: Option<u64>,

    /// Keep full-pipeline project directories after successful runs.
    #[arg(long)]
    keep_projects: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum BackendArg {
    Auto,
    Cpu,
    Wgpu,
    Cuda,
    Rocm,
}

impl BackendArg {
    fn as_value(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Cpu => "cpu",
            Self::Wgpu => "wgpu",
            Self::Cuda => "cuda",
            Self::Rocm => "rocm",
        }
    }
}

fn main() -> ExitCode {
    let arguments = Arguments::parse();
    if arguments.timeout_secs <= 0.0 {
        eprintln!("--timeout-secs must be greater than zero");
        return ExitCode::FAILURE;
    }

    let worker_config = WorkerConfig::from_env().with_compute_selection(
        arguments.backend.map(BackendArg::as_value),
        arguments.adapter.as_deref(),
    );

    if let Some(request_file) = arguments.request_file {
        let config = BenchmarkConfig {
            request_file,
            repeats: arguments.repeats,
            timeout: Duration::from_secs_f64(arguments.timeout_secs),
            label: arguments.label,
        };
        return match run_benchmark(config, &worker_config) {
            Ok(report) => print_report(&report, arguments.json, render_report),
            Err(error) => benchmark_failure(error),
        };
    }

    let config = PipelineConfig {
        input: arguments
            .input
            .unwrap_or_else(|| PathBuf::from(DEFAULT_INPUT)),
        project_root: arguments.project_root,
        repeats: arguments.repeats,
        timeout: Duration::from_secs_f64(arguments.timeout_secs),
        label: arguments.label,
        epochs: arguments.epochs,
        max_output_frames: arguments.max_output_frames,
        keep_projects: arguments.keep_projects,
    };
    match run_full_pipeline(config, &worker_config) {
        Ok(report) => print_report(&report, arguments.json, render_pipeline_report),
        Err(error) => benchmark_failure(error),
    }
}

fn print_report<T: serde::Serialize>(
    report: &T,
    json: bool,
    render: impl Fn(&T) -> String,
) -> ExitCode {
    if json {
        match serde_json::to_string_pretty(report) {
            Ok(text) => println!("{text}"),
            Err(error) => {
                eprintln!("failed to serialize the benchmark report: {error}");
                return ExitCode::FAILURE;
            }
        }
    } else {
        println!("{}", render(report));
    }
    ExitCode::SUCCESS
}

fn benchmark_failure(error: impl std::fmt::Display) -> ExitCode {
    eprintln!("benchmark failed: {error}");
    ExitCode::FAILURE
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[test]
    fn backend_flag_accepts_rocm() {
        let arguments = Arguments::try_parse_from([
            "feathertalk-benchmark",
            "--backend",
            "rocm",
            "--repeats",
            "0",
        ])
        .expect("rocm is a supported backend");
        assert_eq!(arguments.backend, Some(BackendArg::Rocm));
        assert_eq!(arguments.backend.unwrap().as_value(), "rocm");
    }
}
