use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Duration;

use clap::{Parser, ValueEnum};
use feathertalk_benchmark::{BenchmarkConfig, render_report, run_benchmark};
use feathertalk_worker::WorkerConfig;

#[derive(Debug, Parser)]
#[command(
    name = "feathertalk-benchmark",
    version,
    about = "Benchmark FeatherTalk worker commands in-process"
)]
struct Arguments {
    /// JSON array of worker Request objects. String paths may contain {{repeat}}.
    #[arg(long, value_name = "PATH")]
    request_file: PathBuf,

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
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum BackendArg {
    Auto,
    Cpu,
    Wgpu,
    Cuda,
}

impl BackendArg {
    fn as_value(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Cpu => "cpu",
            Self::Wgpu => "wgpu",
            Self::Cuda => "cuda",
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
    let config = BenchmarkConfig {
        request_file: arguments.request_file,
        repeats: arguments.repeats,
        timeout: Duration::from_secs_f64(arguments.timeout_secs),
        label: arguments.label,
    };

    match run_benchmark(config, &worker_config) {
        Ok(report) => {
            if arguments.json {
                match serde_json::to_string_pretty(&report) {
                    Ok(text) => println!("{text}"),
                    Err(error) => {
                        eprintln!("failed to serialize the benchmark report: {error}");
                        return ExitCode::FAILURE;
                    }
                }
            } else {
                println!("{}", render_report(&report));
            }
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("benchmark failed: {error}");
            ExitCode::FAILURE
        }
    }
}
