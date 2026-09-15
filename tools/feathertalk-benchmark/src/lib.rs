use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use feathertalk_domain::Request;
use feathertalk_media::CancellationToken;
use feathertalk_worker::{CommandOutcome, NoReporter, TaskReporter, WorkerConfig, execute};
use serde::Serialize;
use thiserror::Error;

pub type BenchmarkExecutor = Arc<
    dyn Fn(&Request, &WorkerConfig, &CancellationToken, &dyn TaskReporter) -> CommandOutcome
        + Send
        + Sync
        + 'static,
>;

#[derive(Debug, Clone)]
pub struct BenchmarkConfig {
    pub request_file: PathBuf,
    pub repeats: usize,
    pub timeout: Duration,
    pub label: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct BenchmarkReport {
    pub label: String,
    pub request_file: PathBuf,
    pub repeats: usize,
    pub cases: Vec<BenchmarkCase>,
}

#[derive(Debug, Clone, Serialize)]
pub struct BenchmarkCase {
    pub command: String,
    pub first: BenchmarkSample,
    pub warm: Vec<BenchmarkSample>,
    pub warm_median_seconds: Option<f64>,
    pub warm_min_seconds: Option<f64>,
    pub warm_max_seconds: Option<f64>,
}

#[derive(Debug, Clone, Copy, Serialize)]
pub struct BenchmarkSample {
    pub repeat: usize,
    pub seconds: f64,
}

#[derive(Debug, Error)]
pub enum BenchmarkError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("invalid request file {}: {source}", path.display())]
    RequestFile {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },
    #[error("the request file must contain at least one request")]
    EmptyRequestFile,
    #[error("the request file changed command order between repeats")]
    ChangedRequestOrder,
    #[error("{command} did not finish within {:.3}s", timeout.as_secs_f64())]
    TaskTimeout { command: String, timeout: Duration },
    #[error("{command} failed ({code}): {summary}")]
    TaskFailed {
        command: String,
        code: String,
        summary: String,
    },
    #[error("{command} was cancelled")]
    TaskCancelled { command: String },
    #[error("{command} panicked: {reason}")]
    CommandPanic { command: String, reason: String },
    #[error("the command thread failed: {0}")]
    CommandThread(String),
}

pub fn run_benchmark(
    config: BenchmarkConfig,
    worker_config: &WorkerConfig,
) -> Result<BenchmarkReport, BenchmarkError> {
    run_benchmark_with_executor(config, worker_config, Arc::new(execute))
}

pub fn run_benchmark_with_executor(
    config: BenchmarkConfig,
    worker_config: &WorkerConfig,
    executor: BenchmarkExecutor,
) -> Result<BenchmarkReport, BenchmarkError> {
    let initial_requests = load_requests(&config.request_file, 0)?;
    let mut cases = Vec::with_capacity(initial_requests.len());

    for (index, initial_request) in initial_requests.iter().enumerate() {
        let command = initial_request.kind().as_slug().to_owned();
        let first_seconds = run_sample(
            initial_request,
            worker_config,
            &executor,
            config.timeout,
            &command,
        )?;
        let mut warm = Vec::with_capacity(config.repeats);

        for repeat in 1..=config.repeats {
            let requests = load_requests(&config.request_file, repeat)?;
            let Some(request) = requests.get(index) else {
                return Err(BenchmarkError::EmptyRequestFile);
            };
            if request.kind() != initial_requests[index].kind() {
                return Err(BenchmarkError::ChangedRequestOrder);
            }
            let seconds = run_sample(request, worker_config, &executor, config.timeout, &command)?;
            warm.push(BenchmarkSample { repeat, seconds });
        }

        let warm_seconds: Vec<_> = warm.iter().map(|sample| sample.seconds).collect();
        cases.push(BenchmarkCase {
            command,
            first: BenchmarkSample {
                repeat: 0,
                seconds: first_seconds,
            },
            warm_median_seconds: median(&warm_seconds),
            warm_min_seconds: warm_seconds.iter().copied().reduce(f64::min),
            warm_max_seconds: warm_seconds.iter().copied().reduce(f64::max),
            warm,
        });
    }

    Ok(BenchmarkReport {
        label: config.label,
        request_file: config.request_file,
        repeats: config.repeats,
        cases,
    })
}

pub fn render_report(report: &BenchmarkReport) -> String {
    let mut lines = vec![
        "FeatherTalk worker command benchmark".to_owned(),
        format!("label: {}", report.label),
        format!("requests: {}", report.request_file.display()),
        format!("repeats: {}", report.repeats),
        String::new(),
        format!(
            "{:<24} {:>12} {:>16} {:>12} {:>12} {:>12}",
            "command", "first", "warm median", "warm min", "warm max", "warm n"
        ),
    ];
    for case in &report.cases {
        lines.push(format!(
            "{:<24} {:>12} {:>16} {:>12} {:>12} {:>12}",
            case.command,
            optional_seconds(Some(case.first.seconds)),
            optional_seconds(case.warm_median_seconds),
            optional_seconds(case.warm_min_seconds),
            optional_seconds(case.warm_max_seconds),
            case.warm.len()
        ));
    }
    lines.join("\n")
}

fn load_requests(path: &Path, repeat: usize) -> Result<Vec<Request>, BenchmarkError> {
    let template = std::fs::read_to_string(path)?;
    let text = template.replace("{{repeat}}", &repeat.to_string());
    let requests: Vec<Request> =
        serde_json::from_str(&text).map_err(|source| BenchmarkError::RequestFile {
            path: path.to_owned(),
            source,
        })?;
    if requests.is_empty() {
        return Err(BenchmarkError::EmptyRequestFile);
    }
    Ok(requests)
}

fn run_sample(
    request: &Request,
    worker_config: &WorkerConfig,
    executor: &BenchmarkExecutor,
    timeout: Duration,
    command: &str,
) -> Result<f64, BenchmarkError> {
    let token = CancellationToken::new();
    let command_token = token.clone();
    let request = request.clone();
    let worker_config = worker_config.clone();
    let executor = executor.clone();
    let command = command.to_owned();

    let handle = thread::spawn(move || {
        let started = Instant::now();
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            executor(&request, &worker_config, &command_token, &NoReporter)
        }))
        .map_err(|payload| panic_reason(payload.as_ref()));
        (started.elapsed(), outcome)
    });

    let started = Instant::now();
    while !handle.is_finished() {
        if started.elapsed() >= timeout {
            token.cancel();
            let (_, outcome) = join_command(handle, &command)?;
            let _ = outcome;
            return Err(BenchmarkError::TaskTimeout { command, timeout });
        }
        thread::sleep(Duration::from_millis(1));
    }

    let (seconds, outcome) = join_command(handle, &command)?;
    let outcome = outcome.map_err(|reason| BenchmarkError::CommandPanic {
        command: command.clone(),
        reason,
    })?;
    match outcome {
        CommandOutcome::Completed(_) => Ok(seconds.as_secs_f64()),
        CommandOutcome::Cancelled => Err(BenchmarkError::TaskCancelled { command }),
        CommandOutcome::Failed(error) => Err(BenchmarkError::TaskFailed {
            command,
            code: error.code.as_wire().to_owned(),
            summary: error.summary,
        }),
    }
}

fn join_command(
    handle: thread::JoinHandle<(Duration, Result<CommandOutcome, String>)>,
    command: &str,
) -> Result<(Duration, Result<CommandOutcome, String>), BenchmarkError> {
    handle
        .join()
        .map_err(|_| BenchmarkError::CommandThread(command.to_owned()))
}

fn panic_reason(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(text) = payload.downcast_ref::<&str>() {
        (*text).to_owned()
    } else if let Some(text) = payload.downcast_ref::<String>() {
        text.clone()
    } else {
        "unknown panic payload".to_owned()
    }
}

fn median(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let middle = sorted.len() / 2;
    if sorted.len() % 2 == 1 {
        Some(sorted[middle])
    } else {
        Some((sorted[middle - 1] + sorted[middle]) / 2.0)
    }
}

fn optional_seconds(value: Option<f64>) -> String {
    value.map_or_else(|| "-".to_owned(), |seconds| format!("{seconds:.3}s"))
}
