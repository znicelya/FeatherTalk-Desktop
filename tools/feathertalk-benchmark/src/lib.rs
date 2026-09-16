use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use feathertalk_domain::{
    ExtractFeaturesParams, ExtractFramesParams, NormalizeMediaParams, ProbeMediaParams,
    ProjectDirParams, RenderParams, Request, TrainParams, TrainingMode, UnetVariant,
};
use feathertalk_media::CancellationToken;
use feathertalk_project::{ModelSelection, ProjectManifest, write_project_manifest_atomic};
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

#[derive(Debug, Clone)]
pub struct PipelineConfig {
    pub input: PathBuf,
    pub project_root: PathBuf,
    pub repeats: usize,
    pub timeout: Duration,
    pub label: String,
    pub epochs: u32,
    pub max_output_frames: Option<u64>,
    pub keep_projects: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct BenchmarkReport {
    pub label: String,
    pub request_file: PathBuf,
    pub repeats: usize,
    pub cases: Vec<BenchmarkCase>,
}

#[derive(Debug, Clone, Serialize)]
pub struct PipelineReport {
    pub label: String,
    pub input: PathBuf,
    pub project_root: PathBuf,
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
    #[error("full pipeline epochs must be greater than zero")]
    InvalidEpochs,
    #[error("failed to prepare full pipeline project {path}: {source}")]
    PipelineProject {
        path: PathBuf,
        #[source]
        source: feathertalk_project::ProjectError,
    },
    #[error("train did not report a checkpoint directory for {project}")]
    MissingCheckpoint { project: PathBuf },
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

pub fn run_full_pipeline(
    config: PipelineConfig,
    worker_config: &WorkerConfig,
) -> Result<PipelineReport, BenchmarkError> {
    run_full_pipeline_with_executor(config, worker_config, Arc::new(execute))
}

pub fn run_full_pipeline_with_executor(
    config: PipelineConfig,
    worker_config: &WorkerConfig,
    executor: BenchmarkExecutor,
) -> Result<PipelineReport, BenchmarkError> {
    if config.epochs == 0 {
        return Err(BenchmarkError::InvalidEpochs);
    }

    let input = std::fs::canonicalize(&config.input)?;
    std::fs::create_dir_all(&config.project_root)?;
    let project_root = std::fs::canonicalize(&config.project_root)?;
    let mut runs = Vec::with_capacity(config.repeats.saturating_add(1));

    for repeat in 0..=config.repeats {
        runs.push(run_pipeline_once(
            &input,
            &project_root,
            repeat,
            &config,
            worker_config,
            &executor,
        )?);
    }

    Ok(PipelineReport {
        label: config.label,
        input,
        project_root,
        repeats: config.repeats,
        cases: pipeline_cases(runs)?,
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

pub fn render_pipeline_report(report: &PipelineReport) -> String {
    let mut lines = vec![
        "FeatherTalk full pipeline benchmark".to_owned(),
        format!("label: {}", report.label),
        format!("input: {}", report.input.display()),
        format!("project root: {}", report.project_root.display()),
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

fn run_pipeline_once(
    input: &Path,
    project_root: &Path,
    repeat: usize,
    config: &PipelineConfig,
    worker_config: &WorkerConfig,
    executor: &BenchmarkExecutor,
) -> Result<Vec<(String, f64)>, BenchmarkError> {
    let project_dir = prepare_pipeline_project(project_root, repeat)?;
    let result = run_pipeline_commands(input, &project_dir, config, worker_config, executor);
    if result.is_ok() && !config.keep_projects {
        let _ = std::fs::remove_dir_all(&project_dir);
    }
    result
}

fn run_pipeline_commands(
    input: &Path,
    project_dir: &Path,
    config: &PipelineConfig,
    worker_config: &WorkerConfig,
    executor: &BenchmarkExecutor,
) -> Result<Vec<(String, f64)>, BenchmarkError> {
    let mut samples = Vec::with_capacity(7);

    record_pipeline_sample(
        &mut samples,
        &Request::ProbeMedia(ProbeMediaParams {
            input: input.to_path_buf(),
        }),
        worker_config,
        executor,
        config.timeout,
    )?;

    let normalized_video = project_dir.join("assets/video_25fps.mp4");
    let normalized_audio = project_dir.join("assets/audio_16k_mono.wav");
    record_pipeline_sample(
        &mut samples,
        &Request::NormalizeMedia(NormalizeMediaParams {
            input: input.to_path_buf(),
            output_dir: project_dir.join("assets"),
        }),
        worker_config,
        executor,
        config.timeout,
    )?;
    record_pipeline_sample(
        &mut samples,
        &Request::ExtractFrames(ExtractFramesParams {
            project_dir: project_dir.to_path_buf(),
            video: normalized_video,
        }),
        worker_config,
        executor,
        config.timeout,
    )?;
    record_pipeline_sample(
        &mut samples,
        &Request::ExtractFeatures(ExtractFeaturesParams {
            project_dir: project_dir.to_path_buf(),
            audio: normalized_audio.clone(),
        }),
        worker_config,
        executor,
        config.timeout,
    )?;
    record_pipeline_sample(
        &mut samples,
        &Request::LockAssetPackage(ProjectDirParams {
            project_dir: project_dir.to_path_buf(),
        }),
        worker_config,
        executor,
        config.timeout,
    )?;

    let train = Request::Train(TrainParams {
        project_dir: project_dir.to_path_buf(),
        mode: TrainingMode::Baseline,
        variant: UnetVariant::OriginalUnet,
        epochs: config.epochs,
        batch_size: 1,
        resume: false,
    });
    let command = train.kind().as_slug().to_owned();
    let (seconds, payload) =
        run_sample_with_result(&train, worker_config, executor, config.timeout, &command)?;
    samples.push((command, seconds));
    let checkpoint = payload
        .as_ref()
        .and_then(|value| value.get("checkpoint_dir"))
        .and_then(serde_json::Value::as_str)
        .map(PathBuf::from)
        .ok_or_else(|| BenchmarkError::MissingCheckpoint {
            project: project_dir.to_path_buf(),
        })?;

    record_pipeline_sample(
        &mut samples,
        &Request::Render(RenderParams {
            project_dir: project_dir.to_path_buf(),
            checkpoint,
            audio: normalized_audio,
            output: project_dir.join("outputs/pipeline.mp4"),
            max_output_frames: config.max_output_frames,
        }),
        worker_config,
        executor,
        config.timeout,
    )?;

    Ok(samples)
}

fn record_pipeline_sample(
    samples: &mut Vec<(String, f64)>,
    request: &Request,
    worker_config: &WorkerConfig,
    executor: &BenchmarkExecutor,
    timeout: Duration,
) -> Result<(), BenchmarkError> {
    let command = request.kind().as_slug().to_owned();
    let seconds = run_sample(request, worker_config, executor, timeout, &command)?;
    samples.push((command, seconds));
    Ok(())
}

fn prepare_pipeline_project(project_root: &Path, repeat: usize) -> Result<PathBuf, BenchmarkError> {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let project_dir = project_root.join(format!("run-{}-{nonce}-{repeat}", std::process::id()));
    std::fs::create_dir(&project_dir)?;
    std::fs::create_dir_all(project_dir.join("assets/frames"))?;
    std::fs::create_dir_all(project_dir.join("assets/landmarks"))?;
    std::fs::create_dir_all(project_dir.join("assets/features"))?;
    std::fs::create_dir_all(project_dir.join("models"))?;
    std::fs::create_dir_all(project_dir.join("outputs"))?;

    let manifest = ProjectManifest {
        schema_version: 1,
        project_id: "benchmark".to_owned(),
        display_name: "FeatherTalk full pipeline benchmark".to_owned(),
        asset_package: "assets/assets.json".to_owned(),
        default_model: ModelSelection::OriginalUnet,
        task_history: Vec::new(),
    };
    write_project_manifest_atomic(&project_dir.join("project.json"), &manifest).map_err(
        |source| BenchmarkError::PipelineProject {
            path: project_dir.clone(),
            source,
        },
    )?;
    std::fs::canonicalize(&project_dir).map_err(Into::into)
}

fn pipeline_cases(runs: Vec<Vec<(String, f64)>>) -> Result<Vec<BenchmarkCase>, BenchmarkError> {
    let Some(first_run) = runs.first() else {
        return Ok(Vec::new());
    };
    let mut cases = Vec::with_capacity(first_run.len());

    for (index, (command, first_seconds)) in first_run.iter().enumerate() {
        let mut warm = Vec::with_capacity(runs.len().saturating_sub(1));
        for (repeat, run) in runs.iter().enumerate().skip(1) {
            let Some((warm_command, seconds)) = run.get(index) else {
                return Err(BenchmarkError::ChangedRequestOrder);
            };
            if warm_command != command {
                return Err(BenchmarkError::ChangedRequestOrder);
            }
            warm.push(BenchmarkSample {
                repeat,
                seconds: *seconds,
            });
        }
        let warm_seconds: Vec<_> = warm.iter().map(|sample| sample.seconds).collect();
        cases.push(BenchmarkCase {
            command: command.clone(),
            first: BenchmarkSample {
                repeat: 0,
                seconds: *first_seconds,
            },
            warm_median_seconds: median(&warm_seconds),
            warm_min_seconds: warm_seconds.iter().copied().reduce(f64::min),
            warm_max_seconds: warm_seconds.iter().copied().reduce(f64::max),
            warm,
        });
    }

    Ok(cases)
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
    run_sample_with_result(request, worker_config, executor, timeout, command)
        .map(|(seconds, _)| seconds)
}

fn run_sample_with_result(
    request: &Request,
    worker_config: &WorkerConfig,
    executor: &BenchmarkExecutor,
    timeout: Duration,
    command: &str,
) -> Result<(f64, Option<serde_json::Value>), BenchmarkError> {
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
        CommandOutcome::Completed(payload) => Ok((seconds.as_secs_f64(), payload)),
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
