//! The training loop and the command that drives it.

use std::path::PathBuf;
use std::time::Instant;

use burn::{
    module::AutodiffModule,
    optim::AdamConfig,
    tensor::Device,
};
use feathertalk_domain::{
    Metrics, Progress, TaskError, TaskStage, TrainParams, TrainingMode as DomainTrainingMode,
    UnetVariant};
use feathertalk_export::ModelConfiguration;
use feathertalk_media::CancellationToken;
use feathertalk_models::unet::{MobileOneUnetConfig, OriginalUnetConfig, TrainableTalkingHead};
use feathertalk_training::{
    CheckpointCompatibility, PerceptualFeatureExtractor, TrainingDataset, TrainingError,
    Vgg19Conv3_3, load_training_checkpoint, load_vgg19_package};
use feathertalk_training_data::{ProjectTrainingDataset, TrainingItem};
use feathertalk_training_run::{StepReport, TrainingRunner, build_preview_artifact};

/// Diagnostic-only: when the env var FEATHERTALK_TRAIN_PERCEPTUAL is off, the
/// perceptual branch is dropped from the autodiff graph so backward_secs
/// measures the UNet (depthwise conv) backward alone. Compare against a normal
/// run to see how much of backward is VGG19. Not a training knob; unset for real runs.
enum DiagnosticExtractor {
    Vgg19(Vgg19Conv3_3),
    Identity}

impl PerceptualFeatureExtractor for DiagnosticExtractor {
    fn forward(&self, image: burn::tensor::Tensor<4>) -> burn::tensor::Tensor<4> {
        match self {
            DiagnosticExtractor::Vgg19(inner) => inner.forward(image),
            DiagnosticExtractor::Identity => image}
    }
}

fn perceptual_enabled_from_env() -> bool {
    !matches!(
        std::env::var("FEATHERTALK_TRAIN_PERCEPTUAL")
            .unwrap_or_default()
            .trim()
            .to_ascii_lowercase()
            .as_str(),
        "0" | "off" | "false" | "no" | "identity"
    )
}

/// Reads the freeze experiment setting for the step profile meta record.
///
/// This only records what the operator requested via the environment; the
/// value is reported verbatim so an experiment log can distinguish its runs.
fn freeze_setting_from_env() -> String {
    match std::env::var("FEATHERTALK_TRAIN_FREEZE") {
        Ok(value) => {
            let trimmed = value.trim();
            if trimmed.is_empty() {
                "none".to_owned()
            } else {
                trimmed.to_ascii_lowercase()
            }
        }
        Err(_) => "none".to_owned()}
}

/// Whether the runner's data-loader prefetch is on, mirroring the same
/// environment variable the runner reads, so the meta record matches behaviour.
fn prefetch_enabled_from_env() -> bool {
    match std::env::var("FEATHERTALK_TRAIN_PREFETCH") {
        Ok(value) => !matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "0" | "false" | "off" | "no"
        ),
        Err(_) => true}
}

/// Appends per-step timing rows to `outputs/metrics/step-profile.jsonl`.
///
/// Enabled only when `FEATHERTALK_STEP_PROFILE` is set. The first line is a
/// `meta` record describing the run; every later line is one committed step.
/// Profiling is best-effort: a write failure disables further logging instead
/// of failing the training run.
struct StepProfileWriter {
    file: std::fs::File,
    healthy: bool}

impl StepProfileWriter {
    fn open(plan: &TrainingPlan, execution_name: &str) -> Option<Self> {
        if std::env::var_os("FEATHERTALK_STEP_PROFILE").is_none() {
            return None;
        }
        let path = plan.paths.step_profile();
        if let Some(parent) = path.parent()
            && let Err(error) = std::fs::create_dir_all(parent)
        {
            eprintln!("step profile disabled: create {}: {error}", parent.display());
            return None;
        }
        // Truncate: each experiment run owns a fresh profile, as the docs advise.
        let file = match std::fs::File::create(&path) {
            Ok(file) => file,
            Err(error) => {
                eprintln!("step profile disabled: open {}: {error}", path.display());
                return None;
            }
        };
        let mut writer = Self {
            file,
            healthy: true};
        let meta = serde_json::json!({
            "record": "meta",
            "backend": execution_name,
            "mode": plan.mode,
            "variant": plan.variant,
            "batch_size": plan.config.batch_size,
            "total_epochs": plan.config.total_epochs,
            "freeze": freeze_setting_from_env(),
            "prefetch": prefetch_enabled_from_env(),
            "perceptual": perceptual_enabled_from_env(),
            "resumed": plan.resume_from.is_some()});
        writer.write_line(&meta);
        Some(writer)
    }

    fn record(&mut self, report: &StepReport) {
        if !self.healthy {
            return;
        }
        let t = report.timings;
        let row = serde_json::json!({
            "record": "step",
            "epoch": report.epoch,
            "global_step": report.global_step,
            "samples_in_batch": report.samples_in_batch,
            "load_secs": t.load_secs,
            "prefetch_wait_secs": t.prefetch_wait_secs,
            "forward_secs": t.forward_secs,
            "backward_secs": t.backward_secs,
            "optim_secs": t.optim_secs,
            "total_secs": t.total_secs});
        self.write_line(&row);
    }

    fn write_line(&mut self, value: &serde_json::Value) {
        use std::io::Write;
        let mut line = value.to_string();
        line.push('\n');
        if let Err(error) = self
            .file
            .write_all(line.as_bytes())
            .and_then(|_| self.file.flush())
        {
            eprintln!("step profile disabled: write: {error}");
            self.healthy = false;
        }
    }
}
use crate::admission::{check_project_dir, invalid_request};
use crate::execution_backend::{execution_name, gpu_memory_bytes};
use crate::train_result::train_to_json_on;
use crate::{
    CommandOutcome, MAX_EPOCHS, TRAINING_SEED, TaskReporter,
    TrainSummary, TrainingPaths, TrainingPlan, TrainingToolchain, WORKER_STATE,
    checkpoint_descriptor, latest_checkpoint, preview_sample, publish_checkpoint, sample_count,
    training_config, training_data_task_error, training_task_error, write_metrics_unless_present,
    write_preview_unless_present};

/// Trains until the plan's epoch count is reached, the task is cancelled, or a
/// step fails.
///
/// Everything the loop needs arrives as a parameter, which is what lets the
/// tests drive it with a stub dataset and a constant extractor instead of a
/// locked project and half a gigabyte of VGG19 weights.
#[allow(clippy::too_many_arguments)]
pub fn run_training<M, O, D, E>(
    plan: &TrainingPlan,
    dataset: D,
    model: M,
    optimizer: O,
    extractor: &E,
    device: &Device,
    token: &CancellationToken,
    reporter: &dyn TaskReporter,
) -> CommandOutcome
where
    M: TrainableTalkingHead + AutodiffModule + Clone,
    O: feathertalk_training::CheckpointableOptimizer + Clone,
    D: TrainingDataset<Item = TrainingItem> + Send + Sync + 'static,
    E: PerceptualFeatureExtractor,
{
    let mut runner = match build_runner(plan, dataset, model, optimizer, device) {
        Ok(runner) => runner,
        Err(error) => {
            return CommandOutcome::Failed(training_task_error(&error, TaskStage::Preparing));
        }
    };

    // The clock starts here, not at admission: `restore` zeroes `samples_seen`,
    // and the throughput this feeds has to describe the run that is starting.
    let started = Instant::now();
    let total = total_steps(plan);
    let mut step_profile = StepProfileWriter::open(plan, execution_name(device));
    let mut published = Published::default();
    let mut stage = TaskStage::Preparing;
    let mut steps: u64 = 0;
    let mut total_loss = None;

    while !runner.is_finished() {
        if token.is_cancelled() {
            // With no step behind it, the checkpoint already on disk is the best
            // one there is and republishing would only copy it.
            if steps > 0
                && let Err(error) = publish(
                    &runner,
                    plan,
                    runner.global_step(),
                    &mut published,
                    reporter,
                    &stage,
                )
            {
                return CommandOutcome::Failed(error);
            }
            return CommandOutcome::Cancelled;
        }

        let report = match runner.step(extractor) {
            Ok(report) => report,
            Err(error) => return CommandOutcome::Failed(training_task_error(&error, stage))};
        steps = steps.saturating_add(1);
        if let Some(profile) = step_profile.as_mut() {
            profile.record(&report);
        }
        total_loss = Some(report.losses.total);
        stage = TaskStage::Training {
            epoch: u32::try_from(report.epoch).unwrap_or(u32::MAX),
            step: report.global_step,
            loss: report.losses.total};
        reporter.report_metrics(
            stage.clone(),
            Some(progress(report.global_step, total)),
            Metrics {
                vram_bytes: gpu_memory_bytes(device),
                ..Default::default()
            },
        );

        // `report.epoch` is the epoch the batch came from, `runner.epoch()` is
        // where the loader stands now, so they differ exactly once per epoch.
        if runner.epoch() > report.epoch {
            let closed = close_epoch(
                &runner,
                plan,
                &report,
                started,
                device,
                &mut published,
                reporter,
                &stage,
            );
            if let Err(error) = closed {
                return CommandOutcome::Failed(error);
            }
        }
    }

    // A resume whose requested epochs are already complete writes no artifact,
    // but loading its GPU checkpoint still needs a health boundary.
    if published.checkpoints == 0
        && let Err(error) = reporter.before_publish()
    {
        return CommandOutcome::Failed(error);
    }
    let summary = TrainSummary {
        mode: plan.mode,
        variant: plan.variant,
        descriptor: &plan.descriptor,
        frame_count: plan.frame_count,
        epochs_requested: plan.epochs_requested,
        epochs_completed: runner.epoch(),
        global_step: runner.global_step(),
        samples_seen: runner.samples_seen(),
        total_loss,
        resumed_from: plan.resume_from.as_deref(),
        checkpoint_dir: published.latest.as_deref(),
        checkpoints_written: published.checkpoints,
        metrics_written: published.metrics,
        previews_written: published.previews};
    CommandOutcome::Completed(Some(train_to_json_on(&summary, execution_name(device))))
}

/// What this run has put on disk.
#[derive(Debug, Default)]
struct Published {
    checkpoints: u64,
    metrics: u64,
    previews: u64,
    latest: Option<PathBuf>}

/// Starts a fresh run, or continues from the checkpoint the plan names.
fn build_runner<M, O, D>(
    plan: &TrainingPlan,
    dataset: D,
    model: M,
    optimizer: O,
    device: &Device,
) -> Result<TrainingRunner<M, O, D>, TrainingError>
where
    M: TrainableTalkingHead + AutodiffModule + Clone,
    O: feathertalk_training::CheckpointableOptimizer + Clone,
    D: TrainingDataset<Item = TrainingItem> + Send + Sync + 'static,
{
    let Some(directory) = plan.resume_from.as_deref() else {
        return TrainingRunner::new(
            dataset,
            model,
            optimizer,
            plan.config.clone(),
            TRAINING_SEED,
            device.clone(),
        );
    };

    // The model and the optimizer go in as templates: the loader reads the
    // records into their shapes and hands the restored pair back.
    let expected = CheckpointCompatibility::new(
        plan.descriptor.clone(),
        plan.config.clone(),
        plan.frame_count,
    );
    let restored =
        load_training_checkpoint::<M, O>(directory, &model, &optimizer, device, &expected)?;
    TrainingRunner::restore(dataset, restored, device.clone())
}

/// Publishes a checkpoint for `global_step` and remembers it as the newest one.
fn publish<M, O, D>(
    runner: &TrainingRunner<M, O, D>,
    plan: &TrainingPlan,
    global_step: u64,
    published: &mut Published,
    reporter: &dyn TaskReporter,
    stage: &TaskStage,
) -> Result<(), TaskError>
where
    M: TrainableTalkingHead + AutodiffModule + Clone,
    O: feathertalk_training::CheckpointableOptimizer + Clone,
    D: TrainingDataset<Item = TrainingItem> + Send + Sync + 'static,
{
    let mut health_failure = None;
    let checkpoint = publish_checkpoint(&plan.paths, global_step, |staged| {
        runner
            .save_checkpoint(staged, plan.descriptor.clone())
            .map(|_| ())?;
        // Saving reads model and optimizer tensors back from the GPU. Check
        // afterwards, while publish_checkpoint still owns only staging files.
        if let Err(error) = reporter.before_publish() {
            let detail = error.detail.clone();
            health_failure = Some(error);
            return Err(TrainingError::Store(detail));
        }
        Ok(())
    });
    if let Some(error) = health_failure {
        return Err(error);
    }
    let checkpoint = checkpoint.map_err(|error| training_task_error(&error, stage.clone()))?;
    published.checkpoints = published.checkpoints.saturating_add(1);
    published.latest = Some(checkpoint);
    Ok(())
}

/// What an epoch boundary owes: the checkpoint first, then the two diagnostics.
#[allow(clippy::too_many_arguments)]
fn close_epoch<M, O, D>(
    runner: &TrainingRunner<M, O, D>,
    plan: &TrainingPlan,
    report: &StepReport,
    started: Instant,
    device: &Device,
    published: &mut Published,
    reporter: &dyn TaskReporter,
    stage: &TaskStage,
) -> Result<(), TaskError>
where
    M: TrainableTalkingHead + AutodiffModule + Clone,
    O: feathertalk_training::CheckpointableOptimizer + Clone,
    D: TrainingDataset<Item = TrainingItem> + Send + Sync + 'static,
{
    let error_at_stage = |error: TrainingError| training_task_error(&error, stage.clone());
    publish(runner, plan, report.global_step, published, reporter, stage)?;

    let metrics = runner
        .metrics(
            report,
            started.elapsed(),
            gpu_memory_bytes(device),
            WORKER_STATE,
        )
        .map_err(error_at_stage)?;
    if write_metrics_unless_present(&plan.paths, report.global_step, &metrics)
        .map_err(error_at_stage)?
    {
        published.metrics = published.metrics.saturating_add(1);
    }

    let artifact = build_preview_artifact::<M, D>(
        runner.model().map_err(error_at_stage)?,
        runner.dataset(),
        device,
        &preview_sample(plan.frame_count),
        report.epoch,
        report.global_step,
        &plan.descriptor.model_kind,
        &plan.descriptor.model_config_sha256,
        WORKER_STATE,
    )
    .map_err(error_at_stage)?;
    reporter.before_publish()?;
    if write_preview_unless_present(&plan.paths, report.global_step, &artifact)
        .map_err(error_at_stage)?
    {
        published.previews = published.previews.saturating_add(1);
    }
    Ok(())
}

/// The number of steps the whole lineage will reach, or `None` when it does not
/// fit in a `u64` -- in which case progress reports what it has done and no
/// total, which is what `Progress.total` being optional is for.
fn total_steps(plan: &TrainingPlan) -> Option<u64> {
    let samples = sample_count(plan.mode, plan.frame_count);
    // `TrainingConfig::validate` rejects a zero batch size, but a divide here
    // must not be the place that finds out.
    let steps_per_epoch = samples.div_ceil(plan.config.batch_size.max(1));
    plan.config.total_epochs.checked_mul(steps_per_epoch)
}

/// A resumed run keeps the lineage's step numbers, so `completed` is clamped
/// rather than trusted: if the loader and the total ever disagree, a total that
/// is reached early beats a progress bar past one hundred percent.
fn progress(global_step: u64, total: Option<u64>) -> Progress {
    Progress {
        completed: match total {
            Some(total) => global_step.min(total),
            None => global_step},
        total}
}

/// Trains the U-Net of a locked project.
///
/// The toolchain arrives by reference instead of being read from the
/// environment here: `commands.rs` already holds the validated config, and a
/// command that reads its own environment cannot be tested.
pub fn execute_train(
    params: &TrainParams,
    token: &CancellationToken,
    reporter: &dyn TaskReporter,
    toolchain: &TrainingToolchain,
) -> CommandOutcome {
    execute_train_on(params, token, reporter, toolchain, &crate::cpu_training_device())
}

/// Runs training on the caller's already-selected tensor device.
pub fn execute_train_on(
    params: &TrainParams,
    token: &CancellationToken,
    reporter: &dyn TaskReporter,
    toolchain: &TrainingToolchain,
    device: &Device,
) -> CommandOutcome {
    // Admission reads the asset manifest, the whole feature file and then half a
    // gigabyte of VGG19 weights, so the stage goes out before any of it.
    reporter.report(TaskStage::Preparing, None);
    if let Err(error) = check_project_dir(&params.project_dir) {
        return CommandOutcome::Failed(error);
    }
    if let Err(error) = check_epochs(params.epochs) {
        return CommandOutcome::Failed(error);
    }
    if params.batch_size == 0 {
        return CommandOutcome::Failed(invalid_request(
            "批大小无效",
            "batch_size must be greater than zero".to_owned(),
        ));
    }

    // Autodiff is a device property in burn 0.22: enable it once here so the
    // model initializes with gradient tracking and the whole run shares one
    // device. Callers may pass either a plain or an already-autodiff device.
    let device = &crate::ensure_autodiff_device(device);
    // The variant is a type rather than a value, so each arm monomorphises the
    // whole run for one model. This is the only place that branches on it.
    match params.variant {
        UnetVariant::OriginalUnet => {
            let configuration = OriginalUnetConfig::production();
            let described = ModelConfiguration::original_unet(&configuration);
            start(
                params,
                token,
                reporter,
                toolchain,
                device,
                described,
                |device| configuration.init(device),
            )
        }
        UnetVariant::MobileOneUnet => {
            let configuration = MobileOneUnetConfig::production();
            // Not reparameterized: training needs the multi-branch graph, and
            // fusing the branches is an export-time step.
            let described = ModelConfiguration::mobileone_unet(&configuration, false);
            start(
                params,
                token,
                reporter,
                toolchain,
                device,
                described,
                |device| configuration.init(device),
            )
        }
    }
}

/// The rest of the command, once the model type is known.
fn start<M, F>(
    params: &TrainParams,
    token: &CancellationToken,
    reporter: &dyn TaskReporter,
    toolchain: &TrainingToolchain,
    device: &Device,
    configuration: ModelConfiguration,
    init: F,
) -> CommandOutcome
where
    M: TrainableTalkingHead + AutodiffModule + Clone,
    F: FnOnce(&Device) -> M,
{
    // Deliberately never named: writing out `ProjectTrainingDataset<JpegFrameReader>`
    // would pull `feathertalk-inference` into this crate's dependencies for the
    // sake of one type annotation.
    let dataset = match ProjectTrainingDataset::open(&params.project_dir) {
        Ok(dataset) => dataset,
        Err(error) => return CommandOutcome::Failed(training_data_task_error(&error))};
    let frame_count = dataset.frame_count();
    if let Err(error) = check_frame_count(params.mode, frame_count) {
        return CommandOutcome::Failed(error);
    }

    let paths = TrainingPaths::new(&params.project_dir);
    let found = match latest_checkpoint(&paths) {
        Ok(found) => found,
        Err(error) => return failed_preparing(&error)};
    if params.resume && found.is_none() {
        return CommandOutcome::Failed(invalid_request(
            "未找到可续训的检查点",
            format!("no checkpoint under {}", paths.checkpoints().display()),
        ));
    }
    let descriptor = match checkpoint_descriptor(&configuration) {
        Ok(descriptor) => descriptor,
        Err(error) => return failed_preparing(&error)};
    let plan = TrainingPlan {
        mode: params.mode,
        variant: params.variant,
        epochs_requested: params.epochs,
        frame_count,
        config: training_config(params),
        descriptor,
        paths,
        // Without `--resume`, a checkpoint on disk is not continued: the run
        // starts from fresh weights and republishes over the old names.
        resume_from: if params.resume { found } else { None }};

    let extractor = if perceptual_enabled_from_env() {
        match load_vgg19_package(toolchain.vgg19_dir(), device) {
            Ok(extractor) => DiagnosticExtractor::Vgg19(extractor),
            Err(error) => return failed_preparing(&error)}
    } else {
        DiagnosticExtractor::Identity
    };
    // Loading the weights took seconds; the caller may have given up meanwhile.
    if token.is_cancelled() {
        return CommandOutcome::Cancelled;
    }

    let optimizer = AdamConfig::new().init();
    run_training(
        &plan,
        dataset,
        init(device),
        optimizer,
        &extractor,
        device,
        token,
        reporter,
    )
}

/// `TrainingConfig::validate` also refuses zero, later and with a message about
/// a config the operator never wrote.
fn check_epochs(epochs: u32) -> Result<(), TaskError> {
    if epochs == 0 || epochs > MAX_EPOCHS {
        return Err(invalid_request(
            "训练轮数无效",
            format!("epochs must be between 1 and {MAX_EPOCHS}, got {epochs}"),
        ));
    }
    Ok(())
}

/// The temporal mode pairs each frame with its successor, so one frame yields no
/// samples at all. `DataLoaderConfig::validate` refuses it too, in terms of a
/// sample count rather than of frames.
pub fn check_frame_count(mode: DomainTrainingMode, frame_count: u64) -> Result<(), TaskError> {
    if matches!(mode, DomainTrainingMode::Temporal) && frame_count < 2 {
        return Err(invalid_request(
            "帧数不足，无法做时序训练",
            format!("temporal training needs at least 2 frames, got {frame_count}"),
        ));
    }
    Ok(())
}

/// Everything that goes wrong before the first step is a preparing failure.
fn failed_preparing(error: &TrainingError) -> CommandOutcome {
    CommandOutcome::Failed(training_task_error(error, TaskStage::Preparing))
}
