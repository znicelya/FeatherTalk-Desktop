//! Publish a standard model package from a training checkpoint.

use std::{
    fmt, fs,
    path::{Path, PathBuf},
};

use burn::module::AutodiffModule;
use feathertalk_domain::{ExportModelPackageParams, Progress, TaskStage};
use feathertalk_export::{
    LICENSE_FILE_NAME, ModelDescription, ModelPackageManifest, PackageBuildRequest, SourceManifest,
    TrainingManifest, TrainingMode as PackageTrainingMode, write_model_package,
};
use feathertalk_media::CancellationToken;
use feathertalk_training::{
    CHECKPOINT_MODEL_FILE_NAME, TrainingCheckpointMetadata, TrainingConfig,
    TrainingMode as CheckpointTrainingMode, load_training_checkpoint_model,
    read_training_checkpoint,
};

use crate::inspect_result::package_mode_slug;
use crate::{
    ModelSourceKind, RenderBackend, RenderDevice, RenderVariant, TaskReporter, TrainBackend,
    TrainDevice, WorkerConfig, checkpoint_descriptor, model_source_kind, render_variant,
};

/// What the published manifest calls the thing it was made from. A checkpoint is
/// not a vendor artifact, so it gets its own format name rather than borrowing
/// the legacy importer's `pytorch-pickle-restricted`.
const SOURCE_FORMAT: &str = "feathertalk-training-checkpoint";

#[derive(Debug)]
pub enum ExportModelPackageError {
    Cancelled { stage: TaskStage },
    Failed { detail: String, stage: TaskStage },
}

impl fmt::Display for ExportModelPackageError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cancelled { .. } => formatter.write_str("model package export cancelled"),
            Self::Failed { detail, .. } => formatter.write_str(detail),
        }
    }
}

impl ExportModelPackageError {
    pub fn stage(&self) -> TaskStage {
        match self {
            Self::Cancelled { stage } | Self::Failed { stage, .. } => stage.clone(),
        }
    }

    pub fn is_cancelled(&self) -> bool {
        matches!(self, Self::Cancelled { .. })
    }
}

/// Everything the publication needs once the architecture is known.
///
/// The plan is derived from the checkpoint's own manifest and state, so the
/// published package can be traced back to the training point that produced it
/// without the request carrying any of it.
#[derive(Debug, Clone)]
pub struct ExportPlan {
    pub source: PathBuf,
    pub destination: PathBuf,
    pub licenses: PathBuf,
    pub created_at: String,
    pub minimum_app_version: String,
    pub source_manifest: SourceManifest,
    pub training: TrainingManifest,
    pub model_kind: String,
    pub epoch: u64,
    pub global_step: u64,
}

/// Publishes the package a training checkpoint describes.
///
/// The architecture is resolved from the kind the checkpoint recorded, which is
/// why only the two production UNet configurations can be exported: the
/// configuration digest in the checkpoint descriptor is the gate the record is
/// applied under, and a guessed configuration would pour weights into the wrong
/// shapes.
pub fn execute_export_model_package(
    params: &ExportModelPackageParams,
    config: &WorkerConfig,
    token: &CancellationToken,
    reporter: &dyn TaskReporter,
) -> Result<serde_json::Value, ExportModelPackageError> {
    validate_request(params)?;
    reporter.report(TaskStage::Preparing, None);
    if token.is_cancelled() {
        return Err(cancelled(TaskStage::Preparing));
    }
    let metadata = read_training_checkpoint(&params.source)
        .map_err(|error| failure(TaskStage::Preparing, error.to_string()))?;
    let variant = render_variant(&metadata.manifest.model_kind).ok_or_else(|| {
        failure(
            TaskStage::Preparing,
            format!(
                "checkpoint model kind {} has no exportable architecture",
                metadata.manifest.model_kind
            ),
        )
    })?;
    let plan = export_plan(params, &metadata, config.worker_version())?;
    publish_checkpoint_package(&plan, &variant, token, reporter)
}

/// Derives the plan from an admitted request and the checkpoint it names.
pub fn export_plan(
    params: &ExportModelPackageParams,
    metadata: &TrainingCheckpointMetadata,
    minimum_app_version: &str,
) -> Result<ExportPlan, ExportModelPackageError> {
    let created_at = time::OffsetDateTime::now_utc()
        .format(&time::format_description::well_known::Rfc3339)
        .map_err(|error| failure(TaskStage::Preparing, error.to_string()))?;
    let licenses = licenses_path(&params.source)?;
    let epoch = metadata.state.epoch;
    let global_step = metadata.state.global_step;
    Ok(ExportPlan {
        source: params.source.clone(),
        destination: params.destination.clone(),
        licenses,
        created_at,
        minimum_app_version: minimum_app_version.to_owned(),
        source_manifest: SourceManifest {
            format: SOURCE_FORMAT.to_owned(),
            identifier: metadata.manifest.model_kind.clone(),
            // The training point, not the architecture: the architecture is a
            // manifest field of its own.
            version: format!("epoch-{epoch}-step-{global_step}"),
            file_name: CHECKPOINT_MODEL_FILE_NAME.to_owned(),
            sha256: metadata.manifest.model.sha256.clone(),
            url: None,
        },
        training: training_manifest(&metadata.state.training_config),
        model_kind: metadata.manifest.model_kind.clone(),
        epoch,
        global_step,
    })
}

/// Restores the record, fuses MobileOne, and publishes the package.
///
/// The variant is a parameter rather than a lookup, the way `run_render` takes
/// the model the command resolved: it keeps the whole publication reachable with
/// a micro configuration instead of a production-sized checkpoint.
pub fn publish_checkpoint_package(
    plan: &ExportPlan,
    variant: &RenderVariant,
    token: &CancellationToken,
    reporter: &dyn TaskReporter,
) -> Result<serde_json::Value, ExportModelPackageError> {
    reporter.report(
        TaskStage::Exporting,
        Some(Progress {
            completed: 0,
            total: Some(1),
        }),
    );
    if token.is_cancelled() {
        return Err(cancelled(TaskStage::Exporting));
    }
    // The descriptor describes the checkpoint, so MobileOne is named in its
    // unfused training shape here even though the package publishes the fused
    // graph below.
    let descriptor = checkpoint_descriptor(&variant.configuration())
        .map_err(|error| failure(TaskStage::Exporting, error.to_string()))?;
    let load_device = TrainDevice::default();
    let device = RenderDevice::default();
    let manifest = match variant {
        RenderVariant::OriginalUnet(configuration) => {
            let template = configuration.init::<TrainBackend>(&load_device);
            let restored = load_training_checkpoint_model::<TrainBackend, _>(
                &plan.source,
                &template,
                &load_device,
                &descriptor,
            )
            .map_err(|error| failure(TaskStage::Exporting, error.to_string()))?;
            if token.is_cancelled() {
                return Err(cancelled(TaskStage::Exporting));
            }
            // The record was written by a module on the autodiff backend; the
            // shell is dropped here because a package carries weights only.
            let model = restored.model.valid();
            let factory = configuration.clone();
            write_model_package::<RenderBackend, _, _>(
                &build_request(plan, ModelDescription::original_unet(configuration.clone())),
                &model,
                &device,
                move |device| factory.init::<RenderBackend>(device),
            )
        }
        RenderVariant::MobileOneUnet(configuration) => {
            let template = configuration.init::<TrainBackend>(&load_device);
            let restored = load_training_checkpoint_model::<TrainBackend, _>(
                &plan.source,
                &template,
                &load_device,
                &descriptor,
            )
            .map_err(|error| failure(TaskStage::Exporting, error.to_string()))?;
            if token.is_cancelled() {
                return Err(cancelled(TaskStage::Exporting));
            }
            // A package is an inference artifact, and both the renderer and the
            // ONNX exporter fuse the multi-branch blocks before use, so the
            // fused graph is what gets published.
            let model = restored.model.valid().reparameterize();
            let factory = configuration.clone();
            write_model_package::<RenderBackend, _, _>(
                &build_request(
                    plan,
                    ModelDescription::mobileone_unet(configuration.clone(), true),
                ),
                &model,
                &device,
                move |device| factory.init::<RenderBackend>(device).reparameterize(),
            )
        }
    }
    .map_err(|error| failure(TaskStage::Exporting, error.to_string()))?
    .manifest;
    reporter.report(
        TaskStage::Exporting,
        Some(Progress {
            completed: 1,
            total: Some(1),
        }),
    );
    Ok(report_json(plan, &manifest))
}

/// What has to hold before a checkpoint is read: an absolute checkpoint
/// directory, a license bundle beside it, and a free destination.
fn validate_request(params: &ExportModelPackageParams) -> Result<(), ExportModelPackageError> {
    let kind = model_source_kind(&params.source)
        .map_err(|error| failure(TaskStage::Preparing, error.detail))?;
    if kind != ModelSourceKind::TrainingCheckpoint {
        return Err(failure(
            TaskStage::Preparing,
            format!(
                "export requires a training checkpoint, not a {}",
                kind.as_slug()
            ),
        ));
    }
    // A checkpoint directory is allowed exactly four entries, so the bundle
    // cannot live inside it; beside it is where the legacy importer looks too.
    let licenses = licenses_path(&params.source)?;
    let metadata = fs::symlink_metadata(&licenses).map_err(|error| {
        failure(
            TaskStage::Preparing,
            format!("{}: {error}", licenses.display()),
        )
    })?;
    if metadata.file_type().is_symlink() || !metadata.file_type().is_file() {
        return Err(failure(
            TaskStage::Preparing,
            format!("{} must be a regular non-symlink file", licenses.display()),
        ));
    }
    if !params.destination.is_absolute() {
        return Err(failure(
            TaskStage::Preparing,
            "destination path must be absolute",
        ));
    }
    match fs::symlink_metadata(&params.destination) {
        Ok(_) => {
            return Err(failure(
                TaskStage::Preparing,
                "destination must not already exist",
            ));
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(failure(TaskStage::Preparing, error.to_string())),
    }
    let parent = params
        .destination
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .ok_or_else(|| failure(TaskStage::Preparing, "destination parent is unavailable"))?;
    let parent_metadata = fs::symlink_metadata(parent)
        .map_err(|error| failure(TaskStage::Preparing, error.to_string()))?;
    if parent_metadata.file_type().is_symlink() || !parent_metadata.is_dir() {
        return Err(failure(
            TaskStage::Preparing,
            "destination parent must be an existing non-symlink directory",
        ));
    }
    Ok(())
}

/// The bundle beside the checkpoint directory.
fn licenses_path(source: &Path) -> Result<PathBuf, ExportModelPackageError> {
    source
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .map(|parent| parent.join(LICENSE_FILE_NAME))
        .ok_or_else(|| {
            failure(
                TaskStage::Preparing,
                "checkpoint parent directory is unavailable",
            )
        })
}

/// The recipe the weights were trained under, as the package manifest records
/// it. A checkpoint has no `inference` mode, so the mapping is total.
fn training_manifest(config: &TrainingConfig) -> TrainingManifest {
    TrainingManifest {
        mode: match config.mode {
            CheckpointTrainingMode::Baseline => PackageTrainingMode::Baseline,
            CheckpointTrainingMode::MouthRoi => PackageTrainingMode::MouthRoi,
            CheckpointTrainingMode::MouthRoiTemporal => PackageTrainingMode::MouthRoiTemporal,
        },
        mouth_weight: config.mouth_weight,
        temporal_weight: config.temporal_weight,
        temporal_mouth_weight: config.temporal_mouth_weight,
        perceptual_weight: config.perceptual_weight,
    }
}

fn build_request(plan: &ExportPlan, description: ModelDescription) -> PackageBuildRequest {
    PackageBuildRequest {
        destination: plan.destination.clone(),
        description,
        // The writer re-hashes this file, so a checkpoint that changed under the
        // export fails instead of publishing.
        source_path: plan.source.join(CHECKPOINT_MODEL_FILE_NAME),
        source: plan.source_manifest.clone(),
        licenses_path: plan.licenses.clone(),
        created_at: plan.created_at.clone(),
        minimum_app_version: plan.minimum_app_version.clone(),
        training: plan.training.clone(),
    }
}

fn report_json(plan: &ExportPlan, manifest: &ModelPackageManifest) -> serde_json::Value {
    serde_json::json!({
        "kind": "export_model_package",
        "model_kind": manifest.model_type,
        "architecture_version": manifest.architecture_version,
        "source": plan.source,
        "destination": plan.destination,
        "epoch": plan.epoch,
        "global_step": plan.global_step,
        "training_mode": package_mode_slug(manifest.training.mode),
        "source_sha256": manifest.source.sha256,
        "model_sha256": manifest.model.sha256,
        "tensor_count": manifest.tensors.tensor_count,
        "total_elements": manifest.tensors.total_elements,
    })
}

fn cancelled(stage: TaskStage) -> ExportModelPackageError {
    ExportModelPackageError::Cancelled { stage }
}

fn failure(stage: TaskStage, detail: impl Into<String>) -> ExportModelPackageError {
    ExportModelPackageError::Failed {
        detail: detail.into(),
        stage,
    }
}
