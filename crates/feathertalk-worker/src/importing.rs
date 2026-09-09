//! Import supported legacy checkpoints into standard model packages.

use std::{fmt, fs, path::Path};

use feathertalk_domain::{ImportLegacyModelParams, LegacyModelKind, Progress, TaskStage};
use feathertalk_export::{
    FeatherHubertPackageRequest, ModelDescription, PackageBuildRequest, SourceManifest,
    TrainingManifest, build_feather_hubert_package, write_model_package,
};
use feathertalk_media::CancellationToken;
use feathertalk_models::{
    backend::CpuBackend,
    unet::{OriginalUnet, OriginalUnetConfig},
};
use feathertalk_weights::{LegacyImportRequest, LegacyModelKind as WeightModelKind, import_into};
use time::{OffsetDateTime, format_description::well_known::Rfc3339};

use crate::{TaskReporter, WorkerConfig};

const SOURCE_FORMAT: &str = "pytorch-pickle-restricted";

#[derive(Debug)]
pub enum ImportLegacyModelError {
    Cancelled,
    Failed { detail: String, stage: TaskStage },
}

impl fmt::Display for ImportLegacyModelError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cancelled => formatter.write_str("legacy model import cancelled"),
            Self::Failed { detail, .. } => formatter.write_str(detail),
        }
    }
}

impl ImportLegacyModelError {
    pub fn stage(&self) -> TaskStage {
        match self {
            Self::Cancelled => TaskStage::Preparing,
            Self::Failed { stage, .. } => stage.clone(),
        }
    }

    pub fn is_cancelled(&self) -> bool {
        matches!(self, Self::Cancelled)
    }
}

pub fn execute_import_legacy_model(
    params: &ImportLegacyModelParams,
    config: &WorkerConfig,
    token: &CancellationToken,
    reporter: &dyn TaskReporter,
) -> Result<serde_json::Value, ImportLegacyModelError> {
    validate_request(params)?;
    if token.is_cancelled() {
        return Err(ImportLegacyModelError::Cancelled);
    }
    reporter.report(
        TaskStage::Importing,
        Some(Progress {
            completed: 0,
            total: Some(1),
        }),
    );
    let created_at = OffsetDateTime::now_utc()
        .format(&Rfc3339)
        .map_err(|error| failure(TaskStage::Preparing, error.to_string()))?;
    let licenses = params
        .source
        .parent()
        .ok_or_else(|| failure(TaskStage::Preparing, "source parent is unavailable"))?
        .join(feathertalk_export::LICENSE_FILE_NAME);
    let result = match params.kind {
        LegacyModelKind::FeatherHubert => {
            import_feather_hubert(params, &licenses, &created_at, config, token)
        }
        LegacyModelKind::OriginalUnet => {
            import_original_unet(params, &licenses, &created_at, config, token)
        }
        LegacyModelKind::Pfld | LegacyModelKind::MobileOneUnet => Err(failure(
            TaskStage::Preparing,
            "legacy model kind is not supported by the standard package writer",
        )),
    }?;
    if token.is_cancelled() {
        return Err(ImportLegacyModelError::Cancelled);
    }
    reporter.report(
        TaskStage::Importing,
        Some(Progress {
            completed: 1,
            total: Some(1),
        }),
    );
    Ok(result)
}

fn validate_request(params: &ImportLegacyModelParams) -> Result<(), ImportLegacyModelError> {
    if matches!(
        params.kind,
        LegacyModelKind::Pfld | LegacyModelKind::MobileOneUnet
    ) {
        return Err(failure(
            TaskStage::Preparing,
            "legacy model kind is not supported by the standard package writer",
        ));
    }
    let source = &params.source;
    if !source.is_absolute() {
        return Err(failure(
            TaskStage::Preparing,
            "source path must be absolute",
        ));
    }
    let metadata = fs::symlink_metadata(source)
        .map_err(|error| failure(TaskStage::Preparing, error.to_string()))?;
    if metadata.file_type().is_symlink() || !metadata.file_type().is_file() {
        return Err(failure(
            TaskStage::Preparing,
            "source must be a regular non-symlink file",
        ));
    }
    let file_name = source
        .file_name()
        .and_then(|name| name.to_str())
        .filter(|name| !name.is_empty())
        .ok_or_else(|| failure(TaskStage::Preparing, "source file name must be valid UTF-8"))?;
    if !file_name.ends_with(".pth") && !file_name.ends_with(".pth.tar") {
        return Err(failure(
            TaskStage::Preparing,
            "legacy model source must end in .pth or .pth.tar",
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
        .filter(|path| !path.as_os_str().is_empty())
        .ok_or_else(|| failure(TaskStage::Preparing, "destination parent is unavailable"))?;
    let parent_metadata = fs::symlink_metadata(parent)
        .map_err(|error| failure(TaskStage::Preparing, error.to_string()))?;
    if parent_metadata.file_type().is_symlink() || !parent_metadata.file_type().is_dir() {
        return Err(failure(
            TaskStage::Preparing,
            "destination parent must be a real directory",
        ));
    }
    Ok(())
}

fn import_feather_hubert(
    params: &ImportLegacyModelParams,
    licenses: &Path,
    created_at: &str,
    config: &WorkerConfig,
    token: &CancellationToken,
) -> Result<serde_json::Value, ImportLegacyModelError> {
    if token.is_cancelled() {
        return Err(ImportLegacyModelError::Cancelled);
    }
    let report = build_feather_hubert_package(&FeatherHubertPackageRequest {
        source: params.source.clone(),
        licenses: licenses.to_owned(),
        destination: params.destination.clone(),
        created_at: created_at.to_owned(),
        minimum_app_version: config.worker_version().to_owned(),
    })
    .map_err(|error| failure(TaskStage::Importing, error.to_string()))?;
    Ok(report_json(&ImportReportData {
        kind: LegacyModelKind::FeatherHubert,
        architecture_version: &report.manifest.architecture_version,
        source: &params.source,
        destination: &params.destination,
        source_sha256: &report.manifest.source.sha256,
        model_sha256: &report.manifest.model.sha256,
        tensor_count: report.manifest.tensors.tensor_count,
        total_elements: report.manifest.tensors.total_elements,
    }))
}

fn import_original_unet(
    params: &ImportLegacyModelParams,
    licenses: &Path,
    created_at: &str,
    config: &WorkerConfig,
    token: &CancellationToken,
) -> Result<serde_json::Value, ImportLegacyModelError> {
    let device = Default::default();
    let model_config = OriginalUnetConfig::production();
    let mut model = model_config.clone().init::<CpuBackend>(&device);
    let import = import_into::<CpuBackend, OriginalUnet<CpuBackend>>(
        &mut model,
        &LegacyImportRequest {
            path: params.source.clone(),
            kind: WeightModelKind::OriginalUnet,
            ..Default::default()
        },
    )
    .map_err(|error| failure(TaskStage::Importing, error.to_string()))?;
    if token.is_cancelled() {
        return Err(ImportLegacyModelError::Cancelled);
    }
    let file_name = source_file_name(&params.source)?;
    let source_version = legacy_version(&file_name)?;
    let package = write_model_package::<CpuBackend, _, _>(
        &PackageBuildRequest {
            destination: params.destination.clone(),
            description: ModelDescription::original_unet(model_config.clone()),
            source_path: params.source.clone(),
            source: SourceManifest {
                format: SOURCE_FORMAT.to_owned(),
                identifier: "feathertalk-original-unet".to_owned(),
                version: source_version,
                file_name,
                sha256: import.source_sha256,
                url: None,
            },
            licenses_path: licenses.to_owned(),
            created_at: created_at.to_owned(),
            minimum_app_version: config.worker_version().to_owned(),
            training: TrainingManifest::default(),
        },
        &model,
        &device,
        move |device| model_config.clone().init::<CpuBackend>(device),
    )
    .map_err(|error| failure(TaskStage::Importing, error.to_string()))?;
    Ok(report_json(&ImportReportData {
        kind: LegacyModelKind::OriginalUnet,
        architecture_version: &package.manifest.architecture_version,
        source: &params.source,
        destination: &params.destination,
        source_sha256: &package.manifest.source.sha256,
        model_sha256: &package.manifest.model.sha256,
        tensor_count: package.manifest.tensors.tensor_count,
        total_elements: package.manifest.tensors.total_elements,
    }))
}

struct ImportReportData<'a> {
    kind: LegacyModelKind,
    architecture_version: &'a str,
    source: &'a Path,
    destination: &'a Path,
    source_sha256: &'a str,
    model_sha256: &'a str,
    tensor_count: usize,
    total_elements: u64,
}

fn report_json(report: &ImportReportData<'_>) -> serde_json::Value {
    let model_kind = match report.kind {
        LegacyModelKind::FeatherHubert => "feather_hubert",
        LegacyModelKind::OriginalUnet => "original_unet",
        LegacyModelKind::Pfld => "pfld",
        LegacyModelKind::MobileOneUnet => "mobileone_unet",
    };
    serde_json::json!({
        "kind": "import_legacy_model",
        "model_kind": model_kind,
        "architecture_version": report.architecture_version,
        "source": report.source,
        "destination": report.destination,
        "source_sha256": report.source_sha256,
        "model_sha256": report.model_sha256,
        "tensor_count": report.tensor_count,
        "total_elements": report.total_elements,
    })
}

fn source_file_name(source: &Path) -> Result<String, ImportLegacyModelError> {
    source
        .file_name()
        .and_then(|value| value.to_str())
        .filter(|value| !value.is_empty())
        .map(str::to_owned)
        .ok_or_else(|| failure(TaskStage::Preparing, "source file name must be valid UTF-8"))
}

fn legacy_version(file_name: &str) -> Result<String, ImportLegacyModelError> {
    file_name
        .strip_suffix(".pth.tar")
        .or_else(|| file_name.strip_suffix(".pth"))
        .filter(|value| !value.is_empty())
        .map(str::to_owned)
        .ok_or_else(|| {
            failure(
                TaskStage::Preparing,
                "legacy source must have a non-empty version stem",
            )
        })
}

fn failure(stage: TaskStage, detail: impl Into<String>) -> ImportLegacyModelError {
    ImportLegacyModelError::Failed {
        detail: detail.into(),
        stage,
    }
}
