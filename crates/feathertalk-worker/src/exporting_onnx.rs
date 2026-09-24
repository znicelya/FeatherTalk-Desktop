//! Export a published model package as an opset 17 ONNX model.

use std::{fmt, fs, path::Path};

use feathertalk_domain::{ExportOnnxParams, OnnxExportKind, Progress, TaskStage};
use feathertalk_export::{
    ModelConfiguration, ModelDescription, ModelPackageManifest, OnnxArtifact,
    export_feather_hubert_onnx, export_mobileone_unet_onnx, export_original_unet_onnx,
    load_model_package, onnx::OnnxModelKind, publish_onnx_model, read_package_manifest,
};
use feathertalk_media::CancellationToken;
use feathertalk_models::{
    feather_hubert::FeatherHubertEncoder,
    unet::{
        MobileOneUnet, MobileOneUnetConfig, MobileOneUnetInference, OriginalUnet,
        OriginalUnetConfig,
    },
};

use crate::features::feather_hubert_config;
use crate::{ModelSourceKind, TaskReporter, model_source_kind};

#[derive(Debug)]
pub enum ExportOnnxError {
    Cancelled { stage: TaskStage },
    Failed { detail: String, stage: TaskStage },
}

impl fmt::Display for ExportOnnxError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cancelled { .. } => formatter.write_str("onnx export cancelled"),
            Self::Failed { detail, .. } => formatter.write_str(detail),
        }
    }
}

impl ExportOnnxError {
    pub fn stage(&self) -> TaskStage {
        match self {
            Self::Cancelled { stage } | Self::Failed { stage, .. } => stage.clone(),
        }
    }

    pub fn is_cancelled(&self) -> bool {
        matches!(self, Self::Cancelled { .. })
    }
}

/// Publishes the ONNX model a package describes.
///
/// The configuration comes from the manifest rather than from a production
/// literal: `load_model_package` re-hashes every file the manifest names and
/// validates a save/load round trip, so the record it returns is the one the
/// manifest describes. That is also what makes a micro package testable.
pub fn execute_export_onnx(
    params: &ExportOnnxParams,
    token: &CancellationToken,
    reporter: &dyn TaskReporter,
) -> Result<serde_json::Value, ExportOnnxError> {
    validate_request(params)?;
    reporter.report(TaskStage::Preparing, None);
    if token.is_cancelled() {
        return Err(cancelled(TaskStage::Preparing));
    }
    let manifest = read_package_manifest(&params.source)
        .map_err(|error| failure(TaskStage::Preparing, error.to_string()))?;
    let expected = model_type(params.kind);
    if manifest.model_type != expected {
        return Err(failure(
            TaskStage::Preparing,
            format!(
                "package holds {} weights, not the requested {expected}",
                manifest.model_type
            ),
        ));
    }
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
    let bytes = build_graph(&params.source, &manifest.configuration, params.kind)?;
    if token.is_cancelled() {
        return Err(cancelled(TaskStage::Exporting));
    }
    let artifact = publish_onnx_model(&params.destination, onnx_kind(params.kind), &bytes)
        .map_err(|error| failure(TaskStage::Exporting, error.to_string()))?;
    reporter.report(
        TaskStage::Exporting,
        Some(Progress {
            completed: 1,
            total: Some(1),
        }),
    );
    Ok(report_json(params, &manifest, &artifact))
}

/// Restores the record the package carries and serialises its inference graph.
fn build_graph(
    source: &Path,
    configuration: &ModelConfiguration,
    kind: OnnxExportKind,
) -> Result<Vec<u8>, ExportOnnxError> {
    let device = Default::default();
    match kind {
        OnnxExportKind::FeatherHubert => {
            let config = feather_hubert_config(configuration)
                .map_err(|error| failure(TaskStage::Exporting, error.to_string()))?;
            let (model, _) = load_model_package::<FeatherHubertEncoder, _>(
                source,
                &ModelDescription::feather_hubert(config.clone()),
                &device,
                |device| config.init(device),
            )
            .map_err(|error| failure(TaskStage::Exporting, error.to_string()))?;
            export_feather_hubert_onnx(&model, &config)
                .map_err(|error| failure(TaskStage::Exporting, error.to_string()))
        }
        OnnxExportKind::OriginalUnet => {
            let ModelConfiguration::OriginalUnet { channels } = configuration else {
                return Err(mismatch(configuration, kind));
            };
            let config = OriginalUnetConfig {
                channels: *channels,
            };
            let (model, _) = load_model_package::<OriginalUnet, _>(
                source,
                &ModelDescription::original_unet(config.clone()),
                &device,
                |device| config.init(device),
            )
            .map_err(|error| failure(TaskStage::Exporting, error.to_string()))?;
            export_original_unet_onnx(&model, &config)
                .map_err(|error| failure(TaskStage::Exporting, error.to_string()))
        }
        OnnxExportKind::MobileOneUnet => {
            let ModelConfiguration::MobileOneUnet {
                channels,
                num_conv_branches,
                reparameterized,
            } = configuration
            else {
                return Err(mismatch(configuration, kind));
            };
            let config = MobileOneUnetConfig {
                channels: *channels,
                num_conv_branches: *num_conv_branches,
            };
            // Migration design section 5.6 publishes the fused inference graph. A
            // package written by `export_model_package` is already fused; one that
            // still carries the training branches is fused here.
            //
            // No package written today takes the second path: a manifest names
            // every tensor, and the branched graph pushes that file past
            // `MAX_MANIFEST_BYTES`, so `write_model_package` refuses it. The arm
            // stays because the flag is part of the manifest schema, and a
            // branched package must export rather than fail obscurely.
            let inference = if *reparameterized {
                load_model_package::<MobileOneUnetInference, _>(
                    source,
                    &ModelDescription::mobileone_unet(config.clone(), true),
                    &device,
                    |device| config.init(device).reparameterize(),
                )
                .map_err(|error| failure(TaskStage::Exporting, error.to_string()))?
                .0
            } else {
                let (training, _) = load_model_package::<MobileOneUnet, _>(
                    source,
                    &ModelDescription::mobileone_unet(config.clone(), false),
                    &device,
                    |device| config.init(device),
                )
                .map_err(|error| failure(TaskStage::Exporting, error.to_string()))?;
                training.reparameterize()
            };
            export_mobileone_unet_onnx(&inference, &config)
                .map_err(|error| failure(TaskStage::Exporting, error.to_string()))
        }
    }
}

/// What has to hold before a package is read: an absolute package directory and
/// a free destination inside an existing directory.
fn validate_request(params: &ExportOnnxParams) -> Result<(), ExportOnnxError> {
    let kind = model_source_kind(&params.source)
        .map_err(|error| failure(TaskStage::Preparing, error.detail))?;
    if kind != ModelSourceKind::ModelPackage {
        return Err(failure(
            TaskStage::Preparing,
            format!(
                "onnx export requires a model package, not a {}; publish the checkpoint first",
                kind.as_slug()
            ),
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
    // The parent is named in the detail because the operating system's own
    // message is localised and says nothing about which path it means.
    let metadata = fs::symlink_metadata(parent).map_err(|error| {
        failure(
            TaskStage::Preparing,
            format!(
                "destination parent is unavailable: {}: {error}",
                parent.display()
            ),
        )
    })?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        return Err(failure(
            TaskStage::Preparing,
            format!(
                "destination parent must be an existing non-symlink directory: {}",
                parent.display()
            ),
        ));
    }
    Ok(())
}

/// The manifest `model_type` each requested kind expects.
fn model_type(kind: OnnxExportKind) -> &'static str {
    match kind {
        OnnxExportKind::FeatherHubert => "feather_hubert",
        OnnxExportKind::OriginalUnet => "original_unet",
        OnnxExportKind::MobileOneUnet => "mobileone_unet",
    }
}

fn onnx_kind(kind: OnnxExportKind) -> OnnxModelKind {
    match kind {
        OnnxExportKind::FeatherHubert => OnnxModelKind::FeatherHubert,
        OnnxExportKind::OriginalUnet => OnnxModelKind::OriginalUnet,
        OnnxExportKind::MobileOneUnet => OnnxModelKind::MobileOneUnet,
    }
}

/// The manifest agreed on the kind but not on the configuration shape. The
/// admission check makes this unreachable in practice; it is a refusal rather
/// than an assertion so a future configuration variant cannot panic here.
fn mismatch(configuration: &ModelConfiguration, kind: OnnxExportKind) -> ExportOnnxError {
    failure(
        TaskStage::Exporting,
        format!(
            "package configuration {} does not match the requested {}",
            configuration.model_type(),
            model_type(kind)
        ),
    )
}

fn report_json(
    params: &ExportOnnxParams,
    manifest: &ModelPackageManifest,
    artifact: &OnnxArtifact,
) -> serde_json::Value {
    serde_json::json!({
        "kind": "export_onnx",
        "model_kind": manifest.model_type,
        "architecture_version": manifest.architecture_version,
        "source": params.source,
        "destination": params.destination,
        "opset": artifact.opset(),
        "bytes": artifact.bytes(),
        "sha256": artifact.sha256(),
        // The package's own weight digest ties this file to the package it was
        // built from without re-hashing the weights.
        "source_model_sha256": manifest.model.sha256})
}

fn cancelled(stage: TaskStage) -> ExportOnnxError {
    ExportOnnxError::Cancelled { stage }
}

fn failure(stage: TaskStage, detail: impl Into<String>) -> ExportOnnxError {
    ExportOnnxError::Failed {
        detail: detail.into(),
        stage,
    }
}
