//! Strict, auditable FeatherTalk model packages.

mod error;
mod feather_hubert;
mod io;
mod manifest;
pub mod onnx;
mod onnx_feather_hubert;
mod onnx_publish;
mod onnx_unet;
mod package;

pub use error::PackageError;
pub use feather_hubert::{
    FeatherHubertPackageReport, FeatherHubertPackageRequest, build_feather_hubert_package};
pub use manifest::{
    FEATHER_HUBERT_ARCHITECTURE_VERSION, FileManifest, LICENSE_FILE_NAME, LicenseBundle,
    LicenseEntry, MANIFEST_FILE_NAME, MAX_LICENSE_BYTES, MAX_MANIFEST_BYTES, MAX_MODEL_BYTES,
    MAX_SOURCE_BYTES, MOBILEONE_UNET_ARCHITECTURE_VERSION, MODEL_FILE_NAME,
    MODEL_LICENSE_SCHEMA_VERSION, MODEL_PACKAGE_SCHEMA_VERSION, ModelConfiguration,
    ModelDescription, ModelPackageManifest, OPTIMIZER_FILE_NAME,
    ORIGINAL_UNET_ARCHITECTURE_VERSION, SourceManifest, TRAINING_STATE_FILE_NAME, TensorContract,
    TensorSpec, TrainingManifest, TrainingMode};
pub use onnx_feather_hubert::export_feather_hubert_onnx;
pub use onnx_publish::{OnnxArtifact, OnnxPublishError, publish_onnx_model};
pub use onnx_unet::{export_mobileone_unet_onnx, export_original_unet_onnx};
pub use package::{
    PackageBuildReport, PackageBuildRequest, load_model_package, read_package_manifest,
    write_model_package};
