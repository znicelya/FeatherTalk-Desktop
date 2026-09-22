//! The FeatherTalk worker: a JSON Lines command server over stdin/stdout.
//!
//! CPU is the default; model commands can use an explicitly selected native
//! WGPU adapter. Commands that shell out to ffmpeg or need a model directory are
//! announced only once their toolchain is configured, and are rejected if a
//! client asks for them anyway; the rest are announced unconditionally.

mod adapters;
mod admission;
mod asset_scan;
mod commands;
mod compute;
mod compute_commands;
mod config;
mod error_map;
mod execution_backend;
mod exporting;
mod exporting_onnx;
mod extract_features;
mod extract_frames;
mod feature_result;
mod features;
mod handshake;
mod importing;
mod inspect;
mod inspect_result;
mod inspecting;
mod lock_asset_package;
mod lock_result;
mod migrating_features;
mod models;
mod normalize_result;
mod probe_result;
mod quality_result;
mod render;
mod render_result;
mod rendering;
mod reporter;
mod runtime;
mod train;
mod train_result;
mod training;

pub use adapters::{AdapterLockError, AdapterLocks};
pub use commands::{CommandOutcome, execute, execute_with_runner};
pub use compute::{ComputeRegistry, GpuContext, GpuFailure, WgpuContext};
pub use config::{
    DEFAULT_MEDIA_TIMEOUT_MS, ENV_ADAPTER, ENV_BACKEND, ENV_FFMPEG, ENV_FFPROBE, ENV_HUBERT_DIR,
    ENV_MEDIA_TIMEOUT_MS, ENV_PFLD_DIR, ENV_SCRFD_DIR, ENV_VGG19_DIR, FeatureToolchain,
    ModelToolchain, TrainingToolchain, WorkerConfig};
pub use error_map::{
    audio_task_error, export_task_error, is_audio_cancellation, is_inference_cancellation,
    is_media_cancellation, is_pipeline_cancellation, legacy_feature_task_error, legacy_task_error,
    media_task_error, onnx_task_error, package_task_error, pipeline_task_error, project_task_error,
    quality_task_error, render_task_error, training_data_task_error, training_task_error};
pub use execution_backend::{execution_name, gpu_memory_bytes};
pub use exporting::{
    ExportModelPackageError, ExportPlan, execute_export_model_package, export_plan,
    publish_checkpoint_package};
pub use exporting_onnx::{ExportOnnxError, execute_export_onnx};
pub use extract_features::execute_extract_features;
pub use extract_frames::execute_extract_frames;
pub use feature_result::feature_to_json;
pub use features::FeatureModel;
pub use handshake::{CPU_ADAPTER_ID, cpu_adapter, ready_frame, supported_commands};
pub use importing::{ImportLegacyModelError, execute_import_legacy_model};
pub use inspect::execute_inspect_model;
pub use inspect_result::{InspectSummary, InspectedModel, inspect_to_json};
pub use inspecting::{
    InspectedFile, ModelSourceKind, checkpoint_files, checkpoint_incompatibilities,
    model_source_kind, package_files, package_incompatibilities};
pub use lock_asset_package::execute_lock_asset_package;
pub use lock_result::lock_to_json;
pub use migrating_features::{MigrateLegacyFeaturesError, execute_migrate_legacy_features};
pub use models::FrameModels;
pub use normalize_result::normalize_to_json;
pub use probe_result::probe_to_json;
pub use quality_result::quality_to_json;
pub use render::{execute_render, execute_render_on, run_render};
pub use render_result::{RenderSummary, render_to_json};
pub use rendering::{
    ProjectAssets, RENDER_BACKEND_NAME, RENDER_FPS, RenderBackend, RenderDevice, RenderJob,
    RenderVariant, check_max_output_frames, check_render_paths, progress_total, project_assets,
    render_job, render_variant, staging_task_id};
pub use reporter::{NoReporter, TaskReporter};
pub use runtime::{JobExecutor, serve, serve_with_executor};
pub use train::{check_frame_count, execute_train, execute_train_on, run_training};
pub use train_result::{TrainSummary, train_to_json};
pub use training::{
    DEFAULT_BATCH_SIZE, DEFAULT_LEARNING_RATE, MAX_EPOCHS, TRAIN_BACKEND_NAME, TRAINING_SEED,
    TrainBackend, TrainDevice, TrainingPaths, TrainingPlan, WORKER_STATE, checkpoint_descriptor,
    latest_checkpoint, preview_sample, publish_checkpoint, sample_count, training_config,
    training_mode, write_metrics_unless_present, write_preview_unless_present};

/// The CPU tensor device the worker trains and renders on by default.
///
/// The unified default device prefers a GPU backend whenever one is compiled
/// (this workspace always enables ulkan, and cuda on Windows/Linux), so
/// the CPU path names the Flex device explicitly rather than relying on the
/// default, which would otherwise route `CPU` work onto the GPU.
pub(crate) fn cpu_device() -> burn::tensor::Device {
    burn::tensor::Device::flex()
}

/// The autodiff-enabled CPU device used when training without a GPU.
pub(crate) fn cpu_training_device() -> burn::tensor::Device {
    cpu_device().autodiff()
}

/// Returns an autodiff-enabled clone of device, leaving one that already has
/// autodiff enabled untouched.
///
/// Autodiff is a device property in burn 0.22 and enabling it twice panics, so
/// training enables it here once regardless of whether the caller selected a
/// plain or an already-autodiff device.
pub(crate) fn ensure_autodiff_device(device: &burn::tensor::Device) -> burn::tensor::Device {
    if device.is_autodiff() {
        device.clone()
    } else {
        device.clone().autodiff()
    }
}
