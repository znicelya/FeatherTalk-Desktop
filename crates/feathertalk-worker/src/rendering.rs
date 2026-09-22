//! The pure decisions a render makes before the first frame: where the project
//! keeps its inference inputs, which architecture a checkpoint holds, and what
//! the request's numbers mean.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use burn::tensor::Device;
use feathertalk_domain::{RenderParams, TaskError, TaskStage};
use feathertalk_export::ModelConfiguration;
use feathertalk_inference::OfflineRenderRequest;
use feathertalk_models::backend::CpuBackend;
use feathertalk_models::unet::{MobileOneUnetConfig, OriginalUnetConfig};
use feathertalk_training::CheckpointDescriptor;

use crate::admission::invalid_request;
use crate::error_map::render_task_error;

/// The default backend for CPU convenience functions. Generic rendering runs
/// one forward pass per frame on the caller's selected device without autodiff.
pub type RenderBackend = CpuBackend;

/// The device type the render command hands to the model.
pub type RenderDevice = Device;

/// The backend name in results from the CPU convenience functions.
pub const RENDER_BACKEND_NAME: &str = "flex-cpu";

/// The frame rate inference writes into the container.
pub const RENDER_FPS: u32 = 25;

/// Where a locked project keeps the cropped frames, the landmarks and the audio
/// features, relative to the project root. These are component arrays rather
/// than slash-joined literals so a Windows join produces native separators.
const FRAME_DIR: [&str; 2] = ["assets", "frames"];
const LANDMARK_DIR: [&str; 2] = ["assets", "landmarks"];
const FEATURE_PATH: [&str; 3] = ["assets", "features", "feather_hubert.f32"];

/// Joins one component at a time, which is the only way to keep the separator
/// native on Windows.
fn project_path(root: &Path, components: &[&str]) -> PathBuf {
    let mut path = root.to_path_buf();
    for component in components {
        path.push(component);
    }
    path
}

/// The three inference inputs a locked project already contains.
#[derive(Debug, Clone)]
pub struct ProjectAssets {
    pub frame_dir: PathBuf,
    pub landmark_dir: PathBuf,
    pub feature_path: PathBuf}

pub fn project_assets(project_dir: &Path) -> ProjectAssets {
    ProjectAssets {
        frame_dir: project_path(project_dir, &FRAME_DIR),
        landmark_dir: project_path(project_dir, &LANDMARK_DIR),
        feature_path: project_path(project_dir, &FEATURE_PATH)}
}

/// The two architectures a training checkpoint can hold.
pub enum RenderVariant {
    OriginalUnet(OriginalUnetConfig),
    MobileOneUnet(MobileOneUnetConfig)}

impl RenderVariant {
    /// The descriptor identity of this variant. `mobileone_unet` is described in
    /// its training shape (`false`), because that is how the checkpoint was
    /// written; fusing the branches happens after the record is restored.
    pub fn configuration(&self) -> ModelConfiguration {
        match self {
            Self::OriginalUnet(config) => ModelConfiguration::original_unet(config),
            Self::MobileOneUnet(config) => ModelConfiguration::mobileone_unet(config, false)}
    }
}

/// Resolves the `model_kind` a checkpoint manifest recorded.
///
/// The comparison is against `ModelConfiguration::model_type` rather than a
/// literal, so the worker cannot drift from the name the checkpoint was written
/// with. The configurations are the production ones for the same reason: the
/// digest in the descriptor is the digest of the configuration training used.
pub fn render_variant(model_kind: &str) -> Option<RenderVariant> {
    let original = RenderVariant::OriginalUnet(OriginalUnetConfig::production());
    if original.configuration().model_type() == model_kind {
        return Some(original);
    }
    let mobileone = RenderVariant::MobileOneUnet(MobileOneUnetConfig::production());
    if mobileone.configuration().model_type() == model_kind {
        return Some(mobileone);
    }
    None
}

/// Inference stages its output next to the destination under a name built from
/// the task id, so the id has to be unique within the process. The pid keeps two
/// workers apart, the counter keeps two renders in one worker apart.
pub fn staging_task_id() -> String {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let ordinal = COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("render-{}-{ordinal}", std::process::id())
}

/// The number of frames the render will write, from the locked manifest and the
/// caller's cap. This mirrors `RenderPlan::new`, which takes the same minimum.
pub fn progress_total(frame_count: u64, max_output_frames: Option<u64>) -> u64 {
    match max_output_frames {
        Some(max) => max.min(frame_count),
        None => frame_count}
}

/// Every path in a render request is absolute, because the worker resolves
/// nothing against its own working directory.
pub fn check_render_paths(params: &RenderParams) -> Result<(), TaskError> {
    if !params.checkpoint.is_absolute() {
        return Err(invalid_request(
            "检查点目录必须是绝对路径",
            format!(
                "checkpoint is not absolute: {}",
                params.checkpoint.display()
            ),
        ));
    }
    if !params.audio.is_absolute() {
        return Err(invalid_request(
            "音频文件必须是绝对路径",
            format!("audio is not absolute: {}", params.audio.display()),
        ));
    }
    if !params.output.is_absolute() {
        return Err(invalid_request(
            "输出文件必须是绝对路径",
            format!("output is not absolute: {}", params.output.display()),
        ));
    }
    Ok(())
}

/// `max_output_frames` is `Option<u64>` on the wire and `Option<usize>` in
/// inference. Zero is refused rather than clamped, and a value that does not fit
/// the host word size is refused rather than truncated.
pub fn check_max_output_frames(max: Option<u64>) -> Result<Option<usize>, TaskError> {
    let Some(max) = max else {
        return Ok(None);
    };
    if max == 0 {
        return Err(invalid_request(
            "最大输出帧数必须大于 0",
            "max_output_frames is zero".to_owned(),
        ));
    }
    let max = usize::try_from(max).map_err(|_| {
        invalid_request(
            "最大输出帧数超出本机可表示范围",
            format!("max_output_frames does not fit in usize: {max}"),
        )
    })?;
    Ok(Some(max))
}

/// Everything the render loop needs once admission is done: the inference
/// request, the progress denominator, and the checkpoint identity the result
/// payload reports.
#[derive(Debug, Clone)]
pub struct RenderJob {
    pub request: OfflineRenderRequest,
    pub progress_total: u64,
    pub descriptor: CheckpointDescriptor,
    pub checkpoint_dir: PathBuf,
    pub checkpoint_epoch: u64,
    pub checkpoint_global_step: u64,
    pub source_frame_count: u64,
    pub max_output_frames: Option<u64>}

/// Turns an admitted request plus the locked manifest's frame count into a job.
///
/// The frame count comes from the manifest and not from the feature file, so a
/// project whose features were regenerated cannot silently change the total.
pub fn render_job(
    params: &RenderParams,
    frame_count: u64,
    ffmpeg: &Path,
    descriptor: CheckpointDescriptor,
    checkpoint_epoch: u64,
    checkpoint_global_step: u64,
) -> Result<RenderJob, TaskError> {
    /// Inference walks the source frames forwards and back, which needs two.
    const MINIMUM_FRAMES: u64 = 2;

    if frame_count < MINIMUM_FRAMES {
        return Err(invalid_request(
            "工程帧数不足，无法渲染",
            format!("frame_count is {frame_count}, the minimum is {MINIMUM_FRAMES}"),
        ));
    }
    let source_frames = usize::try_from(frame_count).map_err(|_| {
        invalid_request(
            "工程帧数超出本机可表示范围",
            format!("frame_count does not fit in usize: {frame_count}"),
        )
    })?;
    let max_output_frames = check_max_output_frames(params.max_output_frames)?;
    let assets = project_assets(&params.project_dir);
    let request = OfflineRenderRequest::new(
        assets.frame_dir,
        assets.landmark_dir,
        assets.feature_path,
        params.audio.clone(),
        ffmpeg.to_path_buf(),
        params.output.clone(),
        staging_task_id(),
        source_frames,
        max_output_frames,
    )
    .map_err(|error| render_task_error(&error, TaskStage::Preparing))?;

    Ok(RenderJob {
        request,
        progress_total: progress_total(frame_count, params.max_output_frames),
        descriptor,
        checkpoint_dir: params.checkpoint.clone(),
        checkpoint_epoch,
        checkpoint_global_step,
        source_frame_count: frame_count,
        max_output_frames: params.max_output_frames})
}
