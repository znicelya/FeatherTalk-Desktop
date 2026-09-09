use std::{any::Any, fmt::Display, io};

use feathertalk_audio::AudioError;
use feathertalk_domain::{ErrorCode, MAX_DETAIL_CHARS, TaskError, TaskStage};
use feathertalk_export::PackageError;
use feathertalk_frame_pipeline::{AnomalyCode, FrameAnomaly, PipelineError};
use feathertalk_inference::InferenceError;
use feathertalk_media::MediaError;
use feathertalk_project::ProjectError;
use feathertalk_training::TrainingError;
use feathertalk_training_data::TrainingDataError;

use crate::ENV_HUBERT_DIR;

/// Both commands in this slice are single-shot checks that run before any long
/// pipeline stage exists, so every failure is reported as happening while the
/// task was being prepared. `TaskError.stage` must never be terminal.
const FAILURE_STAGE: TaskStage = TaskStage::Preparing;

/// How many rejected frames the detail names before it stops. Enough to see a
/// pattern, short enough to stay inside `MAX_DETAIL_CHARS`.
const MAX_REPORTED_ANOMALIES: usize = 3;

pub(crate) fn panic_detail(payload: &(dyn Any + Send)) -> &str {
    if let Some(message) = payload.downcast_ref::<String>() {
        message
    } else if let Some(message) = payload.downcast_ref::<&str>() {
        message
    } else {
        "execution panicked with a non-string payload"
    }
}

pub(crate) fn panic_task_error(payload: &(dyn Any + Send), stage: TaskStage) -> TaskError {
    TaskError::new(
        ErrorCode::WorkerCrashed,
        "任务执行异常，请重试或从检查点恢复",
        &clamp(panic_detail(payload)),
        stage,
    )
}

pub fn project_task_error(error: &ProjectError) -> TaskError {
    let code = project_error_code(error);
    TaskError::new(
        code,
        project_summary(error),
        &clamp(&error.to_string()),
        FAILURE_STAGE,
    )
}

pub fn media_task_error(error: &MediaError) -> TaskError {
    let code = media_error_code(error);
    TaskError::new(
        code,
        media_summary(error),
        &clamp(&error.to_string()),
        FAILURE_STAGE,
    )
}

pub fn is_media_cancellation(error: &MediaError) -> bool {
    matches!(error, MediaError::ToolCancelled { .. })
}

pub fn pipeline_task_error(error: &PipelineError) -> TaskError {
    let code = pipeline_error_code(error);
    TaskError::new(
        code,
        pipeline_summary(error),
        &clamp(&error.to_string()),
        FAILURE_STAGE,
    )
}

pub fn is_pipeline_cancellation(error: &PipelineError) -> bool {
    matches!(error, PipelineError::Cancelled { .. })
}

/// Maps a quality report the pipeline accepted but the caller must reject.
///
/// The first anomaly decides the code and the summary: it is the earliest frame
/// the user has to fix, and mixing codes would leave the CLI without a single
/// recovery hint.
pub fn quality_task_error(anomalies: &[FrameAnomaly]) -> TaskError {
    let first = anomalies.first();
    let code = first.map_or(ErrorCode::MediaInvalid, |anomaly| {
        anomaly_error_code(anomaly.code())
    });
    let summary = first.map_or("抽帧质检未通过", |anomaly| {
        anomaly_summary(anomaly.code())
    });
    let mut detail = format!("{} frame(s) rejected", anomalies.len());
    for anomaly in anomalies.iter().take(MAX_REPORTED_ANOMALIES) {
        detail.push_str(&format!(
            "; frame {} {:?}: {}",
            anomaly.frame_index(),
            anomaly.code(),
            anomaly.summary()
        ));
    }
    TaskError::new(code, summary, &clamp(&detail), FAILURE_STAGE)
}

pub fn audio_task_error(error: &AudioError) -> TaskError {
    let code = audio_error_code(error);
    TaskError::new(
        code,
        audio_summary(error),
        &clamp(&error.to_string()),
        FAILURE_STAGE,
    )
}

pub fn is_audio_cancellation(error: &AudioError) -> bool {
    matches!(error, AudioError::Cancelled { .. })
}

/// Maps a model-package failure the feature command could not recover from.
///
/// Every variant reports `ModelIncompatible`, including `Io`: the request named
/// no path, so a missing or unreadable file under the package directory is a
/// misconfigured model directory, not a broken disk. The detail names the
/// variable that points at it.
pub fn package_task_error(error: &PackageError) -> TaskError {
    TaskError::new(
        ErrorCode::ModelIncompatible,
        "特征模型加载失败",
        &clamp(&package_detail(error)),
        FAILURE_STAGE,
    )
}

/// Maps a legacy model import or package publication failure onto the stable
/// model recovery contract, preserving the phase in which it occurred.
pub fn legacy_task_error(error: &impl Display, stage: TaskStage) -> TaskError {
    TaskError::new(
        ErrorCode::ModelIncompatible,
        "模型导入失败",
        &clamp(&error.to_string()),
        stage,
    )
}

/// Maps a legacy feature migration failure onto the model recovery contract.
pub fn legacy_feature_task_error(error: &impl Display, stage: TaskStage) -> TaskError {
    TaskError::new(
        ErrorCode::ModelIncompatible,
        "特征迁移失败",
        &clamp(&error.to_string()),
        stage,
    )
}

/// Maps a model package export failure onto the model recovery contract.
///
/// The stage is a parameter for the same reason the legacy mappers take one: an
/// export that fails while reading the checkpoint is `Preparing`, and one that
/// fails while publishing is `Exporting`.
pub fn export_task_error(error: &impl Display, stage: TaskStage) -> TaskError {
    TaskError::new(
        ErrorCode::ModelIncompatible,
        "模型导出失败",
        &clamp(&error.to_string()),
        stage,
    )
}

/// Maps an ONNX export failure onto the model recovery contract.
///
/// Its own summary rather than the one `export_task_error` carries: the two
/// commands fail for different reasons, and the client shows the summary.
pub fn onnx_task_error(error: &impl Display, stage: TaskStage) -> TaskError {
    TaskError::new(
        ErrorCode::ModelIncompatible,
        "ONNX 导出失败",
        &clamp(&error.to_string()),
        stage,
    )
}

/// Maps a training failure onto the wire.
///
/// The stage is a parameter, unlike every other mapper in this file: a run that
/// fails at step 3000 has to report the step it failed in, not `Preparing`. The
/// command body passes `TaskStage::Preparing` while it is assembling the run and
/// the last reported `Training { .. }` once the loop has started.
pub fn training_task_error(error: &TrainingError, stage: TaskStage) -> TaskError {
    TaskError::new(
        training_error_code(error),
        training_summary(error),
        &clamp(&error.to_string()),
        stage,
    )
}

/// Maps a dataset failure. Opening the dataset is the only place one can
/// surface, and that happens during admission, so the stage is fixed.
pub fn training_data_task_error(error: &TrainingDataError) -> TaskError {
    TaskError::new(
        training_data_error_code(error),
        training_data_summary(error),
        &clamp(&error.to_string()),
        FAILURE_STAGE,
    )
}

fn training_error_code(error: &TrainingError) -> ErrorCode {
    match error {
        TrainingError::Io(source) => io_error_code(source),
        // `InvalidInput` is wide: a corrupt sample and a diverged loss both land
        // here, and both are the user's data or hyperparameters.
        TrainingError::InvalidInput(_) | TrainingError::CheckpointDirectory(_) => {
            ErrorCode::MediaInvalid
        }
        TrainingError::InvalidPackage(_)
        | TrainingError::HashMismatch { .. }
        | TrainingError::InvalidCheckpoint(_)
        | TrainingError::CheckpointCompatibility(_) => ErrorCode::ModelIncompatible,
        // Every input to the config variants is a worker constant, so they can
        // only be worker bugs; `Store` is left with storage-layer faults once
        // the checkpoint preflight has rejected the incompatible cases.
        TrainingError::InvalidConfig(_)
        | TrainingError::InvalidDataLoaderConfig(_)
        | TrainingError::InvalidDataLoaderState(_)
        | TrainingError::DataLoaderOverflow { .. }
        | TrainingError::PermutationAllocation { .. }
        | TrainingError::BatchAllocation { .. }
        | TrainingError::StalePreparedBatch
        | TrainingError::Store(_) => ErrorCode::WorkerCrashed,
    }
}

fn training_summary(error: &TrainingError) -> &'static str {
    match error {
        TrainingError::Io(source) => io_summary(source),
        TrainingError::InvalidInput(_) => "训练输入无效",
        TrainingError::InvalidConfig(_) | TrainingError::InvalidDataLoaderConfig(_) => {
            "训练配置无效"
        }
        TrainingError::InvalidDataLoaderState(_)
        | TrainingError::DataLoaderOverflow { .. }
        | TrainingError::PermutationAllocation { .. }
        | TrainingError::BatchAllocation { .. }
        | TrainingError::StalePreparedBatch => "训练运行状态异常",
        TrainingError::InvalidPackage(_) | TrainingError::HashMismatch { .. } => {
            "感知损失模型加载失败"
        }
        TrainingError::Store(_) => "检查点读写失败",
        TrainingError::InvalidCheckpoint(_) | TrainingError::CheckpointCompatibility(_) => {
            "检查点与当前训练不兼容"
        }
        TrainingError::CheckpointDirectory(_) => "检查点目录无效",
    }
}

fn training_data_error_code(error: &TrainingDataError) -> ErrorCode {
    match error {
        // The name lines up with the domain code: the token count and the frame
        // count disagree, which deserves better than a generic media error.
        TrainingDataError::FeatureShape { .. } => ErrorCode::FeatureShapeMismatch,
        // Index and stacking invariants a user cannot reach from a request.
        TrainingDataError::FrameIndexOutOfRange { .. } | TrainingDataError::Batch { .. } => {
            ErrorCode::WorkerCrashed
        }
        TrainingDataError::Project { .. }
        | TrainingDataError::Features { .. }
        | TrainingDataError::Frame { .. }
        | TrainingDataError::Landmarks { .. }
        | TrainingDataError::Sample { .. } => ErrorCode::MediaInvalid,
    }
}

fn training_data_summary(error: &TrainingDataError) -> &'static str {
    match error {
        TrainingDataError::Project { .. } => "工程目录不可用于训练",
        TrainingDataError::Features { .. } => "音频特征文件不可读",
        TrainingDataError::FeatureShape { .. } => "特征长度与帧数不匹配",
        TrainingDataError::FrameIndexOutOfRange { .. } => "样本帧号越界",
        TrainingDataError::Frame { .. } => "训练帧不可读",
        TrainingDataError::Landmarks { .. } => "关键点文件不可读",
        TrainingDataError::Sample { .. } => "训练样本构造失败",
        TrainingDataError::Batch { .. } => "训练批次堆叠失败",
    }
}

fn project_error_code(error: &ProjectError) -> ErrorCode {
    match error {
        ProjectError::Io { source, .. } => io_error_code(source),
        ProjectError::ManifestTooLarge { .. }
        | ProjectError::InvalidUtf8 { .. }
        | ProjectError::InvalidJson { .. }
        | ProjectError::UnsupportedSchemaVersion { .. }
        | ProjectError::InvalidField { .. }
        | ProjectError::UnsafeRelativePath { .. }
        | ProjectError::Symlink { .. }
        | ProjectError::InvalidFilesystemEntry { .. }
        | ProjectError::EmptyArtifact { .. }
        | ProjectError::LockedAssetMutation { .. } => ErrorCode::MediaInvalid,
        ProjectError::AtomicReplacementUnsupported { .. } => ErrorCode::WorkerCrashed,
    }
}

fn project_summary(error: &ProjectError) -> &'static str {
    match error {
        ProjectError::Io { source, .. } => io_summary(source),
        ProjectError::ManifestTooLarge { .. } => "项目清单过大",
        ProjectError::InvalidUtf8 { .. } => "项目清单不是有效的 UTF-8 文本",
        ProjectError::InvalidJson { .. } => "项目清单 JSON 格式错误",
        ProjectError::UnsupportedSchemaVersion { .. } => "项目清单版本不受支持",
        ProjectError::InvalidField { .. } => "项目清单字段无效",
        ProjectError::UnsafeRelativePath { .. } => "项目清单包含不安全的相对路径",
        ProjectError::Symlink { .. } => "项目目录包含符号链接",
        ProjectError::InvalidFilesystemEntry { .. } => "项目目录结构不符合要求",
        ProjectError::EmptyArtifact { .. } => "项目素材文件为空",
        ProjectError::LockedAssetMutation { .. } => "素材包已锁定，无法修改",
        ProjectError::AtomicReplacementUnsupported { .. } => "当前文件系统不支持原子替换",
    }
}

fn media_error_code(error: &MediaError) -> ErrorCode {
    match error {
        MediaError::Io { source, .. } => io_error_code(source),
        MediaError::InputMissing { .. }
        | MediaError::InputNotRegularFile { .. }
        | MediaError::SymlinkNotAllowed { .. }
        | MediaError::InvalidToolchain { .. }
        | MediaError::ProbeTooLarge { .. }
        | MediaError::ProbeJson { .. }
        | MediaError::ProbeContract { .. }
        | MediaError::MissingStream { .. }
        | MediaError::DuplicateStream { .. } => ErrorCode::MediaInvalid,
        // The runtime intercepts cancellation before it reaches this mapper.
        // The arm exists so the mapping stays total and so a cancellation that
        // somehow arrives here is not mislabelled as a crash.
        MediaError::ToolCancelled { .. } => ErrorCode::TaskCancelled,
        MediaError::OutputDirectoryInvalid { .. }
        | MediaError::OutputInsideInput { .. }
        | MediaError::OutputConflictsWithInput { .. }
        | MediaError::OutputDestinationInvalid { .. }
        | MediaError::UnsupportedTarget { .. }
        | MediaError::ToolFailed { .. }
        | MediaError::ToolTimedOut { .. }
        | MediaError::ToolOutputTooLarge { .. }
        | MediaError::ToolSpawn { .. }
        | MediaError::NormalizationVerificationFailed { .. }
        | MediaError::OutputCommitFailed { .. }
        | MediaError::OutputRollbackFailed { .. } => ErrorCode::WorkerCrashed,
    }
}

fn media_summary(error: &MediaError) -> &'static str {
    match error {
        MediaError::Io { source, .. } => io_summary(source),
        MediaError::InputMissing { .. } => "找不到输入文件",
        MediaError::InputNotRegularFile { .. } => "输入不是常规文件",
        MediaError::SymlinkNotAllowed { .. } => "输入路径包含符号链接",
        MediaError::InvalidToolchain { .. } => "媒体工具链配置无效",
        MediaError::ProbeTooLarge { .. } => "媒体探测输出过大",
        MediaError::ProbeJson { .. } => "媒体探测输出不是有效 JSON",
        MediaError::ProbeContract { .. } => "媒体探测结果缺少必需字段",
        MediaError::MissingStream { .. } => "媒体文件缺少必需的音视频流",
        MediaError::DuplicateStream { .. } => "媒体文件包含重复的音视频流",
        MediaError::ToolCancelled { .. } => "任务已取消",
        MediaError::OutputDirectoryInvalid { .. }
        | MediaError::OutputInsideInput { .. }
        | MediaError::OutputConflictsWithInput { .. }
        | MediaError::OutputDestinationInvalid { .. } => "输出路径无效",
        MediaError::UnsupportedTarget { .. } => "不支持的媒体转换目标",
        MediaError::ToolFailed { .. } => "媒体工具执行失败",
        MediaError::ToolTimedOut { .. } => "媒体工具执行超时",
        MediaError::ToolOutputTooLarge { .. } => "媒体工具输出过大",
        MediaError::ToolSpawn { .. } => "无法启动媒体工具",
        MediaError::NormalizationVerificationFailed { .. } => "媒体规范化结果校验失败",
        MediaError::OutputCommitFailed { .. } => "写入输出文件失败",
        MediaError::OutputRollbackFailed { .. } => "写入失败后回滚也失败",
    }
}

fn pipeline_error_code(error: &PipelineError) -> ErrorCode {
    match error {
        // Bad input or an output directory the user has to clear: the media,
        // not the worker, is what has to change.
        PipelineError::InvalidField { .. }
        | PipelineError::InvalidReport { .. }
        | PipelineError::OutputDestinationExists { .. }
        | PipelineError::FrameMissing { .. }
        | PipelineError::FrameNotRegular { .. }
        | PipelineError::FrameEmpty { .. }
        | PipelineError::FrameTooLarge { .. }
        | PipelineError::FrameUndecodable { .. }
        | PipelineError::LandmarkNotRegular { .. }
        | PipelineError::InvalidLandmark { .. } => ErrorCode::MediaInvalid,
        PipelineError::Io { source, .. } => io_error_code(source),
        PipelineError::Adapter { .. } => ErrorCode::ModelIncompatible,
        PipelineError::Cancelled { .. } => ErrorCode::TaskCancelled,
        // Everything left is the worker's own machinery misbehaving.
        PipelineError::ToolFailed { .. }
        | PipelineError::ToolTimedOut { .. }
        | PipelineError::ToolOutputTooLarge { .. }
        | PipelineError::ToolSpawn { .. }
        | PipelineError::ReportJson { .. }
        | PipelineError::ReportNotRegular { .. }
        | PipelineError::ReportTooLarge { .. }
        | PipelineError::PublishFailed { .. }
        | PipelineError::PublishRollbackFailed { .. }
        | PipelineError::QualityRejected { .. } => ErrorCode::WorkerCrashed,
    }
}

fn pipeline_summary(error: &PipelineError) -> &'static str {
    match error {
        PipelineError::InvalidField { .. } | PipelineError::InvalidReport { .. } => {
            "抽帧参数不合法"
        }
        PipelineError::OutputDestinationExists { .. } => "素材目录已存在抽帧结果",
        PipelineError::FrameMissing { .. }
        | PipelineError::FrameNotRegular { .. }
        | PipelineError::FrameEmpty { .. }
        | PipelineError::FrameTooLarge { .. } => "抽出的帧不可用",
        PipelineError::FrameUndecodable { .. } => "素材帧无法解码",
        PipelineError::LandmarkNotRegular { .. } | PipelineError::InvalidLandmark { .. } => {
            "关键点文件不可用"
        }
        PipelineError::Io { source, .. } => io_summary(source),
        PipelineError::Adapter { .. } => "模型推理失败",
        PipelineError::Cancelled { .. } => "任务已取消",
        PipelineError::ToolFailed { .. } | PipelineError::ToolSpawn { .. } => "ffmpeg 抽帧失败",
        PipelineError::ToolTimedOut { .. } => "ffmpeg 抽帧超时",
        PipelineError::ToolOutputTooLarge { .. } => "ffmpeg 输出过大",
        PipelineError::ReportJson { .. }
        | PipelineError::ReportNotRegular { .. }
        | PipelineError::ReportTooLarge { .. } => "质检报告写入失败",
        PipelineError::PublishFailed { .. } | PipelineError::PublishRollbackFailed { .. } => {
            "抽帧结果发布失败"
        }
        PipelineError::QualityRejected { .. } => "抽帧质检未通过",
    }
}

fn anomaly_error_code(code: AnomalyCode) -> ErrorCode {
    match code {
        AnomalyCode::FaceNotFound | AnomalyCode::MultipleFaces | AnomalyCode::BboxOutOfBounds => {
            ErrorCode::FaceNotFound
        }
        AnomalyCode::LandmarkInvalid => ErrorCode::LandmarkInvalid,
        AnomalyCode::BlurredFrame
        | AnomalyCode::FrameDecodeFailed
        | AnomalyCode::FrameWriteFailed => ErrorCode::MediaInvalid,
        AnomalyCode::ModelFailed => ErrorCode::ModelIncompatible,
    }
}

fn anomaly_summary(code: AnomalyCode) -> &'static str {
    match code {
        AnomalyCode::FaceNotFound => "有帧未检测到人脸",
        AnomalyCode::MultipleFaces => "有帧检测到多张人脸",
        AnomalyCode::BboxOutOfBounds => "人脸框超出画面范围",
        AnomalyCode::LandmarkInvalid => "关键点不合法",
        AnomalyCode::BlurredFrame => "有帧过于模糊",
        AnomalyCode::FrameDecodeFailed => "有帧无法解码",
        AnomalyCode::FrameWriteFailed => "有帧写入失败",
        AnomalyCode::ModelFailed => "模型推理失败",
    }
}

fn audio_error_code(error: &AudioError) -> ErrorCode {
    match error {
        // The audio or the feature file on disk is what has to change.
        AudioError::WavNotRegular { .. }
        | AudioError::WavTooLarge { .. }
        | AudioError::InvalidRiffHeader
        | AudioError::InvalidWavHeader { .. }
        | AudioError::MissingWavChunk { .. }
        | AudioError::UnsupportedWavFormat { .. }
        | AudioError::UnsupportedWavChannels { .. }
        | AudioError::UnsupportedWavSampleRate { .. }
        | AudioError::UnsupportedWavBitDepth { .. }
        | AudioError::WavPayloadTruncated { .. }
        | AudioError::EmptyWav
        | AudioError::EmptyWaveform
        | AudioError::NonFiniteWaveform { .. }
        | AudioError::ConstantWaveform
        | AudioError::FeatureNotRegular { .. }
        | AudioError::FeatureTooLarge { .. }
        | AudioError::InvalidFeatureMagic
        | AudioError::UnsupportedFeatureVersion { .. }
        | AudioError::FeatureHeaderTruncated { .. }
        | AudioError::FeaturePayloadTruncated { .. }
        | AudioError::FeatureTrailingBytes { .. }
        | AudioError::InvalidFeaturePayloadSize
        | AudioError::InvalidFeaturePairWidth { .. }
        | AudioError::LockedAssetMutation { .. } => ErrorCode::MediaInvalid,
        AudioError::WavIo { source, .. } | AudioError::FeatureIo { source, .. } => {
            io_error_code(source)
        }
        // A feature file whose width or values disagree with the encoder is a
        // model mismatch, not a bad request.
        AudioError::InvalidFeatureDimension
        | AudioError::FeatureLengthMismatch { .. }
        | AudioError::NonFiniteFeature { .. } => ErrorCode::ModelIncompatible,
        AudioError::FeatureShapeMismatch { .. } => ErrorCode::FeatureShapeMismatch,
        // The runtime intercepts cancellation before it reaches this mapper, so
        // this arm exists for the caller that maps an error without asking
        // `is_audio_cancellation` first.
        AudioError::Cancelled { .. } => ErrorCode::TaskCancelled,
        // Everything left is the worker's own machinery misbehaving.
        AudioError::InvalidChunkSize
        | AudioError::ArithmeticOverflow
        | AudioError::TooManyChunks { .. }
        | AudioError::FeatureSizeOverflow
        | AudioError::CommitFailed { .. }
        | AudioError::CommitRollbackFailed { .. }
        | AudioError::StagingCollision { .. } => ErrorCode::WorkerCrashed,
    }
}

fn audio_summary(error: &AudioError) -> &'static str {
    match error {
        AudioError::WavIo { source, .. } | AudioError::FeatureIo { source, .. } => {
            io_summary(source)
        }
        AudioError::WavNotRegular { .. } => "音频文件不是常规文件",
        AudioError::WavTooLarge { .. } => "音频文件过大",
        AudioError::InvalidRiffHeader => "音频文件不是有效的 WAV",
        AudioError::InvalidWavHeader { .. } => "WAV 头部字段无效",
        AudioError::MissingWavChunk { .. } => "WAV 缺少必需的数据块",
        AudioError::UnsupportedWavFormat { .. } => "WAV 编码格式不受支持，需要 16 位 PCM",
        AudioError::UnsupportedWavChannels { .. } => "音频必须是单声道",
        AudioError::UnsupportedWavSampleRate { .. } => "音频采样率必须是 16kHz",
        AudioError::UnsupportedWavBitDepth { .. } => "音频位深必须是 16 位",
        AudioError::WavPayloadTruncated { .. } => "WAV 数据被截断",
        AudioError::EmptyWav | AudioError::EmptyWaveform => "音频没有采样点",
        AudioError::NonFiniteWaveform { .. } => "音频包含非有限采样值",
        AudioError::ConstantWaveform => "音频是恒定值，无法归一化",
        AudioError::InvalidChunkSize
        | AudioError::ArithmeticOverflow
        | AudioError::TooManyChunks { .. } => "音频分块规划失败",
        AudioError::InvalidFeatureDimension | AudioError::FeatureLengthMismatch { .. } => {
            "特征维度与模型不一致"
        }
        AudioError::FeatureShapeMismatch { .. } => "特征长度与帧数不匹配",
        AudioError::NonFiniteFeature { .. } => "特征包含非有限值",
        AudioError::FeatureSizeOverflow => "特征文件尺寸溢出",
        AudioError::FeatureNotRegular { .. } => "特征文件不是常规文件",
        AudioError::FeatureTooLarge { .. } => "特征文件过大",
        AudioError::InvalidFeatureMagic | AudioError::UnsupportedFeatureVersion { .. } => {
            "特征文件格式不受支持"
        }
        AudioError::FeatureHeaderTruncated { .. }
        | AudioError::FeaturePayloadTruncated { .. }
        | AudioError::FeatureTrailingBytes { .. }
        | AudioError::InvalidFeaturePayloadSize
        | AudioError::InvalidFeaturePairWidth { .. } => "特征文件内容损坏",
        AudioError::LockedAssetMutation { .. } => "素材包已锁定，无法修改",
        AudioError::CommitFailed { .. } => "特征文件写入失败",
        AudioError::CommitRollbackFailed { .. } => "写入失败后回滚也失败",
        AudioError::StagingCollision { .. } => "暂存文件已存在",
        AudioError::Cancelled { .. } => "任务已取消",
    }
}

/// The package loader's message plus the variable a user has to fix.
fn package_detail(error: &PackageError) -> String {
    format!("{error} (check {ENV_HUBERT_DIR})")
}

fn io_error_code(source: &io::Error) -> ErrorCode {
    match source.kind() {
        io::ErrorKind::StorageFull | io::ErrorKind::QuotaExceeded => ErrorCode::DiskSpaceLow,
        _ => ErrorCode::WorkerCrashed,
    }
}

fn io_summary(source: &io::Error) -> &'static str {
    match source.kind() {
        io::ErrorKind::StorageFull | io::ErrorKind::QuotaExceeded => "磁盘空间不足",
        _ => "文件读写失败",
    }
}

/// `TaskError::validate` counts characters, not bytes, so the detail is clamped
/// on a character boundary.
pub(crate) fn clamp(detail: &str) -> String {
    detail.chars().take(MAX_DETAIL_CHARS).collect()
}

/// The `field` name inference uses for the landmark artifact. It is the one
/// input whose failure the protocol reports as a landmark problem rather than a
/// generic media problem.
const LANDMARK_FIELD: &str = "landmark_path";

/// Maps an inference failure onto the wire, keeping the stage the render was in
/// when it happened.
pub fn render_task_error(error: &InferenceError, stage: TaskStage) -> TaskError {
    TaskError::new(
        inference_error_code(error),
        inference_summary(error),
        &clamp(&error.to_string()),
        stage,
    )
}

/// True when the render stopped because the task was cancelled, which the
/// command reports as `CommandOutcome::Cancelled` rather than as a failure.
pub fn is_inference_cancellation(error: &InferenceError) -> bool {
    matches!(error, InferenceError::Cancelled { .. })
}

/// Exhaustive on purpose: a new inference failure has to be classified here
/// rather than defaulting to `MediaInvalid` through a `_` arm.
fn inference_error_code(error: &InferenceError) -> ErrorCode {
    match error {
        // The guard comes first: a landmark file that will not parse is the
        // client's data problem, and the protocol has a code that says so.
        InferenceError::InvalidInputArtifact { field, .. } if *field == LANDMARK_FIELD => {
            ErrorCode::LandmarkInvalid
        }
        InferenceError::InvalidBbox { .. } => ErrorCode::LandmarkInvalid,
        InferenceError::InvalidFeatureShape { .. } => ErrorCode::FeatureShapeMismatch,
        InferenceError::InvalidInputDirectory { .. }
        | InferenceError::InvalidInputArtifact { .. }
        | InferenceError::InvalidField { .. }
        | InferenceError::ArithmeticOverflow
        | InferenceError::OutputExists { .. }
        | InferenceError::OutputNotRegular { .. }
        | InferenceError::OutputSymlink { .. }
        | InferenceError::OutputParentInvalid { .. }
        | InferenceError::InvalidTaskId { .. }
        | InferenceError::FfmpegPathNotAbsolute { .. }
        | InferenceError::EmptyFfmpegPath
        | InferenceError::FrameCountTooSmall { .. }
        | InferenceError::EmptyFeatures
        | InferenceError::FrameIndexOutOfRange { .. }
        | InferenceError::OutputFrameOutOfRange { .. }
        | InferenceError::InvalidAudioWindowIndex { .. }
        | InferenceError::FrameDimensionsMismatch { .. }
        | InferenceError::InvalidFrameDimensions { .. }
        | InferenceError::InvalidResizeTarget { .. }
        | InferenceError::PixelOutOfRange { .. }
        | InferenceError::PasteOutOfBounds { .. }
        | InferenceError::FrameBufferLengthMismatch { .. } => ErrorCode::MediaInvalid,
        InferenceError::TensorShapeMismatch { .. }
        | InferenceError::NonFiniteModelInput { .. }
        | InferenceError::ModelTensorData { .. }
        | InferenceError::NonFiniteModelOutput { .. }
        | InferenceError::ModelOutputOutOfRange { .. }
        | InferenceError::NonFinitePrediction { .. } => ErrorCode::ModelIncompatible,
        InferenceError::SinkStart { .. }
        | InferenceError::SinkWrite { .. }
        | InferenceError::SinkFinish { .. }
        | InferenceError::ToolFailed { .. }
        | InferenceError::StagingCollision { .. }
        | InferenceError::StagingOutputInvalid { .. }
        | InferenceError::AtomicPublishFailed { .. }
        | InferenceError::FrameReader { .. }
        | InferenceError::AllocationFailure { .. } => ErrorCode::WorkerCrashed,
        InferenceError::Cancelled { .. } => ErrorCode::TaskCancelled,
    }
}

/// The user-facing half of the mapping. Grouped by what an operator can do about
/// it, which is why these groups do not match the code groups one for one.
fn inference_summary(error: &InferenceError) -> &'static str {
    match error {
        InferenceError::InvalidInputArtifact { field, .. } if *field == LANDMARK_FIELD => {
            "关键点数据无效"
        }
        InferenceError::InvalidBbox { .. } => "关键点数据无效",
        InferenceError::InvalidFeatureShape { .. } => "音频特征形状不符",
        InferenceError::InvalidField { .. }
        | InferenceError::ArithmeticOverflow
        | InferenceError::OutputExists { .. }
        | InferenceError::OutputNotRegular { .. }
        | InferenceError::OutputSymlink { .. }
        | InferenceError::OutputParentInvalid { .. }
        | InferenceError::InvalidTaskId { .. }
        | InferenceError::FfmpegPathNotAbsolute { .. }
        | InferenceError::EmptyFfmpegPath
        | InferenceError::InvalidResizeTarget { .. } => "渲染请求无效",
        InferenceError::InvalidInputDirectory { .. }
        | InferenceError::InvalidInputArtifact { .. }
        | InferenceError::FrameCountTooSmall { .. }
        | InferenceError::EmptyFeatures
        | InferenceError::FrameIndexOutOfRange { .. }
        | InferenceError::OutputFrameOutOfRange { .. }
        | InferenceError::InvalidAudioWindowIndex { .. }
        | InferenceError::FrameDimensionsMismatch { .. }
        | InferenceError::InvalidFrameDimensions { .. }
        | InferenceError::FrameBufferLengthMismatch { .. } => "渲染素材不可用",
        InferenceError::PixelOutOfRange { .. } | InferenceError::PasteOutOfBounds { .. } => {
            "渲染几何越界"
        }
        InferenceError::TensorShapeMismatch { .. }
        | InferenceError::NonFiniteModelInput { .. }
        | InferenceError::ModelTensorData { .. }
        | InferenceError::NonFiniteModelOutput { .. }
        | InferenceError::ModelOutputOutOfRange { .. }
        | InferenceError::NonFinitePrediction { .. } => "模型推理结果异常",
        InferenceError::SinkStart { .. }
        | InferenceError::SinkWrite { .. }
        | InferenceError::SinkFinish { .. }
        | InferenceError::ToolFailed { .. } => "视频编码进程失败",
        InferenceError::StagingCollision { .. }
        | InferenceError::StagingOutputInvalid { .. }
        | InferenceError::AtomicPublishFailed { .. } => "产物发布失败",
        InferenceError::FrameReader { .. } => "视频帧解码失败",
        InferenceError::AllocationFailure { .. } => "内存不足",
        InferenceError::Cancelled { .. } => "任务已取消",
    }
}
