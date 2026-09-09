use std::{io, path::PathBuf};

use feathertalk_audio::AudioError;
use feathertalk_domain::{ErrorCode, MAX_DETAIL_CHARS, TaskStage};
use feathertalk_export::PackageError;
use feathertalk_frame_pipeline::{AnomalyCode, FrameAnomaly, PipelineError, RecoveryAction};
use feathertalk_inference::InferenceError;
use feathertalk_media::MediaError;
use feathertalk_project::ProjectError;
use feathertalk_training::TrainingError;
use feathertalk_training_data::TrainingDataError;
use feathertalk_worker::{
    audio_task_error, is_audio_cancellation, is_inference_cancellation, is_media_cancellation,
    is_pipeline_cancellation, legacy_task_error, media_task_error, package_task_error,
    pipeline_task_error, project_task_error, quality_task_error, render_task_error,
    training_data_task_error, training_task_error,
};

fn io_error(kind: io::ErrorKind) -> io::Error {
    io::Error::new(kind, "synthetic")
}

fn json_error() -> serde_json::Error {
    serde_json::from_str::<serde_json::Value>("{").unwrap_err()
}

fn path() -> PathBuf {
    PathBuf::from("C:/tmp/x")
}

#[test]
fn every_project_error_maps_to_a_code_and_a_valid_payload() {
    let cases = vec![
        (
            ProjectError::Io {
                operation: "read",
                path: path(),
                source: io_error(io::ErrorKind::PermissionDenied),
            },
            ErrorCode::WorkerCrashed,
        ),
        (
            ProjectError::Io {
                operation: "write",
                path: path(),
                source: io_error(io::ErrorKind::StorageFull),
            },
            ErrorCode::DiskSpaceLow,
        ),
        (
            ProjectError::Io {
                operation: "write",
                path: path(),
                source: io_error(io::ErrorKind::QuotaExceeded),
            },
            ErrorCode::DiskSpaceLow,
        ),
        (
            ProjectError::ManifestTooLarge {
                path: path(),
                limit: 1024,
            },
            ErrorCode::MediaInvalid,
        ),
        (
            ProjectError::InvalidUtf8 { path: path() },
            ErrorCode::MediaInvalid,
        ),
        (
            ProjectError::InvalidJson {
                path: path(),
                source: json_error(),
            },
            ErrorCode::MediaInvalid,
        ),
        (
            ProjectError::UnsupportedSchemaVersion {
                path: path(),
                version: 9,
            },
            ErrorCode::MediaInvalid,
        ),
        (
            ProjectError::InvalidField {
                field: "project_id".to_owned(),
                message: "empty".to_owned(),
            },
            ErrorCode::MediaInvalid,
        ),
        (
            ProjectError::UnsafeRelativePath {
                path: "../x".to_owned(),
            },
            ErrorCode::MediaInvalid,
        ),
        (
            ProjectError::Symlink { path: path() },
            ErrorCode::MediaInvalid,
        ),
        (
            ProjectError::InvalidFilesystemEntry { path: path() },
            ErrorCode::MediaInvalid,
        ),
        (
            ProjectError::EmptyArtifact { path: path() },
            ErrorCode::MediaInvalid,
        ),
        (
            ProjectError::LockedAssetMutation { path: path() },
            ErrorCode::MediaInvalid,
        ),
        (
            ProjectError::AtomicReplacementUnsupported { path: path() },
            ErrorCode::WorkerCrashed,
        ),
    ];

    for (error, expected) in cases {
        let mapped = project_task_error(&error);
        assert_eq!(mapped.code, expected, "{error:?}");
        assert_eq!(mapped.stage, TaskStage::Preparing, "{error:?}");
        assert_eq!(mapped.recovery, expected.default_recovery(), "{error:?}");
        assert!(!mapped.summary.trim().is_empty(), "{error:?}");
        assert!(!mapped.detail.is_empty(), "{error:?}");
        mapped.validate().unwrap();
    }
}

#[test]
fn every_media_error_maps_to_a_code_and_a_valid_payload() {
    let cases = vec![
        (
            MediaError::Io {
                operation: "read",
                path: path(),
                source: io_error(io::ErrorKind::NotFound),
            },
            ErrorCode::WorkerCrashed,
        ),
        (
            MediaError::Io {
                operation: "write",
                path: path(),
                source: io_error(io::ErrorKind::StorageFull),
            },
            ErrorCode::DiskSpaceLow,
        ),
        (
            MediaError::InputMissing { path: path() },
            ErrorCode::MediaInvalid,
        ),
        (
            MediaError::InputNotRegularFile { path: path() },
            ErrorCode::MediaInvalid,
        ),
        (
            MediaError::SymlinkNotAllowed { path: path() },
            ErrorCode::MediaInvalid,
        ),
        (
            MediaError::OutputDirectoryInvalid { path: path() },
            ErrorCode::WorkerCrashed,
        ),
        (
            MediaError::OutputInsideInput {
                input: path(),
                output: path(),
            },
            ErrorCode::WorkerCrashed,
        ),
        (
            MediaError::OutputConflictsWithInput { path: path() },
            ErrorCode::WorkerCrashed,
        ),
        (
            MediaError::OutputDestinationInvalid { path: path() },
            ErrorCode::WorkerCrashed,
        ),
        (
            MediaError::UnsupportedTarget {
                field: "fps",
                expected: "25",
                actual: "30".to_owned(),
            },
            ErrorCode::WorkerCrashed,
        ),
        (
            MediaError::InvalidToolchain {
                field: "ffprobe",
                message: "relative".to_owned(),
            },
            ErrorCode::MediaInvalid,
        ),
        (
            MediaError::ProbeTooLarge {
                limit: 16,
                actual: 32,
            },
            ErrorCode::MediaInvalid,
        ),
        (
            MediaError::ProbeJson {
                message: "bad".to_owned(),
            },
            ErrorCode::MediaInvalid,
        ),
        (
            MediaError::ProbeContract {
                field: "width".to_owned(),
                message: "missing".to_owned(),
            },
            ErrorCode::MediaInvalid,
        ),
        (
            MediaError::MissingStream { stream: "video" },
            ErrorCode::MediaInvalid,
        ),
        (
            MediaError::DuplicateStream { stream: "audio" },
            ErrorCode::MediaInvalid,
        ),
        (
            MediaError::ToolFailed {
                operation: "probe",
                exit_code: Some(1),
                stderr: "boom".to_owned(),
            },
            ErrorCode::WorkerCrashed,
        ),
        (
            MediaError::ToolTimedOut {
                operation: "probe",
                timeout_ms: 10,
            },
            ErrorCode::WorkerCrashed,
        ),
        (
            MediaError::ToolOutputTooLarge {
                operation: "probe",
                stream: "stdout",
                limit: 16,
                actual: 32,
            },
            ErrorCode::WorkerCrashed,
        ),
        (
            MediaError::ToolSpawn {
                operation: "probe",
                message: "not found".to_owned(),
            },
            ErrorCode::WorkerCrashed,
        ),
        (
            MediaError::ToolCancelled { operation: "probe" },
            ErrorCode::TaskCancelled,
        ),
        (
            MediaError::NormalizationVerificationFailed {
                field: "fps",
                expected: "25".to_owned(),
                actual: "30".to_owned(),
            },
            ErrorCode::WorkerCrashed,
        ),
        (
            MediaError::OutputCommitFailed {
                operation: "commit",
                message: "busy".to_owned(),
            },
            ErrorCode::WorkerCrashed,
        ),
        (
            MediaError::OutputRollbackFailed {
                operation: "rollback",
                primary: "a".to_owned(),
                rollback: "b".to_owned(),
            },
            ErrorCode::WorkerCrashed,
        ),
    ];

    for (error, expected) in cases {
        let mapped = media_task_error(&error);
        assert_eq!(mapped.code, expected, "{error:?}");
        assert_eq!(mapped.stage, TaskStage::Preparing, "{error:?}");
        assert_eq!(mapped.recovery, expected.default_recovery(), "{error:?}");
        assert!(!mapped.summary.trim().is_empty(), "{error:?}");
        mapped.validate().unwrap();
    }
}

#[test]
fn an_oversized_detail_is_clamped_to_the_wire_limit() {
    let mapped = media_task_error(&MediaError::ProbeJson {
        message: "x".repeat(MAX_DETAIL_CHARS * 2),
    });
    assert_eq!(mapped.detail.chars().count(), MAX_DETAIL_CHARS);
    mapped.validate().unwrap();
}

#[test]
fn only_tool_cancelled_counts_as_cancellation() {
    assert!(is_media_cancellation(&MediaError::ToolCancelled {
        operation: "probe"
    }));
    assert!(!is_media_cancellation(&MediaError::ToolTimedOut {
        operation: "probe",
        timeout_ms: 10
    }));
    assert!(!is_media_cancellation(&MediaError::InputMissing {
        path: path()
    }));
}

fn anomaly(index: u64, code: AnomalyCode) -> FrameAnomaly {
    FrameAnomaly::new(index, code, "摘要", "detail", RecoveryAction::ExcludeFrame).unwrap()
}

#[test]
fn every_pipeline_error_maps_to_a_code_and_a_valid_payload() {
    let cases = vec![
        (
            PipelineError::InvalidField {
                field: "frame_count",
                message: "must be greater than zero".to_owned(),
            },
            ErrorCode::MediaInvalid,
        ),
        (
            PipelineError::OutputDestinationExists { path: path() },
            ErrorCode::MediaInvalid,
        ),
        (
            PipelineError::FrameMissing { path: path() },
            ErrorCode::MediaInvalid,
        ),
        (
            PipelineError::Io {
                operation: "create_dir",
                path: path(),
                source: io_error(io::ErrorKind::StorageFull),
            },
            ErrorCode::DiskSpaceLow,
        ),
        (
            PipelineError::Io {
                operation: "create_dir",
                path: path(),
                source: io_error(io::ErrorKind::PermissionDenied),
            },
            ErrorCode::WorkerCrashed,
        ),
        (
            PipelineError::Adapter {
                component: "scrfd",
                message: "device lost".to_owned(),
            },
            ErrorCode::ModelIncompatible,
        ),
        (
            PipelineError::Cancelled {
                operation: "extract_frames",
            },
            ErrorCode::TaskCancelled,
        ),
        (
            PipelineError::ToolTimedOut {
                operation: "extract_frames",
                timeout_ms: 300_000,
            },
            ErrorCode::WorkerCrashed,
        ),
        (
            PipelineError::QualityRejected { count: 4 },
            ErrorCode::WorkerCrashed,
        ),
        (
            PipelineError::FrameUndecodable {
                path: path(),
                message: "no SOI marker".to_owned(),
            },
            ErrorCode::MediaInvalid,
        ),
        (
            PipelineError::LandmarkNotRegular { path: path() },
            ErrorCode::MediaInvalid,
        ),
        (
            PipelineError::InvalidLandmark {
                path: path(),
                message: "expected 110 lines, found 109".to_owned(),
            },
            ErrorCode::MediaInvalid,
        ),
    ];

    for (error, expected) in cases {
        let mapped = pipeline_task_error(&error);
        assert_eq!(mapped.code, expected, "{error:?}");
        assert_eq!(mapped.stage, TaskStage::Preparing, "{error:?}");
        assert_eq!(mapped.recovery, expected.default_recovery(), "{error:?}");
        assert!(!mapped.summary.trim().is_empty(), "{error:?}");
        assert!(!mapped.detail.is_empty(), "{error:?}");
        mapped.validate().unwrap();
    }
}

#[test]
fn only_cancellation_is_reported_as_cancellation() {
    assert!(is_pipeline_cancellation(&PipelineError::Cancelled {
        operation: "evaluate_frames"
    }));
    assert!(!is_pipeline_cancellation(&PipelineError::QualityRejected {
        count: 1
    }));
}

#[test]
fn the_asset_lock_failures_read_as_media_problems() {
    let undecodable = pipeline_task_error(&PipelineError::FrameUndecodable {
        path: path(),
        message: "no SOI marker".to_owned(),
    });
    assert_eq!(undecodable.summary, "素材帧无法解码");
    assert!(
        undecodable.detail.contains("no SOI marker"),
        "{}",
        undecodable.detail
    );

    let not_regular = pipeline_task_error(&PipelineError::LandmarkNotRegular { path: path() });
    assert_eq!(not_regular.summary, "关键点文件不可用");

    let malformed = pipeline_task_error(&PipelineError::InvalidLandmark {
        path: path(),
        message: "expected 110 lines, found 109".to_owned(),
    });
    assert_eq!(malformed.summary, "关键点文件不可用");
    malformed.validate().unwrap();
}

#[test]
fn a_rejected_quality_report_reports_the_first_anomaly() {
    let anomalies = vec![
        anomaly(7, AnomalyCode::LandmarkInvalid),
        anomaly(8, AnomalyCode::FaceNotFound),
        anomaly(9, AnomalyCode::BlurredFrame),
        anomaly(10, AnomalyCode::ModelFailed),
    ];

    let mapped = quality_task_error(&anomalies);

    assert_eq!(mapped.code, ErrorCode::LandmarkInvalid);
    assert_eq!(mapped.stage, TaskStage::Preparing);
    assert!(
        mapped.detail.contains("4 frame(s) rejected"),
        "{}",
        mapped.detail
    );
    // Only the first three are named, so a run with thousands of bad frames
    // still produces a readable detail.
    assert!(mapped.detail.contains("frame 7"), "{}", mapped.detail);
    assert!(mapped.detail.contains("frame 9"), "{}", mapped.detail);
    assert!(!mapped.detail.contains("frame 10"), "{}", mapped.detail);
    mapped.validate().unwrap();
}

#[test]
fn an_empty_anomaly_list_still_produces_a_valid_error() {
    let mapped = quality_task_error(&[]);

    assert_eq!(mapped.code, ErrorCode::MediaInvalid);
    mapped.validate().unwrap();
}

#[test]
fn audio_errors_map_onto_wire_codes() {
    let cases = vec![
        (AudioError::InvalidRiffHeader, ErrorCode::MediaInvalid),
        (
            AudioError::UnsupportedWavSampleRate {
                actual: 44_100,
                expected: 16_000,
            },
            ErrorCode::MediaInvalid,
        ),
        (AudioError::EmptyWav, ErrorCode::MediaInvalid),
        (AudioError::ConstantWaveform, ErrorCode::MediaInvalid),
        (
            AudioError::WavIo {
                operation: "read",
                path: path(),
                source: io_error(io::ErrorKind::StorageFull),
            },
            ErrorCode::DiskSpaceLow,
        ),
        (
            AudioError::WavIo {
                operation: "read",
                path: path(),
                source: io_error(io::ErrorKind::PermissionDenied),
            },
            ErrorCode::WorkerCrashed,
        ),
        (
            AudioError::InvalidFeatureDimension,
            ErrorCode::ModelIncompatible,
        ),
        (
            AudioError::FeatureShapeMismatch {
                frame_count: 4,
                tokens: 7,
                dims: 1024,
            },
            ErrorCode::FeatureShapeMismatch,
        ),
        (
            AudioError::TooManyChunks {
                actual: 2_000_000,
                limit: 1_000_000,
            },
            ErrorCode::WorkerCrashed,
        ),
        (
            AudioError::Cancelled {
                operation: "extract_features",
            },
            ErrorCode::TaskCancelled,
        ),
    ];

    for (error, expected) in cases {
        let mapped = audio_task_error(&error);
        assert_eq!(mapped.code, expected, "{error:?}");
        assert_eq!(mapped.stage, TaskStage::Preparing, "{error:?}");
        assert_eq!(mapped.recovery, expected.default_recovery(), "{error:?}");
        assert!(!mapped.summary.trim().is_empty(), "{error:?}");
        mapped.validate().unwrap();
    }
}

#[test]
fn only_cancellation_is_audio_cancellation() {
    let cancelled = AudioError::Cancelled {
        operation: "extract_features",
    };

    assert!(is_audio_cancellation(&cancelled));
    assert!(!is_audio_cancellation(&AudioError::EmptyWav));
}

#[test]
fn a_package_failure_names_the_hubert_variable() {
    // Not an I/O failure, and still ModelIncompatible: the request carried no
    // path, so the directory is the only thing a user can act on.
    let error = PackageError::InvalidRequest("no manifest".to_owned());

    let mapped = package_task_error(&error);

    assert_eq!(mapped.code, ErrorCode::ModelIncompatible);
    assert_eq!(mapped.summary, "特征模型加载失败");
    assert_eq!(mapped.stage, TaskStage::Preparing);
    assert!(
        mapped.detail.contains("FEATHERTALK_WORKER_HUBERT_DIR"),
        "{}",
        mapped.detail
    );
    mapped.validate().unwrap();
}

#[test]
fn legacy_import_failures_use_the_reimport_recovery() {
    let package = PackageError::InvalidRequest("synthetic".to_owned());
    let detail = package.to_string();
    let mapped = legacy_task_error(&detail, TaskStage::Importing);
    assert_eq!(mapped.code, ErrorCode::ModelIncompatible);
    assert_eq!(mapped.summary, "模型导入失败");
    assert_eq!(mapped.stage, TaskStage::Importing);
    assert_eq!(mapped.recovery, feathertalk_domain::Recovery::ReimportModel);
    mapped.validate().unwrap();
}

fn try_reserve_error() -> std::collections::TryReserveError {
    Vec::<u64>::new()
        .try_reserve_exact(usize::MAX)
        .expect_err("an impossible reservation fails")
}

#[test]
fn every_training_error_maps_to_a_code_and_a_valid_payload() {
    let cases = vec![
        (
            TrainingError::Io(io_error(io::ErrorKind::StorageFull)),
            ErrorCode::DiskSpaceLow,
        ),
        (
            TrainingError::Io(io_error(io::ErrorKind::PermissionDenied)),
            ErrorCode::WorkerCrashed,
        ),
        (
            TrainingError::InvalidInput("loss is not finite".to_owned()),
            ErrorCode::MediaInvalid,
        ),
        (
            TrainingError::InvalidConfig("batch_size".to_owned()),
            ErrorCode::WorkerCrashed,
        ),
        (
            TrainingError::InvalidDataLoaderConfig("stride".to_owned()),
            ErrorCode::WorkerCrashed,
        ),
        (
            TrainingError::InvalidDataLoaderState("epoch".to_owned()),
            ErrorCode::WorkerCrashed,
        ),
        (
            TrainingError::DataLoaderOverflow {
                operation: "counting steps",
            },
            ErrorCode::WorkerCrashed,
        ),
        (
            TrainingError::PermutationAllocation {
                samples: 8,
                source: try_reserve_error(),
            },
            ErrorCode::WorkerCrashed,
        ),
        (
            TrainingError::BatchAllocation {
                items: 2,
                source: try_reserve_error(),
            },
            ErrorCode::WorkerCrashed,
        ),
        (TrainingError::StalePreparedBatch, ErrorCode::WorkerCrashed),
        (
            TrainingError::InvalidPackage("manifest".to_owned()),
            ErrorCode::ModelIncompatible,
        ),
        (
            TrainingError::HashMismatch {
                file: "model.safetensors".to_owned(),
                expected: "a".repeat(64),
                actual: "b".repeat(64),
            },
            ErrorCode::ModelIncompatible,
        ),
        (
            TrainingError::Store("record write failed".to_owned()),
            ErrorCode::WorkerCrashed,
        ),
        (
            TrainingError::InvalidCheckpoint("manifest".to_owned()),
            ErrorCode::ModelIncompatible,
        ),
        (
            TrainingError::CheckpointCompatibility("frame_count".to_owned()),
            ErrorCode::ModelIncompatible,
        ),
        (
            TrainingError::CheckpointDirectory("already exists".to_owned()),
            ErrorCode::MediaInvalid,
        ),
    ];

    for (error, expected) in cases {
        let mapped = training_task_error(&error, TaskStage::Preparing);
        assert_eq!(mapped.code, expected, "{error:?}");
        assert_eq!(mapped.stage, TaskStage::Preparing, "{error:?}");
        assert!(!mapped.summary.trim().is_empty(), "{error:?}");
        assert!(!mapped.detail.is_empty(), "{error:?}");
        mapped.validate().unwrap();
    }
}

#[test]
fn a_mid_run_training_failure_keeps_the_stage_it_failed_in() {
    let stage = TaskStage::Training {
        epoch: 3,
        step: 3000,
        loss: 0.125,
    };
    let mapped = training_task_error(&TrainingError::StalePreparedBatch, stage.clone());

    assert_eq!(mapped.stage, stage);
    mapped.validate().unwrap();
}

#[test]
fn a_long_training_detail_is_clamped() {
    let error = TrainingError::InvalidInput("x".repeat(MAX_DETAIL_CHARS * 2));
    let mapped = training_task_error(&error, TaskStage::Preparing);

    assert_eq!(mapped.detail.chars().count(), MAX_DETAIL_CHARS);
    mapped.validate().unwrap();
}

#[test]
fn every_training_data_error_maps_to_a_code_and_a_valid_payload() {
    let cases = vec![
        (
            TrainingDataError::Project {
                path: path(),
                message: "not locked".to_owned(),
            },
            ErrorCode::MediaInvalid,
        ),
        (
            TrainingDataError::Features {
                path: path(),
                message: "truncated".to_owned(),
            },
            ErrorCode::MediaInvalid,
        ),
        (
            TrainingDataError::FeatureShape {
                path: path(),
                expected_tokens: 8,
                actual_tokens: 98,
                dims: 1024,
            },
            ErrorCode::FeatureShapeMismatch,
        ),
        (
            TrainingDataError::FrameIndexOutOfRange {
                index: 9,
                frame_count: 4,
            },
            ErrorCode::WorkerCrashed,
        ),
        (
            TrainingDataError::Frame {
                index: 0,
                path: path(),
                message: "not a jpeg".to_owned(),
            },
            ErrorCode::MediaInvalid,
        ),
        (
            TrainingDataError::Landmarks {
                index: 0,
                path: path(),
                message: "short line".to_owned(),
            },
            ErrorCode::MediaInvalid,
        ),
        (
            TrainingDataError::Sample {
                index: 0,
                message: "image plane".to_owned(),
            },
            ErrorCode::MediaInvalid,
        ),
        (
            TrainingDataError::Batch {
                message: "shape".to_owned(),
            },
            ErrorCode::WorkerCrashed,
        ),
    ];

    for (error, expected) in cases {
        let mapped = training_data_task_error(&error);
        assert_eq!(mapped.code, expected, "{error:?}");
        assert_eq!(mapped.stage, TaskStage::Preparing, "{error:?}");
        assert!(!mapped.summary.trim().is_empty(), "{error:?}");
        mapped.validate().unwrap();
    }
}

/// One row per `InferenceError` variant, in the declaration order of
/// `feathertalk-inference/src/error.rs`. A `Vec` gives no exhaustiveness check,
/// so a forgotten row here is a mapping nobody ever ran.
#[test]
fn every_inference_error_maps_to_a_render_task_error() {
    let stage = TaskStage::Rendering { frame: 3, total: 8 };
    let cases: Vec<(InferenceError, ErrorCode)> = vec![
        (
            InferenceError::InvalidInputDirectory {
                field: "frame_dir",
                path: path(),
            },
            ErrorCode::MediaInvalid,
        ),
        (
            InferenceError::InvalidInputArtifact {
                field: "landmark_path",
                path: path(),
                message: "bad".to_owned(),
            },
            ErrorCode::LandmarkInvalid,
        ),
        (
            InferenceError::InvalidInputArtifact {
                field: "feature_path",
                path: path(),
                message: "bad".to_owned(),
            },
            ErrorCode::MediaInvalid,
        ),
        (
            InferenceError::FrameIndexOutOfRange { index: 4, count: 2 },
            ErrorCode::MediaInvalid,
        ),
        (
            InferenceError::FrameDimensionsMismatch {
                index: 1,
                expected_width: 1280,
                expected_height: 720,
                actual_width: 640,
                actual_height: 480,
            },
            ErrorCode::MediaInvalid,
        ),
        (
            InferenceError::FrameReader {
                index: 0,
                path: path(),
                message: "decode".to_owned(),
            },
            ErrorCode::WorkerCrashed,
        ),
        (
            InferenceError::SinkStart {
                message: "spawn".to_owned(),
            },
            ErrorCode::WorkerCrashed,
        ),
        (
            InferenceError::SinkWrite {
                message: "broken pipe".to_owned(),
            },
            ErrorCode::WorkerCrashed,
        ),
        (
            InferenceError::SinkFinish {
                message: "exit".to_owned(),
            },
            ErrorCode::WorkerCrashed,
        ),
        (
            InferenceError::StagingCollision { path: path() },
            ErrorCode::WorkerCrashed,
        ),
        (
            InferenceError::StagingOutputInvalid {
                path: path(),
                message: "empty".to_owned(),
            },
            ErrorCode::WorkerCrashed,
        ),
        (
            InferenceError::AtomicPublishFailed {
                path: path(),
                message: "rename".to_owned(),
            },
            ErrorCode::WorkerCrashed,
        ),
        (
            InferenceError::ToolFailed {
                operation: "render",
                exit_code: Some(1),
                stderr: "ffmpeg".to_owned(),
            },
            ErrorCode::WorkerCrashed,
        ),
        (
            InferenceError::FrameCountTooSmall {
                actual: 1,
                minimum: 2,
            },
            ErrorCode::MediaInvalid,
        ),
        (InferenceError::EmptyFeatures, ErrorCode::MediaInvalid),
        (
            InferenceError::OutputFrameOutOfRange { index: 9, count: 2 },
            ErrorCode::MediaInvalid,
        ),
        (
            InferenceError::InvalidField {
                field: "task_id",
                message: "empty".to_owned(),
            },
            ErrorCode::MediaInvalid,
        ),
        (InferenceError::ArithmeticOverflow, ErrorCode::MediaInvalid),
        (
            InferenceError::OutputExists { path: path() },
            ErrorCode::MediaInvalid,
        ),
        (
            InferenceError::OutputNotRegular { path: path() },
            ErrorCode::MediaInvalid,
        ),
        (
            InferenceError::OutputSymlink { path: path() },
            ErrorCode::MediaInvalid,
        ),
        (
            InferenceError::OutputParentInvalid { path: path() },
            ErrorCode::MediaInvalid,
        ),
        (
            InferenceError::InvalidTaskId {
                task_id: "..".to_owned(),
            },
            ErrorCode::MediaInvalid,
        ),
        (
            InferenceError::FfmpegPathNotAbsolute { path: path() },
            ErrorCode::MediaInvalid,
        ),
        (InferenceError::EmptyFfmpegPath, ErrorCode::MediaInvalid),
        (
            InferenceError::InvalidFrameDimensions {
                width: 0,
                height: 720,
            },
            ErrorCode::MediaInvalid,
        ),
        (
            InferenceError::FrameBufferLengthMismatch {
                expected: 100,
                actual: 99,
            },
            ErrorCode::MediaInvalid,
        ),
        (
            InferenceError::PixelOutOfRange {
                x: 9,
                y: 9,
                width: 4,
                height: 4,
            },
            ErrorCode::MediaInvalid,
        ),
        (
            InferenceError::InvalidBbox {
                xmin: 4,
                ymin: 4,
                xmax: 0,
                ymax: 0,
                frame_width: 8,
                frame_height: 8,
            },
            ErrorCode::LandmarkInvalid,
        ),
        (
            InferenceError::InvalidResizeTarget {
                width: 0,
                height: 0,
            },
            ErrorCode::MediaInvalid,
        ),
        (
            InferenceError::TensorShapeMismatch {
                context: "audio",
                expected: vec![1, 32, 32, 32],
                actual: vec![1, 2, 3, 4],
            },
            ErrorCode::ModelIncompatible,
        ),
        (
            InferenceError::InvalidFeatureShape {
                tokens: 3,
                dims: 1024,
            },
            ErrorCode::FeatureShapeMismatch,
        ),
        (
            InferenceError::InvalidAudioWindowIndex {
                slot: 1,
                index: 9,
                frame_count: 2,
            },
            ErrorCode::MediaInvalid,
        ),
        (
            InferenceError::NonFiniteModelInput {
                context: "reference",
                index: 7,
            },
            ErrorCode::ModelIncompatible,
        ),
        (
            InferenceError::ModelTensorData {
                context: "prediction",
                message: "read".to_owned(),
            },
            ErrorCode::ModelIncompatible,
        ),
        (
            InferenceError::NonFiniteModelOutput { index: 3 },
            ErrorCode::ModelIncompatible,
        ),
        (
            InferenceError::ModelOutputOutOfRange {
                index: 3,
                value: 1.5,
            },
            ErrorCode::ModelIncompatible,
        ),
        (
            InferenceError::NonFinitePrediction { index: 2 },
            ErrorCode::ModelIncompatible,
        ),
        (
            InferenceError::PasteOutOfBounds {
                x: 9,
                y: 9,
                source_width: 4,
                source_height: 4,
                destination_width: 2,
                destination_height: 2,
            },
            ErrorCode::MediaInvalid,
        ),
        (
            InferenceError::AllocationFailure { bytes: 1 << 40 },
            ErrorCode::WorkerCrashed,
        ),
        (
            InferenceError::Cancelled {
                operation: "render",
            },
            ErrorCode::TaskCancelled,
        ),
    ];

    for (error, expected) in cases {
        let mapped = render_task_error(&error, stage.clone());
        assert_eq!(mapped.code, expected, "{error:?}");
        mapped.validate().unwrap();
        assert!(!mapped.summary.trim().is_empty(), "{error:?}");
        // Summaries are user-facing, so they are Chinese, never English.
        assert!(!mapped.summary.is_ascii(), "{}", mapped.summary);
        assert!(!mapped.detail.trim().is_empty(), "{error:?}");
        // The stage is the caller's, echoed rather than replaced.
        assert_eq!(mapped.stage, stage, "{error:?}");
    }
}

#[test]
fn only_a_cancelled_render_counts_as_a_cancellation() {
    assert!(is_inference_cancellation(&InferenceError::Cancelled {
        operation: "render",
    }));
    assert!(!is_inference_cancellation(&InferenceError::EmptyFeatures));
    assert!(!is_inference_cancellation(
        &InferenceError::ArithmeticOverflow
    ));
}
