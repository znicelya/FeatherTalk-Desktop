//! Locate the worker, run one task, choose the exit code.

use std::path::Path;

use feathertalk_client::{
    CancelToken, ComputeOptions, EventSink, SessionOptions, SessionOutcome, WorkerLocator,
    WorkerSession, generate_task_id,
};
use feathertalk_domain::{
    ExportModelPackageParams, ExportOnnxParams, ExtractFeaturesParams, ExtractFramesParams,
    ImportLegacyModelParams, InspectModelParams, LegacyModelKind, MigrateLegacyFeaturesParams,
    NormalizeMediaParams, OnnxExportKind, ProbeMediaParams, ProjectDirParams, RenderParams,
    Request, TaskId, TrainParams, TrainingMode, UnetVariant,
};

use crate::cli::{Cli, Command, LegacyModelKindArg, OnnxExportKindArg, TrainMode, TrainVariant};
use crate::render::{HumanSink, JsonSink, capabilities_report, failure_block, render_client_error};

/// The four exit codes, fixed by the spec. Nothing else is ever returned.
pub const EXIT_COMPLETED: i32 = 0;
pub const EXIT_TASK_FAILED: i32 = 1;
pub const EXIT_CANCELLED: i32 = 2;
pub const EXIT_SESSION_ERROR: i32 = 3;

pub fn run(cli: Cli) -> i32 {
    let compute =
        match ComputeOptions::from_flags(cli.backend.map(Into::into), cli.adapter.as_deref()) {
            Ok(compute) => compute,
            Err(error) => {
                eprintln!("计算设备配置无效：{error}");
                return EXIT_SESSION_ERROR;
            }
        };
    let request = match build_request(&cli.command) {
        Ok(request) => request,
        Err(message) => {
            eprintln!("{message}");
            return EXIT_SESSION_ERROR;
        }
    };
    let path = match WorkerLocator::from_env(cli.worker.clone()).resolve() {
        Ok(path) => path,
        Err(error) => {
            eprintln!("{}", render_client_error(&error));
            return EXIT_SESSION_ERROR;
        }
    };
    // No flags preserve environment configuration. An explicit choice overrides
    // both compute keys only on the child, clearing any inherited adapter.
    let env = compute
        .as_ref()
        .map(ComputeOptions::env_overrides)
        .unwrap_or_default();
    let mut session = match WorkerSession::spawn_with_env(&path, SessionOptions::default(), &env) {
        Ok(session) => session,
        Err(error) => {
            eprintln!("{}", render_client_error(&error));
            return EXIT_SESSION_ERROR;
        }
    };
    if cli.json {
        // The handshake is a protocol frame too, so machine consumers get it.
        println!("{}", session.ready_raw());
    }
    let code = match request {
        // `capabilities` needs no task: the handshake already answered it.
        None => {
            if !cli.json {
                println!("{}", capabilities_report(session.ready()));
            }
            EXIT_COMPLETED
        }
        Some(request) => run_task(&mut session, &cli, request),
    };
    let _ = session.shutdown();
    code
}

/// Build the request, or `None` for `capabilities`.
///
/// Only empty arguments are rejected here. Whether a path exists, is a project,
/// or is decodable media is the worker's judgement, and duplicating it in the
/// CLI would produce two answers that can disagree.
fn build_request(command: &Command) -> Result<Option<Request>, String> {
    match command {
        Command::Capabilities => Ok(None),
        Command::ValidateProject { project_dir } => {
            reject_empty(project_dir, "工程目录")?;
            Ok(Some(Request::ValidateProject(ProjectDirParams {
                project_dir: project_dir.clone(),
            })))
        }
        Command::ProbeMedia { input } => {
            reject_empty(input, "输入文件")?;
            Ok(Some(Request::ProbeMedia(ProbeMediaParams {
                input: input.clone(),
            })))
        }
        Command::NormalizeMedia { input, output_dir } => {
            reject_empty(input, "输入文件")?;
            reject_empty(output_dir, "输出目录")?;
            Ok(Some(Request::NormalizeMedia(NormalizeMediaParams {
                input: input.clone(),
                output_dir: output_dir.clone(),
            })))
        }
        Command::ExtractFrames { project_dir, video } => {
            reject_empty(project_dir, "工程目录")?;
            reject_empty(video, "输入文件")?;
            Ok(Some(Request::ExtractFrames(ExtractFramesParams {
                project_dir: project_dir.clone(),
                video: video.clone(),
            })))
        }
        Command::ExtractFeatures { project_dir, audio } => {
            reject_empty(project_dir, "工程目录")?;
            reject_empty(audio, "音频文件")?;
            Ok(Some(Request::ExtractFeatures(ExtractFeaturesParams {
                project_dir: project_dir.clone(),
                audio: audio.clone(),
            })))
        }
        Command::LockAssetPackage { project_dir } => {
            reject_empty(project_dir, "工程目录")?;
            Ok(Some(Request::LockAssetPackage(ProjectDirParams {
                project_dir: project_dir.clone(),
            })))
        }
        Command::Train {
            project_dir,
            mode,
            variant,
            epochs,
            batch_size,
            resume,
        } => {
            reject_empty(project_dir, "工程目录")?;
            // The epoch range is the worker's judgement, like every path here.
            Ok(Some(Request::Train(TrainParams {
                project_dir: project_dir.clone(),
                mode: training_mode(*mode),
                variant: unet_variant(*variant),
                epochs: *epochs,
                batch_size: *batch_size,
                resume: *resume,
            })))
        }
        Command::Render {
            project_dir,
            checkpoint,
            audio,
            output,
            max_output_frames,
        } => {
            reject_empty(project_dir, "工程目录")?;
            reject_empty(checkpoint, "检查点目录")?;
            reject_empty(audio, "音频文件")?;
            reject_empty(output, "输出文件")?;
            // Whether these paths are absolute, and whether the cap is a number
            // this machine can render, is the worker's judgement.
            Ok(Some(Request::Render(RenderParams {
                project_dir: project_dir.clone(),
                checkpoint: checkpoint.clone(),
                audio: audio.clone(),
                output: output.clone(),
                max_output_frames: *max_output_frames,
            })))
        }
        Command::InspectModel { source } => {
            reject_empty(source, "模型目录")?;
            // Which of the two layouts this directory is, and whether the path is
            // absolute, is the worker's judgement.
            Ok(Some(Request::InspectModel(InspectModelParams {
                source: source.clone(),
            })))
        }
        Command::ImportLegacyModel {
            source,
            kind,
            destination,
        } => {
            reject_empty(source, "旧模型文件")?;
            reject_empty(destination, "目标目录")?;
            Ok(Some(Request::ImportLegacyModel(ImportLegacyModelParams {
                source: source.clone(),
                kind: legacy_model_kind(*kind),
                destination: destination.clone(),
            })))
        }
        Command::MigrateLegacyFeatures {
            source,
            destination,
        } => {
            reject_empty(source, "旧特征文件")?;
            reject_empty(destination, "目标文件")?;
            Ok(Some(Request::MigrateLegacyFeatures(
                MigrateLegacyFeaturesParams {
                    source: source.clone(),
                    destination: destination.clone(),
                },
            )))
        }
        Command::ExportModelPackage {
            source,
            destination,
        } => {
            reject_empty(source, "检查点目录")?;
            reject_empty(destination, "目标目录")?;
            Ok(Some(Request::ExportModelPackage(
                ExportModelPackageParams {
                    source: source.clone(),
                    destination: destination.clone(),
                },
            )))
        }
        Command::ExportOnnx {
            source,
            kind,
            destination,
        } => {
            reject_empty(source, "模型包目录")?;
            reject_empty(destination, "目标文件")?;
            Ok(Some(Request::ExportOnnx(ExportOnnxParams {
                source: source.clone(),
                kind: onnx_export_kind(*kind),
                destination: destination.clone(),
            })))
        }
    }
}

fn reject_empty(path: &Path, label: &str) -> Result<(), String> {
    if path.as_os_str().is_empty() {
        return Err(format!("{label}不能为空。"));
    }
    Ok(())
}

/// The only place where clap's kebab-cased values meet the domain enums.
fn training_mode(mode: TrainMode) -> TrainingMode {
    match mode {
        TrainMode::Baseline => TrainingMode::Baseline,
        TrainMode::MouthRoi => TrainingMode::MouthRoi,
        TrainMode::Temporal => TrainingMode::Temporal,
    }
}

fn unet_variant(variant: TrainVariant) -> UnetVariant {
    match variant {
        TrainVariant::OriginalUnet => UnetVariant::OriginalUnet,
        TrainVariant::MobileOneUnet => UnetVariant::MobileOneUnet,
    }
}

fn legacy_model_kind(kind: LegacyModelKindArg) -> LegacyModelKind {
    match kind {
        LegacyModelKindArg::FeatherHubert => LegacyModelKind::FeatherHubert,
        LegacyModelKindArg::Pfld => LegacyModelKind::Pfld,
        LegacyModelKindArg::OriginalUnet => LegacyModelKind::OriginalUnet,
        LegacyModelKindArg::MobileOneUnet => LegacyModelKind::MobileOneUnet,
    }
}

fn onnx_export_kind(kind: OnnxExportKindArg) -> OnnxExportKind {
    match kind {
        OnnxExportKindArg::FeatherHubert => OnnxExportKind::FeatherHubert,
        OnnxExportKindArg::OriginalUnet => OnnxExportKind::OriginalUnet,
        OnnxExportKindArg::MobileOneUnet => OnnxExportKind::MobileOneUnet,
    }
}

fn run_task(session: &mut WorkerSession, cli: &Cli, request: Request) -> i32 {
    let task_id = match resolve_task_id(cli.task_id.as_deref()) {
        Ok(task_id) => task_id,
        Err(message) => {
            eprintln!("{message}");
            return EXIT_SESSION_ERROR;
        }
    };
    let cancel = CancelToken::new();
    install_cancel_handler(&cancel);
    let mut human = HumanSink::new(cli.quiet);
    let mut json = JsonSink;
    let sink: &mut dyn EventSink = if cli.json { &mut json } else { &mut human };
    match session.run(task_id, request, &cancel, sink) {
        SessionOutcome::Completed { result } => {
            // Under --json the completed frame already carried the result.
            if !cli.json {
                let value = result.unwrap_or_else(|| serde_json::json!({}));
                let text =
                    serde_json::to_string_pretty(&value).unwrap_or_else(|_| value.to_string());
                println!("{text}");
            }
            EXIT_COMPLETED
        }
        SessionOutcome::Failed(error) => {
            eprintln!("{}", failure_block(&error));
            EXIT_TASK_FAILED
        }
        SessionOutcome::Cancelled => {
            eprintln!("任务已取消。");
            EXIT_CANCELLED
        }
        SessionOutcome::SessionError(error) => {
            eprintln!("{}", render_client_error(&error));
            EXIT_SESSION_ERROR
        }
    }
}

fn resolve_task_id(requested: Option<&str>) -> Result<TaskId, String> {
    match requested {
        Some(text) => TaskId::parse(text).map_err(|error| {
            format!("任务 ID 无效：{error}\n格式为 13 位毫秒时间戳、连字符、8 位小写十六进制。")
        }),
        None => generate_task_id().map_err(|error| format!("无法生成任务 ID：{error}")),
    }
}

/// Ctrl-C bumps the token and does nothing else: one atomic add, no allocation,
/// no locks. All the escalation lives in the client's run loop.
///
/// Failing to install is reported and ignored — the task can still run, it just
/// cannot be interrupted politely, and refusing to work would be worse.
fn install_cancel_handler(cancel: &CancelToken) {
    let token = cancel.clone();
    if let Err(error) = ctrlc::set_handler(move || token.request()) {
        eprintln!("无法注册 Ctrl-C 处理器：{error}");
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;

    #[test]
    fn capabilities_needs_no_task() {
        assert!(
            build_request(&Command::Capabilities)
                .expect("capabilities always builds")
                .is_none()
        );
    }

    #[test]
    fn an_empty_path_is_rejected_in_chinese() {
        // Clap catches this first on the command line, but `run` is a library
        // entry point too, so the guard has to hold on its own.
        let error = build_request(&Command::ValidateProject {
            project_dir: PathBuf::new(),
        })
        .expect_err("an empty project directory is refused");
        assert_eq!(error, "工程目录不能为空。");

        let error = build_request(&Command::ProbeMedia {
            input: PathBuf::new(),
        })
        .expect_err("an empty input is refused");
        assert_eq!(error, "输入文件不能为空。");
    }

    #[test]
    fn normalize_media_refuses_empty_arguments() {
        let error = build_request(&Command::NormalizeMedia {
            input: PathBuf::new(),
            output_dir: PathBuf::from("assets"),
        })
        .expect_err("an empty input is refused");
        assert_eq!(error, "输入文件不能为空。");

        let error = build_request(&Command::NormalizeMedia {
            input: PathBuf::from("clip.mp4"),
            output_dir: PathBuf::new(),
        })
        .expect_err("an empty output directory is refused");
        assert_eq!(error, "输出目录不能为空。");
    }

    #[test]
    fn extract_frames_refuses_empty_arguments() {
        let error = build_request(&Command::ExtractFrames {
            project_dir: PathBuf::new(),
            video: PathBuf::from("project/assets/video_25fps.mp4"),
        })
        .expect_err("an empty project directory is refused");
        assert_eq!(error, "工程目录不能为空。");

        let error = build_request(&Command::ExtractFrames {
            project_dir: PathBuf::from("project"),
            video: PathBuf::new(),
        })
        .expect_err("an empty video is refused");
        assert_eq!(error, "输入文件不能为空。");
    }

    #[test]
    fn extract_frames_carries_both_paths() {
        let request = build_request(&Command::ExtractFrames {
            project_dir: PathBuf::from("project"),
            video: PathBuf::from("project/assets/video_25fps.mp4"),
        })
        .expect("both paths are accepted")
        .expect("extract-frames needs a task");
        let Request::ExtractFrames(params) = request else {
            panic!("extract-frames must build an ExtractFrames request");
        };
        assert_eq!(params.project_dir, PathBuf::from("project"));
        assert_eq!(
            params.video,
            PathBuf::from("project/assets/video_25fps.mp4")
        );
    }

    #[test]
    fn extract_features_refuses_empty_arguments() {
        let error = build_request(&Command::ExtractFeatures {
            project_dir: PathBuf::new(),
            audio: PathBuf::from("project/assets/audio_16k_mono.wav"),
        })
        .expect_err("an empty project directory is refused");
        assert_eq!(error, "工程目录不能为空。");

        let error = build_request(&Command::ExtractFeatures {
            project_dir: PathBuf::from("project"),
            audio: PathBuf::new(),
        })
        .expect_err("an empty audio file is refused");
        assert_eq!(error, "音频文件不能为空。");
    }

    #[test]
    fn extract_features_carries_both_paths() {
        let request = build_request(&Command::ExtractFeatures {
            project_dir: PathBuf::from("project"),
            audio: PathBuf::from("project/assets/audio_16k_mono.wav"),
        })
        .expect("both paths are accepted")
        .expect("extract-features needs a task");
        let Request::ExtractFeatures(params) = request else {
            panic!("extract-features must build an ExtractFeatures request");
        };
        assert_eq!(params.project_dir, PathBuf::from("project"));
        assert_eq!(
            params.audio,
            PathBuf::from("project/assets/audio_16k_mono.wav")
        );
    }

    #[test]
    fn lock_asset_package_refuses_an_empty_project_directory() {
        let error = build_request(&Command::LockAssetPackage {
            project_dir: PathBuf::new(),
        })
        .expect_err("an empty project directory is refused");
        assert_eq!(error, "工程目录不能为空。");
    }

    #[test]
    fn lock_asset_package_carries_the_project_directory() {
        let request = build_request(&Command::LockAssetPackage {
            project_dir: PathBuf::from("project"),
        })
        .expect("the path is accepted")
        .expect("lock-asset-package needs a task");
        let Request::LockAssetPackage(params) = request else {
            panic!("lock-asset-package must build a LockAssetPackage request");
        };
        assert_eq!(params.project_dir, PathBuf::from("project"));
    }

    #[test]
    fn a_malformed_task_id_explains_the_format() {
        let error = resolve_task_id(Some("nope")).expect_err("a short id is refused");
        assert!(error.contains("任务 ID 无效"), "{error}");
        assert!(error.contains("13 位毫秒时间戳"), "{error}");
    }

    #[test]
    fn train_refuses_an_empty_project_directory() {
        let error = build_request(&Command::Train {
            project_dir: PathBuf::new(),
            mode: TrainMode::Baseline,
            variant: TrainVariant::OriginalUnet,
            epochs: 1,
            batch_size: 1,
            resume: false,
        })
        .expect_err("an empty project directory is refused");
        assert_eq!(error, "工程目录不能为空。");
    }

    #[test]
    fn train_carries_every_flag_into_the_request() {
        let request = build_request(&Command::Train {
            project_dir: PathBuf::from("project"),
            mode: TrainMode::Temporal,
            variant: TrainVariant::MobileOneUnet,
            epochs: 3,
            batch_size: 4,
            resume: true,
        })
        .expect("the arguments are accepted")
        .expect("train needs a task");
        let Request::Train(params) = request else {
            panic!("train must build a Train request");
        };
        assert_eq!(params.project_dir, PathBuf::from("project"));
        assert_eq!(params.mode, TrainingMode::Temporal);
        assert_eq!(params.variant, UnetVariant::MobileOneUnet);
        assert_eq!(params.epochs, 3);
        assert_eq!(params.batch_size, 4);
        assert!(params.resume);
    }

    #[test]
    fn train_batch_size_flag_and_default_reach_the_request() {
        use clap::Parser;
        for (args, expected) in [
            (vec!["feathertalk", "train", "project", "--epochs", "1"], 1),
            (
                vec![
                    "feathertalk",
                    "train",
                    "project",
                    "--epochs",
                    "1",
                    "--batch-size",
                    "4",
                ],
                4,
            ),
        ] {
            let cli = Cli::try_parse_from(args).expect("valid training arguments");
            let Some(Request::Train(params)) = build_request(&cli.command).unwrap() else {
                panic!("expected a training request");
            };
            assert_eq!(params.batch_size, expected);
        }
    }

    #[test]
    fn train_batch_size_rejects_non_integer_cli_values() {
        use clap::Parser;
        for value in ["-1", "1.5", "NaN", "4294967296"] {
            assert!(
                Cli::try_parse_from([
                    "feathertalk",
                    "train",
                    "project",
                    "--epochs",
                    "1",
                    "--batch-size",
                    value,
                ])
                .is_err(),
                "{value}"
            );
        }
    }

    #[test]
    fn an_out_of_range_epoch_count_is_left_to_the_worker() {
        // The CLI does not know `MAX_EPOCHS`, and two answers that can disagree
        // are worse than one; the worker rejects it with a Chinese summary.
        let request = build_request(&Command::Train {
            project_dir: PathBuf::from("project"),
            mode: TrainMode::Baseline,
            variant: TrainVariant::OriginalUnet,
            epochs: 0,
            batch_size: 1,
            resume: false,
        })
        .expect("zero epochs still builds a request");
        assert!(request.is_some());
    }

    #[test]
    fn every_mirrored_value_maps_onto_the_domain() {
        assert_eq!(training_mode(TrainMode::Baseline), TrainingMode::Baseline);
        assert_eq!(training_mode(TrainMode::MouthRoi), TrainingMode::MouthRoi);
        assert_eq!(training_mode(TrainMode::Temporal), TrainingMode::Temporal);
        assert_eq!(
            unet_variant(TrainVariant::OriginalUnet),
            UnetVariant::OriginalUnet
        );
        assert_eq!(
            unet_variant(TrainVariant::MobileOneUnet),
            UnetVariant::MobileOneUnet
        );
    }

    /// The four paths a render names, with one of them blanked out.
    fn render_with(
        project_dir: PathBuf,
        checkpoint: PathBuf,
        audio: PathBuf,
        output: PathBuf,
    ) -> Command {
        Command::Render {
            project_dir,
            checkpoint,
            audio,
            output,
            max_output_frames: None,
        }
    }

    #[test]
    fn render_refuses_an_empty_path_by_name() {
        // One case per path, because the label is what tells the operator which
        // of four arguments was the empty one.
        let cases = [
            (
                render_with(
                    PathBuf::new(),
                    PathBuf::from("checkpoint"),
                    PathBuf::from("voice.wav"),
                    PathBuf::from("preview.mp4"),
                ),
                "工程目录不能为空。",
            ),
            (
                render_with(
                    PathBuf::from("project"),
                    PathBuf::new(),
                    PathBuf::from("voice.wav"),
                    PathBuf::from("preview.mp4"),
                ),
                "检查点目录不能为空。",
            ),
            (
                render_with(
                    PathBuf::from("project"),
                    PathBuf::from("checkpoint"),
                    PathBuf::new(),
                    PathBuf::from("preview.mp4"),
                ),
                "音频文件不能为空。",
            ),
            (
                render_with(
                    PathBuf::from("project"),
                    PathBuf::from("checkpoint"),
                    PathBuf::from("voice.wav"),
                    PathBuf::new(),
                ),
                "输出文件不能为空。",
            ),
        ];

        for (command, expected) in cases {
            let error = build_request(&command).expect_err("an empty path is refused");
            assert_eq!(error, expected);
        }
    }

    #[test]
    fn render_carries_every_flag_into_the_request() {
        let request = build_request(&Command::Render {
            project_dir: PathBuf::from("project"),
            checkpoint: PathBuf::from("project/models/unet/checkpoint-00000004"),
            audio: PathBuf::from("voice.wav"),
            output: PathBuf::from("preview.mp4"),
            max_output_frames: Some(2),
        })
        .expect("the arguments are accepted")
        .expect("render needs a task");
        let Request::Render(params) = request else {
            panic!("render must build a Render request");
        };
        assert_eq!(params.project_dir, PathBuf::from("project"));
        assert_eq!(
            params.checkpoint,
            PathBuf::from("project/models/unet/checkpoint-00000004")
        );
        assert_eq!(params.audio, PathBuf::from("voice.wav"));
        assert_eq!(params.output, PathBuf::from("preview.mp4"));
        // The cap passes through untouched: whether a number is acceptable is the
        // worker's judgement, like every path here.
        assert_eq!(params.max_output_frames, Some(2));
    }

    #[test]
    fn a_render_without_a_cap_asks_for_the_whole_project() {
        let request = build_request(&render_with(
            PathBuf::from("project"),
            PathBuf::from("checkpoint"),
            PathBuf::from("voice.wav"),
            PathBuf::from("preview.mp4"),
        ))
        .expect("the arguments are accepted")
        .expect("render needs a task");
        let Request::Render(params) = request else {
            panic!("render must build a Render request");
        };
        assert_eq!(params.max_output_frames, None);
    }

    #[test]
    fn inspect_model_refuses_an_empty_path_by_name() {
        let error = build_request(&Command::InspectModel {
            source: PathBuf::new(),
        })
        .expect_err("an empty source is refused");
        assert_eq!(error, "模型目录不能为空。");
    }

    #[test]
    fn inspect_model_carries_the_source_into_the_request() {
        let request = build_request(&Command::InspectModel {
            source: PathBuf::from("models/hubert"),
        })
        .expect("the arguments are accepted")
        .expect("inspect-model needs a task");
        let Request::InspectModel(params) = request else {
            panic!("inspect-model must build an InspectModel request");
        };
        // Relative here, absolute demanded by the worker: whether a path is
        // usable is the worker's judgement, like every other path in this file.
        assert_eq!(params.source, PathBuf::from("models/hubert"));
    }

    #[test]
    fn import_legacy_model_carries_paths_and_kind() {
        let request = build_request(&Command::ImportLegacyModel {
            source: PathBuf::from("legacy/model.pth"),
            kind: LegacyModelKindArg::OriginalUnet,
            destination: PathBuf::from("models/original"),
        })
        .expect("the arguments are accepted")
        .expect("import needs a task");
        let Request::ImportLegacyModel(params) = request else {
            panic!("import-legacy-model must build an ImportLegacyModel request");
        };
        assert_eq!(params.source, PathBuf::from("legacy/model.pth"));
        assert_eq!(params.kind, LegacyModelKind::OriginalUnet);
        assert_eq!(params.destination, PathBuf::from("models/original"));
    }

    #[test]
    fn import_legacy_model_refuses_empty_paths() {
        let error = build_request(&Command::ImportLegacyModel {
            source: PathBuf::new(),
            kind: LegacyModelKindArg::FeatherHubert,
            destination: PathBuf::from("models/package"),
        })
        .expect_err("an empty source is refused");
        assert_eq!(error, "旧模型文件不能为空。");

        let error = build_request(&Command::ImportLegacyModel {
            source: PathBuf::from("model.pth"),
            kind: LegacyModelKindArg::FeatherHubert,
            destination: PathBuf::new(),
        })
        .expect_err("an empty destination is refused");
        assert_eq!(error, "目标目录不能为空。");
    }

    #[test]
    fn migrate_legacy_features_carries_both_paths() {
        let request = build_request(&Command::MigrateLegacyFeatures {
            source: PathBuf::from("legacy/aud_hu.npy"),
            destination: PathBuf::from("assets/features/features.f32"),
        })
        .expect("the arguments are accepted")
        .expect("migration needs a task");
        let Request::MigrateLegacyFeatures(params) = request else {
            panic!("migrate-legacy-features must build a MigrateLegacyFeatures request");
        };
        // Relative here, absolute demanded by the worker: which paths are usable
        // is the worker's judgement, like every other path in this file.
        assert_eq!(params.source, PathBuf::from("legacy/aud_hu.npy"));
        assert_eq!(
            params.destination,
            PathBuf::from("assets/features/features.f32")
        );
    }

    #[test]
    fn migrate_legacy_features_refuses_empty_paths() {
        let error = build_request(&Command::MigrateLegacyFeatures {
            source: PathBuf::new(),
            destination: PathBuf::from("features.f32"),
        })
        .expect_err("an empty source is refused");
        assert_eq!(error, "旧特征文件不能为空。");

        let error = build_request(&Command::MigrateLegacyFeatures {
            source: PathBuf::from("aud_hu.npy"),
            destination: PathBuf::new(),
        })
        .expect_err("an empty destination is refused");
        assert_eq!(error, "目标文件不能为空。");
    }

    #[test]
    fn export_model_package_carries_both_paths() {
        let request = build_request(&Command::ExportModelPackage {
            source: PathBuf::from("models/unet/checkpoint-00000004"),
            destination: PathBuf::from("models/unet/original_unet_v1"),
        })
        .expect("the arguments are accepted")
        .expect("an export needs a task");
        let Request::ExportModelPackage(params) = request else {
            panic!("export-model-package must build an ExportModelPackage request");
        };
        // Relative here, absolute demanded by the worker: which paths are usable
        // is the worker's judgement, like every other path in this file.
        assert_eq!(
            params.source,
            PathBuf::from("models/unet/checkpoint-00000004")
        );
        assert_eq!(
            params.destination,
            PathBuf::from("models/unet/original_unet_v1")
        );
    }

    #[test]
    fn export_model_package_refuses_empty_paths() {
        let error = build_request(&Command::ExportModelPackage {
            source: PathBuf::new(),
            destination: PathBuf::from("models/unet/original_unet_v1"),
        })
        .expect_err("an empty source is refused");
        assert_eq!(error, "检查点目录不能为空。");

        let error = build_request(&Command::ExportModelPackage {
            source: PathBuf::from("models/unet/checkpoint-00000004"),
            destination: PathBuf::new(),
        })
        .expect_err("an empty destination is refused");
        assert_eq!(error, "目标目录不能为空。");
    }

    #[test]
    fn export_onnx_carries_the_kind_and_both_paths() {
        let request = build_request(&Command::ExportOnnx {
            source: PathBuf::from("models/unet/original_unet_v1"),
            kind: OnnxExportKindArg::OriginalUnet,
            destination: PathBuf::from("exports/original_unet.onnx"),
        })
        .expect("the arguments are accepted")
        .expect("an export needs a task");
        let Request::ExportOnnx(params) = request else {
            panic!("export-onnx must build an ExportOnnx request");
        };
        assert_eq!(params.source, PathBuf::from("models/unet/original_unet_v1"));
        assert_eq!(params.kind, OnnxExportKind::OriginalUnet);
        assert_eq!(
            params.destination,
            PathBuf::from("exports/original_unet.onnx")
        );
    }

    #[test]
    fn export_onnx_refuses_empty_paths() {
        let error = build_request(&Command::ExportOnnx {
            source: PathBuf::new(),
            kind: OnnxExportKindArg::FeatherHubert,
            destination: PathBuf::from("exports/feather_hubert.onnx"),
        })
        .expect_err("an empty source is refused");
        assert_eq!(error, "模型包目录不能为空。");

        let error = build_request(&Command::ExportOnnx {
            source: PathBuf::from("models/hubert"),
            kind: OnnxExportKindArg::FeatherHubert,
            destination: PathBuf::new(),
        })
        .expect_err("an empty destination is refused");
        assert_eq!(error, "目标文件不能为空。");
    }

    #[test]
    fn the_onnx_kinds_map_onto_the_domain_enum() {
        assert_eq!(
            onnx_export_kind(OnnxExportKindArg::FeatherHubert),
            OnnxExportKind::FeatherHubert
        );
        assert_eq!(
            onnx_export_kind(OnnxExportKindArg::OriginalUnet),
            OnnxExportKind::OriginalUnet
        );
        assert_eq!(
            onnx_export_kind(OnnxExportKindArg::MobileOneUnet),
            OnnxExportKind::MobileOneUnet
        );
    }
}
