mod support;

use std::{
    fs,
    path::Path,
    sync::{Mutex, OnceLock},
};

use burn::{module::Module, optim::AdamConfig};
use feathertalk_audio::read_feature_file;
use feathertalk_domain::{
    Backend, ExtractFeaturesParams, RenderParams, Request, TrainingMode, UnetVariant,
};
use feathertalk_export::ModelConfiguration;
use feathertalk_media::CancellationToken;
use feathertalk_models::{
    backend::{GpuAutodiffBackend, GpuBackend},
    unet::{MobileOneUnetConfig, OriginalUnetConfig},
};
use feathertalk_worker::{
    CommandOutcome, ComputeRegistry, FrameModels, NoReporter, TrainDevice, WgpuContext,
    WorkerConfig, checkpoint_descriptor, execute, render_job, run_render, run_training,
};
use serde_json::Value;
use support::{
    IdentityExtractor, MemorySinkFactory, Recorder, StubDataset, StubFrameReader, micro_plan,
    model, on_step_stack, published_package, render_audio, render_tree,
};

static GPU_LOCK: Mutex<()> = Mutex::new(());

fn registry() -> &'static ComputeRegistry {
    static REGISTRY: OnceLock<ComputeRegistry> = OnceLock::new();
    REGISTRY.get_or_init(ComputeRegistry::discover)
}

fn gpu() -> WgpuContext {
    let registry = registry();
    let adapter = registry
        .resolve(Backend::Wgpu, None)
        .expect("a certified native GPU is required");
    assert_eq!(adapter.backend, Backend::Wgpu);
    assert!(adapter.certified);
    registry
        .open_wgpu(&adapter.id)
        .expect("the selected GPU opens")
}

fn completed(outcome: CommandOutcome) -> Value {
    match outcome {
        CommandOutcome::Completed(Some(result)) => result,
        other => panic!("expected completion: {other:?}"),
    }
}

#[test]
#[ignore = "requires a certified native WGPU adapter"]
fn a_training_checkpoint_resumes_cpu_to_gpu_and_back_to_cpu() {
    let _guard = GPU_LOCK.lock().unwrap();
    on_step_stack("wgpu-checkpoint-portability", || {
        let root = tempfile::tempdir().unwrap();
        let project = root.path().join("project");
        let cpu = TrainDevice::default();
        let ctx = gpu();
        let plan = micro_plan(&project, TrainingMode::Baseline, 3, 1, None);
        let cancel = CancellationToken::new();
        let first = run_training(
            &plan,
            StubDataset::new(1),
            model(&cpu),
            AdamConfig::new().init(),
            &IdentityExtractor,
            &cpu,
            &cancel,
            &Recorder::cancelling_after(1, cancel.clone()),
        );
        assert!(matches!(first, CommandOutcome::Cancelled));

        let plan = micro_plan(
            &project,
            TrainingMode::Baseline,
            3,
            1,
            Some(plan.paths.checkpoint(1)),
        );
        let cancel = CancellationToken::new();
        let second = run_training(
            &plan,
            StubDataset::new(1),
            OriginalUnetConfig::parity_micro().init::<GpuAutodiffBackend>(&ctx.device),
            AdamConfig::new().init(),
            &IdentityExtractor,
            &ctx.device,
            &cancel,
            &Recorder::cancelling_after(1, cancel.clone()),
        );
        assert!(matches!(second, CommandOutcome::Cancelled), "{second:?}");
        let metrics: Value =
            serde_json::from_slice(&fs::read(plan.paths.metrics(2)).unwrap()).unwrap();
        assert!(
            metrics["gpu_memory_bytes"]
                .as_u64()
                .is_some_and(|bytes| bytes > 0),
            "{metrics}"
        );

        let plan = micro_plan(
            &project,
            TrainingMode::Baseline,
            3,
            1,
            Some(plan.paths.checkpoint(2)),
        );
        let result = completed(run_training(
            &plan,
            StubDataset::new(1),
            model(&cpu),
            AdamConfig::new().init(),
            &IdentityExtractor,
            &cpu,
            &CancellationToken::new(),
            &Recorder::new(),
        ));
        assert_eq!(result["global_step"], 3);
        assert_eq!(result["samples_seen"], 1);
        assert_eq!(result["backend"], "ndarray-cpu");
        assert!(result["total_loss"].as_f64().unwrap().is_finite());
        ctx.check().unwrap();
    });
}

#[test]
#[ignore = "requires a certified native WGPU adapter"]
fn both_unets_train_all_three_modes_on_wgpu() {
    let _guard = GPU_LOCK.lock().unwrap();
    on_step_stack("wgpu-training-modes", || {
        let ctx = gpu();
        for mode in [
            TrainingMode::Baseline,
            TrainingMode::MouthRoi,
            TrainingMode::Temporal,
        ] {
            for variant in [UnetVariant::OriginalUnet, UnetVariant::MobileOneUnet] {
                let root = tempfile::tempdir().unwrap();
                let mut plan = micro_plan(root.path(), mode, 1, 2, None);
                plan.variant = variant;
                let result = match variant {
                    UnetVariant::OriginalUnet => completed(run_training(
                        &plan,
                        StubDataset::new(2),
                        OriginalUnetConfig::parity_micro().init::<GpuAutodiffBackend>(&ctx.device),
                        AdamConfig::new().init(),
                        &IdentityExtractor,
                        &ctx.device,
                        &CancellationToken::new(),
                        &Recorder::new(),
                    )),
                    UnetVariant::MobileOneUnet => {
                        let configuration = MobileOneUnetConfig::parity_micro();
                        plan.descriptor = checkpoint_descriptor(
                            &ModelConfiguration::mobileone_unet(&configuration, false),
                        )
                        .unwrap();
                        completed(run_training(
                            &plan,
                            StubDataset::new(2),
                            configuration.init::<GpuAutodiffBackend>(&ctx.device),
                            AdamConfig::new().init(),
                            &IdentityExtractor,
                            &ctx.device,
                            &CancellationToken::new(),
                            &Recorder::new(),
                        ))
                    }
                };
                assert_eq!(result["backend"], "wgpu");
                assert_eq!(result["epochs_completed"], 1);
                assert_eq!(result["checkpoints_written"], 1);
                assert_eq!(result["previews_written"], 1);
                assert!(result["total_loss"].as_f64().unwrap().is_finite());
                ctx.check().unwrap();
            }
        }
    });
}

#[test]
#[ignore = "requires a certified native WGPU adapter"]
fn a_wgpu_render_publishes_frames_and_reports_its_backend() {
    let _guard = GPU_LOCK.lock().unwrap();
    on_step_stack("wgpu-render", || {
        let ctx = gpu();
        let (root, project) = render_tree(2, 2);
        let params = RenderParams {
            project_dir: project.clone(),
            checkpoint: project.join("models/unet/checkpoint-00000002"),
            audio: render_audio(&project),
            output: root.path().join("wgpu-render.mp4"),
            max_output_frames: Some(1),
        };
        let configuration = OriginalUnetConfig::parity_micro();
        let descriptor =
            checkpoint_descriptor(&ModelConfiguration::original_unet(&configuration)).unwrap();
        let job = render_job(
            &params,
            2,
            &std::env::current_exe().unwrap(),
            descriptor,
            1,
            2,
        )
        .unwrap();
        let model = configuration
            .init::<GpuBackend>(&ctx.device)
            .fork(&ctx.device);
        let sinks = MemorySinkFactory::default();
        let result = completed(run_render(
            &job,
            &model,
            &ctx.device,
            &CancellationToken::new(),
            &Recorder::new(),
            &StubFrameReader::default(),
            &sinks,
        ));
        assert_eq!(result["backend"], "wgpu");
        assert_eq!(result["frame_count"], 1);
        assert!(params.output.is_file());
        assert_eq!(sinks.frames.lock().unwrap().len(), 1);
        ctx.check().unwrap();

        let configuration = MobileOneUnetConfig::parity_micro();
        let descriptor =
            checkpoint_descriptor(&ModelConfiguration::mobileone_unet(&configuration, false))
                .unwrap();
        let params = RenderParams {
            output: root.path().join("mobileone-render.mp4"),
            ..params
        };
        let job = render_job(
            &params,
            2,
            &std::env::current_exe().unwrap(),
            descriptor,
            1,
            2,
        )
        .unwrap();
        let model = configuration
            .init::<GpuBackend>(&ctx.device)
            .reparameterize();
        let result = completed(run_render(
            &job,
            &model,
            &ctx.device,
            &CancellationToken::new(),
            &Recorder::new(),
            &StubFrameReader::default(),
            &MemorySinkFactory::default(),
        ));
        assert_eq!(result["backend"], "wgpu");
        assert!(params.output.is_file());
        ctx.check().unwrap();
    });
}

#[test]
#[ignore = "requires a certified native WGPU adapter"]
fn feature_command_uses_the_selected_gpu_and_matches_cpu_output() {
    let _guard = GPU_LOCK.lock().unwrap();
    on_step_stack("wgpu-feature-command", || {
        let root = tempfile::tempdir().unwrap();
        let package = published_package(root.path(), "hubert", "0.1.0");
        let base = WorkerConfig::from_values_with_toolchains(
            None,
            None,
            None,
            None,
            None,
            Some(package.display().to_string()),
        )
        .with_compute_registry(registry().clone());
        let adapter = registry().resolve(Backend::Wgpu, None).unwrap();
        let mut outputs = Vec::new();
        for backend in ["cpu", "wgpu"] {
            let project_dir = root.path().join(backend);
            fs::create_dir_all(project_dir.join("assets")).unwrap();
            fs::write(project_dir.join("project.json"), b"{}").unwrap();
            let audio = project_dir.join("assets/audio_16k_mono.wav");
            write_test_wav(&audio);
            let config = base.clone().with_compute_selection(
                Some(backend),
                (backend == "wgpu").then_some(adapter.id.as_str()),
            );
            let result = completed(execute(
                &Request::ExtractFeatures(ExtractFeaturesParams { project_dir, audio }),
                &config,
                &CancellationToken::new(),
                &NoReporter,
            ));
            let expected_backend = if backend == "cpu" {
                "ndarray-cpu"
            } else {
                "wgpu"
            };
            assert_eq!(result["backend"], expected_backend);
            assert_eq!(result["used_cpu_fallback"], false);
            assert_eq!(
                result["adapter"]["id"],
                if backend == "cpu" {
                    "cpu-0"
                } else {
                    &adapter.id
                }
            );
            if backend == "wgpu" {
                assert_eq!(result["graphics_api"], gpu().graphics_api());
            }
            outputs.push(
                read_feature_file(Path::new(result["feature_file"].as_str().unwrap())).unwrap(),
            );
        }
        assert_eq!(outputs[0].values().len(), outputs[1].values().len());
        let max_abs =
            outputs[0]
                .values()
                .iter()
                .zip(outputs[1].values())
                .fold(0.0_f32, |max, (cpu, gpu)| {
                    assert!(gpu.is_finite());
                    max.max((cpu - gpu).abs())
                });
        assert!(max_abs <= 1e-3, "FeatherHuBERT CPU/GPU max_abs={max_abs}");
    });
}

#[test]
#[ignore = "requires a certified native WGPU adapter and the committed SCRFD/PFLD artifacts"]
fn frame_models_load_and_infer_on_the_selected_gpu() {
    let _guard = GPU_LOCK.lock().unwrap();
    on_step_stack("wgpu-frame-models", || {
        let ctx = gpu();
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        let config = WorkerConfig::from_values_with_models(
            None,
            None,
            None,
            Some(
                root.join("../feathertalk-scrfd/artifacts/scrfd_2_5g")
                    .display()
                    .to_string(),
            ),
            Some(
                root.join("../feathertalk-pfld/artifacts/pfld_ghost_one")
                    .display()
                    .to_string(),
            ),
        );
        let models =
            FrameModels::<GpuBackend>::load_on(config.models().unwrap(), ctx.device.clone())
                .unwrap();
        let frame = models
            .decoder()
            .decode(
                0,
                &root.join("../feathertalk-frame-adapters/tests/fixtures/demo_frame_v1/frame.jpg"),
            )
            .unwrap();
        let faces = models.detector().detect(&frame).unwrap();
        assert_eq!(faces.len(), 1);
        let landmarks = models.predictor().predict(&frame, &faces[0]).unwrap();
        assert_eq!(landmarks.points().len(), 110);
        ctx.check().unwrap();

        // Compare with the pinned OpenCV fixture using the adapter suite's
        // existing score/pixel tolerances, including integer landmark rounding.
        let fixture: Value =
            serde_json::from_slice(
                &fs::read(root.join(
                    "../feathertalk-frame-adapters/tests/fixtures/demo_frame_v1/fixture.json",
                ))
                .unwrap(),
            )
            .unwrap();
        let expected = &fixture["frames"]["sharp"];
        let score = expected["detection"]["score"].as_f64().unwrap();
        assert!((f64::from(faces[0].score) - score).abs() <= 0.01);
        for (got, want) in faces[0]
            .bbox
            .iter()
            .zip(expected["detection"]["bbox"].as_array().unwrap())
        {
            assert!((f64::from(*got) - want.as_f64().unwrap()).abs() <= 1.0);
        }
        for (got, want) in landmarks
            .points()
            .iter()
            .zip(expected["landmarks"].as_array().unwrap())
        {
            assert!((i64::from(got.x) - want[0].as_i64().unwrap()).abs() <= 1);
            assert!((i64::from(got.y) - want[1].as_i64().unwrap()).abs() <= 1);
        }
    });
}

fn write_test_wav(path: &Path) {
    let samples: Vec<i16> = (0..3_200)
        .map(|index| (index % 2_000) as i16 - 1_000)
        .collect();
    let payload = samples.len() as u32 * 2;
    let mut bytes = Vec::with_capacity(44 + payload as usize);
    bytes.extend(b"RIFF");
    bytes.extend((36 + payload).to_le_bytes());
    bytes.extend(b"WAVEfmt ");
    bytes.extend(16u32.to_le_bytes());
    bytes.extend(1u16.to_le_bytes());
    bytes.extend(1u16.to_le_bytes());
    bytes.extend(16_000u32.to_le_bytes());
    bytes.extend(32_000u32.to_le_bytes());
    bytes.extend(2u16.to_le_bytes());
    bytes.extend(16u16.to_le_bytes());
    bytes.extend(b"data");
    bytes.extend(payload.to_le_bytes());
    bytes.extend(samples.into_iter().flat_map(i16::to_le_bytes));
    fs::write(path, bytes).unwrap();
}
