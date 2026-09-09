#![allow(dead_code)]

pub mod health;

use std::cell::Cell;
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use burn::{
    module::Module,
    optim::AdamConfig,
    tensor::{Tensor, backend::Backend},
};
use feathertalk_audio::{FeatureMatrix, write_feature_file};
use feathertalk_domain::{
    Progress, TaskStage, TrainParams, TrainingMode as DomainTrainingMode, UnetVariant,
};
use feathertalk_export::{
    LicenseBundle, LicenseEntry, ModelConfiguration, ModelDescription, PackageBuildRequest,
    SourceManifest, TrainingManifest, write_model_package,
};
use feathertalk_inference::{
    BgrFrame, CommandSpec, FrameReader, InferenceError, RawVideoSink, RawVideoSinkFactory,
};
use feathertalk_media::CancellationToken;
use feathertalk_models::{
    feather_hubert::{FeatherHubertConfig, FeatherHubertEncoder},
    unet::{MobileOneUnetConfig, MobileOneUnetInference, OriginalUnet, OriginalUnetConfig},
};
use feathertalk_project::{
    AssetManifest, AssetPackageState, FeatureType, ModelSelection, ProjectManifest,
    TaskHistoryEntry, TaskHistoryStatus, lock_asset_package, write_project_manifest_atomic,
};
use feathertalk_training::{
    CheckpointDescriptor, DATA_LOADER_STATE_SCHEMA_VERSION, DataLoaderConfig, DataLoaderState,
    PerceptualFeatureExtractor, Provenance, RandomAlgorithm, SamplingConfig, SamplingKind,
    TRAINING_STATE_SCHEMA_VERSION, TrainingCheckpointState, TrainingDataset, TrainingError,
    TrainingSample, save_training_checkpoint,
};
use feathertalk_training_data::{FrameSample, TrainingItem};
use feathertalk_worker::{
    RenderBackend, RenderDevice, TRAINING_SEED, TaskReporter, TrainBackend, TrainDevice,
    TrainingPaths, TrainingPlan, checkpoint_descriptor, project_assets, training_config,
};
use sha2::{Digest, Sha256};
use tempfile::TempDir;

/// A 160x160 forward plus backward through burn's autodiff graph overruns the
/// default 2 MiB libtest stack in a debug build and takes the whole binary down
/// with `STATUS_STACK_OVERFLOW`. `feathertalk-training-run/tests/support/mod.rs`
/// solves it the same way, and Task 4 gives the worker's own execution thread
/// the same stack.
const STEP_STACK_BYTES: usize = 64 * 1024 * 1024;

/// Runs `body` on a thread whose stack is large enough for a training step.
/// Panics travel back through `join`, so failed assertions still fail the test.
pub fn on_step_stack(name: &str, body: impl FnOnce() + Send + 'static) {
    std::thread::Builder::new()
        .name(name.to_owned())
        .stack_size(STEP_STACK_BYTES)
        .spawn(body)
        .expect("the step thread starts")
        .join()
        .expect("the step thread does not panic");
}

/// The perceptual term with the weights taken out: it compares the images
/// themselves, which keeps the loss finite and the test independent of VGG19.
#[derive(Debug, Clone, Copy)]
pub struct IdentityExtractor;

impl<B: Backend> PerceptualFeatureExtractor<B> for IdentityExtractor {
    fn forward(&self, image: Tensor<B, 4>) -> Tensor<B, 4> {
        image
    }
}

/// Behaves like `IdentityExtractor` until the given call, then poisons the loss.
///
/// One step calls the extractor more than once, so the threshold is counted in
/// calls rather than steps; the tests only need "not the first step".
#[derive(Debug)]
pub struct PoisonedExtractor {
    calls: Cell<usize>,
    poison_from: usize,
}

impl PoisonedExtractor {
    pub fn after(calls: usize) -> Self {
        Self {
            calls: Cell::new(0),
            poison_from: calls,
        }
    }
}

impl<B: Backend> PerceptualFeatureExtractor<B> for PoisonedExtractor {
    fn forward(&self, image: Tensor<B, 4>) -> Tensor<B, 4> {
        let seen = self.calls.get().saturating_add(1);
        self.calls.set(seen);
        if seen > self.poison_from {
            return image.mul_scalar(f32::NAN);
        }
        image
    }
}

/// The micro model with every parameter already materialised.
///
/// `fork` pushes each `Param` through `val()`; without it a clone would copy the
/// lazy initialiser instead of the weights, and two clones would draw different
/// numbers (burn-core 0.21 `module/param/base.rs`).
pub fn model(device: &TrainDevice) -> OriginalUnet<TrainBackend> {
    OriginalUnetConfig::parity_micro()
        .init::<TrainBackend>(device)
        .fork(device)
}

/// A dataset that synthesises every sample, so the loop can be driven without a
/// locked project on disk. This is what Task 1 opened `FrameSample::new` for.
pub struct StubDataset {
    frame_count: u64,
}

impl StubDataset {
    pub fn new(frame_count: u64) -> Self {
        Self { frame_count }
    }
}

impl TrainingDataset for StubDataset {
    type Item = TrainingItem;

    fn frame_count(&self) -> u64 {
        self.frame_count
    }

    fn load_sample(&self, sample: &TrainingSample) -> Result<Self::Item, TrainingError> {
        Ok(match sample {
            TrainingSample::SingleFrame { target_index, .. } => {
                TrainingItem::SingleFrame(frame(*target_index)?)
            }
            TrainingSample::TemporalPair {
                first_target_index,
                second_target_index,
                ..
            } => TrainingItem::TemporalPair {
                first: frame(*first_target_index)?,
                second: frame(*second_target_index)?,
            },
        })
    }
}

/// Flat planes whose value follows the frame index: enough for a finite loss and
/// for two frames to differ, cheap enough to allocate per sample.
fn frame(index: u64) -> Result<FrameSample, TrainingError> {
    let value = (index % 7) as f32 / 7.0;
    Ok(FrameSample::new(
        vec![value; 6 * 160 * 160],
        vec![value; 16 * 32 * 32],
        vec![value; 3 * 160 * 160],
        vec![1.0; 160 * 160],
    )?)
}

/// Records every event, and can cancel a token once enough have arrived.
pub struct Recorder {
    events: Mutex<Vec<(TaskStage, Option<Progress>)>>,
    cancel_after: Option<(usize, CancellationToken)>,
}

impl Recorder {
    pub fn new() -> Self {
        Self {
            events: Mutex::new(Vec::new()),
            cancel_after: None,
        }
    }

    /// Cancels `token` once `events` events have been reported, which is how a
    /// test interrupts a run at a known step instead of at a known time.
    pub fn cancelling_after(events: usize, token: CancellationToken) -> Self {
        Self {
            events: Mutex::new(Vec::new()),
            cancel_after: Some((events, token)),
        }
    }

    pub fn events(&self) -> Vec<(TaskStage, Option<Progress>)> {
        self.events.lock().expect("the recorder is intact").clone()
    }
}

impl TaskReporter for Recorder {
    fn report(&self, stage: TaskStage, progress: Option<Progress>) {
        let mut events = self.events.lock().expect("the recorder is intact");
        events.push((stage, progress));
        if let Some((limit, token)) = &self.cancel_after
            && events.len() >= *limit
        {
            token.cancel();
        }
    }
}

/// The plan a micro run trains under: `parity_micro`, batch size 1, whatever
/// mode and epoch count the test asks for.
pub fn micro_plan(
    project_dir: &Path,
    mode: DomainTrainingMode,
    epochs: u32,
    frame_count: u64,
    resume_from: Option<PathBuf>,
) -> TrainingPlan {
    let params = TrainParams {
        project_dir: project_dir.to_path_buf(),
        mode,
        variant: UnetVariant::OriginalUnet,
        epochs,
        batch_size: 1,
        resume: resume_from.is_some(),
    };
    let configuration = ModelConfiguration::original_unet(&OriginalUnetConfig::parity_micro());
    TrainingPlan {
        mode,
        variant: UnetVariant::OriginalUnet,
        epochs_requested: epochs,
        frame_count,
        config: training_config(&params),
        descriptor: checkpoint_descriptor(&configuration).expect("the configuration serialises"),
        paths: TrainingPaths::new(project_dir),
        resume_from,
    }
}

/// The three inference inputs a locked project holds, plus a placeholder for the
/// audio a render request names.
///
/// Ported from `feathertalk-inference/tests/executor.rs`, but laid out at the
/// paths `project_assets` resolves, so the worker's own assembly finds them.
/// Returns the temporary root and the project directory inside it, which lets a
/// test put its output next to the project instead of inside it.
pub fn render_tree(frame_count: usize, feature_frames: usize) -> (TempDir, PathBuf) {
    let root = tempfile::tempdir().expect("the temporary root is created");
    let project = root.path().join("project");
    let assets = project_assets(&project);
    std::fs::create_dir_all(&assets.frame_dir).expect("the frame directory is created");
    std::fs::create_dir_all(&assets.landmark_dir).expect("the landmark directory is created");
    for index in 0..frame_count {
        std::fs::write(assets.frame_dir.join(format!("{index:06}.jpg")), b"fixture")
            .expect("the frame is written");
        let mut landmarks = String::new();
        for point in 0..110 {
            // Point 31 is the one the crop geometry measures; 168 keeps the crop
            // inside the frame `StubFrameReader` hands back.
            let x = if point == 31 { 168 } else { 0 };
            landmarks.push_str(&format!("{x} 0\n"));
        }
        std::fs::write(
            assets.landmark_dir.join(format!("{index:06}.lms")),
            landmarks,
        )
        .expect("the landmarks are written");
    }
    if let Some(parent) = assets.feature_path.parent() {
        std::fs::create_dir_all(parent).expect("the feature directory is created");
    }
    // Two tokens per frame, which is the ratio the feature extractor writes.
    let tokens = feature_frames * 2;
    let features =
        FeatureMatrix::new(tokens, 1024, vec![0.0; tokens * 1024]).expect("the features are valid");
    write_feature_file(&assets.feature_path, &features).expect("the features are written");
    std::fs::write(render_audio(&project), b"audio").expect("the audio placeholder is written");
    (root, project)
}

/// The audio file `render_tree` writes. The sink is a stub, so nothing reads it;
/// the request only needs a path that exists.
pub fn render_audio(project_dir: &Path) -> PathBuf {
    project_dir.join("audio.wav")
}

/// The frame size the render fixtures advertise, which is the size
/// `StubFrameReader` hands back.
pub const RENDER_FRAME_SIDE: u32 = 168;

/// Turns a `render_tree` into a project `validate_project_dir` accepts: the two
/// media placeholders it insists on, the project manifest, and a locked asset
/// package.
///
/// The order matters: `write_asset_manifest_atomic` refuses to overwrite a
/// manifest that already validates as locked, so the lock runs last.
pub fn lock_render_tree(project_dir: &Path, frame_count: u64) {
    let assets = project_dir.join("assets");
    std::fs::write(assets.join("video_25fps.mp4"), b"video").expect("the video is written");
    std::fs::write(assets.join("audio_16k_mono.wav"), b"audio").expect("the audio is written");
    let manifest = ProjectManifest {
        schema_version: 1,
        project_id: "render".into(),
        display_name: "Render".into(),
        asset_package: "assets/assets.json".into(),
        default_model: ModelSelection::OriginalUnet,
        task_history: vec![TaskHistoryEntry {
            task_id: "task-1".into(),
            kind: "preprocess".into(),
            status: TaskHistoryStatus::Completed,
            updated_at: "2026-09-04T10:00:00Z".into(),
        }],
    };
    write_project_manifest_atomic(&project_dir.join("project.json"), &manifest)
        .expect("the project manifest is written");
    lock_asset_package(
        project_dir,
        AssetManifest {
            schema_version: 1,
            state: AssetPackageState::Locked,
            video_fps: 25,
            audio_sample_rate: 16_000,
            audio_channels: 1,
            frame_count,
            frame_width: RENDER_FRAME_SIDE,
            frame_height: RENDER_FRAME_SIDE,
            feature_type: FeatureType::FeatherHubert,
            feature_shape: [frame_count, 2, 1024],
            landmark_model_sha256: "a".repeat(64),
            feature_model_sha256: "b".repeat(64),
        },
    )
    .expect("the asset package locks");
}

/// Hands back a 168x168 frame for every index and remembers which indexes the
/// render asked for.
///
/// The size is not arbitrary: the crop the fixture's landmarks describe has to
/// fit inside the frame, or the paste would fall outside it.
#[derive(Debug, Default)]
pub struct StubFrameReader {
    pub frames: Mutex<Vec<usize>>,
    /// The index whose read fails, for the tests that need a failure in the
    /// middle of the loop rather than before it starts.
    pub fail_at: Option<usize>,
}

impl StubFrameReader {
    pub fn failing_at(index: usize) -> Self {
        Self {
            frames: Mutex::new(Vec::new()),
            fail_at: Some(index),
        }
    }
}

impl FrameReader for StubFrameReader {
    fn read(&self, index: usize, path: &Path) -> Result<BgrFrame, InferenceError> {
        let side = RENDER_FRAME_SIDE;
        self.frames
            .lock()
            .expect("the reader is intact")
            .push(index);
        if self.fail_at == Some(index) {
            return Err(InferenceError::FrameReader {
                index,
                path: path.to_owned(),
                message: "injected reader failure".into(),
            });
        }
        // A value that follows the index, so an all-zero frame would be visible.
        let value = (index as u8).wrapping_add(1);
        BgrFrame::new(side, side, vec![value; (side * side * 3) as usize])
    }
}

/// A sink that keeps the frame sizes it was handed and writes a placeholder to
/// the staging path, so the executor's atomic publish has a file to rename.
///
/// This is what makes a real `execute_offline_render` runnable in a unit test
/// without ffmpeg on the machine.
#[derive(Debug, Default)]
pub struct MemorySinkFactory {
    pub frames: Mutex<Vec<usize>>,
    pub staging: Mutex<Option<PathBuf>>,
}

struct MemorySink<'a> {
    factory: &'a MemorySinkFactory,
}

impl RawVideoSink for MemorySink<'_> {
    fn write_frame(&mut self, frame: &BgrFrame) -> Result<(), InferenceError> {
        self.factory
            .frames
            .lock()
            .expect("the sink is intact")
            .push(frame.as_bytes().len());
        Ok(())
    }

    fn finish(self: Box<Self>) -> Result<(), InferenceError> {
        let staging = self.factory.staging.lock().expect("the sink is intact");
        let path = staging.as_ref().expect("the sink knows its staging path");
        std::fs::write(path, b"rendered-video").expect("the staging file is written");
        Ok(())
    }
}

impl RawVideoSinkFactory for MemorySinkFactory {
    fn start(&self, command: &CommandSpec) -> Result<Box<dyn RawVideoSink + '_>, InferenceError> {
        // ffmpeg's last argument is the file it writes, which is the staging
        // path the executor renames once the render succeeds.
        let staging = command
            .arguments()
            .last()
            .expect("the command names an output file")
            .clone();
        *self.staging.lock().expect("the factory is intact") = Some(PathBuf::from(staging));
        Ok(Box::new(MemorySink { factory: self }))
    }
}

/// The micro model a render test runs, with every parameter materialised.
///
/// `parity_micro` rather than `production`: the shapes are the ones inference
/// requires, and the parameter count is small enough for a unit test.
pub fn render_model(device: &RenderDevice) -> OriginalUnet<RenderBackend> {
    OriginalUnetConfig::parity_micro()
        .init::<RenderBackend>(device)
        .fork(device)
}

/// Publishes a micro FeatherHuBERT package under `root/{name}` and returns it.
///
/// Ported from `tests/features.rs::published_package` with the minimum app
/// version as a parameter, because compatibility is exactly what varies here.
/// The real writer rather than hand-written files: the manifest declares the size
/// and digest of everything beside it, and only the writer makes them agree.
pub fn published_package(root: &Path, name: &str, minimum_app_version: &str) -> PathBuf {
    write_hubert_package(
        root,
        name,
        minimum_app_version,
        FeatherHubertConfig::parity_micro(),
    )
}

/// Publishes a FeatherHuBERT package the ONNX exporter accepts.
///
/// `parity_micro` is not exportable: the public ONNX contract fixes the hidden
/// output at 1024 channels, so the exporter refuses any other `output_dim`.
/// Everything else stays micro, which keeps the fixture cheap.
pub fn published_onnx_hubert_package(root: &Path, name: &str) -> PathBuf {
    write_hubert_package(
        root,
        name,
        "0.1.0",
        FeatherHubertConfig {
            channels: 32,
            expansion: 2,
            num_blocks: 1,
            output_dim: 1024,
            dropout: 0.0,
        },
    )
}

fn write_hubert_package(
    root: &Path,
    name: &str,
    minimum_app_version: &str,
    config: FeatherHubertConfig,
) -> PathBuf {
    let source_name = format!("{name}-source.pth");
    let source_path = root.join(&source_name);
    fs::write(&source_path, b"source-fixture").expect("the source fixture is written");
    let source_sha256 = hex::encode(Sha256::digest(b"source-fixture"));
    let licenses_path = root.join(format!("{name}-LICENSES.input.json"));
    let licenses = LicenseBundle {
        schema_version: 1,
        entries: vec![LicenseEntry {
            component: "synthetic FeatherHuBERT fixture".to_owned(),
            license_id: "LicenseRef-Test".to_owned(),
            source_url: "https://example.invalid/feather-hubert".to_owned(),
            notice: "test-only local record".to_owned(),
        }],
    };
    fs::write(
        &licenses_path,
        serde_json::to_vec(&licenses).expect("the bundle serialises"),
    )
    .expect("the licenses fixture is written");
    let device = RenderDevice::default();
    let model = config.init::<RenderBackend>(&device);
    let request = PackageBuildRequest {
        destination: root.join(name),
        description: ModelDescription::feather_hubert(config.clone()),
        source_path,
        source: SourceManifest {
            format: "test".to_owned(),
            identifier: "feather-hubert-fixture".to_owned(),
            version: "1".to_owned(),
            file_name: source_name,
            sha256: source_sha256,
            url: None,
        },
        licenses_path,
        created_at: "2026-08-27T00:00:00Z".to_owned(),
        minimum_app_version: minimum_app_version.to_owned(),
        training: TrainingManifest::default(),
    };
    write_model_package::<RenderBackend, FeatherHubertEncoder<RenderBackend>, _>(
        &request,
        &model,
        &device,
        |device| config.init::<RenderBackend>(device),
    )
    .expect("the package is written");
    request.destination
}

/// Writes a micro Original UNet package, the way `published_package` writes a
/// micro FeatherHuBERT one.
///
/// The ONNX exporter reads the configuration out of the manifest rather than
/// resolving a production literal, so a micro package exercises the whole
/// command.
pub fn published_unet_package(root: &Path, name: &str) -> PathBuf {
    let source_name = format!("{name}-source.pth");
    let source_path = root.join(&source_name);
    fs::write(&source_path, b"unet-source-fixture").expect("the source fixture is written");
    let source_sha256 = hex::encode(Sha256::digest(b"unet-source-fixture"));
    let licenses_path = root.join(format!("{name}-LICENSES.input.json"));
    let licenses = LicenseBundle {
        schema_version: 1,
        entries: vec![LicenseEntry {
            component: "synthetic UNet fixture".to_owned(),
            license_id: "LicenseRef-Test".to_owned(),
            source_url: "https://example.invalid/original-unet".to_owned(),
            notice: "test-only local record".to_owned(),
        }],
    };
    fs::write(
        &licenses_path,
        serde_json::to_vec(&licenses).expect("the bundle serialises"),
    )
    .expect("the licenses fixture is written");
    let config = OriginalUnetConfig::parity_micro();
    let device = RenderDevice::default();
    let model = config.init::<RenderBackend>(&device);
    let request = PackageBuildRequest {
        destination: root.join(name),
        description: ModelDescription::original_unet(config.clone()),
        source_path,
        source: SourceManifest {
            format: "test".to_owned(),
            identifier: "original-unet-fixture".to_owned(),
            version: "1".to_owned(),
            file_name: source_name,
            sha256: source_sha256,
            url: None,
        },
        licenses_path,
        created_at: "2026-09-05T00:00:00Z".to_owned(),
        minimum_app_version: "0.1.0".to_owned(),
        training: TrainingManifest::default(),
    };
    write_model_package::<RenderBackend, OriginalUnet<RenderBackend>, _>(
        &request,
        &model,
        &device,
        |device| config.init::<RenderBackend>(device),
    )
    .expect("the package is written");
    request.destination
}

/// Writes a micro fused MobileOne UNet package.
///
/// Fused is the only shape a package can hold: a manifest names every tensor,
/// and the training graph's branches push that file past `MAX_MANIFEST_BYTES`,
/// so `write_model_package` refuses a branched MobileOne package outright.
pub fn published_mobileone_package(root: &Path, name: &str) -> PathBuf {
    let source_name = format!("{name}-source.pth");
    let source_path = root.join(&source_name);
    fs::write(&source_path, b"mobileone-source-fixture").expect("the source fixture is written");
    let source_sha256 = hex::encode(Sha256::digest(b"mobileone-source-fixture"));
    let licenses_path = root.join(format!("{name}-LICENSES.input.json"));
    let licenses = LicenseBundle {
        schema_version: 1,
        entries: vec![LicenseEntry {
            component: "synthetic MobileOne UNet fixture".to_owned(),
            license_id: "LicenseRef-Test".to_owned(),
            source_url: "https://example.invalid/mobileone-unet".to_owned(),
            notice: "test-only local record".to_owned(),
        }],
    };
    fs::write(
        &licenses_path,
        serde_json::to_vec(&licenses).expect("the bundle serialises"),
    )
    .expect("the licenses fixture is written");
    let config = MobileOneUnetConfig::parity_micro();
    let device = RenderDevice::default();
    let request = PackageBuildRequest {
        destination: root.join(name),
        description: ModelDescription::mobileone_unet(config.clone(), true),
        source_path,
        source: SourceManifest {
            format: "test".to_owned(),
            identifier: "mobileone-unet-fixture".to_owned(),
            version: "1".to_owned(),
            file_name: source_name,
            sha256: source_sha256,
            url: None,
        },
        licenses_path,
        created_at: "2026-09-05T00:00:00Z".to_owned(),
        minimum_app_version: "0.1.0".to_owned(),
        training: TrainingManifest::default(),
    };
    let model = config.init::<RenderBackend>(&device).reparameterize();
    write_model_package::<RenderBackend, MobileOneUnetInference<RenderBackend>, _>(
        &request,
        &model,
        &device,
        |device| config.init::<RenderBackend>(device).reparameterize(),
    )
    .expect("the package is written");
    request.destination
}

/// The training state a checkpoint carries.
///
/// A copy of the private `state()` in `tests/render.rs`: every field is one
/// `TrainingCheckpointState::validate` insists on, and the three cross-checked
/// values come from `training_config` rather than from a literal. `tests/render.rs`
/// keeps its own copy -- folding them together would mean re-verifying a committed
/// test for no change in behaviour.
pub fn checkpoint_state(project_dir: &Path, frame_count: u64) -> TrainingCheckpointState {
    let params = TrainParams {
        project_dir: project_dir.to_path_buf(),
        mode: DomainTrainingMode::Baseline,
        variant: UnetVariant::OriginalUnet,
        epochs: 1,
        batch_size: 1,
        resume: false,
    };
    let config = training_config(&params);
    let batch_size = config.batch_size;
    let temporal_stride = config.temporal_stride;
    TrainingCheckpointState {
        schema_version: TRAINING_STATE_SCHEMA_VERSION,
        epoch: 1,
        global_step: 2,
        random_seed: TRAINING_SEED,
        data_loader: DataLoaderState {
            schema_version: DATA_LOADER_STATE_SCHEMA_VERSION,
            random_algorithm: RandomAlgorithm::Splitmix64FisherYatesV1,
            config: DataLoaderConfig {
                batch_size,
                seed: TRAINING_SEED,
                sampling: SamplingConfig {
                    kind: SamplingKind::SingleFrame,
                    temporal_stride,
                },
            },
            frame_count,
            epoch: 1,
            next_position: 0,
        },
        training_config: config,
        asset_provenance: Provenance {
            entries: BTreeMap::new(),
        },
        model_provenance: Provenance {
            entries: BTreeMap::new(),
        },
    }
}

/// Writes a real checkpoint whose manifest carries `descriptor`. `directory` must
/// not exist: `save_training_checkpoint` creates it.
pub fn write_checkpoint(directory: &Path, descriptor: CheckpointDescriptor) {
    let device = TrainDevice::default();
    save_training_checkpoint::<TrainBackend, _, _>(
        directory,
        &model(&device),
        &AdamConfig::new().init(),
        descriptor,
        checkpoint_state(directory, 2),
    )
    .expect("the checkpoint is written");
}
