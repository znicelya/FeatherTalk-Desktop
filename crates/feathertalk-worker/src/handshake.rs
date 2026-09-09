use feathertalk_domain::{
    AdapterInfo, AdapterKind, Backend, Capabilities, PROTOCOL_VERSION, ReadyFrame, TaskKind,
};

use crate::WorkerConfig;

/// Stable identity of the CPU adapter. The adapter
/// lock table keys on it, so it must not change between worker restarts.
pub const CPU_ADAPTER_ID: &str = "cpu-0";

pub fn cpu_adapter() -> AdapterInfo {
    AdapterInfo {
        id: CPU_ADAPTER_ID.to_owned(),
        name: "CPU".to_owned(),
        backend: Backend::Cpu,
        kind: AdapterKind::Cpu,
        certified: true,
        vram_bytes: None,
    }
}

pub fn supported_commands(config: &WorkerConfig) -> Vec<TaskKind> {
    // None of these needs a toolchain: they walk project directories, read model
    // manifests, and write model files, so they are always available. The ONNX
    // export is here too -- serialising a graph needs no runtime.
    let mut commands = vec![
        TaskKind::ValidateProject,
        TaskKind::InspectModel,
        TaskKind::ImportLegacyModel,
        TaskKind::MigrateLegacyFeatures,
        TaskKind::ExportModelPackage,
        TaskKind::ExportOnnx,
    ];
    // Both media commands shell out to the same two binaries, so they are
    // available together or not at all.
    if config.media().is_some() {
        commands.push(TaskKind::ProbeMedia);
        commands.push(TaskKind::NormalizeMedia);
        // Rendering shells out to ffmpeg and to nothing else: the frames, the
        // landmarks and the audio features are already inside the locked
        // project, and inference computes no perceptual loss, so neither the
        // frame models nor a model package are preconditions.
        commands.push(TaskKind::Render);
        // Extraction needs the media toolchain *and* both model directories.
        if config.models().is_some() {
            commands.push(TaskKind::ExtractFrames);
        }
    }
    // Feature extraction needs no media tools: it reads the wav the media
    // commands already wrote, so its only requirement is the model directory.
    if config.features().is_some() {
        commands.push(TaskKind::ExtractFeatures);
        // The lock needs the same package for a different reason: it reads the
        // encoder's digest out of the package manifest and writes it into
        // `assets.json`, which is what later runs compare against.
        commands.push(TaskKind::LockAssetPackage);
    }
    // Training needs no media tools and no frame models: the frames, landmarks
    // and audio features are already inside the locked project, so the
    // perceptual-loss package is its only requirement.
    if config.training().is_some() {
        commands.push(TaskKind::Train);
    }
    commands
}

pub fn ready_frame(config: &WorkerConfig) -> ReadyFrame {
    let adapters = config.compute().adapters().to_vec();
    let mut backends = vec![Backend::Cpu];
    if adapters
        .iter()
        .any(|adapter| adapter.backend == Backend::Wgpu)
    {
        backends.push(Backend::Wgpu);
    }
    let wgpu_training = config.training().is_some()
        && adapters
            .iter()
            .any(|adapter| adapter.backend == Backend::Wgpu && adapter.certified);
    ReadyFrame {
        protocol_version: PROTOCOL_VERSION,
        worker_version: config.worker_version().to_owned(),
        backends,
        adapters,
        supported_commands: supported_commands(config),
        capabilities: Capabilities {
            training: config.training().is_some(),
            wgpu_training,
            onnx_validation: false,
            ffmpeg: config.media().is_some(),
        },
    }
}
