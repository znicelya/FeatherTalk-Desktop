use std::{
    ffi::OsString,
    path::{Path, PathBuf},
    time::Duration,
};

use feathertalk_domain::{AdapterInfo, Backend, TaskKind};
use feathertalk_media::MediaToolchain;

use crate::{ComputeRegistry, cpu_adapter};

pub const ENV_FFPROBE: &str = "FEATHERTALK_WORKER_FFPROBE";
pub const ENV_FFMPEG: &str = "FEATHERTALK_WORKER_FFMPEG";
pub const ENV_MEDIA_TIMEOUT_MS: &str = "FEATHERTALK_WORKER_MEDIA_TIMEOUT_MS";
pub const DEFAULT_MEDIA_TIMEOUT_MS: u64 = 300_000;
pub const ENV_SCRFD_DIR: &str = "FEATHERTALK_WORKER_SCRFD_DIR";
pub const ENV_PFLD_DIR: &str = "FEATHERTALK_WORKER_PFLD_DIR";
pub const ENV_HUBERT_DIR: &str = "FEATHERTALK_WORKER_HUBERT_DIR";
pub const ENV_VGG19_DIR: &str = "FEATHERTALK_WORKER_VGG19_DIR";
pub const ENV_BACKEND: &str = "FEATHERTALK_WORKER_BACKEND";
pub const ENV_ADAPTER: &str = "FEATHERTALK_WORKER_ADAPTER";

#[derive(Debug, Clone)]
struct ComputeChoice {
    backend: Backend,
    adapter_id: Option<String>,
}

/// Where the worker finds the two model artifact directories.
///
/// Only the shape of the paths is checked here. Whether the directories hold a
/// loadable manifest and weights is discovered when the first job loads them,
/// because a directory can disappear between startup and the first job.
#[derive(Debug, Clone)]
pub struct ModelToolchain {
    scrfd_dir: PathBuf,
    pfld_dir: PathBuf,
}

impl ModelToolchain {
    pub fn scrfd_dir(&self) -> &Path {
        &self.scrfd_dir
    }

    pub fn pfld_dir(&self) -> &Path {
        &self.pfld_dir
    }
}

/// Where the worker finds the FeatherHuBERT model package.
///
/// Only the shape of the path is checked here, for the same reason as
/// `ModelToolchain`: a directory can disappear between startup and the first
/// job, so the manifest and the weights are validated when a job loads them.
#[derive(Debug, Clone)]
pub struct FeatureToolchain {
    hubert_dir: PathBuf,
}

impl FeatureToolchain {
    pub fn hubert_dir(&self) -> &Path {
        &self.hubert_dir
    }
}

/// Where the worker finds the VGG19 perceptual-loss package.
///
/// Only the shape of the path is checked here, for the same reason as
/// `FeatureToolchain`: the manifest, the licence bundle and the safetensors
/// weights are validated when a training job loads them, because a directory
/// can disappear between startup and the first job.
#[derive(Debug, Clone)]
pub struct TrainingToolchain {
    vgg19_dir: PathBuf,
}

impl TrainingToolchain {
    pub fn vgg19_dir(&self) -> &Path {
        &self.vgg19_dir
    }
}

/// Everything the worker learns from its environment at startup.
///
/// A missing or unusable media toolchain is not a startup failure: the worker
/// still serves `validate_project` and simply reports `probe_media` as
/// unsupported, with the reason kept for the rejection message.
#[derive(Debug, Clone)]
pub struct WorkerConfig {
    worker_version: String,
    media: Option<MediaToolchain>,
    media_rejection: Option<String>,
    models: Option<ModelToolchain>,
    model_rejection: Option<String>,
    features: Option<FeatureToolchain>,
    feature_rejection: Option<String>,
    training: Option<TrainingToolchain>,
    training_rejection: Option<String>,
    compute: ComputeRegistry,
    compute_choice: Result<ComputeChoice, String>,
}

impl WorkerConfig {
    pub fn from_env() -> Self {
        let executable = std::env::current_exe().ok();
        // Invalid Unicode must not turn an explicit device request into an
        // absent setting. Adapter IDs and the two backend names are ASCII.
        let backend =
            std::env::var_os(ENV_BACKEND).map(|value| value.to_string_lossy().into_owned());
        let adapter =
            std::env::var_os(ENV_ADAPTER).map(|value| value.to_string_lossy().into_owned());
        Self::from_values_with_training(
            media_tool_path(
                std::env::var_os(ENV_FFPROBE),
                executable.as_deref(),
                "ffprobe",
            ),
            media_tool_path(
                std::env::var_os(ENV_FFMPEG),
                executable.as_deref(),
                "ffmpeg",
            ),
            std::env::var(ENV_MEDIA_TIMEOUT_MS).ok(),
            model_package_path(
                std::env::var_os(ENV_SCRFD_DIR),
                executable.as_deref(),
                "scrfd_2_5g",
            ),
            model_package_path(
                std::env::var_os(ENV_PFLD_DIR),
                executable.as_deref(),
                "pfld_ghost_one",
            ),
            model_package_path(
                std::env::var_os(ENV_HUBERT_DIR),
                executable.as_deref(),
                "feather_hubert",
            ),
            model_package_path(
                std::env::var_os(ENV_VGG19_DIR),
                executable.as_deref(),
                "vgg19",
            ),
        )
        .with_compute_registry(ComputeRegistry::discover())
        .with_compute_selection(backend.as_deref(), adapter.as_deref())
    }

    /// The media-only form: no model directories, so `extract_frames` stays
    /// unsupported.
    pub fn from_values(
        ffprobe: Option<String>,
        ffmpeg: Option<String>,
        timeout_ms: Option<String>,
    ) -> Self {
        Self::from_values_with_models(ffprobe, ffmpeg, timeout_ms, None, None)
    }

    /// The frame form: no FeatherHuBERT directory, so `extract_features` stays
    /// unsupported.
    pub fn from_values_with_models(
        ffprobe: Option<String>,
        ffmpeg: Option<String>,
        timeout_ms: Option<String>,
        scrfd_dir: Option<String>,
        pfld_dir: Option<String>,
    ) -> Self {
        Self::from_values_with_toolchains(ffprobe, ffmpeg, timeout_ms, scrfd_dir, pfld_dir, None)
    }

    /// The toolchain form: no VGG19 directory, so `train` stays unsupported.
    pub fn from_values_with_toolchains(
        ffprobe: Option<String>,
        ffmpeg: Option<String>,
        timeout_ms: Option<String>,
        scrfd_dir: Option<String>,
        pfld_dir: Option<String>,
        hubert_dir: Option<String>,
    ) -> Self {
        Self::from_values_with_training(
            ffprobe, ffmpeg, timeout_ms, scrfd_dir, pfld_dir, hubert_dir, None,
        )
    }

    /// The training form: the VGG19 package the perceptual loss reads. Training
    /// needs no media tools and no frame models, so this is orthogonal to every
    /// other toolchain.
    pub fn from_values_with_training(
        ffprobe: Option<String>,
        ffmpeg: Option<String>,
        timeout_ms: Option<String>,
        scrfd_dir: Option<String>,
        pfld_dir: Option<String>,
        hubert_dir: Option<String>,
        vgg19_dir: Option<String>,
    ) -> Self {
        let (media, media_rejection) = match media_toolchain(ffprobe, ffmpeg, timeout_ms) {
            Ok(toolchain) => (Some(toolchain), None),
            Err(reason) => (None, Some(reason)),
        };
        let (models, model_rejection) = match model_toolchain(scrfd_dir, pfld_dir) {
            Ok(toolchain) => (Some(toolchain), None),
            Err(reason) => (None, Some(reason)),
        };
        let (features, feature_rejection) = match feature_toolchain(hubert_dir) {
            Ok(toolchain) => (Some(toolchain), None),
            Err(reason) => (None, Some(reason)),
        };
        let (training, training_rejection) = match training_toolchain(vgg19_dir) {
            Ok(toolchain) => (Some(toolchain), None),
            Err(reason) => (None, Some(reason)),
        };
        Self {
            worker_version: env!("CARGO_PKG_VERSION").to_owned(),
            media,
            media_rejection,
            models,
            model_rejection,
            features,
            feature_rejection,
            training,
            training_rejection,
            compute: ComputeRegistry::cpu_only(),
            compute_choice: Ok(ComputeChoice {
                backend: Backend::Auto,
                adapter_id: None,
            }),
        }
    }

    /// Supply a discovered registry without making value constructors probe
    /// hardware. Clones keep the same exact native handles and device cache.
    pub fn with_compute_registry(mut self, registry: ComputeRegistry) -> Self {
        self.compute = registry;
        self
    }

    pub fn with_compute_selection(mut self, backend: Option<&str>, adapter: Option<&str>) -> Self {
        self.compute_choice = match backend.map(str::trim) {
            None | Some("auto") => Ok(Backend::Auto),
            Some("cpu") => Ok(Backend::Cpu),
            Some("wgpu") => Ok(Backend::Wgpu),
            Some("cuda") => Ok(Backend::Cuda),
            Some("rocm") => Ok(Backend::Rocm),
            Some(value) => Err(format!(
                "{ENV_BACKEND} must be auto, cpu, wgpu, cuda or rocm, got {value:?}"
            )),
        }
        .map(|backend| ComputeChoice {
            backend,
            adapter_id: adapter
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(str::to_owned),
        });
        self
    }

    pub fn compute(&self) -> &ComputeRegistry {
        &self.compute
    }

    pub fn compute_adapter(&self) -> Result<AdapterInfo, String> {
        let choice = self.compute_choice.as_ref().map_err(Clone::clone)?;
        self.compute
            .resolve(choice.backend, choice.adapter_id.as_deref())
            .map_err(|reason| format!("{ENV_BACKEND}/{ENV_ADAPTER}: {reason}"))
    }

    pub(crate) fn cuda_device_index(&self) -> Result<Option<usize>, String> {
        let adapter = self.compute_adapter()?;
        Ok(self.compute.cuda_device_index(&adapter.id))
    }

    pub(crate) fn adapter_for(&self, kind: TaskKind) -> Result<AdapterInfo, String> {
        if uses_compute(kind) {
            self.compute_adapter()
        } else {
            Ok(cpu_adapter())
        }
    }

    pub fn worker_version(&self) -> &str {
        &self.worker_version
    }

    pub fn media(&self) -> Option<&MediaToolchain> {
        self.media.as_ref()
    }

    pub fn media_rejection(&self) -> Option<&str> {
        self.media_rejection.as_deref()
    }

    pub fn models(&self) -> Option<&ModelToolchain> {
        self.models.as_ref()
    }

    pub fn model_rejection(&self) -> Option<&str> {
        self.model_rejection.as_deref()
    }

    pub fn features(&self) -> Option<&FeatureToolchain> {
        self.features.as_ref()
    }

    pub fn feature_rejection(&self) -> Option<&str> {
        self.feature_rejection.as_deref()
    }

    pub fn training(&self) -> Option<&TrainingToolchain> {
        self.training.as_ref()
    }

    pub fn training_rejection(&self) -> Option<&str> {
        self.training_rejection.as_deref()
    }
}

pub(crate) fn uses_compute(kind: TaskKind) -> bool {
    matches!(
        kind,
        TaskKind::Train
            | TaskKind::Render
            | TaskKind::ExtractFrames
            | TaskKind::ExtractFeatures
            | TaskKind::NormalizeMedia
    )
}

/// Installer and portable distributions keep media executables beside the worker.
/// Explicit configuration, including an invalid value, must keep taking priority.
fn media_tool_path(
    configured: Option<OsString>,
    executable: Option<&Path>,
    name: &str,
) -> Option<String> {
    if let Some(configured) = configured {
        // The configuration parser accepts Unicode paths. An invalid encoding
        // remains an explicit invalid value, rejected by required_path below.
        return Some(configured.into_string().unwrap_or_default());
    }
    let path = executable?
        .parent()?
        .join(format!("{name}{}", std::env::consts::EXE_SUFFIX));
    if path.is_absolute() && path.is_file() {
        path.to_str().map(str::to_owned)
    } else {
        None
    }
}

/// Bundled base models live in the executable's models directory. Validate the
/// package contents when a job loads them, as with explicitly configured paths.
fn model_package_path(
    configured: Option<OsString>,
    executable: Option<&Path>,
    name: &str,
) -> Option<String> {
    if let Some(configured) = configured {
        return Some(configured.into_string().unwrap_or_default());
    }
    let path = executable?.parent()?.join("models").join(name);
    if path.is_absolute()
        && path.join("manifest.json").is_file()
        && path.join("model.safetensors").is_file()
    {
        path.to_str().map(str::to_owned)
    } else {
        None
    }
}

fn media_toolchain(
    ffprobe: Option<String>,
    ffmpeg: Option<String>,
    timeout_ms: Option<String>,
) -> Result<MediaToolchain, String> {
    let ffprobe = required_path(ffprobe, ENV_FFPROBE)?;
    let ffmpeg = required_path(ffmpeg, ENV_FFMPEG)?;
    let timeout_ms = match timeout_ms {
        None => DEFAULT_MEDIA_TIMEOUT_MS,
        Some(value) => value.trim().parse::<u64>().map_err(|_| {
            format!("{ENV_MEDIA_TIMEOUT_MS} must be a whole number of milliseconds, got {value:?}")
        })?,
    };
    if timeout_ms == 0 {
        return Err(format!("{ENV_MEDIA_TIMEOUT_MS} must be greater than zero"));
    }
    MediaToolchain::new(ffmpeg, ffprobe, Duration::from_millis(timeout_ms))
        .map_err(|error| error.to_string())
}

fn model_toolchain(
    scrfd_dir: Option<String>,
    pfld_dir: Option<String>,
) -> Result<ModelToolchain, String> {
    let scrfd_dir = required_path(scrfd_dir, ENV_SCRFD_DIR)?;
    let pfld_dir = required_path(pfld_dir, ENV_PFLD_DIR)?;
    Ok(ModelToolchain {
        scrfd_dir,
        pfld_dir,
    })
}

fn feature_toolchain(hubert_dir: Option<String>) -> Result<FeatureToolchain, String> {
    let hubert_dir = required_path(hubert_dir, ENV_HUBERT_DIR)?;
    Ok(FeatureToolchain { hubert_dir })
}

fn training_toolchain(vgg19_dir: Option<String>) -> Result<TrainingToolchain, String> {
    let vgg19_dir = required_path(vgg19_dir, ENV_VGG19_DIR)?;
    Ok(TrainingToolchain { vgg19_dir })
}

fn required_path(value: Option<String>, variable: &str) -> Result<PathBuf, String> {
    let value = value.ok_or_else(|| format!("{variable} is not set"))?;
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return Err(format!("{variable} must not be empty"));
    }
    let path = PathBuf::from(trimmed);
    if !path.is_absolute() {
        return Err(format!(
            "{variable} must be an absolute path, got {trimmed:?}"
        ));
    }
    Ok(path)
}

#[cfg(test)]
mod bundled_media_tests {
    use super::*;

    #[test]
    fn bundled_media_is_found_beside_the_worker() {
        let directory = tempfile::tempdir().unwrap();
        let worker = directory.path().join("feathertalk-worker.exe");
        let ffmpeg = directory
            .path()
            .join(format!("ffmpeg{}", std::env::consts::EXE_SUFFIX));
        std::fs::write(&ffmpeg, b"bundled executable").unwrap();

        let path = media_tool_path(None, Some(&worker), "ffmpeg").unwrap();
        assert_eq!(PathBuf::from(path), ffmpeg);
    }

    #[test]
    fn bundled_media_does_not_replace_explicit_configuration() {
        let directory = tempfile::tempdir().unwrap();
        let worker = directory.path().join("feathertalk-worker.exe");
        let ffmpeg = directory
            .path()
            .join(format!("ffmpeg{}", std::env::consts::EXE_SUFFIX));
        std::fs::write(ffmpeg, b"bundled executable").unwrap();

        for configured in ["", "relative/tool", "X:/missing/ffmpeg.exe"] {
            assert_eq!(
                media_tool_path(Some(configured.into()), Some(&worker), "ffmpeg"),
                Some(configured.to_owned())
            );
        }
    }

    #[test]
    fn bundled_media_requires_a_file_and_a_known_executable_location() {
        let directory = tempfile::tempdir().unwrap();
        let worker = directory.path().join("feathertalk-worker.exe");
        assert_eq!(media_tool_path(None, Some(&worker), "ffprobe"), None);
        let candidate = directory
            .path()
            .join(format!("ffprobe{}", std::env::consts::EXE_SUFFIX));
        std::fs::create_dir(candidate).unwrap();
        assert_eq!(media_tool_path(None, Some(&worker), "ffprobe"), None);
        assert_eq!(media_tool_path(None, None, "ffprobe"), None);
    }

    #[cfg(windows)]
    #[test]
    fn bundled_media_does_not_replace_a_non_unicode_override() {
        use std::os::windows::ffi::OsStringExt;

        let directory = tempfile::tempdir().unwrap();
        let worker = directory.path().join("feathertalk-worker.exe");
        std::fs::write(directory.path().join("ffmpeg.exe"), b"bundled executable").unwrap();
        let invalid = OsString::from_wide(&[0xd800]);
        let resolved = media_tool_path(Some(invalid), Some(&worker), "ffmpeg");
        let config =
            WorkerConfig::from_values(Some(worker.to_str().unwrap().to_owned()), resolved, None);
        assert!(config.media().is_none());
        assert!(config.media_rejection().unwrap().contains(ENV_FFMPEG));
    }
}

#[cfg(test)]
mod bundled_model_tests {
    use super::*;

    fn package(root: &Path, name: &str) -> PathBuf {
        let path = root.join("models").join(name);
        std::fs::create_dir_all(&path).unwrap();
        std::fs::write(path.join("manifest.json"), b"{}").unwrap();
        std::fs::write(path.join("model.safetensors"), b"model weights").unwrap();
        path
    }

    #[test]
    fn bundled_models_are_found_relative_to_the_worker() {
        let directory = tempfile::tempdir().unwrap();
        let worker = directory.path().join("feathertalk-worker.exe");
        for name in ["scrfd_2_5g", "pfld_ghost_one", "feather_hubert", "vgg19"] {
            let expected = package(directory.path(), name);
            let actual = model_package_path(None, Some(&worker), name).unwrap();
            assert_eq!(PathBuf::from(actual), expected);
        }
    }

    #[test]
    fn bundled_models_require_manifest_weights_and_an_absolute_location() {
        let directory = tempfile::tempdir().unwrap();
        let worker = directory.path().join("feathertalk-worker.exe");
        assert_eq!(model_package_path(None, Some(&worker), "scrfd_2_5g"), None);
        let path = directory.path().join("models/scrfd_2_5g");
        std::fs::create_dir_all(&path).unwrap();
        std::fs::write(path.join("manifest.json"), b"{}").unwrap();
        assert_eq!(model_package_path(None, Some(&worker), "scrfd_2_5g"), None);
        std::fs::create_dir(path.join("model.safetensors")).unwrap();
        assert_eq!(model_package_path(None, Some(&worker), "scrfd_2_5g"), None);
        assert_eq!(model_package_path(None, None, "scrfd_2_5g"), None);
        assert_eq!(
            model_package_path(None, Some(Path::new("worker.exe")), "scrfd_2_5g"),
            None
        );
    }

    #[test]
    fn bundled_models_preserve_explicit_configuration_even_when_invalid() {
        let directory = tempfile::tempdir().unwrap();
        let worker = directory.path().join("feathertalk-worker.exe");
        for name in ["feather_hubert", "vgg19"] {
            package(directory.path(), name);
            for configured in ["", "relative/model", "X:/missing/model"] {
                assert_eq!(
                    model_package_path(Some(configured.into()), Some(&worker), name),
                    Some(configured.to_owned())
                );
            }
        }
    }

    #[cfg(windows)]
    #[test]
    fn bundled_models_do_not_replace_a_non_unicode_override() {
        use std::os::windows::ffi::OsStringExt;

        let directory = tempfile::tempdir().unwrap();
        let worker = directory.path().join("feathertalk-worker.exe");
        package(directory.path(), "feather_hubert");
        let configured = OsString::from_wide(&[0xd800]);
        let resolved = model_package_path(Some(configured), Some(&worker), "feather_hubert");
        let config =
            WorkerConfig::from_values_with_toolchains(None, None, None, None, None, resolved);
        assert!(config.features().is_none());
        assert!(config.feature_rejection().unwrap().contains(ENV_HUBERT_DIR));

        package(directory.path(), "vgg19");
        let configured = OsString::from_wide(&[0xd800]);
        let resolved = model_package_path(Some(configured), Some(&worker), "vgg19");
        let config =
            WorkerConfig::from_values_with_training(None, None, None, None, None, None, resolved);
        assert!(config.training().is_none());
        assert!(config.training_rejection().unwrap().contains(ENV_VGG19_DIR));
    }
}
