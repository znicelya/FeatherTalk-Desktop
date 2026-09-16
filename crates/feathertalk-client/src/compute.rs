//! Compute configuration shared by the CLI and desktop, without model dependencies.

use feathertalk_domain::{AdapterInfo, Backend, ReadyFrame, TaskKind};
use thiserror::Error;

pub const ENV_WORKER_BACKEND: &str = "FEATHERTALK_WORKER_BACKEND";
pub const ENV_WORKER_ADAPTER: &str = "FEATHERTALK_WORKER_ADAPTER";

/// A requested backend and, optionally, an exact advertised device identity.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ComputeOptions {
    pub backend: Backend,
    pub adapter: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ComputeError {
    #[error("invalid {ENV_WORKER_BACKEND} value {0:?}; expected auto, cpu, wgpu, cuda or rocm")]
    InvalidBackend(String),
    #[error("{0} is not valid Unicode")]
    InvalidEnvironment(&'static str),
    #[error("CPU selection requires cpu-0, not adapter {0}")]
    CpuAdapter(String),
    #[error("adapter cpu-0 requires the CPU backend")]
    WgpuCpuAdapter,
    #[error("backend {0:?} is not advertised by this worker")]
    BackendUnavailable(Backend),
    #[error("adapter {0} is no longer advertised by this worker")]
    UnknownAdapter(String),
    #[error("adapter {adapter} belongs to {actual:?}, not the requested {requested:?} backend")]
    BackendMismatch {
        adapter: String,
        actual: Backend,
        requested: Backend,
    },
    #[error("adapter {0} is experimental or a software device and cannot be selected")]
    UnavailableAdapter(String),
    #[error("no certified hardware adapter is available for backend {0:?}")]
    NoAdapter(Backend),
    #[error("this worker does not support command {0:?}")]
    UnsupportedCommand(TaskKind),
    #[error("this worker does not advertise {0:?} training support")]
    TrainingUnavailable(Backend),
}

impl Default for ComputeOptions {
    fn default() -> Self {
        Self {
            backend: Backend::Auto,
            adapter: None,
        }
    }
}

impl ComputeOptions {
    pub fn new(backend: Backend, adapter: Option<String>) -> Result<Self, ComputeError> {
        let adapter = adapter.and_then(|id| {
            let id = id.trim();
            (!id.is_empty()).then(|| id.to_owned())
        });
        match (backend, adapter.as_deref()) {
            (Backend::Cpu, Some(id)) if id != "cpu-0" => {
                return Err(ComputeError::CpuAdapter(id.to_owned()));
            }
            (Backend::Wgpu | Backend::Cuda | Backend::Rocm, Some("cpu-0")) => {
                return Err(ComputeError::WgpuCpuAdapter);
            }
            _ => {}
        }
        Ok(Self { backend, adapter })
    }

    /// `None` means no flags: retain the worker's inherited environment exactly.
    /// Adapter-only flags infer CUDA/ROCm from stable ID prefixes, otherwise
    /// WGPU except for the reserved CPU identity.
    pub fn from_flags(
        backend: Option<Backend>,
        adapter: Option<&str>,
    ) -> Result<Option<Self>, ComputeError> {
        if backend.is_none() && adapter.is_none() {
            return Ok(None);
        }
        let backend = backend.unwrap_or_else(|| match adapter.map(str::trim) {
            Some("cpu-0" | "") | None => Backend::Cpu,
            Some(id) if id.starts_with("cuda-") => Backend::Cuda,
            Some(id) if id.starts_with("rocm-") => Backend::Rocm,
            Some(_) => Backend::Wgpu,
        });
        Self::new(backend, adapter.map(str::to_owned)).map(Some)
    }

    /// Match the worker's environment semantics: absent backend means automatic,
    /// while an empty or unknown backend is an error. Adapter-only environment
    /// configuration resolves the exact ID among advertised backends.
    pub fn from_environment_values(
        backend: Option<&str>,
        adapter: Option<&str>,
    ) -> Result<Self, ComputeError> {
        let backend = match backend.map(str::trim) {
            None | Some("auto") => Backend::Auto,
            Some("cpu") => Backend::Cpu,
            Some("wgpu") => Backend::Wgpu,
            Some("cuda") => Backend::Cuda,
            Some("rocm") => Backend::Rocm,
            Some(value) => return Err(ComputeError::InvalidBackend(value.into())),
        };
        Self::new(backend, adapter.map(str::to_owned))
    }

    pub fn from_env() -> Result<Self, ComputeError> {
        Self::from_child_env(&[])
    }

    /// Resolve the same effective values as `Command::envs`: the final override
    /// wins, and missing overrides retain the parent's environment. Snapshot
    /// these at spawn time so a session validates the configuration it launched.
    pub(crate) fn from_child_env(env: &[(String, String)]) -> Result<Self, ComputeError> {
        fn value(
            key: &'static str,
            env: &[(String, String)],
        ) -> Result<Option<String>, ComputeError> {
            if let Some((_, value)) = env.iter().rev().find(|(name, _)| {
                if cfg!(windows) {
                    name.eq_ignore_ascii_case(key)
                } else {
                    name == key
                }
            }) {
                return Ok(Some(value.clone()));
            }
            match std::env::var(key) {
                Ok(value) => Ok(Some(value)),
                Err(std::env::VarError::NotPresent) => Ok(None),
                Err(std::env::VarError::NotUnicode(_)) => {
                    Err(ComputeError::InvalidEnvironment(key))
                }
            }
        }
        Self::from_environment_values(
            value(ENV_WORKER_BACKEND, env)?.as_deref(),
            value(ENV_WORKER_ADAPTER, env)?.as_deref(),
        )
    }

    /// Always override both keys so choosing a backend cannot inherit an
    /// incompatible adapter. These values belong on the child process only.
    pub fn env_overrides(&self) -> Vec<(String, String)> {
        vec![
            (ENV_WORKER_BACKEND.into(), backend_name(self.backend).into()),
            (
                ENV_WORKER_ADAPTER.into(),
                self.adapter.clone().unwrap_or_default(),
            ),
        ]
    }

    /// Revalidate on every fresh handshake. Automatic choices prefer CUDA,
    /// ROCm, wgpu, then CPU; an explicit device ID never switches devices.
    pub fn resolve_adapter<'a>(
        &self,
        ready: &'a ReadyFrame,
    ) -> Result<&'a AdapterInfo, ComputeError> {
        if self.backend != Backend::Auto && !ready.backends.contains(&self.backend) {
            return Err(ComputeError::BackendUnavailable(self.backend));
        }
        let selected = match self.adapter.as_deref() {
            Some(id) => ready
                .adapters
                .iter()
                .find(|adapter| adapter.id == id)
                .ok_or_else(|| ComputeError::UnknownAdapter(id.into()))?,
            None => ready
                .adapters
                .iter()
                .filter(|adapter| {
                    (self.backend == Backend::Auto || adapter.backend == self.backend)
                        && ready.backends.contains(&adapter.backend)
                        && adapter_is_eligible(adapter)
                })
                .min_by(|left, right| {
                    (left.backend.selection_priority(), &left.id)
                        .cmp(&(right.backend.selection_priority(), &right.id))
                })
                .ok_or(ComputeError::NoAdapter(self.backend))?,
        };
        if self.backend != Backend::Auto && selected.backend != self.backend {
            return Err(ComputeError::BackendMismatch {
                adapter: selected.id.clone(),
                actual: selected.backend,
                requested: self.backend,
            });
        }
        if !ready.backends.contains(&selected.backend) {
            return Err(ComputeError::BackendUnavailable(selected.backend));
        }
        if !adapter_is_eligible(selected) {
            return Err(ComputeError::UnavailableAdapter(selected.id.clone()));
        }
        Ok(selected)
    }

    pub fn validate_for(&self, kind: TaskKind, ready: &ReadyFrame) -> Result<(), ComputeError> {
        if !ready.supported_commands.contains(&kind) {
            return Err(ComputeError::UnsupportedCommand(kind));
        }
        if !uses_compute_selection(kind) {
            return Ok(());
        }
        let selected = self.resolve_adapter(ready)?;
        if kind == TaskKind::Train
            && (!ready.capabilities.training
                || (selected.backend == Backend::Wgpu && !ready.capabilities.wgpu_training))
        {
            return Err(ComputeError::TrainingUnavailable(self.backend));
        }
        Ok(())
    }
}

pub fn adapter_is_eligible(adapter: &AdapterInfo) -> bool {
    adapter.is_selectable()
}

pub fn backend_name(backend: Backend) -> &'static str {
    match backend {
        Backend::Auto => "auto",
        Backend::Cpu => "cpu",
        Backend::Wgpu => "wgpu",
        Backend::Cuda => "cuda",
        Backend::Rocm => "rocm",
    }
}

pub fn uses_compute_selection(kind: TaskKind) -> bool {
    matches!(
        kind,
        TaskKind::Train
            | TaskKind::Render
            | TaskKind::ExtractFrames
            | TaskKind::ExtractFeatures
            | TaskKind::NormalizeMedia
    )
}
