use std::path::{Path, PathBuf};

use crate::{ClientError, ProbedPath, WorkerPathSource};

/// Environment variable naming the worker executable.
pub const ENV_WORKER_BIN: &str = "FEATHERTALK_WORKER_BIN";

/// File stem of the worker executable, without a platform suffix.
pub const WORKER_FILE_STEM: &str = "feathertalk-worker";

/// The three places a worker executable is looked for, in priority order.
#[derive(Debug, Clone, Default)]
pub struct WorkerLocator {
    cli_option: Option<PathBuf>,
    env_var: Option<PathBuf>,
    sibling: Option<PathBuf>,
}

impl WorkerLocator {
    /// Read the environment once and build the candidate list.
    pub fn from_env(cli_option: Option<PathBuf>) -> Self {
        let env_var = std::env::var_os(ENV_WORKER_BIN)
            .filter(|value| !value.is_empty())
            .map(PathBuf::from);
        let sibling = std::env::current_exe()
            .ok()
            .as_deref()
            .and_then(Self::sibling_of);
        Self::from_parts(cli_option, env_var, sibling)
    }

    /// Test seam: the same logic with the environment supplied by the caller.
    pub fn from_parts(
        cli_option: Option<PathBuf>,
        env_var: Option<PathBuf>,
        sibling: Option<PathBuf>,
    ) -> Self {
        Self {
            cli_option,
            env_var,
            sibling,
        }
    }

    /// The worker that would sit next to `exe` in the same directory.
    pub fn sibling_of(exe: &Path) -> Option<PathBuf> {
        let directory = exe.parent()?;
        Some(directory.join(format!(
            "{WORKER_FILE_STEM}{}",
            std::env::consts::EXE_SUFFIX
        )))
    }

    /// Every source in priority order, whether or not it was set.
    pub fn candidates(&self) -> Vec<ProbedPath> {
        vec![
            ProbedPath {
                source: WorkerPathSource::CliOption,
                path: self.cli_option.clone(),
            },
            ProbedPath {
                source: WorkerPathSource::EnvVar,
                path: self.env_var.clone(),
            },
            ProbedPath {
                source: WorkerPathSource::SiblingOfCurrentExe,
                path: self.sibling.clone(),
            },
        ]
    }

    /// Resolve the worker executable.
    ///
    /// The highest-priority source that is *set* decides the outcome. A path
    /// that was configured but does not exist is an error rather than a
    /// fall-through, because silently running a different binary than the
    /// operator named is worse than failing.
    pub fn resolve(&self) -> Result<PathBuf, ClientError> {
        let candidates = self.candidates();
        let configured = candidates
            .iter()
            .find_map(|candidate| candidate.path.clone());
        match configured {
            Some(path) if path.is_file() => Ok(path),
            _ => Err(ClientError::WorkerNotFound { probed: candidates }),
        }
    }
}
