//! Device discovery and the desktop's persistent compute choice. No UI or model types.

use async_channel::Receiver;
use feathertalk_client::{
    ComputeError, ComputeOptions, SessionOptions, WorkerLocator, WorkerSession,
    uses_compute_selection,
};
use feathertalk_domain::{AdapterInfo, Backend, ReadyFrame, TaskKind};

use crate::worker_status::WorkerStatus;

#[derive(Debug, Clone)]
pub struct ComputeState {
    selection: Result<ComputeOptions, String>,
    ready: Option<ReadyFrame>,
    discovery_error: Option<String>,
    refreshing: bool,
}

impl ComputeState {
    pub fn new(selection: Result<ComputeOptions, ComputeError>) -> Self {
        Self {
            selection: selection.map_err(|error| error.to_string()),
            ready: None,
            discovery_error: None,
            refreshing: false,
        }
    }

    pub fn begin_discovery(&mut self) {
        self.refreshing = true;
        self.ready = None;
        self.discovery_error = None;
    }

    pub fn finish_discovery(&mut self, result: Result<ReadyFrame, String>) {
        self.refreshing = false;
        match result {
            Ok(ready) => {
                // Pin explicitly requested backends once they resolve. Auto
                // stays automatic so a refresh can select an available backend.
                if let Ok(selection) = &mut self.selection
                    && selection.backend != Backend::Auto
                    && selection.adapter.is_none()
                    && let Ok(adapter) = selection.resolve_adapter(&ready)
                {
                    selection.adapter = Some(adapter.id.clone());
                }
                self.ready = Some(ready);
                self.discovery_error = None;
            }
            Err(error) => {
                self.ready = None;
                self.discovery_error = Some(error);
            }
        }
    }

    pub fn is_refreshing(&self) -> bool {
        self.refreshing
    }

    pub fn is_available(&self) -> bool {
        self.ready.is_some()
    }

    pub fn adapters(&self) -> &[AdapterInfo] {
        self.ready
            .as_ref()
            .map(|ready| ready.adapters.as_slice())
            .unwrap_or_default()
    }

    pub fn requested(&self) -> Option<&ComputeOptions> {
        self.selection.as_ref().ok()
    }

    pub fn selected_adapter(&self) -> Result<&AdapterInfo, String> {
        let selection = self.selection.as_ref().map_err(Clone::clone)?;
        let ready = self.ready.as_ref().ok_or_else(|| {
            self.discovery_error
                .clone()
                .unwrap_or_else(|| "device discovery has not completed".into())
        })?;
        selection
            .resolve_adapter(ready)
            .map_err(|error| error.to_string())
    }

    pub fn can_select(&self, id: &str) -> bool {
        self.options_for(id).is_ok()
    }

    pub fn select(&mut self, id: &str) -> Result<(), String> {
        let options = self.options_for(id)?;
        self.selection = Ok(options);
        Ok(())
    }

    pub fn select_automatic(&mut self) {
        self.selection = Ok(ComputeOptions::default());
    }

    fn options_for(&self, id: &str) -> Result<ComputeOptions, String> {
        let ready = self
            .ready
            .as_ref()
            .ok_or("device discovery has not completed")?;
        let adapter = ready
            .adapters
            .iter()
            .find(|adapter| adapter.id == id)
            .ok_or_else(|| format!("adapter {id} is no longer advertised by this worker"))?;
        let options = ComputeOptions::new(adapter.backend, Some(adapter.id.clone()))
            .map_err(|error| error.to_string())?;
        options
            .resolve_adapter(ready)
            .map_err(|error| error.to_string())?;
        Ok(options)
    }

    /// Keep a malformed environment request visible even while discovery runs.
    pub fn error(&self) -> Option<String> {
        if let Err(error) = &self.selection {
            return Some(error.clone());
        }
        if let Some(error) = &self.discovery_error {
            return Some(error.clone());
        }
        self.ready
            .as_ref()
            .and_then(|_| self.selected_adapter().err())
    }

    pub fn environment_for(&self, kind: TaskKind) -> Result<Vec<(String, String)>, String> {
        if !uses_compute_selection(kind) {
            return Ok(ComputeOptions {
                backend: Backend::Cpu,
                adapter: None,
            }
            .env_overrides());
        }
        let selection = self.selection.as_ref().map_err(Clone::clone)?;
        let ready = self.ready.as_ref().ok_or_else(|| {
            self.discovery_error
                .clone()
                .unwrap_or_else(|| "device discovery has not completed".into())
        })?;
        selection
            .validate_for(kind, ready)
            .map_err(|error| error.to_string())?;
        Ok(selection.env_overrides())
    }

    /// Existing pages share this admission key and keep their own prerequisite
    /// checks. Detailed configuration errors stay next to the device selector.
    pub fn blocked_key(&self, kind: TaskKind) -> Option<&'static str> {
        if !uses_compute_selection(kind) || self.environment_for(kind).is_ok() {
            None
        } else if self.refreshing {
            Some("compute.discovering")
        } else {
            Some("compute.blocked")
        }
    }

    /// Admission for every worker command, independent of whether it uses the
    /// selected GPU. A missing media tool/model disables only affected commands.
    pub fn command_blocked_key(&self, kind: TaskKind) -> Option<&'static str> {
        if self.refreshing {
            return Some("compute.discovering");
        }
        let Some(ready) = &self.ready else {
            return Some("compute.unavailable");
        };
        if !ready.supported_commands.contains(&kind) {
            return Some("compute.unsupported");
        }
        self.blocked_key(kind)
    }
}

#[derive(Debug)]
pub struct Discovery {
    pub worker: WorkerStatus,
    pub ready: Result<ReadyFrame, String>,
}

/// Handshake and shutdown both happen off the window thread. Even if the
/// receiver goes away, the probe reaps its worker before trying to publish.
pub fn spawn_discovery(
    locator: WorkerLocator,
    options: SessionOptions,
) -> std::io::Result<Receiver<Discovery>> {
    let (sender, receiver) = async_channel::bounded(1);
    std::thread::Builder::new()
        .name("compute-discovery".into())
        .spawn(move || {
            let worker = WorkerStatus::probe(&locator);
            let ready = match &worker {
                WorkerStatus::Ready { path } => {
                    // Discovery must still work when the selected environment is
                    // invalid; the selection itself remains an error in ComputeState.
                    match WorkerSession::spawn_with_env(
                        path,
                        options,
                        &ComputeOptions {
                            backend: Backend::Cpu,
                            adapter: None,
                        }
                        .env_overrides(),
                    ) {
                        Ok(session) => {
                            let ready = session.ready().clone();
                            match session.shutdown() {
                                Some(0) => Ok(ready),
                                status => Err(format!(
                                    "device probe worker did not shut down cleanly: {status:?}"
                                )),
                            }
                        }
                        Err(error) => Err(error.to_string()),
                    }
                }
                WorkerStatus::Missing { .. } => {
                    Err("worker executable is unavailable; refresh after configuring it".into())
                }
            };
            let _ = sender.send_blocking(Discovery { worker, ready });
        })?;
    Ok(receiver)
}
