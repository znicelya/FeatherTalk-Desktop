use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use crate::{DomainError, Event, Request, TaskId, TaskKind, check_protocol_version};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Backend {
    Cpu,
    Wgpu,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AdapterKind {
    Discrete,
    Integrated,
    Cpu,
    Other,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AdapterInfo {
    /// Stable identity. Slice 2 keys the "one training or inference task per
    /// adapter" rule on this value, so it must survive a worker restart.
    pub id: String,
    pub name: String,
    pub backend: Backend,
    pub kind: AdapterKind,
    /// False for adapters shown for experimental detection only. Launch support
    /// is promised for the certified set alone.
    pub certified: bool,
    pub vram_bytes: Option<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Capabilities {
    pub training: bool,
    pub wgpu_training: bool,
    pub onnx_validation: bool,
    pub ffmpeg: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReadyFrame {
    pub protocol_version: u32,
    pub worker_version: String,
    pub backends: Vec<Backend>,
    pub adapters: Vec<AdapterInfo>,
    /// Commands this worker will actually accept. A `start` frame naming any
    /// other command is rejected, so the desktop can grey out unsupported
    /// actions instead of discovering them through a failed task.
    pub supported_commands: Vec<TaskKind>,
    pub capabilities: Capabilities,
}

impl ReadyFrame {
    pub fn validate(&self) -> Result<(), DomainError> {
        check_protocol_version(self.protocol_version)?;
        if self.backends.is_empty() {
            return Err(DomainError::InvalidField {
                field: "backends",
                reason: "a worker must report at least one backend".into(),
            });
        }
        let mut seen = BTreeSet::new();
        for adapter in &self.adapters {
            if adapter.id.is_empty() {
                return Err(DomainError::InvalidField {
                    field: "adapters",
                    reason: "adapter id must not be empty".into(),
                });
            }
            if !seen.insert(adapter.id.as_str()) {
                return Err(DomainError::InvalidField {
                    field: "adapters",
                    reason: format!("duplicate adapter id {}", adapter.id),
                });
            }
        }
        if self.supported_commands.is_empty() {
            return Err(DomainError::InvalidField {
                field: "supported_commands",
                reason: "a worker must report at least one supported command".into(),
            });
        }
        let mut commands = BTreeSet::new();
        for command in &self.supported_commands {
            if !commands.insert(*command) {
                return Err(DomainError::InvalidField {
                    field: "supported_commands",
                    reason: format!("duplicate supported command {}", command.as_slug()),
                });
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StartFrame {
    pub protocol_version: u32,
    pub task_id: TaskId,
    pub request: Request,
}

impl StartFrame {
    pub fn validate(&self) -> Result<(), DomainError> {
        check_protocol_version(self.protocol_version)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CancelFrame {
    pub protocol_version: u32,
    pub task_id: TaskId,
}

impl CancelFrame {
    pub fn validate(&self) -> Result<(), DomainError> {
        check_protocol_version(self.protocol_version)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ShutdownFrame {
    pub protocol_version: u32,
}

impl ShutdownFrame {
    pub fn validate(&self) -> Result<(), DomainError> {
        check_protocol_version(self.protocol_version)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RejectedFrame {
    pub protocol_version: u32,
    pub reason: String,
}

impl RejectedFrame {
    pub fn validate(&self) -> Result<(), DomainError> {
        check_protocol_version(self.protocol_version)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(
    tag = "frame",
    content = "data",
    rename_all = "snake_case",
    deny_unknown_fields
)]
pub enum ClientFrame {
    Start(StartFrame),
    Cancel(CancelFrame),
    Shutdown(ShutdownFrame),
}

impl ClientFrame {
    pub fn protocol_version(&self) -> u32 {
        match self {
            Self::Start(frame) => frame.protocol_version,
            Self::Cancel(frame) => frame.protocol_version,
            Self::Shutdown(frame) => frame.protocol_version,
        }
    }

    pub fn validate(&self) -> Result<(), DomainError> {
        match self {
            Self::Start(frame) => frame.validate(),
            Self::Cancel(frame) => frame.validate(),
            Self::Shutdown(frame) => frame.validate(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(
    tag = "frame",
    content = "data",
    rename_all = "snake_case",
    deny_unknown_fields
)]
pub enum ServerFrame {
    Ready(ReadyFrame),
    Event(Event),
    Rejected(RejectedFrame),
}

impl ServerFrame {
    pub fn protocol_version(&self) -> u32 {
        match self {
            Self::Ready(frame) => frame.protocol_version,
            Self::Event(event) => event.protocol_version,
            Self::Rejected(frame) => frame.protocol_version,
        }
    }

    pub fn validate(&self) -> Result<(), DomainError> {
        match self {
            Self::Ready(frame) => frame.validate(),
            Self::Event(event) => event.validate(),
            Self::Rejected(frame) => frame.validate(),
        }
    }
}
