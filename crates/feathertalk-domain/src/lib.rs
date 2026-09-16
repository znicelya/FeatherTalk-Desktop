mod codec;
mod error;
mod event;
mod frame;
mod lifecycle;
mod request;
mod stage;
mod stream;
mod task;
mod task_error;

/// Version of the worker task protocol this crate speaks.
///
/// Version 2 added `supported_commands` to the handshake and `result` to
/// completed events. Version 4 added the Linux ROCm execution backend.
pub const PROTOCOL_VERSION: u32 = 4;

pub use codec::{MAX_FRAME_BYTES, check_protocol_version, decode_line, encode_line};
pub use error::DomainError;
pub use event::{Event, Metrics, Progress};
pub use frame::{
    AdapterInfo, AdapterKind, Backend, CancelFrame, Capabilities, ClientFrame, ReadyFrame,
    RejectedFrame, ServerFrame, ShutdownFrame, StartFrame,
};
pub use lifecycle::TaskLifecycle;
pub use request::{
    DEFAULT_BATCH_SIZE, ExportModelPackageParams, ExportOnnxParams, ExtractFeaturesParams,
    ExtractFramesParams, ImportLegacyModelParams, InspectModelParams, LegacyModelKind,
    MigrateLegacyFeaturesParams, NormalizeMediaParams, OnnxExportKind, ProbeMediaParams,
    ProjectDirParams, RenderParams, Request, TrainParams, TrainingMode, UnetVariant,
};
pub use stage::TaskStage;
pub use stream::{FrameReader, FrameWriter};
pub use task::{
    TASK_ID_LEN, TASK_ID_MILLIS_DIGITS, TASK_ID_SUFFIX_DIGITS, TaskId, TaskKind, TaskStatus,
};
pub use task_error::{ErrorCode, MAX_DETAIL_CHARS, MAX_SUMMARY_CHARS, Recovery, TaskError};
