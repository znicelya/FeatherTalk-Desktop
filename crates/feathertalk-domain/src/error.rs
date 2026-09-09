#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum DomainError {
    #[error("invalid task id: {reason}")]
    InvalidTaskId { reason: String },
    #[error("invalid task transition from {from} to {to}")]
    InvalidTransition {
        from: &'static str,
        to: &'static str,
    },
    #[error("frame exceeds the {limit} byte limit")]
    FrameTooLong { limit: usize },
    #[error("malformed frame: {reason}")]
    MalformedFrame { reason: String },
    #[error("protocol version mismatch: expected {expected}, got {actual}")]
    ProtocolVersion { expected: u32, actual: u32 },
    #[error("invalid {field}: {reason}")]
    InvalidField { field: &'static str, reason: String },
}
