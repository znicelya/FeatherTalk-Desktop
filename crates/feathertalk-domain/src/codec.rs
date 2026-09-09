use serde::{Serialize, de::DeserializeOwned};

use crate::{DomainError, PROTOCOL_VERSION};

pub const MAX_FRAME_BYTES: usize = 1_048_576;

/// Serialize one value as compact JSON without its trailing line delimiter.
///
/// This is a syntax/framing-layer helper. It checks the serialized byte
/// length (and therefore rejects values that cannot fit in one frame), but it
/// does not run any protocol-specific semantic validator on `value`.
pub fn encode_line<T: Serialize>(value: &T) -> Result<String, DomainError> {
    let line = serde_json::to_string(value).map_err(|error| DomainError::MalformedFrame {
        reason: error.to_string(),
    })?;
    if line.len() > MAX_FRAME_BYTES {
        return Err(DomainError::FrameTooLong {
            limit: MAX_FRAME_BYTES,
        });
    }
    Ok(line)
}

/// Decode one compact JSON line after applying the frame-size and syntax
/// checks.
///
/// This function is intentionally syntax-only: it strips one optional `\n`,
/// rejects blank input, and deserializes with serde, but it does not call a
/// decoded frame's semantic `validate()` method. After decoding a
/// [`crate::ClientFrame`], callers must call [`crate::ClientFrame::validate`];
/// after decoding a [`crate::ServerFrame`], callers must call
/// [`crate::ServerFrame::validate`] before dispatching or handling the frame.
pub fn decode_line<T: DeserializeOwned>(line: &str) -> Result<T, DomainError> {
    let line = line.strip_suffix('\n').unwrap_or(line);
    if line.len() > MAX_FRAME_BYTES {
        return Err(DomainError::FrameTooLong {
            limit: MAX_FRAME_BYTES,
        });
    }
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return Err(DomainError::MalformedFrame {
            reason: "empty line".into(),
        });
    }
    serde_json::from_str(trimmed).map_err(|error| DomainError::MalformedFrame {
        reason: error.to_string(),
    })
}

/// Check that a protocol version exactly matches this crate's wire version.
pub fn check_protocol_version(actual: u32) -> Result<(), DomainError> {
    if actual == PROTOCOL_VERSION {
        Ok(())
    } else {
        Err(DomainError::ProtocolVersion {
            expected: PROTOCOL_VERSION,
            actual,
        })
    }
}
