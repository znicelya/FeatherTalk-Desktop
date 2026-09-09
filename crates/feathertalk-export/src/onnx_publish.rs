//! Publish a validated ONNX model file.

use std::{fs, io::Write, path::Path};

use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::onnx::{
    ONNX_OPSET_VERSION, OnnxModelKind, OnnxValidationError, validate_model_contract,
};

#[derive(Debug, Error)]
pub enum OnnxPublishError {
    #[error("invalid ONNX publication request: {0}")]
    InvalidRequest(String),
    #[error("ONNX contract violation: {0}")]
    Contract(#[from] OnnxValidationError),
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}

/// A published ONNX file, as the client should record it.
#[derive(Debug, Clone)]
pub struct OnnxArtifact {
    kind: OnnxModelKind,
    opset: i64,
    bytes: u64,
    sha256: String,
}

impl OnnxArtifact {
    pub fn kind(&self) -> OnnxModelKind {
        self.kind
    }

    pub fn opset(&self) -> i64 {
        self.opset
    }

    pub fn bytes(&self) -> u64 {
        self.bytes
    }

    pub fn sha256(&self) -> &str {
        &self.sha256
    }
}

/// Validates a serialised graph and publishes it exactly once.
///
/// The contract check runs before anything touches the filesystem, so a file
/// exists only if its IR version, opset, graph name, inputs, outputs and
/// initializer references all hold. The bytes are staged beside the destination
/// and persisted without clobbering: an interrupted publication drops the
/// staging file rather than leaving a partial model where the client will look.
pub fn publish_onnx_model(
    destination: &Path,
    kind: OnnxModelKind,
    bytes: &[u8],
) -> Result<OnnxArtifact, OnnxPublishError> {
    validate_model_contract(bytes, &kind.public_contract())?;
    let parent = destination
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .ok_or_else(|| {
            OnnxPublishError::InvalidRequest(format!(
                "destination has no parent directory: {}",
                destination.display()
            ))
        })?;
    let metadata = fs::symlink_metadata(parent).map_err(|error| {
        OnnxPublishError::InvalidRequest(format!(
            "destination parent is unavailable: {}: {error}",
            parent.display()
        ))
    })?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        return Err(OnnxPublishError::InvalidRequest(format!(
            "destination parent must be an existing non-symlink directory: {}",
            parent.display()
        )));
    }
    let mut temporary = tempfile::NamedTempFile::new_in(parent)?;
    temporary.write_all(bytes)?;
    temporary.as_file_mut().sync_all()?;
    temporary
        .persist_noclobber(destination)
        .map_err(|error| OnnxPublishError::Io(error.error))?;
    let published = u64::try_from(bytes.len()).map_err(|_| {
        OnnxPublishError::InvalidRequest("ONNX model size overflowed u64".to_owned())
    })?;
    Ok(OnnxArtifact {
        kind,
        opset: ONNX_OPSET_VERSION,
        bytes: published,
        sha256: hex::encode(Sha256::digest(bytes)),
    })
}
