//! What a model directory is, and whether this build can use it.

use std::{fs, path::Path};

use feathertalk_domain::{ErrorCode, TaskError, TaskStage};
use feathertalk_export::{MODEL_FILE_NAME, ModelPackageManifest};
use feathertalk_training::{CHECKPOINT_MODEL_FILE_NAME, TrainingCheckpointManifest};

use crate::{
    admission::check_model_source, error_map::clamp, rendering::render_variant,
    training::checkpoint_descriptor,
};

/// The two directory layouts `inspect_model` accepts (design section 2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelSourceKind {
    ModelPackage,
    TrainingCheckpoint,
}

impl ModelSourceKind {
    /// The wire spelling. It goes into the payload's `source_kind`, so it is
    /// part of the protocol and must not drift.
    pub fn as_slug(self) -> &'static str {
        match self {
            Self::ModelPackage => "model_package",
            Self::TrainingCheckpoint => "training_checkpoint",
        }
    }
}

/// Decide the layout from the one file only that layout has. An exported package
/// carries `model.safetensors`; a checkpoint carries `model.bin`. Neither file is
/// opened -- the digests in the manifests are the source of truth for content,
/// and reading weights is out of scope (design section 3).
pub fn model_source_kind(source: &Path) -> Result<ModelSourceKind, TaskError> {
    check_model_source(source)?;
    let package = is_regular_file(&source.join(MODEL_FILE_NAME));
    let checkpoint = is_regular_file(&source.join(CHECKPOINT_MODEL_FILE_NAME));
    match (package, checkpoint) {
        (true, false) => Ok(ModelSourceKind::ModelPackage),
        (false, true) => Ok(ModelSourceKind::TrainingCheckpoint),
        // Both means the directory is two things at once and neither reader
        // would be right; neither means it is not a model directory at all.
        _ => Err(unrecognized(source)),
    }
}

fn is_regular_file(path: &Path) -> bool {
    fs::symlink_metadata(path)
        .map(|metadata| metadata.is_file())
        .unwrap_or(false)
}

fn unrecognized(source: &Path) -> TaskError {
    TaskError::new(
        ErrorCode::ModelIncompatible,
        "无法识别的模型目录",
        &clamp(&format!(
            "{} holds neither exactly one {MODEL_FILE_NAME} nor exactly one \
             {CHECKPOINT_MODEL_FILE_NAME}",
            source.display()
        )),
        TaskStage::Preparing,
    )
}

/// The reason codes a client may see. They are identifiers, not sentences: the
/// CLI and the UI own the wording, and a rename here is a protocol change.
const REASON_MINIMUM_APP_VERSION: &str = "minimum_app_version";
const REASON_MODEL_KIND: &str = "model_kind";
const REASON_ARCHITECTURE_VERSION: &str = "architecture_version";
const REASON_MODEL_CONFIG_SHA256: &str = "model_config_sha256";
const REASON_FILE_SIZE: &str = "file_size";

/// One file a manifest names, as the manifest describes it and as the disk
/// answers. `bytes_on_disk` is `None` when the file could not be stated at all,
/// which is reported as a size disagreement rather than as a zero.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InspectedFile {
    pub file_name: String,
    pub bytes: u64,
    pub sha256: String,
    pub bytes_on_disk: Option<u64>,
}

impl InspectedFile {
    /// Whether the disk agrees with the manifest about the size. The digest is
    /// not recomputed: hashing gigabytes to answer a listing would turn an
    /// interactive command into a long job (design section 3).
    pub fn agrees(&self) -> bool {
        self.bytes_on_disk == Some(self.bytes)
    }

    fn from_manifest(directory: &Path, file_name: &str, bytes: u64, sha256: &str) -> Self {
        Self {
            file_name: file_name.to_owned(),
            bytes,
            sha256: sha256.to_owned(),
            bytes_on_disk: fs::symlink_metadata(directory.join(file_name))
                .ok()
                .filter(|metadata| metadata.is_file())
                .map(|metadata| metadata.len()),
        }
    }
}

/// The two files an exported package manifest names, in manifest order.
pub fn package_files(directory: &Path, manifest: &ModelPackageManifest) -> Vec<InspectedFile> {
    [&manifest.model, &manifest.licenses]
        .into_iter()
        .map(|file| {
            InspectedFile::from_manifest(directory, &file.file_name, file.bytes, &file.sha256)
        })
        .collect()
}

/// The three files a checkpoint manifest names, in manifest order.
pub fn checkpoint_files(
    directory: &Path,
    manifest: &TrainingCheckpointManifest,
) -> Vec<InspectedFile> {
    [
        &manifest.model,
        &manifest.optimizer,
        &manifest.training_state,
    ]
    .into_iter()
    .map(|file| InspectedFile::from_manifest(directory, &file.file_name, file.bytes, &file.sha256))
    .collect()
}

/// Why this build cannot use this package, in a fixed order.
pub fn package_incompatibilities(
    manifest: &ModelPackageManifest,
    files: &[InspectedFile],
    worker_version: &str,
) -> Vec<&'static str> {
    let mut reasons = Vec::new();
    if !version_at_least(worker_version, &manifest.minimum_app_version) {
        reasons.push(REASON_MINIMUM_APP_VERSION);
    }
    push_file_size(&mut reasons, files);
    reasons
}

/// Why this build cannot use this checkpoint, in a fixed order.
pub fn checkpoint_incompatibilities(
    manifest: &TrainingCheckpointManifest,
    files: &[InspectedFile],
) -> Vec<&'static str> {
    let mut reasons = Vec::new();
    match render_variant(&manifest.model_kind) {
        // Without a known kind there is no configuration to compare against, so
        // the architecture and digest checks are not "passed", they are unasked.
        None => reasons.push(REASON_MODEL_KIND),
        Some(variant) => match checkpoint_descriptor(&variant.configuration()) {
            Err(_) => reasons.push(REASON_MODEL_KIND),
            Ok(mine) => {
                if manifest.architecture_version != mine.architecture_version {
                    reasons.push(REASON_ARCHITECTURE_VERSION);
                }
                if manifest.model_config_sha256 != mine.model_config_sha256 {
                    reasons.push(REASON_MODEL_CONFIG_SHA256);
                }
            }
        },
    }
    push_file_size(&mut reasons, files);
    reasons
}

fn push_file_size(reasons: &mut Vec<&'static str>, files: &[InspectedFile]) {
    if files.iter().any(|file| !file.agrees()) {
        reasons.push(REASON_FILE_SIZE);
    }
}

/// Whether `have` is at least `want`, both `major.minor.patch`. Written here
/// rather than pulled in: `semver` is not a workspace dependency, and
/// `ModelPackageManifest::validate` has already pinned the manifest side to three
/// numeric components. An unparseable version is not "at least" anything.
fn version_at_least(have: &str, want: &str) -> bool {
    match (parse_version(have), parse_version(want)) {
        (Some(have), Some(want)) => have >= want,
        _ => false,
    }
}

fn parse_version(text: &str) -> Option<(u64, u64, u64)> {
    let mut parts = text.split('.');
    let major = parts.next()?.parse().ok()?;
    let minor = parts.next()?.parse().ok()?;
    let patch = parts.next()?.parse().ok()?;
    if parts.next().is_some() {
        return None;
    }
    Some((major, minor, patch))
}
