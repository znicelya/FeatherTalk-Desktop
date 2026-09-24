//! The project manifest a task needs before it can be recorded.
//!
//! `validate_project_dir` -- the admission every training run goes through --
//! requires `project.json`, and nothing in the workspace writes one: the journal
//! only reads it, and a missing file turns into a `journal_errors` entry nobody
//! sees. So a submission makes sure one is there first.
//!
//! What it will not do is replace a manifest it cannot read. That file may hold a
//! task history and a display name a person chose, and a fresh default is worth
//! less than either.

use std::path::{Path, PathBuf};

use feathertalk_project::{
    read_project_manifest, write_project_manifest_atomic, ModelSelection, ProjectManifest,
};
use thiserror::Error;

/// The schema version a manifest is written with.
///
/// A literal, because `CURRENT_SCHEMA_VERSION` is private to
/// `feathertalk-project`. Writing the wrong number is not a silent mistake:
/// `ProjectManifest::validate` runs before the bytes reach the disk and refuses
/// any version but its own.
const SCHEMA_VERSION: u32 = 1;

/// The asset package path a manifest must declare, which the validator pins to
/// this exact string.
const ASSET_PACKAGE: &str = "assets/assets.json";

/// The identifier used when a directory's name has no usable character in it.
const FALLBACK_ID: &str = "project";

/// The most bytes an identifier may hold, matching `validate_identifier`.
const MAX_ID_BYTES: usize = 128;

/// The most characters a display name may hold, matching the same validator.
const MAX_DISPLAY_CHARS: usize = 256;

/// Where a project keeps its manifest.
pub fn manifest_path(project_dir: &Path) -> PathBuf {
    project_dir.join("project.json")
}

/// Derive an identifier from a project directory's name.
///
/// The validator wants 1 to 128 bytes of ASCII alphanumerics, dots, underscores
/// and hyphens, and a project directory is named by a person: `康辉训练素材` and
/// `my project (1)` are both likely and neither is an identifier. So the name is
/// filtered down to the characters that are allowed, and a name with none of them
/// left becomes `project` -- the identifier is a key in a JSON file, while the
/// name a user reads is the display name, which keeps every character.
pub fn project_id_from_dir(project_dir: &Path) -> String {
    let name = project_dir
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or_default();
    let kept: String = name
        .chars()
        .filter(|character| {
            character.is_ascii_alphanumeric() || matches!(character, '.' | '_' | '-')
        })
        .take(MAX_ID_BYTES)
        .collect();
    if kept.is_empty() {
        FALLBACK_ID.to_owned()
    } else {
        kept
    }
}

/// Derive the name a user reads from the same directory.
///
/// Every character survives here; only the validator's two rules are applied --
/// trimmed, and at most 256 characters. Truncation happens before the trim so a
/// cut that lands in the middle of a run of spaces does not leave one at the end.
/// A name that is empty or nothing but whitespace falls back to `project_id`,
/// which is never empty.
pub fn display_name_from_dir(project_dir: &Path, project_id: &str) -> String {
    let name = project_dir
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or_default();
    let shortened: String = name.chars().take(MAX_DISPLAY_CHARS).collect();
    let trimmed = shortened.trim();
    if trimmed.is_empty() {
        project_id.to_owned()
    } else {
        trimmed.to_owned()
    }
}

/// What `ensure_manifest` found.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Bootstrap {
    /// A readable manifest was already there and was not touched.
    Present,
    /// There was none, so one was derived and written.
    Created,
}

/// Why a project could not be given a manifest.
///
/// English technical detail, like `ClientError`'s: the page pairs it with a
/// Chinese sentence and whoever has to fix the file wants the original words.
#[derive(Debug, Error)]
pub enum ManifestError {
    /// A file is there and could not be read; nothing was written.
    #[error("existing project.json could not be read: {0}")]
    Existing(String),
    /// What was derived from the directory's name did not pass validation.
    #[error("derived manifest is invalid: {0}")]
    Derived(String),
    /// The file could not be written.
    #[error("project.json could not be written: {0}")]
    Write(String),
}

/// Make sure `project_dir` has a manifest, writing one only when there is none.
///
/// Existence is asked of the path itself rather than of what it points at: a
/// broken symbolic link, a directory wearing the name, or a file with a syntax
/// error are all "something is already here", and the answer to all three is to
/// report it rather than to overwrite it.
pub fn ensure_manifest(project_dir: &Path) -> Result<Bootstrap, ManifestError> {
    let path = manifest_path(project_dir);
    if path.symlink_metadata().is_ok() {
        return match read_project_manifest(&path) {
            Ok(_manifest) => Ok(Bootstrap::Present),
            Err(error) => Err(ManifestError::Existing(error.to_string())),
        };
    }
    let project_id = project_id_from_dir(project_dir);
    let display_name = display_name_from_dir(project_dir, &project_id);
    let manifest = ProjectManifest {
        schema_version: SCHEMA_VERSION,
        project_id,
        display_name,
        asset_package: ASSET_PACKAGE.to_owned(),
        // The form's default too. The field is what a later render slice will read
        // to choose a checkpoint; training takes its variant from the request.
        default_model: ModelSelection::OriginalUnet,
        task_history: Vec::new(),
    };
    manifest
        .validate()
        .map_err(|error| ManifestError::Derived(error.to_string()))?;
    write_project_manifest_atomic(&path, &manifest)
        .map_err(|error| ManifestError::Write(error.to_string()))?;
    Ok(Bootstrap::Created)
}
