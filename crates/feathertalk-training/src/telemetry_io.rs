use std::{
    fs::{self, File, OpenOptions},
    io::{Read, Take, Write},
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};

use sha2::{Digest, Sha256};

use crate::{
    PREVIEW_ARTIFACT_FORMAT, PREVIEW_ARTIFACT_SCHEMA_VERSION, PREVIEW_MANIFEST_FILE_NAME,
    PREVIEW_MOUTH_ROI_FILE_NAME, PREVIEW_PREDICTION_FILE_NAME, PREVIEW_TARGET_FILE_NAME,
    PREVIEW_TENSOR_ELEMENTS, PREVIEW_TENSOR_SHAPE, PreviewArtifact, PreviewArtifactManifest,
    PreviewFileManifest, TrainingError, TrainingMetrics,
};

const PREVIEW_MAGIC: [u8; 8] = *b"FTPV32\0\0";
const PREVIEW_FORMAT_VERSION: u32 = 1;
const PREVIEW_HEADER_BYTES: usize = 32;
const PREVIEW_PAYLOAD_BYTES: usize = PREVIEW_TENSOR_ELEMENTS * std::mem::size_of::<f32>();
const PREVIEW_FILE_BYTES: usize = PREVIEW_HEADER_BYTES + PREVIEW_PAYLOAD_BYTES;
const PREVIEW_MAX_FILE_BYTES: u64 = 1024 * 1024;
const JSON_MAX_BYTES: u64 = 64 * 1024;
const MAX_STAGING_ATTEMPTS: usize = 1024;
const MAX_TEMP_ATTEMPTS: usize = 64;

static NEXT_STAGING_ID: AtomicU64 = AtomicU64::new(1);
static NEXT_TEMP_ID: AtomicU64 = AtomicU64::new(1);

pub fn write_training_metrics(
    path: impl AsRef<Path>,
    metrics: &TrainingMetrics,
) -> Result<(), TrainingError> {
    metrics.validate()?;
    let path = path.as_ref();
    let parent = parent_or_current(path);
    reject_symlink_components(parent)?;
    fs::create_dir_all(parent)?;
    reject_existing_destination(path, "metrics destination")?;

    let bytes = serde_json::to_vec(metrics)
        .map_err(|error| TrainingError::Store(format!("serialize training metrics: {error}")))?;
    ensure_json_size(path, bytes.len())?;

    let (temporary, mut file) = create_temp_file(parent, path)?;
    let result = file
        .write_all(&bytes)
        .and_then(|_| file.flush())
        .and_then(|_| file.sync_all())
        .and_then(|_| {
            drop(file);
            fs::rename(&temporary, path)
        })
        .and_then(|_| sync_directory(parent));
    if result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    result.map_err(TrainingError::from)
}

pub fn read_training_metrics(path: impl AsRef<Path>) -> Result<TrainingMetrics, TrainingError> {
    let path = path.as_ref();
    reject_symlink_components(parent_or_current(path))?;
    validate_regular_file(path, "metrics file")?;
    let bytes = read_bounded(path, JSON_MAX_BYTES)?;
    let metrics: TrainingMetrics = serde_json::from_slice(&bytes).map_err(|error| {
        TrainingError::InvalidCheckpoint(format!("training metrics JSON is invalid: {error}"))
    })?;
    metrics.validate()?;
    Ok(metrics)
}

pub fn write_preview_artifact(
    destination: impl AsRef<Path>,
    artifact: &PreviewArtifact,
) -> Result<PreviewArtifactManifest, TrainingError> {
    artifact.validate()?;
    let destination = destination.as_ref();
    let parent = parent_or_current(destination);
    reject_symlink_components(parent)?;
    fs::create_dir_all(parent)?;
    reject_existing_destination(destination, "preview destination")?;

    let mut staging = create_staging_directory(parent)?;
    let staging_path = staging.path().to_owned();

    let prediction_path = staging_path.join(PREVIEW_PREDICTION_FILE_NAME);
    let target_path = staging_path.join(PREVIEW_TARGET_FILE_NAME);
    let mouth_roi_path = staging_path.join(PREVIEW_MOUTH_ROI_FILE_NAME);
    write_tensor_file(&prediction_path, artifact.prediction())?;
    write_tensor_file(&target_path, artifact.target())?;
    write_tensor_file(&mouth_roi_path, artifact.mouth_roi())?;

    let prediction = file_manifest(&prediction_path, PREVIEW_PREDICTION_FILE_NAME)?;
    let target = file_manifest(&target_path, PREVIEW_TARGET_FILE_NAME)?;
    let mouth_roi = file_manifest(&mouth_roi_path, PREVIEW_MOUTH_ROI_FILE_NAME)?;
    let manifest = PreviewArtifactManifest {
        schema_version: PREVIEW_ARTIFACT_SCHEMA_VERSION,
        format: PREVIEW_ARTIFACT_FORMAT.to_owned(),
        sample_index: artifact.sample_index(),
        reference_index: artifact.reference_index(),
        epoch: artifact.epoch(),
        global_step: artifact.global_step(),
        model_kind: artifact.model_kind().to_owned(),
        model_config_sha256: artifact.model_config_sha256().to_owned(),
        worker_state: artifact.worker_state().to_owned(),
        shape: PREVIEW_TENSOR_SHAPE,
        prediction,
        target,
        mouth_roi,
    };
    manifest.validate_against(artifact)?;

    let manifest_bytes = serde_json::to_vec(&manifest)
        .map_err(|error| TrainingError::Store(format!("serialize preview manifest: {error}")))?;
    ensure_json_size(
        &staging_path.join(PREVIEW_MANIFEST_FILE_NAME),
        manifest_bytes.len(),
    )?;
    write_synced_bytes(
        &staging_path.join(PREVIEW_MANIFEST_FILE_NAME),
        &manifest_bytes,
    )?;
    sync_directory(&staging_path)?;
    fs::rename(&staging_path, destination)?;
    staging.disarm();
    sync_directory(parent)?;

    Ok(manifest)
}

pub fn read_preview_artifact(
    directory: impl AsRef<Path>,
    expected_model_kind: &str,
    expected_model_config_sha256: &str,
) -> Result<(PreviewArtifact, PreviewArtifactManifest), TrainingError> {
    let directory = directory.as_ref();
    reject_symlink_components(parent_or_current(directory))?;

    // Keep this order strict: no payload bytes are decoded until the complete
    // directory, manifest, compatibility, length, and hash preflight passes.
    validate_preview_directory(directory)?;
    let manifest_path = directory.join(PREVIEW_MANIFEST_FILE_NAME);
    let manifest_bytes = read_bounded(&manifest_path, JSON_MAX_BYTES)?;
    let manifest: PreviewArtifactManifest =
        serde_json::from_slice(&manifest_bytes).map_err(|error| {
            TrainingError::InvalidCheckpoint(format!("preview manifest JSON is invalid: {error}"))
        })?;
    manifest.validate()?;
    if manifest.model_kind != expected_model_kind {
        return Err(TrainingError::CheckpointCompatibility(format!(
            "preview model_kind mismatch: expected {expected_model_kind}, got {}",
            manifest.model_kind
        )));
    }
    if manifest.model_config_sha256 != expected_model_config_sha256 {
        return Err(TrainingError::CheckpointCompatibility(format!(
            "preview model_config_sha256 mismatch: expected {expected_model_config_sha256}, got {}",
            manifest.model_config_sha256
        )));
    }

    let prediction_path = directory.join(PREVIEW_PREDICTION_FILE_NAME);
    let target_path = directory.join(PREVIEW_TARGET_FILE_NAME);
    let mouth_roi_path = directory.join(PREVIEW_MOUTH_ROI_FILE_NAME);
    validate_declared_file(&prediction_path, &manifest.prediction)?;
    validate_declared_file(&target_path, &manifest.target)?;
    validate_declared_file(&mouth_roi_path, &manifest.mouth_roi)?;

    let prediction = read_tensor_file(&prediction_path)?;
    let target = read_tensor_file(&target_path)?;
    let mouth_roi = read_tensor_file(&mouth_roi_path)?;
    let artifact = PreviewArtifact::new(
        manifest.sample_index,
        manifest.reference_index,
        manifest.epoch,
        manifest.global_step,
        manifest.model_kind.clone(),
        manifest.model_config_sha256.clone(),
        manifest.worker_state.clone(),
        prediction,
        target,
        mouth_roi,
    )?;
    manifest.validate_against(&artifact)?;
    Ok((artifact, manifest))
}

fn write_tensor_file(path: &Path, values: &[f32]) -> Result<(), TrainingError> {
    if values.len() != PREVIEW_TENSOR_ELEMENTS {
        return Err(TrainingError::InvalidCheckpoint(format!(
            "preview tensor must contain {PREVIEW_TENSOR_ELEMENTS} values, got {}",
            values.len()
        )));
    }
    let mut bytes = Vec::with_capacity(PREVIEW_FILE_BYTES);
    bytes.extend_from_slice(&PREVIEW_MAGIC);
    bytes.extend_from_slice(&PREVIEW_FORMAT_VERSION.to_le_bytes());
    bytes.extend_from_slice(&3_u32.to_le_bytes());
    for dimension in PREVIEW_TENSOR_SHAPE {
        bytes.extend_from_slice(&dimension.to_le_bytes());
    }
    bytes.extend_from_slice(&(PREVIEW_PAYLOAD_BYTES as u32).to_le_bytes());
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    if bytes.len() != PREVIEW_FILE_BYTES {
        return Err(TrainingError::InvalidCheckpoint(
            "preview tensor encoded byte count is inconsistent".to_owned(),
        ));
    }
    write_synced_bytes(path, &bytes)
}

fn read_tensor_file(path: &Path) -> Result<Vec<f32>, TrainingError> {
    let bytes = read_bounded(path, PREVIEW_MAX_FILE_BYTES)?;
    if bytes.len() != PREVIEW_FILE_BYTES {
        return Err(TrainingError::InvalidCheckpoint(format!(
            "preview tensor {} must contain exactly {PREVIEW_FILE_BYTES} bytes, got {}",
            path.display(),
            bytes.len()
        )));
    }
    if bytes[..8] != PREVIEW_MAGIC {
        return Err(TrainingError::InvalidCheckpoint(format!(
            "preview tensor {} has an invalid magic header",
            path.display()
        )));
    }
    let version = read_u32(&bytes, 8);
    let rank = read_u32(&bytes, 12);
    let dimensions = [
        read_u32(&bytes, 16),
        read_u32(&bytes, 20),
        read_u32(&bytes, 24),
    ];
    let payload_bytes = read_u32(&bytes, 28);
    if version != PREVIEW_FORMAT_VERSION
        || rank != 3
        || dimensions != PREVIEW_TENSOR_SHAPE
        || payload_bytes as usize != PREVIEW_PAYLOAD_BYTES
    {
        return Err(TrainingError::InvalidCheckpoint(format!(
            "preview tensor {} header does not match [3,160,160] format",
            path.display()
        )));
    }

    let mut values = Vec::with_capacity(PREVIEW_TENSOR_ELEMENTS);
    for (index, chunk) in bytes[PREVIEW_HEADER_BYTES..].chunks_exact(4).enumerate() {
        let value = f32::from_le_bytes(chunk.try_into().expect("chunks_exact gives four bytes"));
        if !value.is_finite() {
            return Err(TrainingError::InvalidCheckpoint(format!(
                "preview tensor {} contains non-finite value at index {index}",
                path.display()
            )));
        }
        values.push(value);
    }
    Ok(values)
}

fn read_u32(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(
        bytes[offset..offset + 4]
            .try_into()
            .expect("fixed header offset"),
    )
}

fn validate_preview_directory(path: &Path) -> Result<(), TrainingError> {
    let metadata = symlink_metadata_or_directory_error(path, "preview directory")?;
    if metadata.file_type().is_symlink() {
        return Err(TrainingError::CheckpointDirectory(format!(
            "preview directory must not be a symbolic link: {}",
            path.display()
        )));
    }
    if !metadata.is_dir() {
        return Err(TrainingError::CheckpointDirectory(format!(
            "preview path is not a directory: {}",
            path.display()
        )));
    }

    let mut entries = fs::read_dir(path)?
        .map(|entry| {
            entry.and_then(|entry| {
                entry.file_name().into_string().map_err(|_| {
                    std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        "preview entry name is not valid UTF-8",
                    )
                })
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    entries.sort();
    let mut expected = vec![
        PREVIEW_MANIFEST_FILE_NAME.to_owned(),
        PREVIEW_MOUTH_ROI_FILE_NAME.to_owned(),
        PREVIEW_PREDICTION_FILE_NAME.to_owned(),
        PREVIEW_TARGET_FILE_NAME.to_owned(),
    ];
    expected.sort();
    if entries != expected {
        return Err(TrainingError::CheckpointDirectory(format!(
            "preview directory entries must be exactly {expected:?}, got {entries:?}"
        )));
    }
    for entry in expected {
        validate_regular_file(&path.join(entry), "preview entry")?;
    }
    Ok(())
}

fn validate_declared_file(
    path: &Path,
    declared: &PreviewFileManifest,
) -> Result<(), TrainingError> {
    validate_regular_file(path, "preview tensor")?;
    if declared.bytes > PREVIEW_MAX_FILE_BYTES {
        return Err(TrainingError::InvalidCheckpoint(format!(
            "preview file {} exceeds maximum size",
            declared.file_name
        )));
    }
    let metadata = fs::symlink_metadata(path)?;
    if metadata.len() > PREVIEW_MAX_FILE_BYTES {
        return Err(TrainingError::InvalidCheckpoint(format!(
            "preview file {} exceeds maximum size",
            declared.file_name,
        )));
    }
    let (bytes, sha256) = sha256_file(path)?;
    if sha256 != declared.sha256 {
        return Err(TrainingError::HashMismatch {
            file: declared.file_name.clone(),
            expected: declared.sha256.clone(),
            actual: sha256,
        });
    }
    if bytes != declared.bytes {
        return Err(TrainingError::InvalidCheckpoint(format!(
            "preview file {} declares {} bytes but contains {}",
            declared.file_name, declared.bytes, bytes
        )));
    }
    Ok(())
}

fn file_manifest(path: &Path, expected_name: &str) -> Result<PreviewFileManifest, TrainingError> {
    let (bytes, sha256) = sha256_file(path)?;
    let manifest = PreviewFileManifest {
        file_name: expected_name.to_owned(),
        bytes,
        sha256,
    };
    manifest.validate(expected_name)?;
    if bytes > PREVIEW_MAX_FILE_BYTES {
        return Err(TrainingError::InvalidCheckpoint(format!(
            "preview file {expected_name} exceeds maximum size"
        )));
    }
    Ok(manifest)
}

fn write_synced_bytes(path: &Path, bytes: &[u8]) -> Result<(), TrainingError> {
    let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
    file.write_all(bytes)?;
    file.flush()?;
    file.sync_all()?;
    Ok(())
}

fn sha256_file(path: &Path) -> Result<(u64, String), TrainingError> {
    let mut file = File::open(path)?;
    let mut digest = Sha256::new();
    let mut bytes = 0_u64;
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        bytes = bytes.checked_add(read as u64).ok_or_else(|| {
            TrainingError::InvalidCheckpoint("preview byte count overflow".to_owned())
        })?;
        digest.update(&buffer[..read]);
    }
    Ok((bytes, hex::encode(digest.finalize())))
}

fn read_bounded(path: &Path, max_bytes: u64) -> Result<Vec<u8>, TrainingError> {
    let file = File::open(path)?;
    let mut reader: Take<File> = file.take(max_bytes.saturating_add(1));
    let mut bytes = Vec::new();
    reader.read_to_end(&mut bytes)?;
    if bytes.len() as u64 > max_bytes {
        return Err(TrainingError::CheckpointDirectory(format!(
            "file {} exceeds maximum size of {max_bytes} bytes",
            path.display()
        )));
    }
    Ok(bytes)
}

fn validate_regular_file(path: &Path, label: &str) -> Result<(), TrainingError> {
    let metadata = symlink_metadata_or_directory_error(path, label)?;
    if metadata.file_type().is_symlink() {
        return Err(TrainingError::CheckpointDirectory(format!(
            "{label} must not be a symbolic link: {}",
            path.display()
        )));
    }
    if !metadata.is_file() {
        return Err(TrainingError::CheckpointDirectory(format!(
            "{label} is not a regular file: {}",
            path.display()
        )));
    }
    Ok(())
}

fn symlink_metadata_or_directory_error(
    path: &Path,
    label: &str,
) -> Result<std::fs::Metadata, TrainingError> {
    fs::symlink_metadata(path).map_err(|error| {
        if error.kind() == std::io::ErrorKind::NotFound {
            TrainingError::CheckpointDirectory(format!(
                "{label} does not exist: {}",
                path.display()
            ))
        } else {
            error.into()
        }
    })
}

fn reject_existing_destination(path: &Path, label: &str) -> Result<(), TrainingError> {
    match fs::symlink_metadata(path) {
        Ok(_) => Err(TrainingError::CheckpointDirectory(format!(
            "{label} already exists: {}",
            path.display()
        ))),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error.into()),
    }
}

fn parent_or_current(path: &Path) -> &Path {
    path.parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."))
}

fn create_temp_file(parent: &Path, destination: &Path) -> Result<(PathBuf, File), TrainingError> {
    let stem = destination
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("metrics");
    for _ in 0..MAX_TEMP_ATTEMPTS {
        let id = NEXT_TEMP_ID.fetch_add(1, Ordering::Relaxed);
        let path = parent.join(format!(".{stem}-{}-{id}.tmp", std::process::id()));
        match OpenOptions::new().write(true).create_new(true).open(&path) {
            Ok(file) => return Ok((path, file)),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(error.into()),
        }
    }
    Err(TrainingError::CheckpointDirectory(
        "unable to allocate a unique metrics temporary file".to_owned(),
    ))
}

struct StagingDirectory {
    path: PathBuf,
    armed: bool,
}

impl StagingDirectory {
    fn path(&self) -> &Path {
        &self.path
    }

    fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for StagingDirectory {
    fn drop(&mut self) {
        if self.armed {
            let _ = fs::remove_dir_all(&self.path);
        }
    }
}

fn create_staging_directory(parent: &Path) -> Result<StagingDirectory, TrainingError> {
    for _ in 0..MAX_STAGING_ATTEMPTS {
        let id = NEXT_STAGING_ID.fetch_add(1, Ordering::Relaxed);
        let path = parent.join(format!(".preview-{}-{id}.staging", std::process::id()));
        match fs::create_dir(&path) {
            Ok(()) => return Ok(StagingDirectory { path, armed: true }),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(error.into()),
        }
    }
    Err(TrainingError::CheckpointDirectory(
        "unable to allocate a unique preview staging directory".to_owned(),
    ))
}

fn reject_symlink_components(path: &Path) -> Result<(), TrainingError> {
    let mut current = PathBuf::new();
    for component in path.components() {
        current.push(component.as_os_str());
        match fs::symlink_metadata(&current) {
            Ok(metadata) if metadata.file_type().is_symlink() => {
                return Err(TrainingError::CheckpointDirectory(format!(
                    "preview path component must not be a symbolic link: {}",
                    current.display()
                )));
            }
            Ok(_) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => break,
            Err(error) => return Err(error.into()),
        }
    }
    Ok(())
}

fn ensure_json_size(path: &Path, length: usize) -> Result<(), TrainingError> {
    if length as u64 > JSON_MAX_BYTES {
        return Err(TrainingError::InvalidCheckpoint(format!(
            "JSON file {} exceeds maximum size of {JSON_MAX_BYTES} bytes",
            path.display()
        )));
    }
    Ok(())
}

fn sync_directory(path: &Path) -> std::io::Result<()> {
    #[cfg(unix)]
    {
        File::open(path)?.sync_all()
    }
    #[cfg(not(unix))]
    {
        let _ = path;
        Ok(())
    }
}
