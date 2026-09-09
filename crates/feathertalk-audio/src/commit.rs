use std::{
    fs,
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};

use feathertalk_project::{
    AssetManifest, AssetPackageState, FeatureType, lock_asset_package, read_asset_manifest,
};

use crate::{
    AudioError, FeatureArtifact, FeatureMatrix, MAX_FEATURE_FILE_BYTES, write_feature_file,
};

const FEATURE_FILE_NAME: &str = "feather_hubert.f32";
const MAX_ATTEMPTS: usize = 32;
static COMMIT_COUNTER: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FeatureCommitSpec {
    pub project_root: PathBuf,
    pub frame_count: u64,
    pub frame_width: u32,
    pub frame_height: u32,
    pub landmark_model_sha256: String,
    pub feature_model_sha256: String,
}

pub fn commit_feature_artifact(
    spec: &FeatureCommitSpec,
    matrix: &FeatureMatrix,
) -> Result<FeatureArtifact, AudioError> {
    validate_spec(spec, matrix)?;
    let assets = spec.project_root.join("assets");
    let features = assets.join("features");
    validate_real_directory(&spec.project_root)?;
    validate_real_directory(&assets)?;
    validate_real_directory(&features)?;
    let final_feature = features.join(FEATURE_FILE_NAME);
    let manifest_path = assets.join("assets.json");
    preflight_manifest(&manifest_path)?;

    let staging = create_unique_dir(&assets, ".feathertalk-feature-build")?;
    let staged_feature = staging.join(FEATURE_FILE_NAME);
    let result = commit_inner(
        spec,
        matrix,
        &assets,
        &staging,
        &staged_feature,
        &final_feature,
        &manifest_path,
    );
    if result.is_err() {
        let _ = fs::remove_dir_all(&staging);
    }
    result
}

#[allow(clippy::too_many_arguments)]
fn commit_inner(
    spec: &FeatureCommitSpec,
    matrix: &FeatureMatrix,
    assets: &Path,
    staging: &Path,
    staged_feature: &Path,
    final_feature: &Path,
    manifest_path: &Path,
) -> Result<FeatureArtifact, AudioError> {
    let staged_artifact = write_feature_file(staged_feature, matrix)?;
    let backup = create_unique_dir(assets, ".feathertalk-feature-backup")?;
    let mut moved = Vec::new();
    let mut installed = Vec::new();

    for destination in [final_feature, manifest_path] {
        match fs::symlink_metadata(destination) {
            Ok(metadata) => {
                if metadata.file_type().is_symlink() || !metadata.is_file() {
                    return Err(rollback(
                        AudioError::CommitFailed {
                            operation: "validate_destination",
                            message: format!(
                                "destination is not a regular file: {}",
                                destination.display()
                            ),
                        },
                        staging,
                        &backup,
                        &moved,
                        &installed,
                    ));
                }
                let backup_path = backup.join(destination.file_name().ok_or_else(|| {
                    AudioError::CommitFailed {
                        operation: "backup_existing",
                        message: format!("missing file name: {}", destination.display()),
                    }
                })?);
                if let Err(source) = fs::rename(destination, &backup_path) {
                    return Err(rollback(
                        AudioError::CommitFailed {
                            operation: "backup_existing",
                            message: source.to_string(),
                        },
                        staging,
                        &backup,
                        &moved,
                        &installed,
                    ));
                }
                moved.push((destination.to_owned(), backup_path));
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(source) => {
                return Err(rollback(
                    io("stat_destination", destination, source),
                    staging,
                    &backup,
                    &moved,
                    &installed,
                ));
            }
        }
    }

    if let Err(source) = fs::rename(staged_feature, final_feature) {
        return Err(rollback(
            AudioError::CommitFailed {
                operation: "install_feature",
                message: source.to_string(),
            },
            staging,
            &backup,
            &moved,
            &installed,
        ));
    }
    installed.push(final_feature.to_owned());
    installed.push(manifest_path.to_owned());
    if let Err(error) = lock_asset_package(&spec.project_root, locked_manifest(spec)) {
        return Err(rollback(
            AudioError::CommitFailed {
                operation: "install_manifest",
                message: error.to_string(),
            },
            staging,
            &backup,
            &moved,
            &installed,
        ));
    }
    if let Err(error) = sync_dir(&assets.join("features")) {
        return Err(rollback(error, staging, &backup, &moved, &installed));
    }
    if let Err(error) = sync_dir(assets) {
        return Err(rollback(error, staging, &backup, &moved, &installed));
    }
    if let Err(source) = fs::remove_dir_all(staging) {
        return Err(rollback(
            io("remove_staging", staging, source),
            staging,
            &backup,
            &moved,
            &installed,
        ));
    }
    if let Err(source) = fs::remove_dir_all(&backup) {
        return Err(rollback(
            io("remove_backup", &backup, source),
            staging,
            &backup,
            &moved,
            &installed,
        ));
    }
    Ok(FeatureArtifact::relocated(
        final_feature.to_owned(),
        staged_artifact.tokens(),
        staged_artifact.dims(),
        staged_artifact.bytes(),
        staged_artifact.sha256().to_owned(),
    ))
}

fn validate_spec(spec: &FeatureCommitSpec, matrix: &FeatureMatrix) -> Result<(), AudioError> {
    if spec.frame_count == 0 || spec.frame_width == 0 || spec.frame_height == 0 {
        return Err(AudioError::CommitFailed {
            operation: "validate_spec",
            message: "frame count and dimensions must be non-zero".into(),
        });
    }
    let expected_tokens = usize::try_from(spec.frame_count)
        .ok()
        .and_then(|count| count.checked_mul(2))
        .ok_or(AudioError::FeatureShapeMismatch {
            frame_count: spec.frame_count,
            tokens: matrix.tokens(),
            dims: matrix.dims(),
        })?;
    if matrix.tokens() != expected_tokens || matrix.dims() != 1024 {
        return Err(AudioError::FeatureShapeMismatch {
            frame_count: spec.frame_count,
            tokens: matrix.tokens(),
            dims: matrix.dims(),
        });
    }
    validate_hash(&spec.landmark_model_sha256)?;
    validate_hash(&spec.feature_model_sha256)?;
    let bytes = matrix
        .values()
        .len()
        .checked_mul(4)
        .and_then(|value| value.checked_add(crate::format::FEATURE_HEADER_BYTES))
        .ok_or(AudioError::FeatureSizeOverflow)?;
    if bytes as u64 > MAX_FEATURE_FILE_BYTES {
        return Err(AudioError::FeatureTooLarge {
            limit: MAX_FEATURE_FILE_BYTES,
            actual: bytes as u64,
        });
    }
    Ok(())
}

fn preflight_manifest(path: &Path) -> Result<(), AudioError> {
    match fs::symlink_metadata(path) {
        Ok(metadata) if metadata.file_type().is_symlink() || !metadata.is_file() => {
            Err(AudioError::CommitFailed {
                operation: "validate_manifest",
                message: format!("manifest is not a regular file: {}", path.display()),
            })
        }
        Ok(_) => match read_asset_manifest(path) {
            Ok(manifest) if manifest.state == AssetPackageState::Locked => {
                Err(AudioError::LockedAssetMutation {
                    path: path.to_owned(),
                })
            }
            Ok(_) => Ok(()),
            Err(error) => Err(AudioError::CommitFailed {
                operation: "read_manifest",
                message: error.to_string(),
            }),
        },
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(source) => Err(io("stat_manifest", path, source)),
    }
}

fn locked_manifest(spec: &FeatureCommitSpec) -> AssetManifest {
    AssetManifest {
        schema_version: 1,
        state: AssetPackageState::Locked,
        video_fps: 25,
        audio_sample_rate: 16_000,
        audio_channels: 1,
        frame_count: spec.frame_count,
        frame_width: spec.frame_width,
        frame_height: spec.frame_height,
        feature_type: FeatureType::FeatherHubert,
        feature_shape: [spec.frame_count, 2, 1024],
        landmark_model_sha256: spec.landmark_model_sha256.clone(),
        feature_model_sha256: spec.feature_model_sha256.clone(),
    }
}

fn create_unique_dir(parent: &Path, prefix: &str) -> Result<PathBuf, AudioError> {
    create_unique_dir_with(
        parent,
        prefix,
        || COMMIT_COUNTER.fetch_add(1, Ordering::Relaxed),
        |path| fs::symlink_metadata(path),
        |path| fs::create_dir(path),
    )
}

fn create_unique_dir_with<Next, Metadata, Create>(
    parent: &Path,
    prefix: &str,
    mut next: Next,
    mut metadata: Metadata,
    mut create: Create,
) -> Result<PathBuf, AudioError>
where
    Next: FnMut() -> u64,
    Metadata: FnMut(&Path) -> std::io::Result<fs::Metadata>,
    Create: FnMut(&Path) -> std::io::Result<()>,
{
    for _ in 0..MAX_ATTEMPTS {
        let path = parent.join(format!("{prefix}-{}-{}", std::process::id(), next()));
        match metadata(&path) {
            Ok(_) => continue,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(source) => return Err(io("stat_staging", &path, source)),
        }
        match create(&path) {
            Ok(()) => return Ok(path),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(source) => return Err(io("create_staging", &path, source)),
        }
    }
    Err(AudioError::StagingCollision {
        path: parent.to_owned(),
    })
}

fn validate_real_directory(path: &Path) -> Result<(), AudioError> {
    let metadata =
        fs::symlink_metadata(path).map_err(|source| io("stat_directory", path, source))?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        return Err(AudioError::CommitFailed {
            operation: "validate_directory",
            message: format!("not a real directory: {}", path.display()),
        });
    }
    Ok(())
}

fn validate_hash(value: &str) -> Result<(), AudioError> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(AudioError::CommitFailed {
            operation: "validate_hash",
            message: "expected lowercase SHA-256".into(),
        });
    }
    Ok(())
}

fn rollback(
    primary: AudioError,
    staging: &Path,
    backup: &Path,
    moved: &[(PathBuf, PathBuf)],
    installed: &[PathBuf],
) -> AudioError {
    let mut errors = Vec::new();
    for path in installed.iter().rev() {
        if let Err(error) = remove_path(path)
            && error.kind() != std::io::ErrorKind::NotFound
        {
            errors.push(format!("remove installed {}: {error}", path.display()));
        }
    }
    for (destination, backup_path) in moved.iter().rev() {
        if let Err(error) = fs::rename(backup_path, destination) {
            errors.push(format!("restore {}: {error}", destination.display()));
        }
    }
    if let Err(error) = fs::remove_dir_all(backup)
        && error.kind() != std::io::ErrorKind::NotFound
    {
        errors.push(format!("remove backup: {error}"));
    }
    if let Err(error) = fs::remove_dir_all(staging)
        && error.kind() != std::io::ErrorKind::NotFound
    {
        errors.push(format!("remove staging: {error}"));
    }
    if errors.is_empty() {
        primary
    } else {
        AudioError::CommitRollbackFailed {
            operation: "rollback",
            primary: primary.to_string(),
            rollback: errors.join("; "),
        }
    }
}

fn remove_path(path: &Path) -> std::io::Result<()> {
    let metadata = fs::symlink_metadata(path)?;
    if metadata.is_dir() && !metadata.file_type().is_symlink() {
        fs::remove_dir_all(path)
    } else {
        fs::remove_file(path)
    }
}

fn sync_dir(path: &Path) -> Result<(), AudioError> {
    #[cfg(unix)]
    {
        std::fs::File::open(path)
            .and_then(|file| file.sync_all())
            .map_err(|source| io("sync_directory", path, source))
    }
    #[cfg(windows)]
    {
        let _ = path;
        Ok(())
    }
}

fn io(operation: &'static str, path: &Path, source: std::io::Error) -> AudioError {
    AudioError::FeatureIo {
        operation,
        path: path.to_owned(),
        source,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allocator_reports_a_real_collision_after_bounded_attempts() {
        let root = tempfile::tempdir().unwrap();
        let mut attempts = 0usize;
        let result = create_unique_dir_with(
            root.path(),
            ".collision",
            || 7,
            |_path| Err(std::io::Error::from(std::io::ErrorKind::NotFound)),
            |_path| {
                attempts += 1;
                Err(std::io::Error::from(std::io::ErrorKind::AlreadyExists))
            },
        );
        assert!(matches!(result, Err(AudioError::StagingCollision { .. })));
        assert_eq!(attempts, MAX_ATTEMPTS);
    }

    #[test]
    fn rollback_preserves_primary_and_reports_restore_failure() {
        let root = tempfile::tempdir().unwrap();
        let staging = root.path().join("staging");
        let backup = root.path().join("backup");
        fs::create_dir_all(&staging).unwrap();
        fs::create_dir_all(&backup).unwrap();
        let destination = root.path().join("destination");
        let missing_backup = backup.join("missing");
        let error = rollback(
            AudioError::CommitFailed {
                operation: "test",
                message: "primary failure".into(),
            },
            &staging,
            &backup,
            &[(destination, missing_backup)],
            &[],
        );
        assert!(matches!(error, AudioError::CommitRollbackFailed { .. }));
    }
}
