use std::{
    fs,
    path::{Path, PathBuf},
};

use crate::MediaError;

const MAX_BACKUP_ATTEMPTS: usize = 32;

pub(crate) trait FileOps {
    fn metadata(&self, path: &Path) -> std::io::Result<fs::Metadata>;
    fn rename(&self, from: &Path, to: &Path) -> std::io::Result<()>;
    fn remove_file(&self, path: &Path) -> std::io::Result<()>;
}

#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct SystemFileOps;

impl FileOps for SystemFileOps {
    fn metadata(&self, path: &Path) -> std::io::Result<fs::Metadata> {
        fs::symlink_metadata(path)
    }

    fn rename(&self, from: &Path, to: &Path) -> std::io::Result<()> {
        fs::rename(from, to)
    }

    fn remove_file(&self, path: &Path) -> std::io::Result<()> {
        fs::remove_file(path)
    }
}

pub(crate) fn commit_output_pair(
    video_temp: &Path,
    audio_temp: &Path,
    video_dest: &Path,
    audio_dest: &Path,
    ops: &dyn FileOps,
) -> Result<(), MediaError> {
    let video_backup = match backup_existing(video_dest, ops) {
        Ok(value) => value,
        Err(error) => {
            remove_temp(video_temp, ops);
            remove_temp(audio_temp, ops);
            return Err(error);
        }
    };
    let audio_backup = match backup_existing(audio_dest, ops) {
        Ok(value) => value,
        Err(error) => {
            let rollback = restore_backup(video_dest, video_backup.as_deref(), ops);
            remove_temp(video_temp, ops);
            remove_temp(audio_temp, ops);
            return Err(with_rollback(error, rollback));
        }
    };
    if let Err(error) = ops.rename(video_temp, video_dest) {
        let rollback = restore_pair(
            video_dest,
            video_backup.as_deref(),
            audio_dest,
            audio_backup.as_deref(),
            ops,
        );
        remove_temp(video_temp, ops);
        remove_temp(audio_temp, ops);
        return Err(commit_error("rename_video", error, rollback));
    }
    if let Err(error) = ops.rename(audio_temp, audio_dest) {
        let remove_video = ops.remove_file(video_dest).err();
        let rollback = restore_pair(
            video_dest,
            video_backup.as_deref(),
            audio_dest,
            audio_backup.as_deref(),
            ops,
        );
        let rollback = combine_rollback(remove_video, rollback);
        remove_temp(video_temp, ops);
        remove_temp(audio_temp, ops);
        return Err(commit_error("rename_audio", error, rollback));
    }
    remove_backups(video_backup, audio_backup, ops)
}

fn backup_existing(destination: &Path, ops: &dyn FileOps) -> Result<Option<PathBuf>, MediaError> {
    match ops.metadata(destination) {
        Ok(metadata) if metadata.file_type().is_symlink() || !metadata.is_file() => {
            Err(MediaError::OutputDestinationInvalid {
                path: destination.to_owned(),
            })
        }
        Ok(_) => {
            for _ in 0..MAX_BACKUP_ATTEMPTS {
                let backup = unique_backup_path(destination);
                match ops.metadata(&backup) {
                    Ok(_) => continue,
                    Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                    Err(source) => {
                        return Err(MediaError::Io {
                            operation: "stat_backup",
                            path: backup,
                            source,
                        });
                    }
                }
                match ops.rename(destination, &backup) {
                    Ok(()) => return Ok(Some(backup)),
                    Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
                    Err(source) => {
                        return Err(MediaError::OutputCommitFailed {
                            operation: "backup_existing",
                            message: source.to_string(),
                        });
                    }
                }
            }
            Err(MediaError::OutputCommitFailed {
                operation: "backup_existing",
                message: format!(
                    "unable to allocate a collision-free backup path after {MAX_BACKUP_ATTEMPTS} attempts"
                ),
            })
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(source) => Err(MediaError::Io {
            operation: "stat_output",
            path: destination.to_owned(),
            source,
        }),
    }
}

fn unique_backup_path(destination: &Path) -> PathBuf {
    let name = destination
        .file_name()
        .map(|value| value.to_string_lossy().into_owned())
        .unwrap_or_else(|| "output".to_owned());
    destination.with_file_name(format!(
        ".{name}.backup-{}",
        crate::normalize::next_temp_id()
    ))
}

fn restore_pair(
    video_dest: &Path,
    video_backup: Option<&Path>,
    audio_dest: &Path,
    audio_backup: Option<&Path>,
    ops: &dyn FileOps,
) -> Option<String> {
    let mut errors = Vec::new();
    if let Some(backup) = video_backup
        && let Err(error) = ops.rename(backup, video_dest)
    {
        errors.push(format!("restore video: {error}"));
    }
    if let Some(backup) = audio_backup
        && let Err(error) = ops.rename(backup, audio_dest)
    {
        errors.push(format!("restore audio: {error}"));
    }
    (!errors.is_empty()).then(|| errors.join("; "))
}

fn restore_backup(destination: &Path, backup: Option<&Path>, ops: &dyn FileOps) -> Option<String> {
    let backup = backup?;
    ops.rename(backup, destination)
        .err()
        .map(|error| format!("restore {destination:?}: {error}"))
}

fn remove_backups(
    video_backup: Option<PathBuf>,
    audio_backup: Option<PathBuf>,
    ops: &dyn FileOps,
) -> Result<(), MediaError> {
    let mut errors = Vec::new();
    for path in [video_backup, audio_backup].into_iter().flatten() {
        if let Err(error) = ops.remove_file(&path) {
            errors.push(format!("{path:?}: {error}"));
        }
    }
    if errors.is_empty() {
        Ok(())
    } else {
        Err(MediaError::OutputCommitFailed {
            operation: "remove_backup",
            message: errors.join("; "),
        })
    }
}

fn remove_temp(path: &Path, ops: &dyn FileOps) {
    let _ = ops.remove_file(path);
}

fn combine_rollback(first: Option<std::io::Error>, second: Option<String>) -> Option<String> {
    match (first, second) {
        (None, None) => None,
        (Some(error), None) => Some(format!("remove partial output: {error}")),
        (None, Some(error)) => Some(error),
        (Some(first), Some(second)) => Some(format!("remove partial output: {first}; {second}")),
    }
}

fn with_rollback(error: MediaError, rollback: Option<String>) -> MediaError {
    match rollback {
        Some(rollback) => MediaError::OutputRollbackFailed {
            operation: "restore_backup",
            primary: error.to_string(),
            rollback,
        },
        None => error,
    }
}

fn commit_error(
    operation: &'static str,
    error: std::io::Error,
    rollback: Option<String>,
) -> MediaError {
    match rollback {
        Some(rollback) => MediaError::OutputRollbackFailed {
            operation,
            primary: error.to_string(),
            rollback,
        },
        None => MediaError::OutputCommitFailed {
            operation,
            message: error.to_string(),
        },
    }
}

#[cfg(test)]
mod tests {
    use std::{
        fs,
        path::{Path, PathBuf},
        sync::Mutex,
    };

    use super::*;

    #[derive(Debug, Clone)]
    struct PairPaths {
        video_temp: PathBuf,
        audio_temp: PathBuf,
        video_dest: PathBuf,
        audio_dest: PathBuf,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum RenamePoint {
        BackupVideo,
        BackupAudio,
        InstallVideo,
        InstallAudio,
        RestoreVideo,
        RestoreAudio,
    }

    struct InjectedOps {
        paths: PairPaths,
        fail_renames: Vec<RenamePoint>,
        backup_collisions: Mutex<usize>,
        fail_backup_removal: bool,
        metadata_source: PathBuf,
    }

    impl InjectedOps {
        fn new(paths: PairPaths, metadata_source: PathBuf) -> Self {
            Self {
                paths,
                fail_renames: Vec::new(),
                backup_collisions: Mutex::new(0),
                fail_backup_removal: false,
                metadata_source,
            }
        }

        fn failing(mut self, points: &[RenamePoint]) -> Self {
            self.fail_renames = points.to_vec();
            self
        }

        fn with_backup_collision(self) -> Self {
            *self.backup_collisions.lock().unwrap() = 1;
            self
        }

        fn with_backup_removal_failure(mut self) -> Self {
            self.fail_backup_removal = true;
            self
        }

        fn rename_point(&self, from: &Path, to: &Path) -> Option<RenamePoint> {
            if from == self.paths.video_dest && is_backup(to) {
                Some(RenamePoint::BackupVideo)
            } else if from == self.paths.audio_dest && is_backup(to) {
                Some(RenamePoint::BackupAudio)
            } else if from == self.paths.video_temp && to == self.paths.video_dest {
                Some(RenamePoint::InstallVideo)
            } else if from == self.paths.audio_temp && to == self.paths.audio_dest {
                Some(RenamePoint::InstallAudio)
            } else if is_backup(from) && to == self.paths.video_dest {
                Some(RenamePoint::RestoreVideo)
            } else if is_backup(from) && to == self.paths.audio_dest {
                Some(RenamePoint::RestoreAudio)
            } else {
                None
            }
        }
    }

    impl FileOps for InjectedOps {
        fn metadata(&self, path: &Path) -> std::io::Result<fs::Metadata> {
            if is_backup(path) {
                let mut collisions = self.backup_collisions.lock().unwrap();
                if *collisions > 0 {
                    *collisions -= 1;
                    return fs::symlink_metadata(&self.metadata_source);
                }
            }
            fs::symlink_metadata(path)
        }

        fn rename(&self, from: &Path, to: &Path) -> std::io::Result<()> {
            let point = self.rename_point(from, to);
            if matches!(
                point,
                Some(RenamePoint::BackupVideo | RenamePoint::BackupAudio)
            ) && *self.backup_collisions.lock().unwrap() > 0
            {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::AlreadyExists,
                    "simulated backup collision",
                ));
            }
            if point.is_some_and(|point| self.fail_renames.contains(&point)) {
                return Err(std::io::Error::other(format!(
                    "simulated {point:?} failure"
                )));
            }
            fs::rename(from, to)
        }

        fn remove_file(&self, path: &Path) -> std::io::Result<()> {
            if self.fail_backup_removal && is_backup(path) {
                return Err(std::io::Error::other("simulated backup removal failure"));
            }
            fs::remove_file(path)
        }
    }

    fn is_backup(path: &Path) -> bool {
        path.file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name.contains(".backup-"))
    }

    fn setup_pair(existing: bool) -> (tempfile::TempDir, PairPaths) {
        let root = tempfile::tempdir().unwrap();
        let paths = PairPaths {
            video_temp: root.path().join("video.tmp.mp4"),
            audio_temp: root.path().join("audio.tmp.wav"),
            video_dest: root.path().join("video_25fps.mp4"),
            audio_dest: root.path().join("audio_16k_mono.wav"),
        };
        fs::write(&paths.video_temp, b"new video").unwrap();
        fs::write(&paths.audio_temp, b"new audio").unwrap();
        if existing {
            fs::write(&paths.video_dest, b"old video").unwrap();
            fs::write(&paths.audio_dest, b"old audio").unwrap();
        }
        (root, paths)
    }

    fn commit(paths: &PairPaths, ops: &dyn FileOps) -> Result<(), MediaError> {
        commit_output_pair(
            &paths.video_temp,
            &paths.audio_temp,
            &paths.video_dest,
            &paths.audio_dest,
            ops,
        )
    }

    fn backup_paths(root: &Path) -> Vec<PathBuf> {
        fs::read_dir(root)
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .filter(|path| is_backup(path))
            .collect()
    }

    fn assert_old_pair_restored(paths: &PairPaths) {
        assert_eq!(fs::read(&paths.video_dest).unwrap(), b"old video");
        assert_eq!(fs::read(&paths.audio_dest).unwrap(), b"old audio");
        assert!(!paths.video_temp.exists());
        assert!(!paths.audio_temp.exists());
    }

    #[test]
    fn installs_pair_when_destinations_are_absent() {
        let (root, paths) = setup_pair(false);

        commit(&paths, &SystemFileOps).unwrap();

        assert_eq!(fs::read(&paths.video_dest).unwrap(), b"new video");
        assert_eq!(fs::read(&paths.audio_dest).unwrap(), b"new audio");
        assert!(!paths.video_temp.exists());
        assert!(!paths.audio_temp.exists());
        assert!(backup_paths(root.path()).is_empty());
    }

    #[test]
    fn replaces_existing_pair_and_removes_backups() {
        let (root, paths) = setup_pair(true);

        commit(&paths, &SystemFileOps).unwrap();

        assert_eq!(fs::read(&paths.video_dest).unwrap(), b"new video");
        assert_eq!(fs::read(&paths.audio_dest).unwrap(), b"new audio");
        assert!(!paths.video_temp.exists());
        assert!(!paths.audio_temp.exists());
        assert_eq!(fs::read_dir(root.path()).unwrap().count(), 2);
    }

    #[test]
    fn preexisting_backup_name_is_skipped_without_overwriting_it() {
        let (root, paths) = setup_pair(true);
        let metadata_source = root.path().join("collision-sentinel");
        fs::write(&metadata_source, b"sentinel").unwrap();
        let ops = InjectedOps::new(paths.clone(), metadata_source).with_backup_collision();

        commit(&paths, &ops).unwrap();

        assert_eq!(fs::read(&paths.video_dest).unwrap(), b"new video");
        assert_eq!(fs::read(&paths.audio_dest).unwrap(), b"new audio");
        assert!(backup_paths(root.path()).is_empty());
    }

    #[test]
    fn second_backup_failure_restores_first_backup_and_removes_temps() {
        let (root, paths) = setup_pair(true);
        let metadata_source = root.path().join("metadata-source");
        fs::write(&metadata_source, b"metadata").unwrap();
        let ops =
            InjectedOps::new(paths.clone(), metadata_source).failing(&[RenamePoint::BackupAudio]);

        assert!(matches!(
            commit(&paths, &ops),
            Err(MediaError::OutputCommitFailed {
                operation: "backup_existing",
                ..
            })
        ));
        assert_old_pair_restored(&paths);
        assert!(backup_paths(root.path()).is_empty());
    }

    #[test]
    fn first_backup_failure_removes_both_owned_temps() {
        let (root, paths) = setup_pair(true);
        let metadata_source = root.path().join("metadata-source");
        fs::write(&metadata_source, b"metadata").unwrap();
        let ops =
            InjectedOps::new(paths.clone(), metadata_source).failing(&[RenamePoint::BackupVideo]);

        assert!(matches!(
            commit(&paths, &ops),
            Err(MediaError::OutputCommitFailed {
                operation: "backup_existing",
                ..
            })
        ));
        assert_old_pair_restored(&paths);
        assert!(backup_paths(root.path()).is_empty());
    }

    #[test]
    fn first_install_failure_restores_both_original_files() {
        let (root, paths) = setup_pair(true);
        let metadata_source = root.path().join("metadata-source");
        fs::write(&metadata_source, b"metadata").unwrap();
        let ops =
            InjectedOps::new(paths.clone(), metadata_source).failing(&[RenamePoint::InstallVideo]);

        assert!(matches!(
            commit(&paths, &ops),
            Err(MediaError::OutputCommitFailed {
                operation: "rename_video",
                ..
            })
        ));
        assert_old_pair_restored(&paths);
        assert!(backup_paths(root.path()).is_empty());
    }

    #[test]
    fn second_install_failure_restores_both_original_files() {
        let (root, paths) = setup_pair(true);
        let metadata_source = root.path().join("metadata-source");
        fs::write(&metadata_source, b"metadata").unwrap();
        let ops =
            InjectedOps::new(paths.clone(), metadata_source).failing(&[RenamePoint::InstallAudio]);

        let error = commit(&paths, &ops).unwrap_err();
        assert!(matches!(
            error,
            MediaError::OutputCommitFailed {
                operation: "rename_audio",
                ..
            }
        ));
        assert_old_pair_restored(&paths);
        assert!(backup_paths(root.path()).is_empty());
    }

    #[test]
    fn late_invalid_audio_destination_restores_video_and_removes_temps() {
        let (root, paths) = setup_pair(true);
        fs::remove_file(&paths.audio_dest).unwrap();
        fs::create_dir(&paths.audio_dest).unwrap();

        assert!(matches!(
            commit(&paths, &SystemFileOps),
            Err(MediaError::OutputDestinationInvalid { .. })
        ));
        assert_eq!(fs::read(&paths.video_dest).unwrap(), b"old video");
        assert!(paths.audio_dest.is_dir());
        assert!(!paths.video_temp.exists());
        assert!(!paths.audio_temp.exists());
        assert!(backup_paths(root.path()).is_empty());
    }

    #[test]
    fn rollback_failure_reports_primary_and_rollback_errors() {
        let (root, paths) = setup_pair(true);
        let metadata_source = root.path().join("metadata-source");
        fs::write(&metadata_source, b"metadata").unwrap();
        let ops = InjectedOps::new(paths.clone(), metadata_source)
            .failing(&[RenamePoint::InstallAudio, RenamePoint::RestoreVideo]);

        assert!(matches!(
            commit(&paths, &ops),
            Err(MediaError::OutputRollbackFailed {
                operation: "rename_audio",
                ..
            })
        ));
        assert!(!paths.video_dest.exists());
        assert_eq!(fs::read(&paths.audio_dest).unwrap(), b"old audio");
        assert!(!paths.video_temp.exists());
        assert!(!paths.audio_temp.exists());
        assert_eq!(backup_paths(root.path()).len(), 1);
    }

    #[test]
    fn backup_removal_failure_is_reported_after_outputs_are_installed() {
        let (root, paths) = setup_pair(true);
        let metadata_source = root.path().join("metadata-source");
        fs::write(&metadata_source, b"metadata").unwrap();
        let ops = InjectedOps::new(paths.clone(), metadata_source).with_backup_removal_failure();

        assert!(matches!(
            commit(&paths, &ops),
            Err(MediaError::OutputCommitFailed {
                operation: "remove_backup",
                ..
            })
        ));
        assert_eq!(fs::read(&paths.video_dest).unwrap(), b"new video");
        assert_eq!(fs::read(&paths.audio_dest).unwrap(), b"new audio");
        assert_eq!(backup_paths(root.path()).len(), 2);
    }
}
