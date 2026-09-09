use std::{
    fs::{self, File, OpenOptions},
    io::{Read, Write},
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};

use sha2::{Digest, Sha256};

use crate::{
    FrameBatch, FrameEvaluation, FramePipelineSpec, FrameQuality, MAX_FRAME_BYTES, PipelineError,
    QualityReport,
};

const MAX_REPORT_BYTES: usize = 16 * 1024 * 1024;
static PUBLISH_COUNTER: AtomicU64 = AtomicU64::new(0);

pub fn publish_frame_artifacts(
    spec: &FramePipelineSpec,
    batch: &mut FrameBatch,
    evaluation: &FrameEvaluation,
) -> Result<QualityReport, PipelineError> {
    let ops = SystemFileOps;
    publish_frame_artifacts_with_ops(spec, batch, evaluation, &ops)
}

fn publish_frame_artifacts_with_ops(
    spec: &FramePipelineSpec,
    batch: &mut FrameBatch,
    evaluation: &FrameEvaluation,
    ops: &dyn FileOps,
) -> Result<QualityReport, PipelineError> {
    if !evaluation.anomalies().is_empty() {
        return Err(PipelineError::QualityRejected {
            count: evaluation.anomalies().len(),
        });
    }
    if evaluation.accepted().len() != spec.frame_count() as usize {
        return Err(PipelineError::PublishFailed {
            operation: "validate_frame_count",
            message: format!(
                "expected {}, got {}",
                spec.frame_count(),
                evaluation.accepted().len()
            ),
        });
    }
    let staging = batch.staging_dir().to_owned();
    validate_evaluation_paths(spec, batch, evaluation)?;
    let landmarks_dir = staging.join("landmarks");
    ops.create_dir(&landmarks_dir)
        .map_err(|source| io("create_landmarks_dir", &landmarks_dir, source))?;

    let mut frames = Vec::with_capacity(evaluation.accepted().len());
    for accepted in evaluation.accepted() {
        let landmark_path = landmarks_dir.join(format!("{:06}.lms", accepted.index()));
        write_synced_new(&landmark_path, accepted.landmark_bytes())?;
        let (frame_bytes, frame_hash) = hash_file(accepted.frame_path())?;
        let extracted = batch
            .frames()
            .iter()
            .find(|frame| frame.index() == accepted.index())
            .ok_or_else(|| PipelineError::PublishFailed {
                operation: "validate_frame_path",
                message: format!("missing extracted frame {}", accepted.index()),
            })?;
        if extracted.bytes() != frame_bytes || extracted.sha256() != frame_hash {
            return Err(PipelineError::PublishFailed {
                operation: "validate_frame_integrity",
                message: format!("frame {} changed after extraction", accepted.index()),
            });
        }
        let (_landmark_bytes, landmark_hash) = hash_file(&landmark_path)?;
        let frame_file = format!("frames/{:06}.jpg", accepted.index());
        let landmark_file = format!("landmarks/{:06}.lms", accepted.index());
        frames.push(FrameQuality::new(
            accepted.index(),
            frame_file,
            landmark_file,
            frame_bytes,
            frame_hash,
            landmark_hash,
            accepted.face_score(),
            accepted.bbox(),
            accepted.blur_variance(),
        )?);
    }
    frames.sort_by_key(|frame| frame.index());
    let report = QualityReport::new(spec.frame_count(), frames, Vec::new())?;
    let report_bytes =
        serde_json::to_vec_pretty(&report).map_err(|source| PipelineError::ReportJson {
            message: source.to_string(),
        })?;
    if report_bytes.len() > MAX_REPORT_BYTES {
        return Err(PipelineError::ReportTooLarge {
            limit: MAX_REPORT_BYTES,
            actual: report_bytes.len(),
        });
    }
    let staged_report = staging.join("quality.json");
    write_synced_new(&staged_report, &report_bytes)?;
    sync_dir(ops, &landmarks_dir)?;
    sync_dir(ops, &staging.join("frames"))?;
    sync_dir(ops, &staging)?;

    commit_staging(spec, batch, &staging, &staged_report, ops)?;
    batch.disarm();
    Ok(report)
}

pub fn read_quality_report(path: &Path) -> Result<QualityReport, PipelineError> {
    let metadata =
        fs::symlink_metadata(path).map_err(|source| io("stat_quality_report", path, source))?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(PipelineError::ReportNotRegular {
            path: path.to_owned(),
        });
    }
    let mut file = File::open(path).map_err(|source| io("open_quality_report", path, source))?;
    let size = file
        .metadata()
        .map_err(|source| io("stat_quality_report", path, source))?
        .len();
    if size > MAX_REPORT_BYTES as u64 {
        return Err(PipelineError::ReportTooLarge {
            limit: MAX_REPORT_BYTES,
            actual: size as usize,
        });
    }
    let mut bytes = Vec::with_capacity(size as usize);
    file.read_to_end(&mut bytes)
        .map_err(|source| io("read_quality_report", path, source))?;
    let report: QualityReport =
        serde_json::from_slice(&bytes).map_err(|source| PipelineError::ReportJson {
            message: source.to_string(),
        })?;
    report.validate()?;
    Ok(report)
}

trait FileOps {
    fn metadata(&self, path: &Path) -> std::io::Result<fs::Metadata>;
    fn create_dir(&self, path: &Path) -> std::io::Result<()>;
    fn rename(&self, from: &Path, to: &Path) -> std::io::Result<()>;
    fn remove_path(&self, path: &Path) -> std::io::Result<()>;
    fn sync_dir(&self, path: &Path) -> std::io::Result<()>;
}

#[derive(Debug, Default, Clone, Copy)]
struct SystemFileOps;

impl FileOps for SystemFileOps {
    fn metadata(&self, path: &Path) -> std::io::Result<fs::Metadata> {
        fs::symlink_metadata(path)
    }

    fn create_dir(&self, path: &Path) -> std::io::Result<()> {
        fs::create_dir(path)
    }

    fn rename(&self, from: &Path, to: &Path) -> std::io::Result<()> {
        fs::rename(from, to)
    }

    fn remove_path(&self, path: &Path) -> std::io::Result<()> {
        remove_file_or_dir(path)
    }

    fn sync_dir(&self, path: &Path) -> std::io::Result<()> {
        #[cfg(unix)]
        {
            File::open(path).and_then(|file| file.sync_all())
        }
        #[cfg(windows)]
        {
            let _ = path;
            Ok(())
        }
    }
}

fn commit_staging(
    spec: &FramePipelineSpec,
    batch: &FrameBatch,
    staging: &Path,
    staged_report: &Path,
    ops: &dyn FileOps,
) -> Result<(), PipelineError> {
    let root = spec.output_root();
    let final_frames = root.join("frames");
    let final_landmarks = root.join("landmarks");
    let final_report = spec.quality_path();
    let backup_root = match create_backup_root(root, ops) {
        Ok(path) => path,
        Err(error) => {
            return Err(cleanup_commit_failure(
                error,
                batch,
                staging,
                None,
                &[],
                ops,
            ));
        }
    };
    let mut moved = Vec::<(PathBuf, PathBuf)>::new();
    let mut installed = Vec::<PathBuf>::new();
    let destinations = [
        (final_frames.clone(), "frames"),
        (final_landmarks.clone(), "landmarks"),
        (final_report.clone(), "report"),
    ];

    for (destination, name) in &destinations {
        match ops.metadata(destination) {
            Ok(metadata) => {
                let expected_kind_ok = match *name {
                    "frames" | "landmarks" => metadata.is_dir(),
                    "report" => metadata.is_file(),
                    _ => false,
                };
                if metadata.file_type().is_symlink() || !expected_kind_ok {
                    let primary = PipelineError::PublishFailed {
                        operation: "validate_destination",
                        message: format!("invalid destination: {}", destination.display()),
                    };
                    return Err(rollback_commit(
                        primary,
                        batch,
                        staging,
                        &backup_root,
                        &moved,
                        &installed,
                        ops,
                    ));
                }
                let backup = backup_root.join(destination.file_name().ok_or_else(|| {
                    PipelineError::PublishFailed {
                        operation: "backup_existing",
                        message: format!("destination has no file name: {}", destination.display()),
                    }
                })?);
                if let Err(source) = ops.rename(destination, &backup) {
                    let primary = PipelineError::PublishFailed {
                        operation: "backup_existing",
                        message: source.to_string(),
                    };
                    return Err(rollback_commit(
                        primary,
                        batch,
                        staging,
                        &backup_root,
                        &moved,
                        &installed,
                        ops,
                    ));
                }
                moved.push((destination.clone(), backup));
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(source) => {
                let primary = io("stat_destination", destination, source);
                return Err(rollback_commit(
                    primary,
                    batch,
                    staging,
                    &backup_root,
                    &moved,
                    &installed,
                    ops,
                ));
            }
        }
    }

    for (destination, name) in &destinations {
        let source = match *name {
            "frames" => staging.join("frames"),
            "landmarks" => staging.join("landmarks"),
            "report" => staged_report.to_owned(),
            _ => unreachable!(),
        };
        if let Err(error) = ops.rename(&source, destination) {
            let primary = PipelineError::PublishFailed {
                operation: match *name {
                    "frames" => "install_frames",
                    "landmarks" => "install_landmarks",
                    "report" => "install_report",
                    _ => unreachable!(),
                },
                message: error.to_string(),
            };
            return Err(rollback_commit(
                primary,
                batch,
                staging,
                &backup_root,
                &moved,
                &installed,
                ops,
            ));
        }
        installed.push(destination.clone());
    }

    if let Err(source) = ops.sync_dir(root) {
        let primary = io("sync_output", root, source);
        return Err(rollback_commit(
            primary,
            batch,
            staging,
            &backup_root,
            &moved,
            &installed,
            ops,
        ));
    }
    if let Err(source) = ops.remove_path(staging) {
        let primary = io("remove_staging", staging, source);
        return Err(rollback_commit(
            primary,
            batch,
            staging,
            &backup_root,
            &moved,
            &installed,
            ops,
        ));
    }
    if let Err(source) = ops.remove_path(&backup_root) {
        let primary = io("remove_backup", &backup_root, source);
        return Err(rollback_commit(
            primary,
            batch,
            staging,
            &backup_root,
            &moved,
            &installed,
            ops,
        ));
    }
    Ok(())
}

fn create_backup_root(root: &Path, ops: &dyn FileOps) -> Result<PathBuf, PipelineError> {
    for _ in 0..32 {
        let path = root.join(format!(
            ".feathertalk-frame-backup-{}-{}",
            std::process::id(),
            PUBLISH_COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        match ops.metadata(&path) {
            Ok(_) => continue,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(source) => return Err(io("stat_backup", &path, source)),
        }
        match ops.create_dir(&path) {
            Ok(()) => return Ok(path),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(source) => return Err(io("create_backup", &path, source)),
        }
    }
    Err(PipelineError::PublishFailed {
        operation: "create_backup",
        message: "unable to allocate collision-free backup directory after 32 attempts".into(),
    })
}

fn cleanup_commit_failure(
    primary: PipelineError,
    _batch: &FrameBatch,
    staging: &Path,
    backup_root: Option<&Path>,
    rollback_errors: &[String],
    ops: &dyn FileOps,
) -> PipelineError {
    let mut errors = rollback_errors.to_vec();
    if let Some(backup_root) = backup_root
        && let Err(error) = ops.remove_path(backup_root)
        && error.kind() != std::io::ErrorKind::NotFound
    {
        errors.push(format!("remove backup: {error}"));
    }
    if let Err(error) = ops.remove_path(staging)
        && error.kind() != std::io::ErrorKind::NotFound
    {
        errors.push(format!("remove staging: {error}"));
    }
    if errors.is_empty() {
        primary
    } else {
        PipelineError::PublishRollbackFailed {
            operation: "rollback",
            primary: primary.to_string(),
            rollback: errors.join("; "),
        }
    }
}

fn rollback_commit(
    primary: PipelineError,
    batch: &FrameBatch,
    staging: &Path,
    backup_root: &Path,
    moved: &[(PathBuf, PathBuf)],
    installed: &[PathBuf],
    ops: &dyn FileOps,
) -> PipelineError {
    let mut errors = Vec::new();
    for destination in installed.iter().rev() {
        match ops.remove_path(destination) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => errors.push(format!(
                "remove installed {}: {error}",
                destination.display()
            )),
        }
    }
    for (destination, backup) in moved.iter().rev() {
        if let Err(error) = ops.rename(backup, destination) {
            errors.push(format!("restore {}: {error}", destination.display()));
        }
    }
    cleanup_commit_failure(primary, batch, staging, Some(backup_root), &errors, ops)
}

fn write_synced_new(path: &Path, bytes: &[u8]) -> Result<(), PipelineError> {
    if bytes.is_empty() {
        return Err(PipelineError::PublishFailed {
            operation: "write_artifact",
            message: format!("empty artifact: {}", path.display()),
        });
    }
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|source| io("create_artifact", path, source))?;
    file.write_all(bytes)
        .and_then(|_| file.sync_all())
        .map_err(|source| io("write_artifact", path, source))?;
    Ok(())
}

fn hash_file(path: &Path) -> Result<(u64, String), PipelineError> {
    let metadata =
        fs::symlink_metadata(path).map_err(|source| io("stat_artifact", path, source))?;
    if metadata.file_type().is_symlink() || !metadata.is_file() || metadata.len() == 0 {
        return Err(PipelineError::PublishFailed {
            operation: "validate_artifact",
            message: format!("invalid artifact: {}", path.display()),
        });
    }
    if metadata.len() > MAX_FRAME_BYTES {
        return Err(PipelineError::PublishFailed {
            operation: "validate_artifact",
            message: format!("artifact too large: {}", path.display()),
        });
    }
    let mut file = File::open(path).map_err(|source| io("open_artifact", path, source))?;
    let mut digest = Sha256::new();
    let mut bytes = 0_u64;
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let read = file
            .read(&mut buffer)
            .map_err(|source| io("hash_artifact", path, source))?;
        if read == 0 {
            break;
        }
        bytes += read as u64;
        digest.update(&buffer[..read]);
    }
    Ok((bytes, hex::encode(digest.finalize())))
}

fn sync_dir(ops: &dyn FileOps, path: &Path) -> Result<(), PipelineError> {
    ops.sync_dir(path)
        .map_err(|source| io("sync_directory", path, source))
}

fn validate_evaluation_paths(
    spec: &FramePipelineSpec,
    batch: &FrameBatch,
    evaluation: &FrameEvaluation,
) -> Result<(), PipelineError> {
    let mut seen = std::collections::HashSet::new();
    for accepted in evaluation.accepted() {
        if accepted.index() >= spec.frame_count() || !seen.insert(accepted.index()) {
            return Err(PipelineError::PublishFailed {
                operation: "validate_frame_path",
                message: format!("invalid or duplicate frame index {}", accepted.index()),
            });
        }
        let expected = batch
            .staging_dir()
            .join("frames")
            .join(format!("{:06}.jpg", accepted.index()));
        if accepted.frame_path() != expected
            || !batch
                .frames()
                .iter()
                .any(|frame| frame.index() == accepted.index() && frame.path() == expected)
        {
            return Err(PipelineError::PublishFailed {
                operation: "validate_frame_path",
                message: format!(
                    "accepted frame {} is not owned by this batch",
                    accepted.index()
                ),
            });
        }
    }
    Ok(())
}

fn io(operation: &'static str, path: &Path, source: std::io::Error) -> PipelineError {
    PipelineError::Io {
        operation,
        path: path.to_owned(),
        source,
    }
}

fn remove_file_or_dir(path: &Path) -> std::io::Result<()> {
    let metadata = fs::symlink_metadata(path)?;
    if metadata.is_dir() && !metadata.file_type().is_symlink() {
        fs::remove_dir_all(path)
    } else {
        fs::remove_file(path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    #[derive(Clone, Copy, PartialEq, Eq)]
    enum FailPoint {
        InstallLandmarks,
        InstallLandmarksAndRestoreFrames,
        RemoveStaging,
    }

    struct InjectedOps {
        fail: FailPoint,
        paths: Mutex<Vec<(PathBuf, PathBuf)>>,
    }

    impl InjectedOps {
        fn new(fail: FailPoint) -> Self {
            Self {
                fail,
                paths: Mutex::new(Vec::new()),
            }
        }
    }

    impl FileOps for InjectedOps {
        fn metadata(&self, path: &Path) -> std::io::Result<fs::Metadata> {
            fs::symlink_metadata(path)
        }

        fn create_dir(&self, path: &Path) -> std::io::Result<()> {
            fs::create_dir(path)
        }

        fn rename(&self, from: &Path, to: &Path) -> std::io::Result<()> {
            let is_install_landmarks = from.file_name().is_some_and(|name| name == "landmarks")
                && to.file_name().is_some_and(|name| name == "landmarks")
                && from.parent().and_then(Path::file_name).is_some_and(|name| {
                    name.to_string_lossy()
                        .starts_with(".feathertalk-frame-build-")
                });
            let is_restore_frames = from.file_name().is_some_and(|name| name == "frames")
                && to.file_name().is_some_and(|name| name == "frames")
                && from.parent().and_then(Path::file_name).is_some_and(|name| {
                    name.to_string_lossy()
                        .starts_with(".feathertalk-frame-backup-")
                });
            if (matches!(
                self.fail,
                FailPoint::InstallLandmarks | FailPoint::InstallLandmarksAndRestoreFrames
            ) && is_install_landmarks)
                || (self.fail == FailPoint::InstallLandmarksAndRestoreFrames && is_restore_frames)
            {
                return Err(std::io::Error::other("injected rename failure"));
            }
            self.paths
                .lock()
                .unwrap()
                .push((from.to_owned(), to.to_owned()));
            fs::rename(from, to)
        }

        fn remove_path(&self, path: &Path) -> std::io::Result<()> {
            if self.fail == FailPoint::RemoveStaging
                && path.file_name().is_some_and(|name| {
                    name.to_string_lossy()
                        .starts_with(".feathertalk-frame-build-")
                })
            {
                return Err(std::io::Error::other("injected staging cleanup failure"));
            }
            remove_file_or_dir(path)
        }

        fn sync_dir(&self, _path: &Path) -> std::io::Result<()> {
            Ok(())
        }
    }

    fn setup_commit() -> (tempfile::TempDir, FramePipelineSpec, PathBuf, PathBuf) {
        let root = tempfile::tempdir().unwrap();
        let video = root.path().join("video.mp4");
        fs::write(&video, b"video").unwrap();
        let output = root.path().join("assets");
        fs::create_dir_all(&output).unwrap();
        let spec = FramePipelineSpec::new(video, output.clone(), 1, 640, 480).unwrap();
        let staging = output.join(".feathertalk-frame-build-test");
        fs::create_dir_all(staging.join("frames")).unwrap();
        fs::create_dir_all(staging.join("landmarks")).unwrap();
        fs::write(staging.join("frames/000000.jpg"), b"new-frame").unwrap();
        fs::write(staging.join("landmarks/000000.lms"), b"new-landmark").unwrap();
        fs::write(staging.join("quality.json"), b"new-report").unwrap();
        fs::create_dir_all(output.join("frames")).unwrap();
        fs::create_dir_all(output.join("landmarks")).unwrap();
        fs::write(output.join("frames/000000.jpg"), b"old-frame").unwrap();
        fs::write(output.join("landmarks/000000.lms"), b"old-landmark").unwrap();
        fs::write(output.join("quality.json"), b"old-report").unwrap();
        (root, spec, staging.clone(), staging.join("quality.json"))
    }

    #[test]
    fn late_install_failure_removes_new_outputs_and_restores_old_outputs() {
        let (_root, spec, staging, report) = setup_commit();
        let batch = FrameBatch::from_staging_dir_for_test(staging.clone());
        let ops = InjectedOps::new(FailPoint::InstallLandmarks);
        let error = commit_staging(&spec, &batch, &staging, &report, &ops).unwrap_err();
        assert!(matches!(
            error,
            PipelineError::PublishFailed {
                operation: "install_landmarks",
                ..
            }
        ));
        assert_eq!(fs::read(spec.frame_path(0)).unwrap(), b"old-frame");
        assert_eq!(fs::read(spec.landmark_path(0)).unwrap(), b"old-landmark");
        assert_eq!(fs::read(spec.quality_path()).unwrap(), b"old-report");
        assert!(!staging.exists());
    }

    #[test]
    fn rollback_failure_reports_primary_and_rollback_errors() {
        let (_root, spec, staging, report) = setup_commit();
        let batch = FrameBatch::from_staging_dir_for_test(staging.clone());
        let ops = InjectedOps::new(FailPoint::InstallLandmarksAndRestoreFrames);
        let error = commit_staging(&spec, &batch, &staging, &report, &ops).unwrap_err();
        assert!(matches!(error, PipelineError::PublishRollbackFailed { .. }));
    }

    #[test]
    fn staging_cleanup_failure_rolls_back_installed_outputs() {
        let (_root, spec, staging, report) = setup_commit();
        let batch = FrameBatch::from_staging_dir_for_test(staging.clone());
        let ops = InjectedOps::new(FailPoint::RemoveStaging);
        let error = commit_staging(&spec, &batch, &staging, &report, &ops).unwrap_err();
        assert!(matches!(error, PipelineError::PublishRollbackFailed { .. }));
        assert_eq!(fs::read(spec.frame_path(0)).unwrap(), b"old-frame");
        assert_eq!(fs::read(spec.landmark_path(0)).unwrap(), b"old-landmark");
        assert_eq!(fs::read(spec.quality_path()).unwrap(), b"old-report");
    }

    struct CollisionOps {
        root: PathBuf,
        collisions: Mutex<usize>,
    }

    impl FileOps for CollisionOps {
        fn metadata(&self, path: &Path) -> std::io::Result<fs::Metadata> {
            if path.file_name().is_some_and(|name| {
                name.to_string_lossy()
                    .starts_with(".feathertalk-frame-backup-")
            }) && *self.collisions.lock().unwrap() > 0
            {
                *self.collisions.lock().unwrap() -= 1;
                return fs::symlink_metadata(&self.root);
            }
            fs::symlink_metadata(path)
        }

        fn create_dir(&self, path: &Path) -> std::io::Result<()> {
            fs::create_dir(path)
        }

        fn rename(&self, from: &Path, to: &Path) -> std::io::Result<()> {
            fs::rename(from, to)
        }

        fn remove_path(&self, path: &Path) -> std::io::Result<()> {
            remove_file_or_dir(path)
        }

        fn sync_dir(&self, _path: &Path) -> std::io::Result<()> {
            Ok(())
        }
    }

    #[test]
    fn backup_directory_collision_is_skipped_without_overwriting_existing_entry() {
        let root = tempfile::tempdir().unwrap();
        let ops = CollisionOps {
            root: root.path().to_owned(),
            collisions: Mutex::new(3),
        };
        let backup = create_backup_root(root.path(), &ops).unwrap();
        assert!(backup.exists());
        assert_eq!(*ops.collisions.lock().unwrap(), 0);
        fs::remove_dir_all(backup).unwrap();
    }
}
