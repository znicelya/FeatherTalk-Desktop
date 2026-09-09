//! Verification of a finished asset package: frames, landmarks, and counts.

use std::fs::{self, File};
use std::io::Read;
use std::path::Path;

use feathertalk_domain::{Progress, TaskStage};
use feathertalk_frame_adapters::probe_jpeg_geometry;
use feathertalk_frame_pipeline::{
    MAX_FRAME_BYTES, PipelineError, QualityReport, read_landmark_file,
};
use feathertalk_media::CancellationToken;

use crate::{CommandOutcome, TaskReporter, admission::invalid_request, pipeline_task_error};

/// How much of a frame is read before its header is parsed.
///
/// A baseline JPEG's SOF marker sits within the first few hundred bytes and a
/// progressive one is not much further in, so 64 KiB is generous. Reading a
/// prefix rather than whole files keeps a 49-frame verification from moving
/// tens of megabytes for two numbers per frame.
const JPEG_HEADER_PROBE_BYTES: u64 = 64 * 1024;

/// Verify every frame the quality report lists and return the shared geometry.
///
/// Structural verification only: nothing is re-hashed and `frame_bytes` is not
/// compared, because operators are allowed to hand-edit frames before locking.
/// What must hold is that every frame exists, is a regular file within the
/// size limit, decodes, has the same dimensions as its siblings, and has a
/// landmark file whose points fall inside it.
pub(crate) fn verify_frames(
    assets: &Path,
    report: &QualityReport,
    token: &CancellationToken,
    reporter: &dyn TaskReporter,
) -> Result<(u32, u32), CommandOutcome> {
    let total = Some(report.frame_count());
    let mut geometry: Option<(u64, u32, u32)> = None;
    for (position, frame) in report.frames().iter().enumerate() {
        if token.is_cancelled() {
            return Err(CommandOutcome::Cancelled);
        }
        let frame_path = assets.join(frame.frame_file());
        let (width, height) = verify_frame(&frame_path).map_err(failed)?;
        match geometry {
            None => geometry = Some((frame.index(), width, height)),
            Some((reference_index, reference_width, reference_height)) => {
                if width != reference_width || height != reference_height {
                    return Err(CommandOutcome::Failed(invalid_request(
                        "素材帧尺寸不一致",
                        format!(
                            "frame {} is {width}x{height} but frame {reference_index} is \
                             {reference_width}x{reference_height}",
                            frame.index()
                        ),
                    )));
                }
            }
        }
        let landmark_path = assets.join(frame.landmark_file());
        read_landmark_file(&landmark_path, width, height).map_err(failed)?;
        let completed = position as u64 + 1;
        reporter.report(TaskStage::Preparing, Some(Progress { completed, total }));
    }
    match geometry {
        Some((_, width, height)) => Ok((width, height)),
        None => Err(CommandOutcome::Failed(invalid_request(
            "素材包没有可用的帧",
            "the quality report lists no frames".to_owned(),
        ))),
    }
}

/// Prove the asset directories hold exactly the files the report declares.
///
/// The report is a list of what should be there; the directories are what is
/// there. A leftover frame from an earlier, longer extraction is invisible to
/// a walk that follows the list, and would desynchronise every consumer that
/// reads the directory instead.
pub(crate) fn count_asset_files(assets: &Path, frame_count: u64) -> Result<(), CommandOutcome> {
    for (directory, extension) in [("frames", "jpg"), ("landmarks", "lms")] {
        let path = assets.join(directory);
        let counted = count_matching(&path, extension).map_err(failed)?;
        if counted != frame_count {
            return Err(CommandOutcome::Failed(invalid_request(
                "素材目录的文件数与质检报告不一致",
                format!(
                    "{} holds {counted} .{extension} files, the quality report declares \
                     {frame_count}",
                    path.display()
                ),
            )));
        }
    }
    Ok(())
}

/// Check one frame file and read its geometry.
fn verify_frame(path: &Path) -> Result<(u32, u32), PipelineError> {
    let metadata = match fs::symlink_metadata(path) {
        Ok(metadata) => metadata,
        Err(source) if source.kind() == std::io::ErrorKind::NotFound => {
            return Err(PipelineError::FrameMissing {
                path: path.to_owned(),
            });
        }
        Err(source) => return Err(io("stat_frame", path, source)),
    };
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(PipelineError::FrameNotRegular {
            path: path.to_owned(),
        });
    }
    let size = metadata.len();
    if size == 0 {
        return Err(PipelineError::FrameEmpty {
            path: path.to_owned(),
        });
    }
    if size > MAX_FRAME_BYTES {
        return Err(PipelineError::FrameTooLarge {
            path: path.to_owned(),
            limit: MAX_FRAME_BYTES,
            actual: size,
        });
    }
    let prefix = read_prefix(path, JPEG_HEADER_PROBE_BYTES)?;
    match probe_jpeg_geometry(path, &prefix) {
        Ok(geometry) => Ok(geometry),
        // A progressive or heavily commented JPEG can push its SOF marker
        // past the prefix. Re-read the whole file once and let the second
        // attempt's diagnostic stand, so the error describes the whole file.
        Err(_) if size > prefix.len() as u64 => {
            let whole = read_prefix(path, size)?;
            probe_jpeg_geometry(path, &whole)
        }
        Err(error) => Err(error),
    }
}

/// Read at most `limit` bytes from `path`.
fn read_prefix(path: &Path, limit: u64) -> Result<Vec<u8>, PipelineError> {
    let file = File::open(path).map_err(|source| io("open_frame", path, source))?;
    let mut bytes = Vec::new();
    file.take(limit)
        .read_to_end(&mut bytes)
        .map_err(|source| io("read_frame", path, source))?;
    Ok(bytes)
}

fn count_matching(path: &Path, extension: &str) -> Result<u64, PipelineError> {
    let entries = fs::read_dir(path).map_err(|source| io("read_assets_dir", path, source))?;
    let mut counted = 0u64;
    for entry in entries {
        let entry = entry.map_err(|source| io("read_assets_dir", path, source))?;
        let name = entry.file_name();
        let Some(name) = name.to_str() else {
            continue;
        };
        if is_indexed_name(name, extension) {
            counted += 1;
        }
    }
    Ok(counted)
}

/// `000123.jpg` and nothing else. Hand-written because the workspace carries
/// no regex dependency and does not need one for six digits and a suffix.
fn is_indexed_name(name: &str, extension: &str) -> bool {
    let Some((stem, suffix)) = name.rsplit_once('.') else {
        return false;
    };
    suffix == extension && stem.len() == 6 && stem.bytes().all(|byte| byte.is_ascii_digit())
}

/// `publish.rs` and `landmark.rs` keep their own copies of this helper, so
/// this module follows the same local pattern.
fn io(operation: &'static str, path: &Path, source: std::io::Error) -> PipelineError {
    PipelineError::Io {
        operation,
        path: path.to_owned(),
        source,
    }
}

/// Nothing in this module can produce `PipelineError::Cancelled`, so there is
/// no cancellation bridge here; cancellation is the caller's own outcome.
fn failed(error: PipelineError) -> CommandOutcome {
    CommandOutcome::Failed(pipeline_task_error(&error))
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::sync::Mutex;

    use feathertalk_domain::ErrorCode;
    use feathertalk_frame_pipeline::FrameQuality;
    use tempfile::TempDir;

    use super::*;
    use crate::NoReporter;

    const SHA256: &str = "1111111111111111111111111111111111111111111111111111111111111111";

    /// 1280x720.
    fn wide_frame() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../feathertalk-frame-adapters/tests/fixtures/demo_frame_v1/frame.jpg")
    }

    /// 640x640, used to force a geometry disagreement.
    fn square_frame() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../feathertalk-frame-adapters/tests/fixtures/opencv_cpu_v1/frame.jpg")
    }

    #[derive(Default)]
    struct Recorder {
        events: Mutex<Vec<(TaskStage, Option<Progress>)>>,
    }

    impl TaskReporter for Recorder {
        fn report(&self, stage: TaskStage, progress: Option<Progress>) {
            self.events
                .lock()
                .expect("the recorder must not be poisoned")
                .push((stage, progress));
        }
    }

    /// 110 points, all inside every frame these tests use.
    fn landmark_text() -> String {
        let mut text = String::new();
        for index in 0..110 {
            text.push_str(&format!("{index} {}\n", index * 2));
        }
        text
    }

    fn frame_quality(index: u64, frame_bytes: u64) -> FrameQuality {
        FrameQuality::new(
            index,
            format!("frames/{index:06}.jpg"),
            format!("landmarks/{index:06}.lms"),
            frame_bytes,
            SHA256,
            SHA256,
            0.9,
            [0.0, 0.0, 64.0, 64.0],
            12.5,
        )
        .expect("the frame quality fixture must be valid")
    }

    /// An asset directory of `count` copies of the wide fixture frame, with a
    /// matching quality report. The `TempDir` is returned so it outlives the
    /// test body.
    fn package(count: u64) -> (TempDir, PathBuf, QualityReport) {
        let dir = TempDir::new().expect("a temporary directory must be available");
        let assets = dir.path().join("assets");
        fs::create_dir_all(assets.join("frames")).unwrap();
        fs::create_dir_all(assets.join("landmarks")).unwrap();
        let source = fs::read(wide_frame()).expect("the wide fixture must be readable");
        let mut frames = Vec::new();
        for index in 0..count {
            fs::write(assets.join(format!("frames/{index:06}.jpg")), &source).unwrap();
            fs::write(
                assets.join(format!("landmarks/{index:06}.lms")),
                landmark_text(),
            )
            .unwrap();
            frames.push(frame_quality(index, source.len() as u64));
        }
        let report =
            QualityReport::new(count, frames, Vec::new()).expect("the report must be valid");
        (dir, assets, report)
    }

    #[test]
    fn a_consistent_package_reports_its_geometry_and_progress() {
        let (_dir, assets, report) = package(3);
        let reporter = Recorder::default();
        let geometry =
            verify_frames(&assets, &report, &CancellationToken::new(), &reporter).unwrap();
        assert_eq!(geometry, (1280, 720));
        count_asset_files(&assets, report.frame_count()).unwrap();

        let events = reporter.events.lock().unwrap().clone();
        assert_eq!(events.len(), 3);
        for (position, (stage, progress)) in events.iter().enumerate() {
            assert_eq!(*stage, TaskStage::Preparing);
            let completed = position as u64 + 1;
            assert_eq!(
                *progress,
                Some(Progress {
                    completed,
                    total: Some(3)
                })
            );
        }
    }

    #[test]
    fn a_missing_frame_is_a_media_failure() {
        let (_dir, assets, report) = package(2);
        fs::remove_file(assets.join("frames/000001.jpg")).unwrap();
        let outcome = verify_frames(&assets, &report, &CancellationToken::new(), &NoReporter)
            .expect_err("a missing frame must fail");
        let CommandOutcome::Failed(error) = outcome else {
            panic!("a missing frame must be a failure");
        };
        assert_eq!(error.code, ErrorCode::MediaInvalid);
        assert_eq!(error.summary, "抽出的帧不可用");
        error.validate().unwrap();
    }

    #[test]
    fn a_frame_of_a_different_size_is_refused() {
        let (_dir, assets, report) = package(3);
        let square = fs::read(square_frame()).expect("the square fixture must be readable");
        fs::write(assets.join("frames/000001.jpg"), &square).unwrap();
        let outcome = verify_frames(&assets, &report, &CancellationToken::new(), &NoReporter)
            .expect_err("mismatched geometry must fail");
        let CommandOutcome::Failed(error) = outcome else {
            panic!("mismatched geometry must be a failure");
        };
        assert_eq!(error.summary, "素材帧尺寸不一致");
        assert!(error.detail.contains("640x640"), "{}", error.detail);
        assert!(error.detail.contains("1280x720"), "{}", error.detail);
    }

    #[test]
    fn a_malformed_landmark_file_is_refused() {
        let (_dir, assets, report) = package(1);
        fs::write(assets.join("landmarks/000000.lms"), "0 0\n").unwrap();
        let outcome = verify_frames(&assets, &report, &CancellationToken::new(), &NoReporter)
            .expect_err("a malformed landmark file must fail");
        let CommandOutcome::Failed(error) = outcome else {
            panic!("a malformed landmark file must be a failure");
        };
        assert_eq!(error.summary, "关键点文件不可用");
        assert_eq!(error.code, ErrorCode::MediaInvalid);
    }

    #[test]
    fn a_report_with_no_frames_has_no_geometry() {
        let dir = TempDir::new().unwrap();
        let assets = dir.path().join("assets");
        fs::create_dir_all(assets.join("frames")).unwrap();
        let report = QualityReport::new(1, Vec::new(), Vec::new()).unwrap();
        let outcome = verify_frames(&assets, &report, &CancellationToken::new(), &NoReporter)
            .expect_err("an empty report must fail");
        let CommandOutcome::Failed(error) = outcome else {
            panic!("an empty report must be a failure");
        };
        assert_eq!(error.summary, "素材包没有可用的帧");
    }

    #[test]
    fn a_leftover_file_breaks_the_count() {
        let (_dir, assets, report) = package(3);
        fs::copy(
            assets.join("frames/000000.jpg"),
            assets.join("frames/000009.jpg"),
        )
        .unwrap();
        let outcome = count_asset_files(&assets, report.frame_count())
            .expect_err("a leftover frame must fail");
        let CommandOutcome::Failed(error) = outcome else {
            panic!("a leftover frame must be a failure");
        };
        assert_eq!(error.summary, "素材目录的文件数与质检报告不一致");
        assert!(error.detail.contains("4 .jpg files"), "{}", error.detail);
    }

    #[test]
    fn files_that_are_not_indexed_assets_do_not_count() {
        let (_dir, assets, report) = package(2);
        fs::write(assets.join("frames/notes.txt"), "scratch").unwrap();
        fs::write(assets.join("frames/00001.jpg"), "a five digit stem").unwrap();
        fs::create_dir(assets.join("frames/nested")).unwrap();
        count_asset_files(&assets, report.frame_count()).unwrap();
    }

    #[test]
    fn a_cancelled_token_stops_before_the_first_frame() {
        let (_dir, assets, report) = package(2);
        let token = CancellationToken::new();
        token.cancel();
        let reporter = Recorder::default();
        let outcome = verify_frames(&assets, &report, &token, &reporter)
            .expect_err("a cancelled token must stop the scan");
        assert!(matches!(outcome, CommandOutcome::Cancelled));
        assert!(reporter.events.lock().unwrap().is_empty());
    }
}
