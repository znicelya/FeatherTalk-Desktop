mod support;

use std::{
    fs,
    path::{Path, PathBuf},
    sync::{Arc, OnceLock},
    time::Duration,
};

use feathertalk_frame_adapters::{
    FrameImageCache, JpegFrameDecoder, PfldLandmarkPredictor, ScrfdFaceDetector,
};
use feathertalk_frame_pipeline::{
    AnomalyCode, BLUR_VARIANCE_THRESHOLD, CommandSpec, FrameAnomaly, FrameEvaluation,
    FrameExtractor, FramePipelineSpec, PipelineError, ProcessOutput, ProcessRunner, RecoveryAction,
    evaluate_frames_with_models, extract_frames_with_runner,
};
use feathertalk_models::backend::CpuBackend;
use feathertalk_scrfd::ScrfdArtifactPaths;

/// The committed SCRFD artifact pair, two crates over.
fn scrfd_artifact_paths() -> ScrfdArtifactPaths {
    let root =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../feathertalk-scrfd/artifacts/scrfd_2_5g");
    ScrfdArtifactPaths {
        manifest: root.join("manifest.json"),
        weights: root.join("model.safetensors"),
    }
}

/// The committed PFLD artifact directory, one crate over.
fn pfld_artifact_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../feathertalk-pfld/artifacts/pfld_ghost_one")
}

/// Stands in for ffmpeg. `extract_frames_with_runner` writes each frame to
/// `<staging>/frames/{index:06}.jpg`, so the frame index selects which
/// committed JPEG this frame carries.
struct FixtureRunner {
    payloads: Vec<Vec<u8>>,
}

impl ProcessRunner for FixtureRunner {
    fn run(
        &self,
        command: &CommandSpec,
        _timeout: Duration,
    ) -> Result<ProcessOutput, PipelineError> {
        for (index, path) in support::chunk_outputs(command) {
            let payload = self
                .payloads
                .get(index as usize)
                .unwrap_or_else(|| panic!("no payload for frame {index}"));
            fs::write(&path, payload).expect("the staging directory is writable");
        }
        Ok(ProcessOutput::new(Some(0), vec![], vec![]))
    }
}

/// One extraction and one evaluation, shared by every test in this file.
///
/// `staged_paths` records what the extractor wrote, in index order. The staging
/// tree and the tempdir above it are already deleted when the tests read this,
/// so those paths are values to compare against `AcceptedFrame::frame_path`,
/// not files to open.
struct Outcome {
    evaluation: FrameEvaluation,
    staged_paths: Vec<PathBuf>,
}

/// Loading the GhostOne graph moves a 125 768-byte module struct through
/// several frames, which overruns the default test-thread stack on Windows.
/// `feathertalk-weights` solves the same problem for its detached clone with a
/// dedicated 64 MiB stack, and `tests/pfld_model.rs` mirrors it; the whole
/// builder runs on such a thread here because the load happens inside it.
const OUTCOME_LOAD_STACK_BYTES: usize = 64 * 1024 * 1024;

fn outcome() -> &'static Outcome {
    static OUTCOME: OnceLock<Outcome> = OnceLock::new();
    OUTCOME.get_or_init(|| {
        std::thread::Builder::new()
            .name("pipeline-outcome".to_owned())
            .stack_size(OUTCOME_LOAD_STACK_BYTES)
            .spawn(build_outcome)
            .expect("the loader thread starts")
            .join()
            .expect("the loader thread does not panic")
    })
}

/// Three SCRFD forwards, two PFLD forwards and one load of each set of weights.
/// Frame 2 stops at the confidence gate, so it never reaches PFLD.
fn build_outcome() -> Outcome {
    let demo = support::load_and_verify_demo_fixture().expect("the demo fixture verifies");
    let synthetic = support::load_and_verify_fixture().expect("the synthetic fixture verifies");
    let payloads = vec![
        fs::read(support::demo_frame(&demo, "sharp").path).expect("the sharp JPEG reads"),
        fs::read(support::demo_frame(&demo, "blurred").path).expect("the blurred JPEG reads"),
        fs::read(synthetic.root.join("frame.jpg")).expect("the synthetic JPEG reads"),
    ];

    // `FrameBatch` has no public constructor, so the batch has to come out of
    // `extract_frames_with_runner`. The stub runner keeps ffmpeg out of the test
    // while still producing a real staging tree.
    let root = tempfile::tempdir().expect("a temporary directory is available");
    let video = root.path().join("stub_video.mp4");
    fs::write(&video, b"never read by the stub runner").expect("the tempdir is writable");

    // 1280x720 describes the demo frames. Nothing compares these numbers with a
    // decoded frame: they only shape the ffmpeg command line, which the stub
    // ignores, so the 640x640 synthetic payload at index 2 is not a mismatch.
    let spec = FramePipelineSpec::new(video, root.path().join("assets"), 3, 1280, 720)
        .expect("the spec is well formed");
    let extractor = FrameExtractor::new(root.path().join("ffmpeg"), Duration::from_secs(1))
        .expect("the extractor path is absolute");
    let batch = extract_frames_with_runner(&spec, &extractor, &FixtureRunner { payloads })
        .expect("the stub runner stages every frame");
    let staged_paths: Vec<PathBuf> = batch
        .frames()
        .iter()
        .map(|frame| frame.path().to_path_buf())
        .collect();

    // One cache for all three adapters, exactly as the pipeline wires them: each
    // frame is decoded once and the pixels are reused by detect and predict.
    let cache = Arc::new(FrameImageCache::new());
    let decoder = JpegFrameDecoder::new(Arc::clone(&cache));
    let detector = ScrfdFaceDetector::<CpuBackend>::load(
        &scrfd_artifact_paths(),
        Default::default(),
        Arc::clone(&cache),
    )
    .expect("the committed SCRFD artifact loads");
    let predictor =
        PfldLandmarkPredictor::<CpuBackend>::load(&pfld_artifact_dir(), Default::default(), cache)
            .expect("the committed PFLD artifact loads");

    let evaluation = evaluate_frames_with_models(&batch, &decoder, &detector, &predictor)
        .expect("no frame raises a hard pipeline error");

    Outcome {
        evaluation,
        staged_paths,
    }
}

fn anomaly_for(evaluation: &FrameEvaluation, index: u64) -> &FrameAnomaly {
    evaluation
        .anomalies()
        .iter()
        .find(|anomaly| anomaly.frame_index() == index)
        .unwrap_or_else(|| panic!("no anomaly for frame {index}"))
}

/// `serialize_landmarks` writes one x-space-y line per point. Reading them back
/// is what proves the published bytes carry this frame's prediction.
fn parse_landmarks(bytes: &[u8]) -> Vec<[i32; 2]> {
    std::str::from_utf8(bytes)
        .expect("landmark bytes are ASCII")
        .lines()
        .map(|line| {
            let (x, y) = line.split_once(' ').expect("every line is two integers");
            [
                x.parse::<i32>().expect("x is an integer"),
                y.parse::<i32>().expect("y is an integer"),
            ]
        })
        .collect()
}

#[test]
fn the_batch_splits_into_one_accepted_frame_and_two_anomalies() {
    let outcome = outcome();

    // `{index:06}.jpg` is the extractor's own naming. Pinning it here is what
    // makes `FixtureRunner`'s payload-to-index mapping verifiable.
    let names: Vec<&str> = outcome
        .staged_paths
        .iter()
        .map(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .expect("staged frames have UTF-8 names")
        })
        .collect();
    assert_eq!(names, ["000000.jpg", "000001.jpg", "000002.jpg"]);

    assert_eq!(outcome.evaluation.accepted().len(), 1);
    assert_eq!(outcome.evaluation.anomalies().len(), 2);
    assert!(!outcome.evaluation.is_success());
}

#[test]
fn the_sharp_demo_frame_is_accepted_with_the_fixture_numbers() {
    let outcome = outcome();
    let fixture = support::load_and_verify_demo_fixture().unwrap();
    let expected = support::demo_frame(&fixture, "sharp");

    let accepted = outcome
        .evaluation
        .accepted()
        .first()
        .expect("the sharp frame is accepted");
    assert_eq!(accepted.index(), 0);
    assert_eq!(accepted.frame_path(), outcome.staged_paths[0].as_path());

    // Task 1's gate fix is what makes this frame accepted: the face covers 3.4%
    // of 1280x720, so the old frame-area denominator scored it 0.034 against a
    // 0.10 minimum and the pipeline reported `bbox_out_of_bounds`.
    assert!(
        (accepted.face_score() - expected.detection.score).abs() <= 0.01,
        "score {} vs {}",
        accepted.face_score(),
        expected.detection.score
    );
    let bbox = accepted.bbox();
    for (axis, (actual, wanted)) in bbox.iter().zip(&expected.detection.bbox).enumerate() {
        assert!(
            (actual - wanted).abs() <= 1.0,
            "bbox[{axis}]: {actual} vs {wanted}"
        );
    }

    // The fixture holds cv2's variance on this JPEG (776.03) while the accepted
    // frame carries `jpeg-decoder`'s (776.09). Design section 8.2 allows 1%.
    assert!(
        (accepted.blur_variance() - expected.laplacian_variance).abs()
            <= expected.laplacian_variance * 0.01,
        "variance {} vs {}",
        accepted.blur_variance(),
        expected.laplacian_variance
    );
    assert!(accepted.blur_variance() > BLUR_VARIANCE_THRESHOLD);

    let points = parse_landmarks(accepted.landmark_bytes());
    assert_eq!(points.len(), support::PFLD_LANDMARK_COUNT);
    for (index, (point, wanted)) in points.iter().zip(&expected.landmarks).enumerate() {
        assert!(
            (point[0] - wanted[0]).abs() <= 2 && (point[1] - wanted[1]).abs() <= 2,
            "point {index}: {point:?} vs {wanted:?}"
        );
    }
}

#[test]
fn the_blurred_demo_frame_is_excluded_as_blurred() {
    let outcome = outcome();
    let fixture = support::load_and_verify_demo_fixture().unwrap();
    let expected = support::demo_frame(&fixture, "blurred");

    // Reaching this code means the blurred frame passed detection, the bbox
    // gate, PFLD and landmark serialisation: the blur gate is the last one.
    let anomaly = anomaly_for(&outcome.evaluation, 1);
    assert_eq!(anomaly.code(), AnomalyCode::BlurredFrame);
    assert_eq!(anomaly.summary(), "Frame is too blurry");
    assert_eq!(anomaly.recovery_action(), RecoveryAction::ExcludeFrame);

    // The detail is the variance this decoder measured, 5.129766, against the
    // fixture's cv2 value of 5.122010. Parsing it keeps the assertion on the
    // number rather than on the decoder's last digit.
    let reported: f64 = anomaly
        .technical_detail()
        .strip_prefix("laplacian_variance=")
        .unwrap_or_else(|| panic!("unexpected detail {}", anomaly.technical_detail()))
        .parse()
        .expect("the detail carries a decimal number");
    assert!(
        reported < BLUR_VARIANCE_THRESHOLD,
        "variance {reported} must be below {BLUR_VARIANCE_THRESHOLD}"
    );
    assert!(
        (reported - expected.laplacian_variance).abs() <= expected.laplacian_variance * 0.01,
        "variance {reported} vs {}",
        expected.laplacian_variance
    );
}

#[test]
fn the_synthetic_frame_is_excluded_as_face_not_found() {
    let outcome = outcome();

    // Task 8's pattern peaks at 0.037637 against the 0.50 gate, so the detector
    // returns an empty list and `choose_primary` returns before NMS.
    let anomaly = anomaly_for(&outcome.evaluation, 2);
    assert_eq!(anomaly.code(), AnomalyCode::FaceNotFound);
    assert_eq!(anomaly.summary(), "No face was detected");
    assert_eq!(
        anomaly.technical_detail(),
        "no detection met confidence threshold"
    );
    assert_eq!(anomaly.recovery_action(), RecoveryAction::ExcludeFrame);
}
