mod support;

use std::{
    path::Path,
    sync::{Arc, OnceLock},
};

use burn::tensor::{Tensor, TensorData};
use feathertalk_face::DetectionConfig;
use feathertalk_frame_adapters::{
    FrameImageCache, JpegFrameDecoder, ScrfdFaceDetector, ScrfdInput, scrfd_input,
};
use feathertalk_frame_pipeline::{
    DecodedFrame, FACE_CONFIDENCE_THRESHOLD, FaceDetection, FaceDetector, FrameDecoder,
    NMS_IOU_THRESHOLD,
};
use feathertalk_models::backend::CpuBackend;
use feathertalk_scrfd::{SCRFD_INPUT_SHAPE, ScrfdArtifactPaths, ScrfdModel};

/// The committed SCRFD artifact pair, two crates over.
fn artifact_paths() -> ScrfdArtifactPaths {
    let root =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../feathertalk-scrfd/artifacts/scrfd_2_5g");
    ScrfdArtifactPaths {
        manifest: root.join("manifest.json"),
        weights: root.join("model.safetensors"),
    }
}

/// One cache for the whole binary, so the decoder and the detector share the
/// decoded pixels exactly as they do in production.
fn shared_cache() -> Arc<FrameImageCache> {
    static CACHE: OnceLock<Arc<FrameImageCache>> = OnceLock::new();
    Arc::clone(CACHE.get_or_init(|| Arc::new(FrameImageCache::new())))
}

/// Reading 3.3 MB of weights takes longer than the forward pass, so the shared
/// detector is loaded once. `&'static` is sound because `FaceDetector` is
/// `Send + Sync`.
fn shared_detector() -> &'static ScrfdFaceDetector<CpuBackend> {
    static DETECTOR: OnceLock<ScrfdFaceDetector<CpuBackend>> = OnceLock::new();
    DETECTOR.get_or_init(|| {
        ScrfdFaceDetector::load(&artifact_paths(), Default::default(), shared_cache())
            .expect("the committed artifact loads")
    })
}

fn decoded(path: &Path) -> DecodedFrame {
    JpegFrameDecoder::new(shared_cache())
        .decode(0, path)
        .unwrap()
}

/// The fixture holds cv2's decode of the same JPEG while this path runs
/// `jpeg-decoder`. The measured gap is 0.00016 on the score and 0.06 px on the
/// box, so 0.01 and 1.0 px leave room without hiding a real regression.
fn assert_matches(actual: &FaceDetection, expected: &support::DemoDetection) {
    assert!(
        (actual.score - expected.score).abs() <= 0.01,
        "score {} vs {}",
        actual.score,
        expected.score
    );
    for (index, (got, want)) in actual.bbox.iter().zip(expected.bbox.iter()).enumerate() {
        assert!((got - want).abs() <= 1.0, "bbox[{index}] {got} vs {want}");
    }
    for (index, (got, want)) in actual
        .keypoints
        .iter()
        .zip(expected.keypoints.iter())
        .enumerate()
    {
        for axis in 0..2 {
            assert!(
                (got[axis] - want[axis]).abs() <= 1.0,
                "keypoint[{index}][{axis}] {} vs {}",
                got[axis],
                want[axis]
            );
        }
    }
}

#[test]
fn the_sharp_demo_frame_yields_one_detection_above_the_threshold() {
    let fixture = support::load_and_verify_demo_fixture().unwrap();
    let expected = support::demo_frame(&fixture, "sharp");
    let frame = decoded(&expected.path);

    let detections = shared_detector().detect(&frame).unwrap();

    assert_eq!(detections.len(), 1, "got {detections:?}");
    assert!(
        detections[0].score >= FACE_CONFIDENCE_THRESHOLD,
        "score {} must clear {FACE_CONFIDENCE_THRESHOLD}",
        detections[0].score
    );
    assert_matches(&detections[0], &expected.detection);
}

#[test]
fn the_blurred_demo_frame_still_finds_the_face() {
    let fixture = support::load_and_verify_demo_fixture().unwrap();
    let expected = support::demo_frame(&fixture, "blurred");
    let frame = decoded(&expected.path);

    let detections = shared_detector().detect(&frame).unwrap();

    // Blur costs this frame almost nothing in confidence: 0.8063 against
    // 0.8108 sharp. Task 15 relies on that to reach the blur gate with a real
    // detection behind it.
    assert_eq!(detections.len(), 1, "got {detections:?}");
    assert_matches(&detections[0], &expected.detection);
}

#[test]
fn the_synthetic_frame_yields_no_detection_at_the_production_threshold() {
    let fixture = support::load_and_verify_fixture().unwrap();
    let frame = decoded(&fixture.root.join("frame.jpg"));

    let detections = shared_detector().detect(&frame).unwrap();

    // The reference decode at 0.02 keeps 12 boxes, but its highest score is
    // 0.037637 and the gate is 0.50. This is why Task 15 expects
    // `face_not_found` for this fixture.
    assert!(detections.is_empty(), "got {detections:?}");
}

#[test]
fn the_level_maxima_match_the_demo_fixture() {
    let fixture = support::load_and_verify_demo_fixture().unwrap();
    let expected = support::demo_frame(&fixture, "sharp");
    let image = shared_cache().load(&expected.path).unwrap();
    let ScrfdInput { data, .. } = scrfd_input(&image).unwrap();

    // Deliberately not through the adapter: this pins the model output itself,
    // so a failure here means preprocessing or weights, and a failure in the
    // tests above with this one passing means postprocessing.
    let device = Default::default();
    let model = ScrfdModel::<CpuBackend>::load(&artifact_paths(), &device).unwrap();
    let input = Tensor::<CpuBackend, 4>::from_data(
        TensorData::new(data, SCRFD_INPUT_SHAPE.to_vec()),
        &device,
    );
    let output = model.forward(input).unwrap();

    for (level, tensor) in output.levels.into_iter().enumerate() {
        let scores = tensor.scores.into_data().to_vec::<f32>().unwrap();
        let maximum = scores.iter().copied().fold(f32::MIN, f32::max);
        let want = expected.level_max_scores[level];
        assert!(
            (maximum - want).abs() <= 0.01,
            "level {level} maximum {maximum} vs {want}"
        );
    }
}

#[test]
fn raising_the_confidence_threshold_rejects_the_demo_face() {
    let fixture = support::load_and_verify_demo_fixture().unwrap();
    let expected = support::demo_frame(&fixture, "sharp");
    let frame = decoded(&expected.path);

    let device = Default::default();
    let model = ScrfdModel::<CpuBackend>::load(&artifact_paths(), &device).unwrap();
    let strict = ScrfdFaceDetector::from_model(model, device, shared_cache())
        .with_detection_config(DetectionConfig {
            confidence_threshold: 0.95,
            nms_iou_threshold: NMS_IOU_THRESHOLD,
        });

    // The demo face scores 0.8108, so a 0.95 gate must drop it. Without this
    // test nothing proves `config` reaches `scrfd_detections`.
    assert!(strict.detect(&frame).unwrap().is_empty());
}
