mod support;

use feathertalk_face::{DetectionConfig, ImageSize, ResizeTransform, resize_with_padding};
use feathertalk_frame_adapters::{LevelHostData, scrfd_detections};
use feathertalk_frame_pipeline::PipelineError;
use feathertalk_scrfd::{SCRFD_ANCHORS, SCRFD_STRIDES};
use serde_json::Value;

/// The committed level tensors were decoded without a letterbox, so the
/// transform that reproduces them is the identity 640x640 one.
fn identity_transform() -> ResizeTransform {
    resize_with_padding(ImageSize {
        width: 640,
        height: 640,
    })
    .unwrap()
}

/// The configuration `detections_thr002.json` was produced with.
fn reference_config() -> DetectionConfig {
    DetectionConfig {
        confidence_threshold: 0.02,
        nms_iou_threshold: 0.40,
    }
}

fn reference_document(fixture: &support::VerifiedFixture) -> Value {
    let bytes = std::fs::read(fixture.root.join("detections_thr002.json")).unwrap();
    serde_json::from_slice(&bytes).unwrap()
}

fn reference_levels() -> [LevelHostData; 3] {
    let mut levels = Vec::with_capacity(3);
    for level in 0..3 {
        let anchors = SCRFD_ANCHORS[level];
        let scores =
            support::flatten_f32(&support::read_reference_array(&format!("out{level}.npy")));
        let boxes = support::flatten_f32(&support::read_reference_array(&format!(
            "out{}.npy",
            level + 3
        )));
        let keypoints = support::flatten_f32(&support::read_reference_array(&format!(
            "out{}.npy",
            level + 6
        )));
        assert_eq!(scores.len(), anchors, "level {level} scores");
        assert_eq!(boxes.len(), anchors * 4, "level {level} boxes");
        assert_eq!(keypoints.len(), anchors * 10, "level {level} keypoints");
        levels.push(LevelHostData {
            level,
            stride: SCRFD_STRIDES[level],
            scores,
            bbox_distances: boxes
                .chunks_exact(4)
                .map(|chunk| chunk.try_into().unwrap())
                .collect(),
            keypoint_distances: keypoints
                .chunks_exact(10)
                .map(|chunk| chunk.try_into().unwrap())
                .collect(),
        });
    }
    levels.try_into().expect("three levels")
}

/// Correctly shaped levels that decode to nothing: every score is below any
/// sane threshold, so individual anchors can be switched on one at a time.
fn zero_levels() -> [LevelHostData; 3] {
    let mut levels = Vec::with_capacity(3);
    for level in 0..3 {
        let anchors = SCRFD_ANCHORS[level];
        levels.push(LevelHostData {
            level,
            stride: SCRFD_STRIDES[level],
            scores: vec![0.0; anchors],
            bbox_distances: vec![[0.0; 4]; anchors],
            keypoint_distances: vec![[0.0; 10]; anchors],
        });
    }
    levels.try_into().expect("three levels")
}

#[test]
fn the_reference_levels_reproduce_the_pinned_detections() {
    let fixture = support::load_and_verify_fixture().unwrap();
    let document = reference_document(&fixture);
    assert_eq!(document["candidate_count"], 17);
    assert_eq!(document["degenerate_count"], 33);
    assert_eq!(
        document["confidence_threshold"].as_f64().unwrap() as f32,
        reference_config().confidence_threshold
    );
    assert_eq!(
        document["nms_iou_threshold"].as_f64().unwrap() as f32,
        reference_config().nms_iou_threshold
    );

    let expected = support::expected_detections(&fixture);
    assert_eq!(expected.len(), 12, "the fixture pins twelve survivors");

    let levels = reference_levels();
    let detections = scrfd_detections(&levels, &identity_transform(), &reference_config()).unwrap();
    assert_eq!(detections.len(), expected.len());

    for (index, (actual, wanted)) in detections.iter().zip(&expected).enumerate() {
        // Exact, not toleranced: the identity transform makes the Rust and
        // NumPy paths the same sequence of f32 operations.
        assert_eq!(actual.score, wanted.score, "score at {index}");
        assert_eq!(actual.bbox, wanted.bbox, "bbox at {index}");
        assert_eq!(actual.keypoints, wanted.keypoints, "keypoints at {index}");
    }

    for pair in detections.windows(2) {
        assert!(
            pair[0].score >= pair[1].score,
            "NMS must return detections score descending"
        );
    }
}

#[test]
fn the_production_threshold_finds_nothing_in_the_synthetic_pattern() {
    let fixture = support::load_and_verify_fixture().unwrap();
    let document = reference_document(&fixture);
    let maxima = document["level_max_scores"].as_array().unwrap();
    assert_eq!(maxima.len(), 3);
    for (level, value) in maxima.iter().enumerate() {
        assert!(
            value.as_f64().unwrap() < 0.05,
            "level {level} maximum is {value}"
        );
    }

    let levels = reference_levels();
    let config = DetectionConfig {
        confidence_threshold: 0.50,
        nms_iou_threshold: 0.40,
    };
    let detections = scrfd_detections(&levels, &identity_transform(), &config).unwrap();
    assert!(
        detections.is_empty(),
        "expected no survivor, got {detections:?}"
    );
}

#[test]
fn a_short_score_vector_is_reported_with_its_level() {
    let mut levels = zero_levels();
    levels[0].scores.truncate(SCRFD_ANCHORS[0] - 1);
    let error = scrfd_detections(&levels, &identity_transform(), &reference_config()).unwrap_err();
    match error {
        PipelineError::Adapter { component, message } => {
            assert_eq!(component, "scrfd");
            assert_eq!(
                message,
                "level 0 scores holds 12799 entries, expected 12800"
            );
        }
        other => panic!("expected an adapter error, got {other}"),
    }
}

#[test]
fn a_degenerate_anchor_is_dropped_instead_of_failing_the_frame() {
    let mut levels = zero_levels();
    // Anchors 0 and 1 of the stride 32 level share the centre (0, 0). The
    // first keeps zero distances, so its box collapses to a point and
    // `decode_level` rejects it; the second decodes to a 32x32 box.
    levels[2].scores[0] = 0.90;
    levels[2].scores[1] = 0.80;
    levels[2].bbox_distances[1] = [1.0, 1.0, 1.0, 1.0];

    let detections = scrfd_detections(&levels, &identity_transform(), &reference_config()).unwrap();
    assert_eq!(detections.len(), 1, "only the second anchor survives");
    assert_eq!(detections[0].score, 0.80);
    assert_eq!(detections[0].bbox, [0.0, 0.0, 32.0, 32.0]);
    assert_eq!(detections[0].keypoints, [[0.0, 0.0]; 5]);
}

#[test]
fn a_non_finite_distance_names_the_real_anchor() {
    let mut levels = zero_levels();
    levels[1].scores[7] = 0.90;
    levels[1].bbox_distances[7] = [f32::NAN, 0.0, 1.0, 1.0];
    let error = scrfd_detections(&levels, &identity_transform(), &reference_config()).unwrap_err();
    match error {
        PipelineError::Adapter { component, message } => {
            assert_eq!(component, "scrfd");
            // `decode_level` only sees a one-anchor slice, so its own index is
            // 0; the wrapper supplies the anchor that actually failed.
            assert_eq!(
                message,
                "level 1 anchor 7: non-finite value at level 1 for bbox_distances, index 0"
            );
        }
        other => panic!("expected an adapter error, got {other}"),
    }
}
