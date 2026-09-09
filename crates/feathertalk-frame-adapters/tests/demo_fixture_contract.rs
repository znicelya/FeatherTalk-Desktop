mod support;

use feathertalk_face::{ImageSize, compute_face_crop_geometry};
use feathertalk_frame_pipeline::{
    BLUR_VARIANCE_THRESHOLD, FACE_CONFIDENCE_THRESHOLD, NMS_IOU_THRESHOLD,
};
use serde_json::Value;

fn committed_manifest() -> Value {
    let bytes = std::fs::read(support::demo_fixture_dir().join("fixture.json")).unwrap();
    serde_json::from_slice(&bytes).unwrap()
}

fn manifest_bytes(manifest: &Value) -> Vec<u8> {
    serde_json::to_vec(manifest).unwrap()
}

#[test]
fn the_committed_demo_fixture_loads_and_verifies() {
    let fixture = support::load_and_verify_demo_fixture().unwrap();
    assert_eq!(fixture.manifest["case"], support::DEMO_CASE);
    assert_eq!(
        fixture.manifest["source"]["frame_index"],
        support::DEMO_FRAME_INDEX
    );
    assert_eq!(fixture.manifest["source"]["width"], 1280);
    assert_eq!(fixture.manifest["source"]["height"], 720);
    assert_eq!(fixture.manifest["source"]["frame_count"], 1511);
    assert_eq!(fixture.manifest["blobs"].as_object().unwrap().len(), 2);
    assert_eq!(fixture.manifest["frames"].as_object().unwrap().len(), 2);
    assert_eq!(fixture.manifest["jpeg"]["quality"], 90);
}

#[test]
fn the_committed_demo_manifest_is_lf_terminated_and_sorted() {
    let bytes = std::fs::read(support::demo_fixture_dir().join("fixture.json")).unwrap();
    assert!(
        bytes.ends_with(b"\n"),
        "the manifest must end with a newline"
    );
    assert!(!bytes.contains(&b'\r'), "the manifest must use LF endings");
    let manifest: Value = serde_json::from_slice(&bytes).unwrap();
    let keys = manifest
        .as_object()
        .unwrap()
        .keys()
        .cloned()
        .collect::<Vec<_>>();
    let mut sorted = keys.clone();
    sorted.sort();
    assert_eq!(keys, sorted, "the generator writes with sort_keys=True");
    assert_eq!(keys.len(), 9, "the manifest holds nine top level keys");
}

#[test]
fn the_pinned_detection_config_matches_the_pipeline_constants() {
    let fixture = support::load_and_verify_demo_fixture().unwrap();
    let config = &fixture.manifest["detection_config"];
    // The constants are f32 and 0.4_f32 widened to f64 is not 0.4_f64, so the
    // comparison has to happen in f32.
    assert_eq!(
        config["confidence_threshold"].as_f64().unwrap() as f32,
        FACE_CONFIDENCE_THRESHOLD
    );
    assert_eq!(
        config["nms_iou_threshold"].as_f64().unwrap() as f32,
        NMS_IOU_THRESHOLD
    );
}

#[test]
fn the_recorded_hashes_match_the_committed_jpegs() {
    let fixture = support::load_and_verify_demo_fixture().unwrap();
    for name in support::DEMO_BLOBS {
        let bytes = std::fs::read(fixture.root.join(name)).unwrap();
        let descriptor = &fixture.manifest["blobs"][name];
        assert_eq!(descriptor["bytes"], bytes.len() as u64, "{name}");
        assert_eq!(
            descriptor["sha256"],
            support::sha256_bytes(&bytes),
            "{name}"
        );
        assert!(
            bytes.starts_with(&[0xFF, 0xD8, 0xFF]),
            "{name} does not start with an SOI marker"
        );
    }
}

#[test]
fn the_two_frames_straddle_the_blur_threshold() {
    let fixture = support::load_and_verify_demo_fixture().unwrap();
    let sharp = support::demo_frame(&fixture, "sharp");
    let blurred = support::demo_frame(&fixture, "blurred");
    assert!(
        sharp.laplacian_variance > BLUR_VARIANCE_THRESHOLD,
        "sharp variance {} must clear {BLUR_VARIANCE_THRESHOLD}",
        sharp.laplacian_variance
    );
    assert!(
        blurred.laplacian_variance < BLUR_VARIANCE_THRESHOLD,
        "blurred variance {} must fall below {BLUR_VARIANCE_THRESHOLD}",
        blurred.laplacian_variance
    );
    assert!(sharp.path.ends_with("frame.jpg"));
    assert!(blurred.path.ends_with("frame_blurred.jpg"));
}

#[test]
fn both_detections_clear_the_confidence_threshold() {
    let fixture = support::load_and_verify_demo_fixture().unwrap();
    for name in support::DEMO_FRAMES {
        let frame = support::demo_frame(&fixture, name);
        assert!(
            frame.detection.score >= FACE_CONFIDENCE_THRESHOLD,
            "{name} score {} must clear {FACE_CONFIDENCE_THRESHOLD}",
            frame.detection.score
        );
        let level_max = frame
            .level_max_scores
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        assert!(
            (level_max - frame.detection.score).abs() <= 1e-6,
            "{name}: level maximum {level_max} must be the surviving score {}",
            frame.detection.score
        );
    }
}

#[test]
fn the_pinned_crops_match_compute_face_crop_geometry() {
    let fixture = support::load_and_verify_demo_fixture().unwrap();
    let image = ImageSize {
        width: 1280,
        height: 720,
    };
    for name in support::DEMO_FRAMES {
        let frame = support::demo_frame(&fixture, name);
        let geometry = compute_face_crop_geometry(image, frame.detection.bbox).unwrap();
        assert_eq!(geometry.size, frame.crop.size, "{name} size");
        assert_eq!(geometry.origin_x, frame.crop.origin_x, "{name} origin_x");
        assert_eq!(geometry.origin_y, frame.crop.origin_y, "{name} origin_y");
        assert_eq!(
            [
                geometry.padding.left,
                geometry.padding.top,
                geometry.padding.right,
                geometry.padding.bottom,
            ],
            frame.crop.padding,
            "{name} padding"
        );
        assert_eq!(
            [
                i64::from(geometry.source.x),
                i64::from(geometry.source.y),
                i64::from(geometry.source.width),
                i64::from(geometry.source.height),
            ],
            frame.crop.source,
            "{name} source"
        );

        // PFLD's normalized output is not bounded to 0..1, so a landmark may sit
        // slightly outside the crop square. Ten percent of the edge is enough
        // slack to catch a decode that landed on the wrong face.
        let size = i32::try_from(frame.crop.size).unwrap();
        let slack = size / 10;
        let x_range = (frame.crop.origin_x - slack)..=(frame.crop.origin_x + size + slack);
        let y_range = (frame.crop.origin_y - slack)..=(frame.crop.origin_y + size + slack);
        assert_eq!(
            frame.landmarks.len(),
            support::PFLD_LANDMARK_COUNT,
            "{name}"
        );
        for (index, [x, y]) in frame.landmarks.iter().copied().enumerate() {
            assert!(
                x_range.contains(&x),
                "{name} landmark {index} x={x} outside {x_range:?}"
            );
            assert!(
                y_range.contains(&y),
                "{name} landmark {index} y={y} outside {y_range:?}"
            );
        }
    }
}

#[test]
fn the_committed_demo_video_still_matches_its_hash() {
    let path = support::demo_video_path();
    let bytes = std::fs::read(&path).unwrap_or_else(|error| panic!("{}: {error}", path.display()));
    assert_eq!(bytes.len(), 7_442_868);
    assert_eq!(support::sha256_bytes(&bytes), support::DEMO_VIDEO_SHA256);
}

#[test]
fn a_mutated_demo_manifest_is_rejected() {
    let mut manifest = committed_manifest();
    manifest.as_object_mut().unwrap().remove("frames");
    let error = support::verify_demo_manifest("mutated", &manifest_bytes(&manifest)).unwrap_err();
    assert!(error.contains("expected keys"), "{error}");

    let mut manifest = committed_manifest();
    manifest["source"]["sha256"] = Value::String("0".repeat(64));
    assert!(support::verify_demo_manifest("mutated", &manifest_bytes(&manifest)).is_err());

    let mut manifest = committed_manifest();
    manifest["blur"]["kernel"] = Value::from(18);
    assert!(support::verify_demo_manifest("mutated", &manifest_bytes(&manifest)).is_err());
}
