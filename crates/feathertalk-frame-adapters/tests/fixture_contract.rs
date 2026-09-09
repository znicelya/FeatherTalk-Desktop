mod support;

use std::path::Path;

use ndarray::IxDyn;
use serde_json::Value;
use tempfile::TempDir;

/// One channel of the generator's `bgr_u8_channel_affine_v1` pattern.
fn pattern_channel(channel: usize, x: usize, y: usize) -> u8 {
    let value = match channel {
        0 => 3 * x + 5 * y + 17,
        1 => 7 * x + 11 * y + 29,
        2 => 13 * x + 17 * y + 43,
        _ => unreachable!(),
    };
    (value % 256) as u8
}

fn committed_manifest() -> Value {
    let bytes = std::fs::read(support::fixture_dir().join("fixture.json")).unwrap();
    serde_json::from_slice(&bytes).unwrap()
}

fn manifest_bytes(manifest: &Value) -> Vec<u8> {
    serde_json::to_vec(manifest).unwrap()
}

/// Copy the committed fixture into a scratch directory so a test can corrupt it.
fn copy_fixture(destination: &Path) {
    std::fs::create_dir_all(destination).unwrap();
    for entry in std::fs::read_dir(support::fixture_dir()).unwrap() {
        let entry = entry.unwrap();
        std::fs::copy(entry.path(), destination.join(entry.file_name())).unwrap();
    }
}

#[test]
fn the_committed_fixture_loads_and_verifies() {
    let fixture = support::load_and_verify_fixture().unwrap();
    assert_eq!(fixture.manifest["case"], support::CASE);
    assert_eq!(fixture.manifest["generator"]["opencv_version"], "5.0.0");
    assert_eq!(fixture.manifest["generator"]["numpy_version"], "2.4.6");
    assert_eq!(fixture.manifest["generator"]["torch_version"], "2.13.0");
    assert_eq!(fixture.manifest["arrays"].as_object().unwrap().len(), 3);
    assert_eq!(fixture.manifest["blobs"].as_object().unwrap().len(), 3);
    assert_eq!(fixture.manifest["jpeg"]["quality"], 90);
}

#[test]
fn the_committed_manifest_is_lf_terminated_and_sorted() {
    let bytes = std::fs::read(support::fixture_dir().join("fixture.json")).unwrap();
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
}

#[test]
fn the_pattern_helper_matches_the_manifest_formula() {
    let fixture = support::load_and_verify_fixture().unwrap();
    assert_eq!(fixture.manifest["source"]["width"], 640);
    assert_eq!(fixture.manifest["source"]["height"], 640);
    assert_eq!(
        fixture.manifest["source"]["pattern"],
        "bgr_u8_channel_affine_v1"
    );

    let image = support::pattern_bgr(1280, 720);
    assert_eq!(image.width(), 1280);
    assert_eq!(image.height(), 720);
    for (x, y) in [(0, 0), (1, 0), (0, 1), (639, 359), (1279, 719)] {
        for channel in 0..3 {
            let offset = (y * 1280 + x) * 3 + channel;
            assert_eq!(
                image.as_bytes()[offset],
                pattern_channel(channel, x, y),
                "x={x}, y={y}, channel={channel}"
            );
        }
    }
}

#[test]
fn the_reference_scrfd_fixture_is_present_and_pinned() {
    let fixture = support::load_and_verify_fixture().unwrap();
    let root = support::reference_fixture_dir();
    assert!(
        root.is_dir(),
        "the feathertalk-scrfd fixture must sit at {}",
        root.display()
    );
    let recorded = fixture.manifest["reference_fixture"]["files"]
        .as_object()
        .unwrap();
    assert_eq!(recorded.len(), support::REFERENCE_ARRAYS.len());
    for (name, shape) in support::REFERENCE_ARRAYS {
        let bytes = std::fs::read(root.join(name)).unwrap();
        assert_eq!(
            support::sha256_bytes(&bytes),
            recorded[name]["sha256"].as_str().unwrap(),
            "{name}"
        );
        assert_eq!(
            bytes.len() as u64,
            recorded[name]["bytes"].as_u64().unwrap()
        );
        let array = support::read_reference_array(name);
        assert_eq!(array.shape(), shape, "{name}");
    }
    assert_eq!(
        recorded["input.npy"]["sha256"].as_str().unwrap(),
        support::REFERENCE_INPUT_SHA256
    );
}

#[test]
fn the_recorded_hashes_match_the_committed_bytes() {
    let fixture = support::load_and_verify_fixture().unwrap();
    for (name, _, _) in support::FIXTURE_ARRAYS {
        let bytes = std::fs::read(fixture.root.join(name)).unwrap();
        assert_eq!(
            support::sha256_bytes(&bytes),
            fixture.manifest["arrays"][name]["sha256"].as_str().unwrap(),
            "{name}"
        );
    }
    for name in support::FIXTURE_BLOBS {
        let bytes = std::fs::read(fixture.root.join(name)).unwrap();
        assert_eq!(
            support::sha256_bytes(&bytes),
            fixture.manifest["blobs"][name]["sha256"].as_str().unwrap(),
            "{name}"
        );
    }
}

#[test]
fn a_missing_manifest_field_is_rejected() {
    let mut manifest = committed_manifest();
    manifest.as_object_mut().unwrap().remove("crops");
    let error = support::verify_manifest("mutated", &manifest_bytes(&manifest)).unwrap_err();
    assert!(error.contains("expected keys"), "{error}");

    let mut manifest = committed_manifest();
    manifest["arrays"]["crop_blob.npy"]["dtype"] = Value::String("uint8".to_owned());
    assert!(support::verify_manifest("mutated", &manifest_bytes(&manifest)).is_err());

    let mut manifest = committed_manifest();
    manifest["generator"]["opencv_version"] = Value::String("4.12.0".to_owned());
    assert!(support::verify_manifest("mutated", &manifest_bytes(&manifest)).is_err());
}

#[test]
fn a_corrupted_array_is_rejected() {
    let scratch = TempDir::new().unwrap();
    let root = scratch.path().join(support::CASE);
    copy_fixture(&root);

    let target = root.join("crop_blob.npy");
    let mut bytes = std::fs::read(&target).unwrap();
    let last = bytes.len() - 1;
    bytes[last] ^= 0x01;
    std::fs::write(&target, &bytes).unwrap();

    let error = support::load_and_verify_fixture_at(&root).unwrap_err();
    assert!(error.contains("SHA-256"), "{error}");
}

#[test]
fn an_unexpected_extra_file_is_rejected() {
    let scratch = TempDir::new().unwrap();
    let root = scratch.path().join(support::CASE);
    copy_fixture(&root);
    std::fs::write(root.join("stray.npy"), b"stray").unwrap();

    let error = support::load_and_verify_fixture_at(&root).unwrap_err();
    assert!(error.contains("expected files"), "{error}");
}

#[test]
fn the_expected_values_parse_into_typed_records() {
    let fixture = support::load_and_verify_fixture().unwrap();

    let detections = support::expected_detections(&fixture);
    assert_eq!(detections.len(), 12);
    for window in detections.windows(2) {
        assert!(
            window[0].score >= window[1].score,
            "detections must be score-descending"
        );
    }
    for detection in &detections {
        assert!(detection.score >= 0.02);
        assert!(detection.bbox[2] > 0.0 && detection.bbox[3] > 0.0);
    }

    let landmarks = support::expected_landmarks(&fixture);
    assert_eq!(landmarks.points.len(), 110);
    assert_eq!(landmarks.size, 210);
    assert_eq!((landmarks.origin_x, landmarks.origin_y), (195, 175));

    let in_bounds = support::crop_case(&fixture, "in_bounds");
    assert_eq!(in_bounds.size, 210);
    assert_eq!(in_bounds.padding, [0, 0, 0, 0]);
    let padded = support::crop_case(&fixture, "padded");
    assert_eq!(padded.size, 945);
    assert_eq!((padded.origin_x, padded.origin_y), (-122, -122));
    assert_eq!(padded.padding, [122, 122, 183, 183]);

    let pin = support::letterbox_pin(&fixture);
    assert_eq!(pin.shape, vec![1, 3, 640, 640]);
    assert_eq!((pin.source_width, pin.source_height), (1280, 720));
    assert_eq!((pin.new_width, pin.new_height), (640, 361));
    assert_eq!((pin.pad_x, pin.pad_y), (0, 139));
    assert_eq!(pin.samples.len(), 8);

    let array = support::read_reference_array("out0.npy");
    assert_eq!(array.shape(), &[1, 12800, 1]);
    assert_eq!(
        fixture.manifest["scalars"]["level_max_scores"]
            .as_array()
            .unwrap()
            .len(),
        3
    );
    let _ = IxDyn(&[0, 0, 0]);
}
