mod support;

use ndarray::{ArrayD, IxDyn};
use serde_json::Value;

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

#[test]
fn the_committed_fixture_satisfies_the_contract() {
    let fixture = support::load_and_verify_fixture().unwrap();
    assert_eq!(fixture.manifest["case"], support::CASE);
    assert_eq!(fixture.manifest["generator"]["numpy_version"], "2.4.6");
    assert_eq!(fixture.manifest["generator"]["opencv_version"], "5.0.0");
    assert_eq!(fixture.manifest["files"].as_object().unwrap().len(), 15);
    assert!(support::scalar(&fixture, "laplacian_variance") > 0.0);
}

#[test]
fn every_source_array_is_a_crop_of_the_fixed_pattern() {
    let fixture = support::load_and_verify_fixture().unwrap();
    let mut checked = 0;
    for (name, dtype, shape) in support::FIXTURE_ARRAYS {
        if !name.ends_with("_src.npy") {
            continue;
        }
        assert_eq!(dtype, "uint8", "{name}");
        let array = support::read_u8_array(&fixture.root.join(name)).unwrap();
        for y in 0..shape[0] {
            for x in 0..shape[1] {
                for channel in 0..3 {
                    assert_eq!(
                        array[IxDyn(&[y, x, channel])],
                        pattern_channel(channel, x, y),
                        "{name}: y={y}, x={x}, channel={channel}"
                    );
                }
            }
        }
        checked += 1;
    }
    assert_eq!(
        checked, 7,
        "every case plus the gray source must be covered"
    );
}

#[test]
fn the_recorded_hashes_match_the_committed_bytes() {
    let fixture = support::load_and_verify_fixture().unwrap();
    for (name, _, _) in support::FIXTURE_ARRAYS {
        let bytes = std::fs::read(fixture.root.join(name)).unwrap();
        assert_eq!(
            support::sha256_bytes(&bytes),
            fixture.manifest["files"][name]["sha256"].as_str().unwrap(),
            "{name}"
        );
    }
}

#[test]
fn the_manifest_schema_rejects_unknown_fields() {
    let mut manifest = committed_manifest();
    manifest["future_field"] = Value::Bool(true);
    let error = support::verify_manifest("mutated", &manifest_bytes(&manifest)).unwrap_err();
    assert!(error.contains("expected keys"), "{error}");
}

#[test]
fn the_manifest_schema_rejects_a_missing_scalar() {
    let mut manifest = committed_manifest();
    manifest["scalars"]
        .as_object_mut()
        .unwrap()
        .remove("laplacian_variance");
    assert!(support::verify_manifest("mutated", &manifest_bytes(&manifest)).is_err());
}

#[test]
fn the_manifest_schema_rejects_a_rewritten_descriptor() {
    let mut manifest = committed_manifest();
    manifest["files"]["gray_dst.npy"]["dtype"] = Value::String("float64".to_owned());
    assert!(support::verify_manifest("mutated", &manifest_bytes(&manifest)).is_err());

    let mut manifest = committed_manifest();
    manifest["files"]["gray_dst.npy"]["shape"] = serde_json::json!([64, 65]);
    assert!(support::verify_manifest("mutated", &manifest_bytes(&manifest)).is_err());

    let mut manifest = committed_manifest();
    manifest["generator"]["opencv_version"] = Value::String("4.12.0".to_owned());
    assert!(support::verify_manifest("mutated", &manifest_bytes(&manifest)).is_err());
}

#[test]
fn the_loader_rejects_non_finite_arrays() {
    let array = ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![0.0, 1.0, f64::NAN, 3.0]).unwrap();
    let error = support::check_finite("synthetic", &array).unwrap_err();
    assert!(error.contains("flattened index 2"), "{error}");
}
