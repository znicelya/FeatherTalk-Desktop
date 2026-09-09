mod support;

fn source_channel(channel: usize, x: usize, y: usize) -> u8 {
    let value = match channel {
        0 => 3 * x + 5 * y + 17,
        1 => 7 * x + 11 * y + 29,
        2 => 13 * x + 17 * y + 43,
        _ => unreachable!(),
    };
    (value % 256) as u8
}

fn expected_nchw(channel: usize, x: usize, y: usize) -> f32 {
    let bgr_channel = [2, 1, 0][channel];
    (f32::from(source_channel(bgr_channel, x, y)) - 127.5) / 128.0
}

#[test]
fn committed_opencv_fixture_has_the_fixed_contract_and_input() {
    let fixture = support::load_and_verify_fixture().unwrap();
    assert_eq!(fixture.manifest.schema_version, 1);
    assert_eq!(fixture.manifest.generator.python_version, "3.11");
    assert_eq!(fixture.manifest.generator.numpy_version, "2.2.6");
    assert_eq!(fixture.manifest.generator.opencv_version, "4.12.0");

    let input = support::read_array(&fixture.root.join("input.npy")).unwrap();
    assert_eq!(input.shape(), &[1, 3, 640, 640]);
    for channel in 0..3 {
        for y in 0..640 {
            for x in 0..640 {
                assert_eq!(
                    input[ndarray::IxDyn(&[0, channel, y, x])],
                    expected_nchw(channel, x, y),
                    "channel={channel}, x={x}, y={y}",
                );
            }
        }
    }
}

#[test]
fn fixture_schema_rejects_unknown_fields() {
    let path = support::fixture_dir().join("fixture.json");
    let mut value: serde_json::Value =
        serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    value["future_field"] = serde_json::json!(true);
    assert!(serde_json::from_value::<support::FixtureManifest>(value).is_err());
}

#[test]
fn fixture_loader_rejects_corrupt_metadata_and_non_finite_arrays() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path();
    std::fs::create_dir_all(root).unwrap();
    std::fs::copy(
        support::fixture_dir().join("fixture.json"),
        root.join("fixture.json"),
    )
    .unwrap();
    for name in [
        "input.npy",
        "out0.npy",
        "out1.npy",
        "out2.npy",
        "out3.npy",
        "out4.npy",
        "out5.npy",
        "out6.npy",
        "out7.npy",
        "out8.npy",
    ] {
        std::fs::copy(support::fixture_dir().join(name), root.join(name)).unwrap();
    }
    let mut bytes = std::fs::read(root.join("out0.npy")).unwrap();
    let payload_start = bytes.len() - 4 * 12_800;
    bytes[payload_start..payload_start + 4].copy_from_slice(&f32::NAN.to_le_bytes());
    std::fs::write(root.join("out0.npy"), bytes).unwrap();
    let mut json: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root.join("fixture.json")).unwrap()).unwrap();
    let bytes = std::fs::read(root.join("out0.npy")).unwrap();
    json["files"]["out0.npy"]["bytes"] = serde_json::json!(bytes.len());
    json["files"]["out0.npy"]["sha256"] = serde_json::Value::String(support::sha256_bytes(&bytes));
    std::fs::write(
        root.join("fixture.json"),
        serde_json::to_vec(&json).unwrap(),
    )
    .unwrap();
    assert!(support::load_and_verify_fixture_at(root).is_err());

    let mut json: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root.join("fixture.json")).unwrap()).unwrap();
    json["files"]["out0.npy"]["sha256"] = serde_json::Value::String("0".repeat(64));
    std::fs::write(
        root.join("fixture.json"),
        serde_json::to_vec(&json).unwrap(),
    )
    .unwrap();
    assert!(support::load_and_verify_fixture_at(root).is_err());
}
