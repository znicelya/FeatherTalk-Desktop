use std::{fs::OpenOptions, path::Path};

use burn::{backend::NdArray, tensor::Tensor};
use feathertalk_scrfd::{ScrfdArtifactPaths, ScrfdError, ScrfdModel};
use tempfile::TempDir;

type CpuBackend = NdArray<f32>;

fn artifact_paths() -> ScrfdArtifactPaths {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("artifacts/scrfd_2_5g");
    ScrfdArtifactPaths {
        manifest: root.join("manifest.json"),
        weights: root.join("model.safetensors"),
    }
}

fn copy_artifacts() -> (TempDir, ScrfdArtifactPaths) {
    let temp = tempfile::tempdir().unwrap();
    let manifest = temp.path().join("manifest.json");
    let weights = temp.path().join("model.safetensors");
    std::fs::copy(artifact_paths().manifest, &manifest).unwrap();
    std::fs::copy(artifact_paths().weights, &weights).unwrap();
    (temp, ScrfdArtifactPaths { manifest, weights })
}

#[test]
fn committed_artifact_loads_and_exposes_the_validated_manifest() {
    let device = Default::default();
    let model = ScrfdModel::<CpuBackend>::load(&artifact_paths(), &device).unwrap();
    assert_eq!(model.manifest().schema_version, 1);
    assert_eq!(model.manifest().levels[0].anchors, 12_800);
}

#[test]
fn forward_rejects_every_non_contract_input_shape_before_graph_execution() {
    let device = Default::default();
    let model = ScrfdModel::<CpuBackend>::load(&artifact_paths(), &device).unwrap();
    for shape in [
        [2, 3, 640, 640],
        [1, 1, 640, 640],
        [1, 3, 639, 640],
        [1, 3, 640, 639],
    ] {
        let input = Tensor::<CpuBackend, 4>::zeros(shape, &device);
        assert!(matches!(
            model.forward(input),
            Err(ScrfdError::InvalidInputShape { actual }) if actual == shape
        ));
    }
}

#[test]
fn committed_model_returns_the_three_fixed_level_shapes() {
    let device = Default::default();
    let model = ScrfdModel::<CpuBackend>::load(&artifact_paths(), &device).unwrap();
    let output = model
        .forward(Tensor::<CpuBackend, 4>::zeros([1, 3, 640, 640], &device))
        .unwrap();
    for (level, stride, anchors) in output
        .levels
        .into_iter()
        .zip([8, 16, 32])
        .zip([12_800, 3_200, 800])
        .map(|((level, stride), anchors)| (level, stride, anchors))
    {
        assert_eq!(level.stride, stride);
        assert_eq!(level.scores.dims(), [1, anchors]);
        assert_eq!(level.bbox_deltas.dims(), [1, anchors, 4]);
        assert_eq!(level.keypoint_deltas.dims(), [1, anchors, 10]);
    }
}

#[test]
fn unknown_manifest_field_is_rejected() {
    let (_temp, paths) = copy_artifacts();
    let mut value: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&paths.manifest).unwrap()).unwrap();
    value["future_field"] = serde_json::json!(true);
    std::fs::write(&paths.manifest, serde_json::to_vec(&value).unwrap()).unwrap();
    let device = Default::default();
    assert!(matches!(
        ScrfdModel::<CpuBackend>::load(&paths, &device),
        Err(ScrfdError::ManifestJson(_))
    ));
}

#[test]
fn changed_weight_hash_is_rejected_as_contract_mismatch() {
    let (_temp, paths) = copy_artifacts();
    let mut value: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&paths.manifest).unwrap()).unwrap();
    value["weights"]["sha256"] = serde_json::Value::String("0".repeat(64));
    std::fs::write(&paths.manifest, serde_json::to_vec(&value).unwrap()).unwrap();
    let device = Default::default();
    assert!(matches!(
        ScrfdModel::<CpuBackend>::load(&paths, &device),
        Err(ScrfdError::ContractMismatch {
            field: "weights.sha256",
            ..
        })
    ));
}

#[test]
fn oversized_manifest_is_rejected_before_json_parsing() {
    let (_temp, paths) = copy_artifacts();
    let bytes = vec![b' '; 65_537];
    std::fs::write(&paths.manifest, bytes).unwrap();
    let device = Default::default();
    assert!(matches!(
        ScrfdModel::<CpuBackend>::load(&paths, &device),
        Err(ScrfdError::ManifestTooLarge { .. })
    ));
}

#[test]
fn oversized_weights_are_rejected_before_allocation() {
    let (_temp, paths) = copy_artifacts();
    let file = OpenOptions::new().write(true).open(&paths.weights).unwrap();
    file.set_len(16 * 1024 * 1024 + 1).unwrap();
    let device = Default::default();
    assert!(matches!(
        ScrfdModel::<CpuBackend>::load(&paths, &device),
        Err(ScrfdError::WeightsTooLarge { .. })
    ));
}

#[test]
fn changed_weight_bytes_are_rejected_by_hash() {
    let (_temp, paths) = copy_artifacts();
    let mut bytes = std::fs::read(&paths.weights).unwrap();
    let last = bytes.len() - 1;
    bytes[last] ^= 1;
    std::fs::write(&paths.weights, bytes).unwrap();
    let device = Default::default();
    assert!(matches!(
        ScrfdModel::<CpuBackend>::load(&paths, &device),
        Err(ScrfdError::HashMismatch {
            artifact: "weights",
            ..
        })
    ));
}

#[test]
fn truncated_weights_are_rejected_by_byte_count() {
    let (_temp, paths) = copy_artifacts();
    let mut bytes = std::fs::read(&paths.weights).unwrap();
    bytes.truncate(bytes.len() - 1);
    std::fs::write(&paths.weights, bytes).unwrap();
    let device = Default::default();
    assert!(matches!(
        ScrfdModel::<CpuBackend>::load(&paths, &device),
        Err(ScrfdError::WeightSizeMismatch { .. })
    ));
}

#[test]
fn missing_manifest_and_weight_paths_return_io_errors() {
    let temp = tempfile::tempdir().unwrap();
    let paths = ScrfdArtifactPaths {
        manifest: temp.path().join("missing-manifest.json"),
        weights: temp.path().join("missing-model.safetensors"),
    };
    let device = Default::default();
    assert!(matches!(
        ScrfdModel::<CpuBackend>::load(&paths, &device),
        Err(ScrfdError::Io { .. })
    ));

    let (_artifacts, paths) = copy_artifacts();
    std::fs::remove_file(&paths.weights).unwrap();
    assert!(matches!(
        ScrfdModel::<CpuBackend>::load(&paths, &device),
        Err(ScrfdError::Io { .. })
    ));
}
