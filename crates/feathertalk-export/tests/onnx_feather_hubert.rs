use burn_store::ModuleSnapshot;
use feathertalk_export::onnx::{
    ONNX_FLOAT_DATA_TYPE, OnnxModelContract, OnnxModelKind, OnnxModelProto, OnnxTensorContract,
    export_feather_hubert_onnx, validate_model_contract,
};
use feathertalk_models::feather_hubert::{FeatherHubertConfig, FeatherHubertEncoder};
use prost::Message;

use feathertalk_models::backend::CpuBackend;

fn contract() -> OnnxModelContract {
    OnnxModelContract::new(
        OnnxModelKind::FeatherHubert,
        vec![OnnxTensorContract::new("waveform", vec![1, -1])],
        vec![OnnxTensorContract::new("hidden", vec![1, -1, 1024])],
    )
}

#[test]
fn feather_hubert_export_contains_inference_graph_and_all_weights() {
    let config = FeatherHubertConfig {
        channels: 32,
        expansion: 2,
        num_blocks: 1,
        output_dim: 1024,
        dropout: 0.0,
    };
    let device = Default::default();
    let model: FeatherHubertEncoder<CpuBackend> = config.init(&device);

    let bytes = export_feather_hubert_onnx(&model, &config).unwrap();
    validate_model_contract(&bytes, &contract()).unwrap();
    let proto = OnnxModelProto::decode(bytes.as_slice()).unwrap();
    let graph = proto.graph.unwrap();

    assert_eq!(graph.input[0].name, "waveform");
    assert_eq!(graph.output[0].name, "hidden");
    assert!(
        graph
            .initializer
            .iter()
            .all(|tensor| { tensor.data_type == ONNX_FLOAT_DATA_TYPE || tensor.data_type == 7 })
    );

    let op_types = graph
        .node
        .iter()
        .map(|node| node.op_type.as_str())
        .collect::<Vec<_>>();
    assert!(op_types.iter().filter(|op| **op == "Conv").count() >= 11);
    assert!(
        op_types
            .iter()
            .filter(|op| **op == "InstanceNormalization")
            .count()
            >= 9
    );
    assert!(op_types.contains(&"Erf"));
    assert!(op_types.contains(&"Add"));
    assert!(!op_types.contains(&"Dropout"));

    let initializer_names = graph
        .initializer
        .iter()
        .map(|tensor| tensor.name.as_str())
        .collect::<std::collections::BTreeSet<_>>();
    for snapshot in model.collect(None, None, false) {
        assert!(initializer_names.contains(snapshot.full_path().as_str()));
    }
}

#[test]
fn tcn_depthwise_convolutions_preserve_residual_length_with_symmetric_padding() {
    let config = FeatherHubertConfig {
        channels: 32,
        expansion: 2,
        num_blocks: 4,
        output_dim: 1024,
        dropout: 0.0,
    };
    let device = Default::default();
    let model: FeatherHubertEncoder<CpuBackend> = config.init(&device);

    let bytes = export_feather_hubert_onnx(&model, &config).unwrap();
    let proto = OnnxModelProto::decode(bytes.as_slice()).unwrap();
    let graph = proto.graph.unwrap();

    for (index, expected_padding) in [2_i64, 4, 8, 16].into_iter().enumerate() {
        let node_name = format!("encoder.{index}.dw_conv");
        let node = graph
            .node
            .iter()
            .find(|node| node.name == node_name)
            .unwrap();
        let pads = node
            .attribute
            .iter()
            .find(|attribute| attribute.name == "pads")
            .unwrap();
        assert_eq!(pads.ints, [expected_padding, expected_padding]);
    }
}

#[test]
fn group_norm_reshapes_by_group_before_instance_normalization() {
    let config = FeatherHubertConfig {
        channels: 32,
        expansion: 2,
        num_blocks: 1,
        output_dim: 1024,
        dropout: 0.0,
    };
    let device = Default::default();
    let model: FeatherHubertEncoder<CpuBackend> = config.init(&device);

    let bytes = export_feather_hubert_onnx(&model, &config).unwrap();
    let proto = OnnxModelProto::decode(bytes.as_slice()).unwrap();
    let graph = proto.graph.unwrap();

    let shape_in = graph
        .initializer
        .iter()
        .find(|tensor| tensor.name == "encoder.0.norm.shape.in")
        .unwrap();
    assert_eq!(decode_i64(shape_in), [0, 32, -1]);
    let shape_out = graph
        .initializer
        .iter()
        .find(|tensor| tensor.name == "encoder.0.norm.shape.out")
        .unwrap();
    assert_eq!(decode_i64(shape_out), [0, 32, -1]);
    let instance_scale = graph
        .initializer
        .iter()
        .find(|tensor| tensor.name == "encoder.0.norm.instance_scale")
        .unwrap();
    assert_eq!(instance_scale.dims, [32]);
}

fn decode_i64(tensor: &feathertalk_export::onnx::OnnxTensorProto) -> Vec<i64> {
    tensor
        .raw_data
        .chunks_exact(8)
        .map(|bytes| i64::from_le_bytes(bytes.try_into().unwrap()))
        .collect()
}
