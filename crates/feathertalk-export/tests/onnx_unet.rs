use burn_store::ModuleSnapshot;
use feathertalk_export::onnx::{
    OnnxModelContract, OnnxModelKind, OnnxModelProto, OnnxTensorContract,
    export_mobileone_unet_onnx, export_original_unet_onnx, validate_model_contract,
};
use feathertalk_models::{
    backend::CpuBackend,
    unet::{MobileOneUnetConfig, OriginalUnetConfig},
};
use prost::Message;

fn contract(kind: OnnxModelKind) -> OnnxModelContract {
    OnnxModelContract::new(
        kind,
        vec![
            OnnxTensorContract::new("input", vec![1, 6, 160, 160]),
            OnnxTensorContract::new("audio", vec![1, 16, 32, 32]),
        ],
        vec![OnnxTensorContract::new("output", vec![1, 3, 160, 160])],
    )
}

fn has_attribute(node: &feathertalk_export::onnx::OnnxNodeProto, name: &str, value: i64) -> bool {
    node.attribute.iter().any(|attribute| {
        attribute.name == name && (attribute.i == value || attribute.ints.contains(&value))
    })
}

#[test]
fn original_unet_export_contains_bn_relu_resize_concat_and_sigmoid() {
    let config = OriginalUnetConfig::parity_micro();
    let device = Default::default();
    let model = config.init::<CpuBackend>(&device);

    let bytes = export_original_unet_onnx(&model, &config).unwrap();
    validate_model_contract(&bytes, &contract(OnnxModelKind::OriginalUnet)).unwrap();
    let graph = OnnxModelProto::decode(bytes.as_slice())
        .unwrap()
        .graph
        .unwrap();
    let op_types = graph
        .node
        .iter()
        .map(|node| node.op_type.as_str())
        .collect::<Vec<_>>();
    assert!(op_types.contains(&"BatchNormalization"));
    assert!(op_types.contains(&"Relu"));
    assert!(op_types.contains(&"Resize"));
    assert!(op_types.contains(&"Concat"));
    assert!(op_types.contains(&"Sigmoid"));
    assert!(
        graph
            .node
            .iter()
            .any(|node| { node.op_type == "Conv" && has_attribute(node, "group", 4) })
    );
    assert!(graph.node.iter().any(|node| {
        node.op_type == "Resize"
            && node
                .attribute
                .iter()
                .any(|attribute| attribute.name == "coordinate_transformation_mode")
    }));
    let names = graph
        .initializer
        .iter()
        .map(|tensor| tensor.name.as_str())
        .collect::<std::collections::BTreeSet<_>>();
    for snapshot in model.collect(None, None, false) {
        assert!(names.contains(snapshot.full_path().as_str()));
    }
}

#[test]
fn mobileone_export_accepts_only_reparameterized_inference_graph() {
    let config = MobileOneUnetConfig::parity_micro();
    let device = Default::default();
    let training = config.init::<CpuBackend>(&device);
    let inference = training.reparameterize();

    let bytes = export_mobileone_unet_onnx(&inference, &config).unwrap();
    validate_model_contract(&bytes, &contract(OnnxModelKind::MobileOneUnet)).unwrap();
    let graph = OnnxModelProto::decode(bytes.as_slice())
        .unwrap()
        .graph
        .unwrap();
    assert!(graph.node.iter().any(|node| node.op_type == "Sigmoid"));
    assert!(graph.node.iter().any(|node| node.op_type == "Concat"));
    assert!(graph.node.iter().any(|node| node.op_type == "Resize"));
    assert!(
        graph
            .node
            .iter()
            .all(|node| node.op_type != "BatchNormalization")
    );
    assert!(graph.node.iter().all(|node| !node.name.contains("branches")
        && !node.name.contains("scale")
        && !node.name.contains("skip")));
    assert!(
        graph
            .node
            .iter()
            .filter(|node| node.op_type == "Conv")
            .all(|node| node.input.len() == 3)
    );
}
