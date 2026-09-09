use feathertalk_export::onnx::{
    ONNX_OPSET_VERSION, OnnxModel, OnnxModelContract, OnnxModelKind, OnnxTensorContract,
    OnnxValidationError, serialize_model, validate_model_contract,
};

fn contract(kind: OnnxModelKind) -> OnnxModelContract {
    OnnxModelContract::new(
        kind,
        match kind {
            OnnxModelKind::FeatherHubert => vec![OnnxTensorContract::new("waveform", vec![1, -1])],
            OnnxModelKind::OriginalUnet | OnnxModelKind::MobileOneUnet => vec![
                OnnxTensorContract::new("input", vec![1, 6, 160, 160]),
                OnnxTensorContract::new("audio", vec![1, 16, 32, 32]),
            ],
        },
        match kind {
            OnnxModelKind::FeatherHubert => {
                vec![OnnxTensorContract::new("hidden", vec![1, -1, 1024])]
            }
            OnnxModelKind::OriginalUnet | OnnxModelKind::MobileOneUnet => {
                vec![OnnxTensorContract::new("output", vec![1, 3, 160, 160])]
            }
        },
    )
}

#[test]
fn valid_feather_hubert_model_has_the_fixed_public_contract() {
    let expected = contract(OnnxModelKind::FeatherHubert);
    let bytes = serialize_model(&OnnxModel::new(expected.kind)).unwrap();

    validate_model_contract(&bytes, &expected).unwrap();
    assert_eq!(ONNX_OPSET_VERSION, 17);
}

#[test]
fn valid_unet_models_have_two_inputs_and_one_output() {
    for kind in [OnnxModelKind::OriginalUnet, OnnxModelKind::MobileOneUnet] {
        let expected = contract(kind);
        let bytes = serialize_model(&OnnxModel::new(kind)).unwrap();
        validate_model_contract(&bytes, &expected).unwrap();
    }
}

#[test]
fn validator_rejects_wrong_opset() {
    let expected = contract(OnnxModelKind::FeatherHubert);
    let mut model = OnnxModel::new(expected.kind);
    model.opset_version = 16;
    let bytes = serialize_model(&model).unwrap();

    assert!(matches!(
        validate_model_contract(&bytes, &expected),
        Err(OnnxValidationError::Opset { .. })
    ));
}

#[test]
fn validator_rejects_wrong_names_and_dtype() {
    let expected = contract(OnnxModelKind::FeatherHubert);
    let mut model = OnnxModel::new(expected.kind);
    model.graph.inputs[0].name = "samples".to_owned();
    let bytes = serialize_model(&model).unwrap();
    assert!(matches!(
        validate_model_contract(&bytes, &expected),
        Err(OnnxValidationError::TensorName { .. })
    ));

    let mut model = OnnxModel::new(expected.kind);
    model.graph.inputs[0].dtype = 10;
    let bytes = serialize_model(&model).unwrap();
    assert!(matches!(
        validate_model_contract(&bytes, &expected),
        Err(OnnxValidationError::DType { .. })
    ));
}

#[test]
fn validator_rejects_missing_graph_and_forbidden_symbolic_dimensions() {
    let expected = contract(OnnxModelKind::FeatherHubert);
    let mut model = OnnxModel::new(expected.kind);
    model.graph_present = false;
    let bytes = serialize_model(&model).unwrap();
    assert!(matches!(
        validate_model_contract(&bytes, &expected),
        Err(OnnxValidationError::MissingGraph)
    ));

    let mut model = OnnxModel::new(expected.kind);
    model.graph.inputs[0].shape[0] = -1;
    let bytes = serialize_model(&model).unwrap();
    assert!(matches!(
        validate_model_contract(&bytes, &expected),
        Err(OnnxValidationError::Dimension { .. })
    ));
}
