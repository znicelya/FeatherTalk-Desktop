use burn::tensor::Tensor;
use feathertalk_models::{PFLD_GhostOne, PFLD_INPUT_CHANNELS, PFLD_OUTPUT_VALUES, PfldConfig};

#[test]
fn pfld_public_api_is_crate_root_only() {
    let device = Default::default();
    let model = PFLD_GhostOne::new(PfldConfig::production(), &device);
    let input = Tensor::<4>::zeros([1, PFLD_INPUT_CHANNELS, 192, 192], &device);
    assert_eq!(model.forward(input).dims(), [1, PFLD_OUTPUT_VALUES]);
}
