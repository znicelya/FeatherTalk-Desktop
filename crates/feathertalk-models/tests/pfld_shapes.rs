use burn::tensor::Tensor;
use burn::{module::Module, tensor::backend::Backend};
use feathertalk_models::backend::CpuBackend;
use feathertalk_models::{
    GhostOneBottleneck, MobileOneBlock, PFLD_GhostOne, PFLD_OUTPUT_VALUES, PfldConfig,
};

fn assert_module<B: Backend, M: Module<B>>() {}

#[test]
fn production_config_is_fixed() {
    let config = PfldConfig::production();
    assert_eq!(config.width_factor, 0.5);
    assert_eq!(config.input_size, 192);
    assert_eq!(config.landmark_count, 110);
    assert_eq!(config.num_conv_branches, 6);
    assert_eq!(PFLD_OUTPUT_VALUES, 220);
}

#[test]
fn pfld_graph_is_a_burn_module() {
    assert_module::<CpuBackend, MobileOneBlock<CpuBackend>>();
    assert_module::<CpuBackend, GhostOneBottleneck<CpuBackend>>();
    assert_module::<CpuBackend, PFLD_GhostOne<CpuBackend>>();
}

#[test]
fn production_model_shape_is_declared() {
    let device = Default::default();
    let model = PFLD_GhostOne::<CpuBackend>::new(PfldConfig::production(), &device);
    let input = Tensor::<CpuBackend, 4>::zeros([1, 3, 192, 192], &device);
    assert_eq!(model.forward(input).dims(), [1, PFLD_OUTPUT_VALUES]);
}

#[test]
fn ghost_one_bottleneck_preserves_expected_stride_shapes() {
    let device = Default::default();
    let down = GhostOneBottleneck::<CpuBackend>::new(32, 48, 40, 2, 6, &device);
    let same = GhostOneBottleneck::<CpuBackend>::new(40, 60, 40, 1, 6, &device);
    let input = Tensor::<CpuBackend, 4>::zeros([1, 32, 96, 96], &device);
    assert_eq!(down.forward(input).dims(), [1, 40, 48, 48]);
    let input = Tensor::<CpuBackend, 4>::zeros([1, 40, 48, 48], &device);
    assert_eq!(same.forward(input).dims(), [1, 40, 48, 48]);
}

#[test]
fn production_model_supports_multiple_batch_items() {
    let device = Default::default();
    let model = PFLD_GhostOne::<CpuBackend>::new(PfldConfig::production(), &device);
    let input = Tensor::<CpuBackend, 4>::zeros([2, 3, 192, 192], &device);
    assert_eq!(model.forward(input).dims(), [2, 220]);
}

#[test]
fn production_pooled_channels_match_output_head_contract() {
    let config = PfldConfig::production();
    assert_eq!(config.pooled_channels(), 32 + 40 + 48 + 72 + 64);
    assert_eq!(config.pooled_channels(), 256);
}
