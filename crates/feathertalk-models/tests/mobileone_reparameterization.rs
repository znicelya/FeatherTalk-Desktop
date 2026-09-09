use burn::{module::Module, tensor::Tensor};
use feathertalk_models::{MobileOneBlock, backend::CpuBackend};

fn assert_module<M: Module<CpuBackend>>() {}

#[test]
fn shared_mobileone_block_is_a_burn_module() {
    assert_module::<MobileOneBlock<CpuBackend>>();
}

#[test]
fn square_stride_keeps_the_existing_pfld_shape_contract() {
    let device = Default::default();
    let block = MobileOneBlock::<CpuBackend>::new(4, 8, 3, 2, 1, 1, 2, false, &device);
    let input = Tensor::zeros([1, 4, 12, 14], &device);
    assert_eq!(block.forward(input).dims(), [1, 8, 6, 7]);
}

#[test]
fn anisotropic_stride_halves_only_width() {
    let device = Default::default();
    let block =
        MobileOneBlock::<CpuBackend>::new_with_stride(4, 8, 3, [1, 2], 1, 1, 2, false, &device);
    let input = Tensor::zeros([1, 4, 12, 14], &device);
    assert_eq!(block.forward(input).dims(), [1, 8, 12, 7]);
}

#[test]
#[should_panic]
fn scaled_branch_rejects_non_centered_padding_during_construction() {
    let device = Default::default();
    let _block =
        MobileOneBlock::<CpuBackend>::new_with_stride(4, 8, 3, [1, 2], 0, 1, 2, false, &device);
}

fn assert_reparameterized_close(block: MobileOneBlock<CpuBackend>, input: Tensor<CpuBackend, 4>) {
    let expected = block
        .forward(input.clone())
        .into_data()
        .to_vec::<f32>()
        .unwrap();
    let actual = block
        .reparameterize()
        .forward(input)
        .into_data()
        .to_vec::<f32>()
        .unwrap();
    let max_abs = expected
        .iter()
        .zip(actual.iter())
        .map(|(left, right)| (left - right).abs())
        .fold(0.0_f32, f32::max);
    assert!(max_abs <= 1.0e-4, "max_abs={max_abs}");
}

#[test]
fn reparameterized_one_by_one_block_matches_training_graph() {
    let device = Default::default();
    let block = MobileOneBlock::<CpuBackend>::new(4, 8, 1, 1, 0, 1, 2, false, &device);
    assert_reparameterized_close(block, Tensor::ones([1, 4, 8, 8], &device));
}

#[test]
fn reparameterized_three_by_three_block_with_skip_matches_training_graph() {
    let device = Default::default();
    let block = MobileOneBlock::<CpuBackend>::new(4, 4, 3, 1, 1, 1, 2, false, &device);
    assert_reparameterized_close(block, Tensor::ones([1, 4, 8, 8], &device));
}

#[test]
fn reparameterized_depthwise_block_matches_training_graph() {
    let device = Default::default();
    let block = MobileOneBlock::<CpuBackend>::new(4, 4, 3, 2, 1, 4, 2, true, &device);
    assert_reparameterized_close(block, Tensor::ones([1, 4, 8, 8], &device));
}

#[test]
fn reparameterized_anisotropic_stride_matches_training_graph() {
    let device = Default::default();
    let block =
        MobileOneBlock::<CpuBackend>::new_with_stride(4, 8, 3, [1, 2], 1, 1, 2, false, &device);
    assert_reparameterized_close(block, Tensor::ones([1, 4, 8, 10], &device));
}
