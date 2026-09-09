use burn::tensor::{Tensor, backend::Backend};
use feathertalk_models::backend::{CpuAutodiffBackend, CpuBackend};

fn assert_backend<B: Backend>() {}

#[test]
fn cpu_backend_aliases_compile_and_execute() {
    assert_backend::<CpuBackend>();
    assert_backend::<CpuAutodiffBackend>();

    let device = Default::default();
    let tensor = Tensor::<CpuBackend, 2>::ones([2, 3], &device);
    assert_eq!(tensor.dims(), [2, 3]);
}
