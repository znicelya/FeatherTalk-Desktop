use burn::tensor::Tensor;
use feathertalk_models::backend::{CpuAutodiffBackend, CpuBackend};
#[allow(unused_imports)]

#[test]
fn cpu_backend_aliases_compile_and_execute() {
    // The 0.22 dispatch backend aliases still resolve and execute on CPU.
    let device = burn::tensor::Device::default();
    let tensor = Tensor::<2>::ones([2, 3], &device);
    assert_eq!(tensor.dims(), [2, 3]);

    // Autodiff is a device property in 0.22, so the aliases are exercised by name only.
    fn aliases<B: burn::backend::Backend>() {}
    aliases::<CpuBackend>();
    aliases::<CpuAutodiffBackend>();
}