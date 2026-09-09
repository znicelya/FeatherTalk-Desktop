use std::{fs, path::Path};

use burn::{
    backend::Wgpu,
    tensor::{Tensor, TensorData},
};
use feathertalk_pfld::PfldRuntime;

type GpuBackend = Wgpu<f32, i32, u32>;

#[test]
#[ignore = "requires a certified WGPU adapter"]
fn committed_pfld_artifact_runs_on_wgpu_without_cpu_fallback() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let fixture = root.join("tests/fixtures/pytorch_cpu_v1");
    let values = fs::read(fixture.join("input.f32"))
        .unwrap()
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect::<Vec<_>>();
    let device = Default::default();
    let input =
        Tensor::<GpuBackend, 4>::from_data(TensorData::new(values, [1, 3, 192, 192]), &device);
    let runtime =
        PfldRuntime::<GpuBackend>::load(&root.join("artifacts/pfld_ghost_one"), &device).unwrap();
    let output = runtime.forward(input).unwrap();
    assert_eq!(output.dims(), [1, 220]);
    let actual = output.into_data().to_vec::<f32>().unwrap();
    let expected = fs::read(fixture.join("output.f32"))
        .unwrap()
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect::<Vec<_>>();
    assert_eq!(actual.len(), 220);
    assert_eq!(expected.len(), 220);
    let mut max_abs = 0.0_f32;
    let mut sum_abs = 0.0_f32;
    for (index, (&actual, &expected)) in actual.iter().zip(&expected).enumerate() {
        assert!(
            actual.is_finite() && expected.is_finite(),
            "non-finite at {index}"
        );
        let difference = (actual - expected).abs();
        max_abs = max_abs.max(difference);
        sum_abs += difference;
    }
    let mean_abs = sum_abs / actual.len() as f32;
    assert!(
        max_abs <= 1e-3,
        "WGPU PFLD parity exceeded max_abs threshold: max_abs={max_abs}, mean_abs={mean_abs}"
    );
}
