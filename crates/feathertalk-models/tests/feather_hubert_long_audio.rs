use burn::tensor::{Tensor, TensorData};
use feathertalk_audio::{ChunkEncoder, drop_odd_token, extract_long_audio};
use feathertalk_models::{
    backend::CpuBackend,
    feather_hubert::{BurnFeatherHubertEncoder, FeatherHubertConfig},
};

#[test]
fn cpu_adapter_encodes_finite_rows_and_supports_long_audio_stitching() {
    let device: burn::tensor::Device<CpuBackend> = Default::default();
    let mut encoder = BurnFeatherHubertEncoder::<CpuBackend>::from_config(
        FeatherHubertConfig::parity_micro(),
        &device,
    );
    assert_eq!(encoder.output_dim(), 64);
    let chunk = vec![0.0_f32; 1360];
    let rows = encoder.encode(0, &chunk).unwrap();
    assert_eq!(rows.len(), 4 * 64);
    assert!(rows.iter().all(|value| value.is_finite()));

    let matrix = extract_long_audio(&chunk, &mut encoder, 1360).unwrap();
    assert_eq!(matrix.tokens(), 4);
    assert_eq!(matrix.dims(), 64);
    assert!(matrix.values().iter().all(|value| value.is_finite()));
    assert_eq!(drop_odd_token(matrix).tokens(), 4);
}

#[test]
fn cpu_adapter_returns_no_rows_for_short_chunks_without_panicking() {
    let device: burn::tensor::Device<CpuBackend> = Default::default();
    let mut encoder = BurnFeatherHubertEncoder::<CpuBackend>::from_config(
        FeatherHubertConfig::parity_micro(),
        &device,
    );
    assert!(encoder.encode(0, &[0.0; 399]).unwrap().is_empty());
}

#[test]
fn cpu_adapter_accepts_tensor_data_shape_contract() {
    let device = Default::default();
    let tensor =
        Tensor::<CpuBackend, 2>::from_data(TensorData::new(vec![0.0_f32; 400], [1, 400]), &device);
    assert_eq!(tensor.dims(), [1, 400]);
}

#[test]
fn cpu_adapter_can_take_ownership_of_an_imported_model() {
    let device = Default::default();
    let model = FeatherHubertConfig::parity_micro().init::<CpuBackend>(&device);
    let mut encoder = BurnFeatherHubertEncoder::from_model(model, &device);

    assert_eq!(encoder.output_dim(), 64);
    assert_eq!(encoder.model().config.output_dim, 64);
    let rows = encoder.encode(0, &[0.0; 1360]).unwrap();
    assert_eq!(rows.len(), 4 * 64);
    assert!(rows.iter().all(|value| value.is_finite()));
}

#[test]
#[ignore = "requires a certified WGPU adapter"]
fn wgpu_adapter_runs_without_cpu_fallback() {
    use burn::backend::Wgpu;

    type GpuBackend = Wgpu<f32, i32, u32>;
    let device = Default::default();
    let mut encoder = BurnFeatherHubertEncoder::<GpuBackend>::from_config(
        FeatherHubertConfig::parity_micro(),
        &device,
    );
    let rows = encoder.encode(0, &vec![0.0_f32; 400]).unwrap();
    assert_eq!(rows.len(), 64);
    assert!(rows.iter().all(|value| value.is_finite()));
}
