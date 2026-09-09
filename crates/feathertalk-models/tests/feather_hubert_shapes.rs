use burn::tensor::{Tensor, TensorData};
use feathertalk_models::{
    backend::CpuBackend,
    feather_hubert::{
        FeatherHubertConfig, expected_hubert_frames, make_even_tokens, normalize_waveform,
    },
};

#[test]
fn hubert_frame_count_matches_python_contract() {
    assert_eq!(expected_hubert_frames(399), 0);
    assert_eq!(expected_hubert_frames(400), 1);
    assert_eq!(expected_hubert_frames(720), 2);
    assert_eq!(expected_hubert_frames(1360), 4);
}

#[test]
fn waveform_normalization_has_zero_mean_and_unit_variance() {
    let device = Default::default();
    let waveform =
        Tensor::<CpuBackend, 2>::from_data(TensorData::from([[1.0_f32, 2.0, 3.0, 4.0]]), &device);
    let values = normalize_waveform(waveform)
        .into_data()
        .to_vec::<f32>()
        .unwrap();
    let mean = values.iter().sum::<f32>() / values.len() as f32;
    let variance = values
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f32>()
        / values.len() as f32;

    assert!(mean.abs() <= 1e-6, "mean={mean}");
    assert!((variance - 1.0).abs() <= 1e-5, "variance={variance}");
}

#[test]
fn waveform_normalization_is_independent_per_batch_item() {
    let device = Default::default();
    let waveform = Tensor::<CpuBackend, 2>::from_data(
        TensorData::from([[1.0_f32, 3.0], [100.0, 102.0]]),
        &device,
    );
    let values = normalize_waveform(waveform)
        .into_data()
        .to_vec::<f32>()
        .unwrap();

    for batch in values.chunks_exact(2) {
        let mean = batch.iter().sum::<f32>() / batch.len() as f32;
        let variance = batch
            .iter()
            .map(|value| (value - mean).powi(2))
            .sum::<f32>()
            / batch.len() as f32;
        assert!(mean.abs() <= 1e-6, "batch mean={mean}");
        assert!((variance - 1.0).abs() <= 1e-5, "batch variance={variance}");
    }
}

#[test]
fn micro_encoder_returns_four_tokens() {
    let device = Default::default();
    let model = FeatherHubertConfig::parity_micro().init::<CpuBackend>(&device);
    let waveform = Tensor::<CpuBackend, 2>::zeros([1, 1360], &device);
    assert_eq!(model.forward(waveform).dims(), [1, 4, 64]);
}

#[test]
fn production_encoder_returns_1024_features() {
    let device = Default::default();
    let model = FeatherHubertConfig::default().init::<CpuBackend>(&device);
    let waveform = Tensor::<CpuBackend, 2>::zeros([1, 1360], &device);
    assert_eq!(model.forward(waveform).dims(), [1, 4, 1024]);
}

#[test]
fn odd_token_count_drops_the_last_token() {
    let device = Default::default();
    let tokens = Tensor::<CpuBackend, 3>::zeros([1, 5, 64], &device);
    assert_eq!(make_even_tokens(tokens).dims(), [1, 4, 64]);
}
