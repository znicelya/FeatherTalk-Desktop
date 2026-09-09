use feathertalk_audio::{AudioError, normalize_waveform};

#[test]
fn normalizes_to_zero_mean_and_unit_variance() {
    let values = normalize_waveform(&[1.0, 2.0, 3.0, 4.0]).unwrap();
    let mean = values.iter().sum::<f32>() / values.len() as f32;
    let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / values.len() as f32;
    assert!(mean.abs() < 1e-6);
    assert!((variance - 1.0).abs() < 1e-5);
}

#[test]
fn rejects_empty_constant_and_non_finite_waveforms() {
    assert!(matches!(
        normalize_waveform(&[]),
        Err(AudioError::EmptyWaveform)
    ));
    assert_eq!(normalize_waveform(&[1.0, 1.0]).unwrap(), vec![0.0, 0.0]);
    assert!(matches!(
        normalize_waveform(&[0.0, f32::NAN]),
        Err(AudioError::NonFiniteWaveform { index: 1 })
    ));
    assert!(matches!(
        normalize_waveform(&[0.0, f32::INFINITY]),
        Err(AudioError::NonFiniteWaveform { index: 1 })
    ));
}
