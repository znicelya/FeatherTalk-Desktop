use crate::AudioError;

pub fn normalize_waveform(samples: &[f32]) -> Result<Vec<f32>, AudioError> {
    if samples.is_empty() {
        return Err(AudioError::EmptyWaveform);
    }
    for (index, value) in samples.iter().enumerate() {
        if !value.is_finite() {
            return Err(AudioError::NonFiniteWaveform { index });
        }
    }
    let mean = samples.iter().map(|value| *value as f64).sum::<f64>() / samples.len() as f64;
    let variance = samples
        .iter()
        .map(|value| {
            let delta = *value as f64 - mean;
            delta * delta
        })
        .sum::<f64>()
        / samples.len() as f64;
    if !variance.is_finite() {
        return Err(AudioError::ConstantWaveform);
    }
    let denominator = (variance + 1e-7_f64).sqrt();
    samples
        .iter()
        .map(|value| ((*value as f64 - mean) / denominator) as f32)
        .collect::<Vec<_>>()
        .into_iter()
        .enumerate()
        .map(|(index, value)| {
            if value.is_finite() {
                Ok(value)
            } else {
                Err(AudioError::NonFiniteWaveform { index })
            }
        })
        .collect()
}
