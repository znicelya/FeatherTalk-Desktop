use crate::{AudioError, FeatureMatrix, plan_chunks};

pub trait ChunkEncoder {
    fn output_dim(&self) -> usize;
    fn encode(&mut self, chunk_index: usize, samples: &[f32]) -> Result<Vec<f32>, AudioError>;
}

pub fn extract_long_audio<E: ChunkEncoder>(
    samples: &[f32],
    encoder: &mut E,
    chunk_samples: usize,
) -> Result<FeatureMatrix, AudioError> {
    for (index, value) in samples.iter().enumerate() {
        if !value.is_finite() {
            return Err(AudioError::NonFiniteWaveform { index });
        }
    }
    let dimension = encoder.output_dim();
    if dimension == 0 {
        return Err(AudioError::InvalidFeatureDimension);
    }
    let plan = plan_chunks(samples.len(), chunk_samples)?;
    let mut values = Vec::new();
    for range in plan.ranges() {
        let output = encoder.encode(range.index(), &samples[range.start()..range.end()])?;
        if !output.len().is_multiple_of(dimension) {
            return Err(AudioError::FeatureLengthMismatch {
                actual: output.len(),
                dimension,
            });
        }
        for (index, value) in output.iter().enumerate() {
            if !value.is_finite() {
                return Err(AudioError::NonFiniteFeature { index });
            }
        }
        values.extend(output);
    }
    let target_values = plan
        .target_tokens()
        .checked_mul(dimension)
        .ok_or(AudioError::FeatureSizeOverflow)?;
    fit_values(&mut values, target_values);
    FeatureMatrix::new(plan.target_tokens(), dimension, values)
}

/// Pad or truncate a feature matrix to exactly `tokens` tokens.
///
/// Same rule as the tail of `extract_long_audio` — short output gains zero
/// vectors, long output loses its tail — exposed for callers that learn the
/// token count from somewhere other than the waveform. The asset lock learns
/// it from the frame count.
pub fn fit_feature_tokens(
    matrix: FeatureMatrix,
    tokens: usize,
) -> Result<FeatureMatrix, AudioError> {
    let dims = matrix.dims();
    let target_values = tokens
        .checked_mul(dims)
        .ok_or(AudioError::FeatureSizeOverflow)?;
    let mut values = matrix.into_values();
    fit_values(&mut values, target_values);
    FeatureMatrix::new(tokens, dims, values)
}

pub fn drop_odd_token(matrix: FeatureMatrix) -> FeatureMatrix {
    if matrix.tokens().is_multiple_of(2) {
        matrix
    } else {
        let tokens = matrix.tokens() - 1;
        let values = matrix.values()[..tokens * matrix.dims()].to_vec();
        FeatureMatrix::new(tokens, matrix.dims(), values).expect("validated feature matrix")
    }
}

/// Pad with zeros or truncate so that `values` holds exactly `target_values`.
fn fit_values(values: &mut Vec<f32>, target_values: usize) {
    if values.len() < target_values {
        values.resize(target_values, 0.0);
    } else {
        values.truncate(target_values);
    }
}
