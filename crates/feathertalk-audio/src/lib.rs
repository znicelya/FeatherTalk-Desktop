mod chunk;
mod commit;
mod error;
mod format;
mod normalize;
mod stitch;
mod wav;

pub use chunk::{
    ChunkPlan, ChunkRange, DEFAULT_CHUNK_SAMPLES, HUBERT_KERNEL, HUBERT_STRIDE,
    expected_hubert_frames, plan_chunks,
};
pub use commit::{FeatureCommitSpec, commit_feature_artifact};
pub use error::AudioError;
pub use format::{
    FeatureArtifact, MAX_FEATURE_FILE_BYTES, read_feature_file, write_feature_file,
    write_feature_file_no_clobber,
};
pub use normalize::normalize_waveform;
pub use stitch::{ChunkEncoder, drop_odd_token, extract_long_audio, fit_feature_tokens};
pub use wav::{MAX_WAV_FILE_BYTES, WAV_SAMPLE_RATE, read_wav_16k_mono};

#[derive(Debug, Clone, PartialEq)]
pub struct FeatureMatrix {
    tokens: usize,
    dims: usize,
    values: Vec<f32>,
}

impl FeatureMatrix {
    pub fn new(tokens: usize, dims: usize, values: Vec<f32>) -> Result<Self, AudioError> {
        if dims == 0 {
            return Err(AudioError::InvalidFeatureDimension);
        }
        let expected = tokens
            .checked_mul(dims)
            .ok_or(AudioError::FeatureSizeOverflow)?;
        if values.len() != expected {
            return Err(AudioError::FeatureLengthMismatch {
                actual: values.len(),
                dimension: dims,
            });
        }
        for (index, value) in values.iter().enumerate() {
            if !value.is_finite() {
                return Err(AudioError::NonFiniteFeature { index });
            }
        }
        Ok(Self {
            tokens,
            dims,
            values,
        })
    }

    pub fn tokens(&self) -> usize {
        self.tokens
    }
    pub fn dims(&self) -> usize {
        self.dims
    }
    pub fn values(&self) -> &[f32] {
        &self.values
    }

    /// Take the backing storage. `pub(crate)` because it is a stepping stone
    /// for in-crate transforms, not part of the crate's public surface.
    pub(crate) fn into_values(self) -> Vec<f32> {
        self.values
    }
}
