use std::sync::{Arc, Mutex};

use feathertalk_audio::{
    AudioError, ChunkEncoder, DEFAULT_CHUNK_SAMPLES, FeatureMatrix, drop_odd_token,
    extract_long_audio, fit_feature_tokens,
};

#[derive(Clone)]
struct FakeEncoder {
    dim: usize,
    calls: Arc<Mutex<Vec<(usize, usize, f32)>>>,
    rows_per_call: Vec<usize>,
}

impl ChunkEncoder for FakeEncoder {
    fn output_dim(&self) -> usize {
        self.dim
    }

    fn encode(&mut self, chunk_index: usize, samples: &[f32]) -> Result<Vec<f32>, AudioError> {
        self.calls
            .lock()
            .unwrap()
            .push((chunk_index, samples.len(), samples[0]));
        let rows = self.rows_per_call[chunk_index];
        Ok((0..rows)
            .flat_map(|row| std::iter::repeat_n((chunk_index * 100 + row) as f32, self.dim))
            .collect())
    }
}

fn waveform(length: usize) -> Vec<f32> {
    (0..length).map(|value| value as f32).collect()
}

#[test]
fn extracts_in_chunk_order_and_uses_extended_full_chunk_ranges() {
    let samples = waveform(DEFAULT_CHUNK_SAMPLES + 720);
    let calls = Arc::new(Mutex::new(Vec::new()));
    let mut encoder = FakeEncoder {
        dim: 2,
        calls: calls.clone(),
        rows_per_call: vec![4, 2],
    };
    let matrix = extract_long_audio(&samples, &mut encoder, DEFAULT_CHUNK_SAMPLES).unwrap();
    assert_eq!(matrix.tokens(), 1002);
    assert_eq!(matrix.dims(), 2);
    assert_eq!(
        calls.lock().unwrap().as_slice(),
        &[
            (0, DEFAULT_CHUNK_SAMPLES + 80, 0.0),
            (1, 720, DEFAULT_CHUNK_SAMPLES as f32),
        ]
    );
    assert_eq!(
        &matrix.values()[..12],
        &[
            0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 100.0, 100.0, 101.0, 101.0
        ]
    );
    assert!(matrix.values()[12..].iter().all(|value| *value == 0.0));
}

#[test]
fn crops_long_encoder_output_pads_short_output_and_drops_odd_token() {
    let samples = waveform(1360);
    let calls = Arc::new(Mutex::new(Vec::new()));
    let mut short = FakeEncoder {
        dim: 2,
        calls,
        rows_per_call: vec![2],
    };
    let padded = extract_long_audio(&samples, &mut short, DEFAULT_CHUNK_SAMPLES).unwrap();
    assert_eq!(padded.tokens(), 4);
    assert_eq!(padded.values(), &[0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]);

    let even = drop_odd_token(FeatureMatrix::new(5, 2, vec![1.0; 10]).unwrap());
    assert_eq!(even.tokens(), 4);
    assert_eq!(even.values().len(), 8);
}

#[test]
fn rejects_encoder_dimension_length_and_non_finite_output() {
    struct Bad;
    impl ChunkEncoder for Bad {
        fn output_dim(&self) -> usize {
            2
        }
        fn encode(&mut self, _: usize, _: &[f32]) -> Result<Vec<f32>, AudioError> {
            Ok(vec![f32::NAN; 2])
        }
    }
    let mut bad = Bad;
    assert!(matches!(
        extract_long_audio(&waveform(720), &mut bad, DEFAULT_CHUNK_SAMPLES),
        Err(AudioError::NonFiniteFeature { .. })
    ));

    struct WrongLength;
    impl ChunkEncoder for WrongLength {
        fn output_dim(&self) -> usize {
            4
        }
        fn encode(&mut self, _: usize, _: &[f32]) -> Result<Vec<f32>, AudioError> {
            Ok(vec![0.0; 3])
        }
    }
    assert!(matches!(
        extract_long_audio(&waveform(720), &mut WrongLength, DEFAULT_CHUNK_SAMPLES),
        Err(AudioError::FeatureLengthMismatch {
            actual: 3,
            dimension: 4
        })
    ));
}

#[test]
fn returns_empty_feature_without_calling_encoder_for_short_input() {
    struct PanicEncoder;
    impl ChunkEncoder for PanicEncoder {
        fn output_dim(&self) -> usize {
            2
        }

        fn encode(&mut self, _: usize, _: &[f32]) -> Result<Vec<f32>, AudioError> {
            panic!("encoder must not be called for a short waveform")
        }
    }

    let mut encoder = PanicEncoder;
    let matrix = extract_long_audio(&waveform(399), &mut encoder, DEFAULT_CHUNK_SAMPLES).unwrap();
    assert_eq!(matrix, FeatureMatrix::new(0, 2, vec![]).unwrap());
}

#[test]
fn fitting_pads_truncates_and_leaves_an_exact_matrix_alone() {
    let matrix = FeatureMatrix::new(2, 4, vec![1.0; 8]).unwrap();

    let padded = fit_feature_tokens(matrix.clone(), 3).unwrap();
    assert_eq!(padded.tokens(), 3);
    assert_eq!(padded.dims(), 4);
    assert_eq!(&padded.values()[..8], &[1.0; 8]);
    assert_eq!(&padded.values()[8..], &[0.0; 4]);

    let truncated = fit_feature_tokens(matrix.clone(), 1).unwrap();
    assert_eq!(truncated.tokens(), 1);
    assert_eq!(truncated.values(), &[1.0; 4]);

    let unchanged = fit_feature_tokens(matrix.clone(), 2).unwrap();
    assert_eq!(unchanged, matrix);
}

#[test]
fn an_impossible_token_count_overflows_instead_of_allocating() {
    let matrix = FeatureMatrix::new(1, 1024, vec![0.5; 1024]).unwrap();
    let error = fit_feature_tokens(matrix, usize::MAX).unwrap_err();
    assert!(
        matches!(error, AudioError::FeatureSizeOverflow),
        "{error:?}"
    );
}

#[test]
fn fitting_to_zero_tokens_empties_the_matrix() {
    let matrix = FeatureMatrix::new(2, 4, vec![1.0; 8]).unwrap();
    let empty = fit_feature_tokens(matrix, 0).unwrap();
    assert_eq!(empty.tokens(), 0);
    assert_eq!(empty.dims(), 4);
    assert!(empty.values().is_empty());
}
