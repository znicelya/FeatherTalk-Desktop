use feathertalk_audio::FeatureMatrix;
use feathertalk_inference::{
    InferenceError, InferenceFramePlan, build_unet_audio_input, build_unet_audio_window,
};

fn features(frame_count: usize) -> FeatureMatrix {
    let tokens = frame_count * 2;
    let values = (0..tokens)
        .flat_map(|token| (0..1024).map(move |dimension| (token * 10_000 + dimension) as f32))
        .collect();
    FeatureMatrix::new(tokens, 1024, values).unwrap()
}

#[test]
fn audio_window_flattens_two_tokens_per_slot_without_transpose() {
    let plan = InferenceFramePlan {
        output_index: 1,
        source_frame_index: 0,
        reference_frame_index: 0,
        audio_window: [None, None, Some(0), Some(1), Some(2), None, None, None],
    };
    let input = build_unet_audio_input(&features(3), &plan).unwrap();

    assert_eq!(input.shape(), [1, 16, 32, 32]);
    assert_eq!(input.as_slice().len(), 16 * 32 * 32);
    assert!(
        input.as_slice()[..2 * 2048]
            .iter()
            .all(|value| *value == 0.0)
    );
    let first = 2 * 2048;
    assert_eq!(&input.as_slice()[first..first + 3], &[0.0, 1.0, 2.0]);
    assert_eq!(input.as_slice()[first + 1024], 10_000.0);
    let second = 3 * 2048;
    assert_eq!(input.as_slice()[second], 20_000.0);
    assert_eq!(input.as_slice()[second + 1024], 30_000.0);
}

#[test]
fn audio_window_rejects_invalid_feature_matrix_contracts() {
    for matrix in [
        FeatureMatrix::new(0, 1024, vec![]).unwrap(),
        FeatureMatrix::new(3, 1024, vec![0.0; 3 * 1024]).unwrap(),
        FeatureMatrix::new(2, 64, vec![0.0; 2 * 64]).unwrap(),
    ] {
        let plan = InferenceFramePlan {
            output_index: 0,
            source_frame_index: 0,
            reference_frame_index: 0,
            audio_window: [None; 8],
        };
        assert!(matches!(
            build_unet_audio_input(&matrix, &plan),
            Err(InferenceError::InvalidFeatureShape { .. })
        ));
    }
}

#[test]
fn audio_window_rejects_output_and_slot_indices_beyond_feature_frames() {
    let matrix = features(2);
    let output_plan = InferenceFramePlan {
        output_index: 2,
        source_frame_index: 0,
        reference_frame_index: 0,
        audio_window: [None; 8],
    };
    assert!(matches!(
        build_unet_audio_input(&matrix, &output_plan),
        Err(InferenceError::OutputFrameOutOfRange { index: 2, count: 2 })
    ));

    let slot_plan = InferenceFramePlan {
        output_index: 0,
        source_frame_index: 0,
        reference_frame_index: 0,
        audio_window: [Some(2), None, None, None, None, None, None, None],
    };
    assert!(matches!(
        build_unet_audio_input(&matrix, &slot_plan),
        Err(InferenceError::InvalidAudioWindowIndex {
            slot: 0,
            index: 2,
            frame_count: 2
        })
    ));
}

#[test]
fn plan_free_audio_window_matches_the_planned_window() {
    let matrix = features(3);
    let audio_window = [None, None, Some(0), Some(1), Some(2), None, None, None];
    let plan = InferenceFramePlan {
        output_index: 1,
        source_frame_index: 0,
        reference_frame_index: 0,
        audio_window,
    };
    let planned = build_unet_audio_input(&matrix, &plan).unwrap();
    let direct = build_unet_audio_window(&matrix, &audio_window).unwrap();
    assert_eq!(direct.shape(), [1, 16, 32, 32]);
    assert_eq!(direct.as_slice(), planned.as_slice());
}

#[test]
fn plan_free_audio_window_rejects_slots_and_feature_shapes() {
    let matrix = features(2);
    let window = [None, None, None, None, Some(2), None, None, None];
    assert!(matches!(
        build_unet_audio_window(&matrix, &window),
        Err(InferenceError::InvalidAudioWindowIndex {
            slot: 4,
            index: 2,
            frame_count: 2
        })
    ));
    let odd = FeatureMatrix::new(3, 1024, vec![0.0; 3 * 1024]).unwrap();
    assert!(matches!(
        build_unet_audio_window(&odd, &[None; 8]),
        Err(InferenceError::InvalidFeatureShape {
            tokens: 3,
            dims: 1024
        })
    ));
}
