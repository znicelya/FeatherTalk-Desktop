use feathertalk_preprocess::{PreprocessError, audio_window_indices};

#[test]
fn returns_padded_window_for_first_middle_and_last_frames() {
    assert_eq!(
        audio_window_indices(0, 10).unwrap(),
        [None, None, None, None, Some(0), Some(1), Some(2), Some(3)]
    );
    assert_eq!(
        audio_window_indices(5, 10).unwrap(),
        [
            Some(1),
            Some(2),
            Some(3),
            Some(4),
            Some(5),
            Some(6),
            Some(7),
            Some(8)
        ]
    );
    assert_eq!(
        audio_window_indices(9, 10).unwrap(),
        [
            Some(5),
            Some(6),
            Some(7),
            Some(8),
            Some(9),
            None,
            None,
            None
        ]
    );
}

#[test]
fn rejects_empty_and_out_of_range_frames() {
    assert!(matches!(
        audio_window_indices(0, 0),
        Err(PreprocessError::FrameIndexOutOfRange { .. })
    ));
    assert!(matches!(
        audio_window_indices(10, 10),
        Err(PreprocessError::FrameIndexOutOfRange { .. })
    ));
}

#[test]
fn handles_large_usize_indices_without_signed_overflow() {
    let frame_count = usize::MAX;
    let frame_index = usize::MAX - 1;
    assert_eq!(
        audio_window_indices(frame_index, frame_count).unwrap(),
        [
            Some(usize::MAX - 5),
            Some(usize::MAX - 4),
            Some(usize::MAX - 3),
            Some(usize::MAX - 2),
            Some(usize::MAX - 1),
            None,
            None,
            None,
        ]
    );
}
