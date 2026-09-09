use feathertalk_inference::{InferenceError, PingPongFrames};

#[test]
fn two_frames_repeat_without_duplicate_endpoints() {
    let mut picker = PingPongFrames::new(2).unwrap();
    let values: Vec<_> = (0..7).map(|_| picker.next()).collect();
    assert_eq!(values, vec![0, 1, 0, 1, 0, 1, 0]);
}

#[test]
fn three_frames_reflect_at_both_boundaries() {
    let mut picker = PingPongFrames::new(3).unwrap();
    let values: Vec<_> = (0..9).map(|_| picker.next()).collect();
    assert_eq!(values, vec![0, 1, 2, 1, 0, 1, 2, 1, 0]);
}

#[test]
fn rejects_fewer_than_two_source_frames() {
    assert!(matches!(
        PingPongFrames::new(0),
        Err(InferenceError::FrameCountTooSmall { .. })
    ));
    assert!(matches!(
        PingPongFrames::new(1),
        Err(InferenceError::FrameCountTooSmall { .. })
    ));
}
