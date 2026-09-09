use feathertalk_inference::{BgrFrame, InferenceError};

#[test]
fn bgr_frame_keeps_dimensions_and_interleaved_bytes() {
    let bytes = vec![10, 20, 30, 40, 50, 60];
    let frame = BgrFrame::new(2, 1, bytes.clone()).unwrap();
    assert_eq!((frame.width(), frame.height()), (2, 1));
    assert_eq!(frame.as_bytes(), bytes.as_slice());
    assert_eq!(frame.pixel(1, 0).unwrap(), [40, 50, 60]);
    assert_eq!(frame.clone().into_bytes(), bytes);
}

#[test]
fn bgr_frame_rejects_zero_dimensions_and_wrong_length() {
    assert!(matches!(
        BgrFrame::new(0, 1, Vec::new()),
        Err(InferenceError::InvalidFrameDimensions { .. })
    ));
    assert!(matches!(
        BgrFrame::new(2, 1, vec![0; 5]),
        Err(InferenceError::FrameBufferLengthMismatch {
            expected: 6,
            actual: 5
        })
    ));
}

#[test]
fn bgr_frame_rejects_out_of_range_pixels() {
    let frame = BgrFrame::new(2, 2, vec![0; 12]).unwrap();
    assert!(matches!(
        frame.pixel(2, 0),
        Err(InferenceError::PixelOutOfRange { x: 2, y: 0, .. })
    ));
    assert!(matches!(
        frame.pixel(0, 2),
        Err(InferenceError::PixelOutOfRange { x: 0, y: 2, .. })
    ));
}
