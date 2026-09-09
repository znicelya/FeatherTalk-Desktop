use feathertalk_inference::{BgrFrame, InferenceError, RenderGeometry, paste_bgr, render_frame};
use feathertalk_preprocess::FaceBoundingBox;

#[test]
fn paste_copies_rows_and_rejects_negative_or_outside_origins() {
    let mut destination = BgrFrame::new(3, 2, vec![0; 18]).unwrap();
    let source = BgrFrame::new(2, 1, vec![1, 2, 3, 4, 5, 6]).unwrap();
    paste_bgr(&mut destination, &source, 1, 1).unwrap();
    assert_eq!(
        destination.as_bytes(),
        &[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6]
    );
    assert!(matches!(
        paste_bgr(&mut destination, &source, -1, 0),
        Err(InferenceError::PasteOutOfBounds { .. })
    ));
    assert!(matches!(
        paste_bgr(&mut destination, &source, 2, 1),
        Err(InferenceError::PasteOutOfBounds { .. })
    ));
}

#[test]
fn render_frame_returns_new_frame_and_leaves_input_unchanged() {
    let geometry = RenderGeometry::standard();
    let frame = BgrFrame::new(2, 2, vec![10; 12]).unwrap();
    let original = frame.clone();
    let bbox = FaceBoundingBox {
        xmin: 0,
        ymin: 0,
        xmax: 2,
        ymax: 2,
    };
    let prediction = vec![1.0; 3 * 160 * 160];
    let rendered = render_frame(&frame, &bbox, &prediction, &geometry).unwrap();
    assert_eq!(frame, original);
    assert_eq!(rendered.as_bytes(), &[255; 12]);
}
