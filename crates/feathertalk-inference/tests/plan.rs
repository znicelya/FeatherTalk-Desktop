use feathertalk_inference::{InferenceError, RenderPlan};

#[test]
fn plan_maps_ping_pong_source_and_current_frame_reference() {
    let plan = RenderPlan::new(3, 6, None).unwrap();
    let frames: Vec<_> = (0..6).map(|i| plan.frame(i).unwrap()).collect();
    assert_eq!(
        frames
            .iter()
            .map(|frame| frame.source_frame_index)
            .collect::<Vec<_>>(),
        vec![0, 1, 2, 1, 0, 1]
    );
    assert!(
        frames
            .iter()
            .all(|frame| frame.source_frame_index == frame.reference_frame_index)
    );
    assert_eq!(
        frames[0].audio_window,
        [None, None, None, None, Some(0), Some(1), Some(2), Some(3)]
    );
    assert_eq!(
        frames[3].audio_window,
        [
            None,
            Some(0),
            Some(1),
            Some(2),
            Some(3),
            Some(4),
            Some(5),
            None
        ]
    );
}

#[test]
fn plan_caps_preview_and_rejects_invalid_requests() {
    let plan = RenderPlan::new(2, 10, Some(4)).unwrap();
    assert_eq!(plan.output_frame_count(), 4);
    assert!(matches!(
        plan.frame(4),
        Err(InferenceError::OutputFrameOutOfRange { index: 4, count: 4 })
    ));
    assert!(matches!(
        RenderPlan::new(2, 0, None),
        Err(InferenceError::EmptyFeatures)
    ));
    assert!(matches!(
        RenderPlan::new(1, 2, None),
        Err(InferenceError::FrameCountTooSmall { .. })
    ));
    assert!(matches!(
        RenderPlan::new(2, 2, Some(0)),
        Err(InferenceError::InvalidField {
            field: "max_output_frames",
            ..
        })
    ));
}
