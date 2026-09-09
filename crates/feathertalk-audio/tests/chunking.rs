use feathertalk_audio::{
    AudioError, DEFAULT_CHUNK_SAMPLES, HUBERT_KERNEL, HUBERT_STRIDE, expected_hubert_frames,
    plan_chunks,
};

#[test]
fn expected_token_count_matches_python_boundaries() {
    assert_eq!(expected_hubert_frames(HUBERT_KERNEL - 1), 0);
    assert_eq!(expected_hubert_frames(HUBERT_KERNEL), 1);
    assert_eq!(expected_hubert_frames(HUBERT_KERNEL + HUBERT_STRIDE - 1), 1);
    assert_eq!(expected_hubert_frames(HUBERT_KERNEL + HUBERT_STRIDE), 2);
    assert_eq!(expected_hubert_frames(1360), 4);
}

#[test]
fn planner_emits_python_compatible_ranges_for_short_and_tail_audio() {
    let plan = plan_chunks(720, DEFAULT_CHUNK_SAMPLES).unwrap();
    assert_eq!(plan.target_tokens(), 2);
    assert_eq!(plan.ranges(), &[(0..720).into()]);

    let total = DEFAULT_CHUNK_SAMPLES + 720;
    let plan = plan_chunks(total, DEFAULT_CHUNK_SAMPLES).unwrap();
    assert_eq!(plan.target_tokens(), expected_hubert_frames(total));
    assert_eq!(plan.ranges().len(), 2);
    assert_eq!(plan.ranges()[0].start(), 0);
    assert_eq!(
        plan.ranges()[0].end(),
        DEFAULT_CHUNK_SAMPLES + HUBERT_KERNEL - HUBERT_STRIDE
    );
    assert_eq!(plan.ranges()[1].start(), DEFAULT_CHUNK_SAMPLES);
    assert_eq!(plan.ranges()[1].end(), total);
}

#[test]
fn planner_uses_exact_chunk_boundary_without_duplicate_tail() {
    let plan = plan_chunks(DEFAULT_CHUNK_SAMPLES, DEFAULT_CHUNK_SAMPLES).unwrap();
    assert_eq!(plan.ranges().len(), 1);
    assert_eq!(plan.ranges()[0].start(), 0);
    assert_eq!(plan.ranges()[0].end(), DEFAULT_CHUNK_SAMPLES);
}

#[test]
fn planner_rejects_zero_chunk_and_checked_overflow() {
    assert!(matches!(
        plan_chunks(100, 0),
        Err(AudioError::InvalidChunkSize)
    ));
    assert!(matches!(
        plan_chunks(usize::MAX, DEFAULT_CHUNK_SAMPLES),
        Err(AudioError::ArithmeticOverflow)
    ));
}

#[test]
fn planner_does_not_schedule_encoder_for_shorter_than_kernel_input() {
    let plan = plan_chunks(HUBERT_KERNEL - 1, DEFAULT_CHUNK_SAMPLES).unwrap();
    assert_eq!(plan.target_tokens(), 0);
    assert!(plan.ranges().is_empty());
}

#[test]
fn planner_skips_a_tail_shorter_than_the_encoder_kernel() {
    let total = DEFAULT_CHUNK_SAMPLES + HUBERT_KERNEL - 1;
    let plan = plan_chunks(total, DEFAULT_CHUNK_SAMPLES).unwrap();
    assert_eq!(plan.ranges().len(), 1);
    assert_eq!(plan.ranges()[0].start(), 0);
    assert_eq!(
        plan.ranges()[0].end(),
        DEFAULT_CHUNK_SAMPLES + HUBERT_KERNEL - HUBERT_STRIDE
    );
}
