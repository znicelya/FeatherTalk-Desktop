mod support;

use std::fs;
use std::path::Path;

use feathertalk_inference::MouthMasking;
use feathertalk_training::{TrainingDataset, TrainingSample};
use feathertalk_training_data::{
    FrameSample, ProjectTrainingDataset, TrainingDataError, TrainingItem,
};
use support::{
    GradientFrameReader, INNER_SIZE, downgrade_to_preparing, inner_planes, locked_project,
    mouth_rect, write_features,
};

fn open_dataset(project_dir: &Path) -> ProjectTrainingDataset<GradientFrameReader> {
    ProjectTrainingDataset::open_with_reader(project_dir, GradientFrameReader).unwrap()
}

fn single_frame(item: &TrainingItem) -> &FrameSample {
    match item {
        TrainingItem::SingleFrame(sample) => sample,
        TrainingItem::TemporalPair { .. } => panic!("expected a single-frame item"),
    }
}

fn temporal_pair(item: &TrainingItem) -> (&FrameSample, &FrameSample) {
    match item {
        TrainingItem::TemporalPair { first, second } => (first, second),
        TrainingItem::SingleFrame(_) => panic!("expected a temporal-pair item"),
    }
}

fn single_frame_sample(target_index: u64, reference_index: u64) -> TrainingSample {
    TrainingSample::SingleFrame {
        target_index,
        reference_index,
    }
}

fn temporal_sample(first: u64, second: u64, reference_index: u64) -> TrainingSample {
    TrainingSample::TemporalPair {
        first_target_index: first,
        second_target_index: second,
        reference_index,
    }
}

#[test]
fn frame_count_and_root_come_from_the_locked_project() {
    let (_temp, project_dir) = locked_project(4);
    let dataset = open_dataset(&project_dir);
    let canonical = project_dir.canonicalize().unwrap();
    assert_eq!(dataset.frame_count(), 4);
    assert_eq!(dataset.root(), canonical.as_path());
}

#[test]
fn single_frame_image_is_the_reference_then_the_masked_target() {
    let (_temp, project_dir) = locked_project(4);
    let dataset = open_dataset(&project_dir);
    let item = dataset.load_sample(&single_frame_sample(2, 0)).unwrap();
    let image = single_frame(&item).image();
    let plane = INNER_SIZE * INNER_SIZE;
    let reference = inner_planes(&project_dir, 0, MouthMasking::Keep);
    let masked = inner_planes(&project_dir, 2, MouthMasking::Blackout);
    let kept = inner_planes(&project_dir, 2, MouthMasking::Keep);
    assert_eq!(&image[..3 * plane], reference.as_slice());
    assert_eq!(&image[3 * plane..], masked.as_slice());
    assert_ne!(&image[3 * plane..], kept.as_slice());
}

#[test]
fn the_masked_half_is_black_where_the_mouth_is() {
    let (_temp, project_dir) = locked_project(4);
    let dataset = open_dataset(&project_dir);
    let item = dataset.load_sample(&single_frame_sample(2, 0)).unwrap();
    let image = single_frame(&item).image();
    let plane = INNER_SIZE * INNER_SIZE;
    let mask = mouth_rect(&project_dir, 2);
    let centre_x = (mask.x + mask.width / 2) as usize;
    let centre_y = (mask.y + mask.height / 2) as usize;
    assert_eq!(image[3 * plane + centre_y * INNER_SIZE + centre_x], 0.0);
}

#[test]
fn the_target_is_the_unmasked_target_frame() {
    let (_temp, project_dir) = locked_project(4);
    let dataset = open_dataset(&project_dir);
    let item = dataset.load_sample(&single_frame_sample(2, 0)).unwrap();
    let expected = inner_planes(&project_dir, 2, MouthMasking::Keep);
    assert_eq!(single_frame(&item).target(), expected.as_slice());
}

#[test]
fn every_tensor_has_the_length_the_losses_expect() {
    let (_temp, project_dir) = locked_project(4);
    let dataset = open_dataset(&project_dir);
    let item = dataset.load_sample(&single_frame_sample(2, 0)).unwrap();
    let sample = single_frame(&item);
    let plane = INNER_SIZE * INNER_SIZE;
    assert_eq!(sample.image().len(), 6 * plane);
    assert_eq!(sample.audio().len(), 16 * 32 * 32);
    assert_eq!(sample.target().len(), 3 * plane);
    assert_eq!(sample.mouth_mask().len(), plane);
}

#[test]
fn the_mouth_mask_is_one_inside_the_roi_and_zero_outside() {
    let (_temp, project_dir) = locked_project(4);
    let dataset = open_dataset(&project_dir);
    let item = dataset.load_sample(&single_frame_sample(2, 0)).unwrap();
    let plane = single_frame(&item).mouth_mask();
    let mask = mouth_rect(&project_dir, 2);
    let ones = plane.iter().filter(|value| **value == 1.0).count();
    let inside = (mask.y as usize) * INNER_SIZE + mask.x as usize;
    assert_eq!(ones, (mask.width * mask.height) as usize);
    assert_eq!(plane[inside], 1.0);
    assert_eq!(plane[0], 0.0);
    assert!(plane.iter().all(|value| *value == 0.0 || *value == 1.0));
}

#[test]
fn the_feature_token_count_must_match_the_frame_count() {
    let (_temp, project_dir) = locked_project(4);
    write_features(&project_dir, 3);
    let error =
        ProjectTrainingDataset::open_with_reader(&project_dir, GradientFrameReader).unwrap_err();
    assert!(matches!(
        error,
        TrainingDataError::FeatureShape {
            expected_tokens: 8,
            actual_tokens: 6,
            dims: 1024,
            ..
        }
    ));
}

#[test]
fn a_project_whose_assets_are_not_locked_is_rejected() {
    let (_temp, project_dir) = locked_project(4);
    downgrade_to_preparing(&project_dir);
    let error =
        ProjectTrainingDataset::open_with_reader(&project_dir, GradientFrameReader).unwrap_err();
    assert!(matches!(error, TrainingDataError::Project { .. }));
}

#[test]
fn a_missing_frame_file_names_its_index() {
    let (_temp, project_dir) = locked_project(4);
    let dataset = open_dataset(&project_dir);
    fs::remove_file(project_dir.join("assets/frames/000002.jpg")).unwrap();
    let error = dataset.load_sample(&single_frame_sample(2, 0)).unwrap_err();
    assert!(error.to_string().contains("unable to read frame 2"));
}

#[test]
fn a_missing_landmark_file_names_its_index() {
    let (_temp, project_dir) = locked_project(4);
    let dataset = open_dataset(&project_dir);
    fs::remove_file(project_dir.join("assets/landmarks/000002.lms")).unwrap();
    let error = dataset.load_sample(&single_frame_sample(2, 0)).unwrap_err();
    let message = error.to_string();
    assert!(message.contains("unable to read landmarks for frame 2"));
}

#[test]
fn a_corrupt_landmark_file_names_its_index() {
    let (_temp, project_dir) = locked_project(4);
    let dataset = open_dataset(&project_dir);
    let lines = vec![String::from("1 2 3"); 110];
    let path = project_dir.join("assets/landmarks/000002.lms");
    fs::write(path, lines.join("\n")).unwrap();
    let error = dataset.load_sample(&single_frame_sample(2, 0)).unwrap_err();
    let message = error.to_string();
    assert!(message.contains("unable to read landmarks for frame 2"));
}

#[test]
fn a_frame_index_past_the_end_is_rejected() {
    let (_temp, project_dir) = locked_project(4);
    let dataset = open_dataset(&project_dir);
    let error = dataset.load_sample(&single_frame_sample(4, 0)).unwrap_err();
    let message = error.to_string();
    assert!(message.contains("frame index 4 is out of range for 4 frames"));
}

#[test]
fn a_temporal_pair_shares_one_reference_frame() {
    let (_temp, project_dir) = locked_project(4);
    let dataset = open_dataset(&project_dir);
    let item = dataset.load_sample(&temporal_sample(1, 3, 0)).unwrap();
    let (first, second) = temporal_pair(&item);
    let plane = INNER_SIZE * INNER_SIZE;
    let reference = inner_planes(&project_dir, 0, MouthMasking::Keep);
    assert_eq!(&first.image()[..3 * plane], reference.as_slice());
    assert_eq!(&second.image()[..3 * plane], reference.as_slice());
}

#[test]
fn a_temporal_pair_masks_each_target_separately() {
    let (_temp, project_dir) = locked_project(4);
    let dataset = open_dataset(&project_dir);
    let item = dataset.load_sample(&temporal_sample(1, 3, 0)).unwrap();
    let (first, second) = temporal_pair(&item);
    let plane = INNER_SIZE * INNER_SIZE;
    let first_masked = inner_planes(&project_dir, 1, MouthMasking::Blackout);
    let second_masked = inner_planes(&project_dir, 3, MouthMasking::Blackout);
    assert_eq!(&first.image()[3 * plane..], first_masked.as_slice());
    assert_eq!(&second.image()[3 * plane..], second_masked.as_slice());
}

#[test]
fn a_temporal_pair_carries_two_targets_masks_and_windows() {
    let (_temp, project_dir) = locked_project(4);
    let dataset = open_dataset(&project_dir);
    let item = dataset.load_sample(&temporal_sample(1, 3, 0)).unwrap();
    let (first, second) = temporal_pair(&item);
    let first_target = inner_planes(&project_dir, 1, MouthMasking::Keep);
    let second_target = inner_planes(&project_dir, 3, MouthMasking::Keep);
    assert_eq!(first.target(), first_target.as_slice());
    assert_eq!(second.target(), second_target.as_slice());
    assert_ne!(first.mouth_mask(), second.mouth_mask());
    assert_ne!(first.audio(), second.audio());
}

#[test]
fn a_temporal_pair_rejects_an_index_past_the_end() {
    let (_temp, project_dir) = locked_project(4);
    let dataset = open_dataset(&project_dir);
    let error = dataset.load_sample(&temporal_sample(1, 9, 0)).unwrap_err();
    let message = error.to_string();
    assert!(message.contains("frame index 9 is out of range for 4 frames"));
}

#[test]
fn a_synthesised_frame_sample_keeps_the_four_planes() {
    let sample = FrameSample::new(
        vec![0.25; 6 * 160 * 160],
        vec![0.5; 16 * 32 * 32],
        vec![0.75; 3 * 160 * 160],
        vec![1.0; 160 * 160],
    )
    .expect("the four planes match the tensor contract");

    assert_eq!(sample.image().len(), 153_600);
    assert_eq!(sample.audio().len(), 16_384);
    assert_eq!(sample.target().len(), 76_800);
    assert_eq!(sample.mouth_mask().len(), 25_600);
    assert_eq!(sample.image().first().copied(), Some(0.25));
    assert_eq!(sample.mouth_mask().last().copied(), Some(1.0));
}

#[test]
fn a_plane_of_the_wrong_length_is_refused_by_name() {
    let error = FrameSample::new(
        vec![0.0; 6 * 160 * 160],
        vec![0.0; 16 * 32 * 32],
        vec![0.0; 3 * 160 * 160],
        vec![0.0; 160],
    )
    .expect_err("a truncated mouth mask cannot be stacked into a batch");

    let message = error.to_string();
    assert!(message.contains("mouth_mask"), "{message}");
    assert!(message.contains("25600"), "{message}");
    assert!(message.contains("160"), "{message}");
}
