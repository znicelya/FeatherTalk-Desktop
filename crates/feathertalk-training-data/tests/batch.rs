mod support;

use std::path::Path;

use feathertalk_training::{TrainingDataset, TrainingSample};
use feathertalk_training_data::{
    FrameSample, ProjectTrainingDataset, TrainingDataError, TrainingItem, stack_single_frame_batch,
    stack_temporal_batch,
};
use support::{GradientFrameReader, INNER_SIZE, locked_project};

type CpuBackend = burn::backend::NdArray<f32>;

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

fn single_frame_items(project_dir: &Path) -> Vec<TrainingItem> {
    let dataset = open_dataset(project_dir);
    vec![
        dataset.load_sample(&single_frame_sample(1, 0)).unwrap(),
        dataset.load_sample(&single_frame_sample(2, 0)).unwrap(),
    ]
}

fn temporal_items(project_dir: &Path) -> Vec<TrainingItem> {
    let dataset = open_dataset(project_dir);
    vec![
        dataset.load_sample(&temporal_sample(1, 2, 0)).unwrap(),
        dataset.load_sample(&temporal_sample(3, 1, 0)).unwrap(),
    ]
}

#[test]
fn a_single_frame_batch_has_one_row_per_item() {
    let (_temp, project_dir) = locked_project(4);
    let items = single_frame_items(&project_dir);
    let device = Default::default();
    let batch = stack_single_frame_batch::<CpuBackend>(&items, &device).unwrap();
    assert_eq!(batch.image.dims(), [2, 6, INNER_SIZE, INNER_SIZE]);
    assert_eq!(batch.audio.dims(), [2, 16, 32, 32]);
    assert_eq!(batch.target.dims(), [2, 3, INNER_SIZE, INNER_SIZE]);
    assert_eq!(batch.mouth_mask.dims(), [2, 1, INNER_SIZE, INNER_SIZE]);
}

#[test]
fn a_single_frame_batch_keeps_the_item_order() {
    let (_temp, project_dir) = locked_project(4);
    let items = single_frame_items(&project_dir);
    let device = Default::default();
    let batch = stack_single_frame_batch::<CpuBackend>(&items, &device).unwrap();
    let values = batch.image.into_data().to_vec::<f32>().unwrap();
    let stride = 6 * INNER_SIZE * INNER_SIZE;
    assert_eq!(&values[..stride], single_frame(&items[0]).image());
    assert_eq!(&values[stride..], single_frame(&items[1]).image());
}

#[test]
fn a_temporal_batch_flattens_both_halves() {
    let (_temp, project_dir) = locked_project(4);
    let items = temporal_items(&project_dir);
    let device = Default::default();
    let batch = stack_temporal_batch::<CpuBackend>(&items, &device).unwrap();
    assert_eq!(batch.image.dims(), [4, 6, INNER_SIZE, INNER_SIZE]);
    assert_eq!(batch.audio.dims(), [4, 16, 32, 32]);
    assert_eq!(batch.target.dims(), [2, 2, 3, INNER_SIZE, INNER_SIZE]);
    assert_eq!(batch.mouth_mask.dims(), [2, 2, 1, INNER_SIZE, INNER_SIZE]);
}

#[test]
fn a_temporal_batch_is_sample_major() {
    let (_temp, project_dir) = locked_project(4);
    let items = temporal_items(&project_dir);
    let device = Default::default();
    let batch = stack_temporal_batch::<CpuBackend>(&items, &device).unwrap();
    let values = batch.target.into_data().to_vec::<f32>().unwrap();
    let stride = 3 * INNER_SIZE * INNER_SIZE;
    let (first, second) = temporal_pair(&items[0]);
    let (third, fourth) = temporal_pair(&items[1]);
    assert_eq!(&values[..stride], first.target());
    assert_eq!(&values[stride..2 * stride], second.target());
    assert_eq!(&values[2 * stride..3 * stride], third.target());
    assert_eq!(&values[3 * stride..], fourth.target());
}

#[test]
fn an_empty_batch_is_rejected() {
    let device = Default::default();
    let error = stack_single_frame_batch::<CpuBackend>(&[], &device).unwrap_err();
    let message = error.to_string();
    assert!(matches!(error, TrainingDataError::Batch { .. }));
    assert!(message.contains("a batch needs at least one item"));
}

#[test]
fn a_batch_of_mixed_item_kinds_is_rejected() {
    let (_temp, project_dir) = locked_project(4);
    let dataset = open_dataset(&project_dir);
    let items = vec![
        dataset.load_sample(&single_frame_sample(1, 0)).unwrap(),
        dataset.load_sample(&temporal_sample(2, 3, 0)).unwrap(),
    ];
    let device = Default::default();
    let single = stack_single_frame_batch::<CpuBackend>(&items, &device).unwrap_err();
    assert!(single.to_string().contains("item 1 is a temporal pair"));
    let temporal = stack_temporal_batch::<CpuBackend>(&items, &device).unwrap_err();
    assert!(temporal.to_string().contains("item 0 is a single frame"));
}
