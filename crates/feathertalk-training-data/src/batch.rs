use burn::tensor::{Tensor, TensorData, backend::Backend};

use crate::{FrameSample, TrainingDataError, TrainingItem};

const INNER_SIZE: usize = 160;
const IMAGE_CHANNELS: usize = 6;
const TARGET_CHANNELS: usize = 3;
const AUDIO_CHANNELS: usize = 16;
const AUDIO_SIZE: usize = 32;

/// One batch of single-frame items, ready for a U-Net forward pass.
#[derive(Debug, Clone)]
pub struct SingleFrameBatch<B: Backend> {
    pub image: Tensor<B, 4>,
    pub audio: Tensor<B, 4>,
    pub target: Tensor<B, 4>,
    pub mouth_mask: Tensor<B, 4>,
}

/// One batch of temporal pairs: the inputs are flattened, the targets keep the pair axis.
#[derive(Debug, Clone)]
pub struct TemporalBatch<B: Backend> {
    pub image: Tensor<B, 4>,
    pub audio: Tensor<B, 4>,
    pub target: Tensor<B, 5>,
    pub mouth_mask: Tensor<B, 5>,
}

fn batch_error(message: String) -> TrainingDataError {
    TrainingDataError::Batch { message }
}

fn single_frame_samples(items: &[TrainingItem]) -> Result<Vec<&FrameSample>, TrainingDataError> {
    if items.is_empty() {
        return Err(batch_error("a batch needs at least one item".to_owned()));
    }
    let mut samples = Vec::with_capacity(items.len());
    for (position, item) in items.iter().enumerate() {
        match item {
            TrainingItem::SingleFrame(sample) => samples.push(sample),
            TrainingItem::TemporalPair { .. } => {
                return Err(batch_error(format!(
                    "item {position} is a temporal pair but the batch is single-frame"
                )));
            }
        }
    }
    Ok(samples)
}

fn temporal_samples(items: &[TrainingItem]) -> Result<Vec<&FrameSample>, TrainingDataError> {
    if items.is_empty() {
        return Err(batch_error("a batch needs at least one item".to_owned()));
    }
    let mut samples = Vec::with_capacity(items.len().saturating_mul(2));
    for (position, item) in items.iter().enumerate() {
        match item {
            TrainingItem::TemporalPair { first, second } => {
                samples.push(first);
                samples.push(second);
            }
            TrainingItem::SingleFrame(_) => {
                return Err(batch_error(format!(
                    "item {position} is a single frame but the batch is temporal"
                )));
            }
        }
    }
    Ok(samples)
}

fn gather(
    samples: &[&FrameSample],
    field: fn(&FrameSample) -> &[f32],
    expected: usize,
) -> Result<Vec<f32>, TrainingDataError> {
    let Some(elements) = samples.len().checked_mul(expected) else {
        return Err(batch_error("the batch element count overflows".to_owned()));
    };
    let mut values: Vec<f32> = Vec::new();
    values
        .try_reserve_exact(elements)
        .map_err(|_| batch_error(format!("cannot allocate {elements} floats")))?;
    for (position, sample) in samples.iter().enumerate() {
        let plane = field(sample);
        if plane.len() != expected {
            return Err(batch_error(format!(
                "item {position} has {} values but {expected} were expected",
                plane.len()
            )));
        }
        values.extend_from_slice(plane);
    }
    Ok(values)
}

fn tensor4<B: Backend>(values: Vec<f32>, shape: [usize; 4], device: &B::Device) -> Tensor<B, 4> {
    Tensor::<B, 4>::from_data(TensorData::new(values, shape), device)
}

fn tensor5<B: Backend>(values: Vec<f32>, shape: [usize; 5], device: &B::Device) -> Tensor<B, 5> {
    Tensor::<B, 5>::from_data(TensorData::new(values, shape), device)
}

/// Stacks single-frame items in the order they were given.
pub fn stack_single_frame_batch<B: Backend>(
    items: &[TrainingItem],
    device: &B::Device,
) -> Result<SingleFrameBatch<B>, TrainingDataError> {
    let samples = single_frame_samples(items)?;
    let count = samples.len();
    let plane = INNER_SIZE * INNER_SIZE;
    let audio = AUDIO_CHANNELS * AUDIO_SIZE * AUDIO_SIZE;
    let image_values = gather(&samples, FrameSample::image, IMAGE_CHANNELS * plane)?;
    let audio_values = gather(&samples, FrameSample::audio, audio)?;
    let target_values = gather(&samples, FrameSample::target, TARGET_CHANNELS * plane)?;
    let mask_values = gather(&samples, FrameSample::mouth_mask, plane)?;
    let image_shape = [count, IMAGE_CHANNELS, INNER_SIZE, INNER_SIZE];
    let audio_shape = [count, AUDIO_CHANNELS, AUDIO_SIZE, AUDIO_SIZE];
    let target_shape = [count, TARGET_CHANNELS, INNER_SIZE, INNER_SIZE];
    let mask_shape = [count, 1, INNER_SIZE, INNER_SIZE];
    Ok(SingleFrameBatch {
        image: tensor4(image_values, image_shape, device),
        audio: tensor4(audio_values, audio_shape, device),
        target: tensor4(target_values, target_shape, device),
        mouth_mask: tensor4(mask_values, mask_shape, device),
    })
}

/// Stacks temporal pairs sample-major, so `temporal_loss` can reshape the flattened rows back.
pub fn stack_temporal_batch<B: Backend>(
    items: &[TrainingItem],
    device: &B::Device,
) -> Result<TemporalBatch<B>, TrainingDataError> {
    let samples = temporal_samples(items)?;
    let pairs = items.len();
    let halves = samples.len();
    let plane = INNER_SIZE * INNER_SIZE;
    let audio = AUDIO_CHANNELS * AUDIO_SIZE * AUDIO_SIZE;
    let image_values = gather(&samples, FrameSample::image, IMAGE_CHANNELS * plane)?;
    let audio_values = gather(&samples, FrameSample::audio, audio)?;
    let target_values = gather(&samples, FrameSample::target, TARGET_CHANNELS * plane)?;
    let mask_values = gather(&samples, FrameSample::mouth_mask, plane)?;
    let image_shape = [halves, IMAGE_CHANNELS, INNER_SIZE, INNER_SIZE];
    let audio_shape = [halves, AUDIO_CHANNELS, AUDIO_SIZE, AUDIO_SIZE];
    let target_shape = [pairs, 2, TARGET_CHANNELS, INNER_SIZE, INNER_SIZE];
    let mask_shape = [pairs, 2, 1, INNER_SIZE, INNER_SIZE];
    Ok(TemporalBatch {
        image: tensor4(image_values, image_shape, device),
        audio: tensor4(audio_values, audio_shape, device),
        target: tensor5(target_values, target_shape, device),
        mouth_mask: tensor5(mask_values, mask_shape, device),
    })
}
