use burn::tensor::{Tensor, TensorData, backend::Backend};
use feathertalk_audio::FeatureMatrix;
use feathertalk_models::unet::TalkingHeadModel;

use crate::{
    BgrFrame, InferenceError, InferenceFramePlan, RenderGeometry, build_face_crop,
    build_unet_image_input, render_frame,
};

const FEATURE_DIMS: usize = 1024;
const TOKENS_PER_FRAME: usize = 2;
const AUDIO_VALUES_PER_SLOT: usize = TOKENS_PER_FRAME * FEATURE_DIMS;
const UNET_AUDIO_VALUES: usize = 16 * 32 * 32;
const UNET_IMAGE_SHAPE: [usize; 4] = [1, 6, 160, 160];
const UNET_AUDIO_SHAPE: [usize; 4] = [1, 16, 32, 32];
const UNET_OUTPUT_SHAPE: [usize; 4] = [1, 3, 160, 160];

#[derive(Debug, Clone, PartialEq)]
pub struct UnetAudioInput {
    values: Vec<f32>,
}

impl UnetAudioInput {
    pub fn shape(&self) -> [usize; 4] {
        [1, 16, 32, 32]
    }

    pub fn as_slice(&self) -> &[f32] {
        &self.values
    }
}

pub fn build_unet_audio_window(
    features: &FeatureMatrix,
    audio_window: &[Option<usize>; 8],
) -> Result<UnetAudioInput, InferenceError> {
    let frame_count = feature_frame_count(features)?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(UNET_AUDIO_VALUES)
        .map_err(|_| InferenceError::AllocationFailure {
            bytes: UNET_AUDIO_VALUES * std::mem::size_of::<f32>(),
        })?;
    values.resize(UNET_AUDIO_VALUES, 0.0);

    for (slot, frame_index) in audio_window.iter().copied().enumerate() {
        let Some(frame_index) = frame_index else {
            continue;
        };
        if frame_index >= frame_count {
            return Err(InferenceError::InvalidAudioWindowIndex {
                slot,
                index: frame_index,
                frame_count,
            });
        }

        let source_start = frame_index
            .checked_mul(AUDIO_VALUES_PER_SLOT)
            .ok_or(InferenceError::ArithmeticOverflow)?;
        let destination_start = slot
            .checked_mul(AUDIO_VALUES_PER_SLOT)
            .ok_or(InferenceError::ArithmeticOverflow)?;
        let source_end = source_start
            .checked_add(AUDIO_VALUES_PER_SLOT)
            .ok_or(InferenceError::ArithmeticOverflow)?;
        let destination_end = destination_start
            .checked_add(AUDIO_VALUES_PER_SLOT)
            .ok_or(InferenceError::ArithmeticOverflow)?;
        values[destination_start..destination_end]
            .copy_from_slice(&features.values()[source_start..source_end]);
    }

    Ok(UnetAudioInput { values })
}

pub fn build_unet_audio_input(
    features: &FeatureMatrix,
    plan: &InferenceFramePlan,
) -> Result<UnetAudioInput, InferenceError> {
    let frame_count = feature_frame_count(features)?;
    if plan.output_index >= frame_count {
        return Err(InferenceError::OutputFrameOutOfRange {
            index: plan.output_index,
            count: frame_count,
        });
    }
    build_unet_audio_window(features, &plan.audio_window)
}

fn feature_frame_count(features: &FeatureMatrix) -> Result<usize, InferenceError> {
    let tokens = features.tokens();
    let dims = features.dims();
    if dims != FEATURE_DIMS || tokens == 0 || !tokens.is_multiple_of(TOKENS_PER_FRAME) {
        return Err(InferenceError::InvalidFeatureShape { tokens, dims });
    }
    Ok(tokens / TOKENS_PER_FRAME)
}

pub fn run_unet_prediction<B, M>(
    model: &M,
    image: &crate::UnetImageInput,
    audio: &UnetAudioInput,
    device: &B::Device,
) -> Result<Vec<f32>, InferenceError>
where
    B: Backend,
    M: TalkingHeadModel<B>,
{
    validate_shape("unet_image_input", image.shape(), UNET_IMAGE_SHAPE)?;
    validate_shape("unet_audio_input", audio.shape(), UNET_AUDIO_SHAPE)?;
    ensure_finite(image.as_slice(), "image")?;
    ensure_finite(audio.as_slice(), "audio")?;

    let image_tensor = Tensor::<B, 4>::from_data(
        TensorData::new(image.as_slice().to_vec(), UNET_IMAGE_SHAPE),
        device,
    );
    let audio_tensor = Tensor::<B, 4>::from_data(
        TensorData::new(audio.as_slice().to_vec(), UNET_AUDIO_SHAPE),
        device,
    );
    let output = model.forward_talking_head(image_tensor, audio_tensor);
    validate_shape("unet_output", output.dims(), UNET_OUTPUT_SHAPE)?;
    let values =
        output
            .into_data()
            .to_vec::<f32>()
            .map_err(|error| InferenceError::ModelTensorData {
                context: "unet_output",
                message: error.to_string(),
            })?;
    if values.len() != 3 * 160 * 160 {
        return Err(InferenceError::TensorShapeMismatch {
            context: "unet_output",
            expected: UNET_OUTPUT_SHAPE.to_vec(),
            actual: vec![values.len()],
        });
    }
    if let Some(index) = values.iter().position(|value| !value.is_finite()) {
        return Err(InferenceError::NonFiniteModelOutput { index });
    }
    if let Some((index, value)) = values
        .iter()
        .copied()
        .enumerate()
        .find(|(_, value)| !(0.0..=1.0).contains(value))
    {
        return Err(InferenceError::ModelOutputOutOfRange { index, value });
    }
    Ok(values)
}

pub fn render_planned_frame<B, M>(
    model: &M,
    frame: &BgrFrame,
    bbox: &feathertalk_preprocess::FaceBoundingBox,
    features: &FeatureMatrix,
    plan: &InferenceFramePlan,
    geometry: &RenderGeometry,
    device: &B::Device,
) -> Result<BgrFrame, InferenceError>
where
    B: Backend,
    M: TalkingHeadModel<B>,
{
    let audio = build_unet_audio_input(features, plan)?;
    let face_crop = build_face_crop(frame, bbox, geometry)?;
    let image = build_unet_image_input(&face_crop, geometry)?;
    let prediction = run_unet_prediction::<B, M>(model, &image, &audio, device)?;
    render_frame(frame, bbox, &prediction, geometry)
}

fn validate_shape(
    context: &'static str,
    actual: [usize; 4],
    expected: [usize; 4],
) -> Result<(), InferenceError> {
    if actual != expected {
        return Err(InferenceError::TensorShapeMismatch {
            context,
            expected: expected.to_vec(),
            actual: actual.to_vec(),
        });
    }
    Ok(())
}

fn ensure_finite(values: &[f32], context: &'static str) -> Result<(), InferenceError> {
    if let Some(index) = values.iter().position(|value| !value.is_finite()) {
        return Err(InferenceError::NonFiniteModelInput { context, index });
    }
    Ok(())
}
