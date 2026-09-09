use burn::tensor::{Tensor, backend::Backend};
use feathertalk_models::unet::TrainableTalkingHead;
use feathertalk_training::{PreviewArtifact, TrainingDataset, TrainingError, TrainingSample};
use feathertalk_training_data::{TrainingItem, stack_single_frame_batch};

/// Renders one single-frame sample into a wire-ready preview artifact.
#[allow(clippy::too_many_arguments)]
pub fn build_preview_artifact<B, M, D>(
    model: &M,
    dataset: &D,
    device: &B::Device,
    sample: &TrainingSample,
    epoch: u64,
    global_step: u64,
    model_kind: &str,
    model_config_sha256: &str,
    worker_state: &str,
) -> Result<PreviewArtifact, TrainingError>
where
    B: Backend,
    M: TrainableTalkingHead<B>,
    D: TrainingDataset<Item = TrainingItem>,
{
    let TrainingSample::SingleFrame {
        target_index,
        reference_index,
    } = sample
    else {
        return Err(TrainingError::InvalidInput(
            "a preview needs a single-frame sample".to_owned(),
        ));
    };

    let item = dataset.load_sample(sample)?;
    let batch = stack_single_frame_batch::<B>(&[item], device)?;
    let prediction = model.forward_training(batch.image, batch.audio).detach();
    let mouth_roi = prediction.clone() * batch.mouth_mask;

    PreviewArtifact::new(
        *target_index,
        *reference_index,
        epoch,
        global_step,
        model_kind,
        model_config_sha256,
        worker_state,
        preview_values(prediction, "prediction")?,
        preview_values(batch.target, "target")?,
        preview_values(mouth_roi, "mouth_roi")?,
    )
}

fn preview_values<B: Backend>(
    tensor: Tensor<B, 4>,
    context: &str,
) -> Result<Vec<f32>, TrainingError> {
    tensor
        .into_data()
        .to_vec::<f32>()
        .map_err(|error| TrainingError::InvalidInput(format!("preview {context}: {error}")))
}
