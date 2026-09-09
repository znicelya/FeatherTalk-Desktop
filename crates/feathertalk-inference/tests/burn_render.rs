use burn::tensor::{Tensor, TensorData};
use feathertalk_audio::FeatureMatrix;
use feathertalk_inference::{
    BgrFrame, InferenceError, InferenceFramePlan, RenderGeometry, render_planned_frame,
};
use feathertalk_models::{backend::CpuBackend, unet::TalkingHeadModel};
use feathertalk_preprocess::FaceBoundingBox;

struct OutputModel {
    value: f32,
}

impl TalkingHeadModel<CpuBackend> for OutputModel {
    fn forward_talking_head(
        &self,
        image: Tensor<CpuBackend, 4>,
        _audio: Tensor<CpuBackend, 4>,
    ) -> Tensor<CpuBackend, 4> {
        let device = image.device();
        Tensor::from_data(
            TensorData::new(vec![self.value; 3 * 160 * 160], [1, 3, 160, 160]),
            &device,
        )
    }
}

fn plan() -> InferenceFramePlan {
    InferenceFramePlan {
        output_index: 0,
        source_frame_index: 0,
        reference_frame_index: 0,
        audio_window: [None, None, None, None, Some(0), None, None, None],
    }
}

#[test]
fn planned_frame_reuses_the_existing_crop_prediction_and_paste_kernel() {
    let device = Default::default();
    let model = OutputModel { value: 1.0 };
    let frame = BgrFrame::new(2, 2, vec![10; 12]).unwrap();
    let original = frame.clone();
    let bbox = FaceBoundingBox {
        xmin: 0,
        ymin: 0,
        xmax: 2,
        ymax: 2,
    };
    let features = FeatureMatrix::new(2, 1024, vec![0.0; 2048]).unwrap();

    let rendered = render_planned_frame::<CpuBackend, _>(
        &model,
        &frame,
        &bbox,
        &features,
        &plan(),
        &RenderGeometry::standard(),
        &device,
    )
    .unwrap();

    assert_eq!(frame, original);
    assert_eq!(rendered.as_bytes(), &[255; 12]);
}

#[test]
fn invalid_model_output_returns_before_any_frame_can_be_published() {
    let device = Default::default();
    let model = OutputModel { value: f32::NAN };
    let frame = BgrFrame::new(2, 2, vec![10; 12]).unwrap();
    let original = frame.clone();
    let bbox = FaceBoundingBox {
        xmin: 0,
        ymin: 0,
        xmax: 2,
        ymax: 2,
    };
    let features = FeatureMatrix::new(2, 1024, vec![0.0; 2048]).unwrap();

    assert!(matches!(
        render_planned_frame::<CpuBackend, _>(
            &model,
            &frame,
            &bbox,
            &features,
            &plan(),
            &RenderGeometry::standard(),
            &device,
        ),
        Err(InferenceError::NonFiniteModelOutput { .. })
    ));
    assert_eq!(frame, original);
}
