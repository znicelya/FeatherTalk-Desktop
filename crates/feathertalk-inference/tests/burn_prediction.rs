use burn::tensor::{Tensor, TensorData};
use feathertalk_audio::FeatureMatrix;
use feathertalk_inference::{
    BgrFrame, InferenceError, InferenceFramePlan, RenderGeometry, build_unet_audio_input,
    build_unet_image_input, run_unet_prediction,
};
use feathertalk_models::{
    backend::CpuBackend,
    unet::{MobileOneUnetConfig, OriginalUnetConfig, TalkingHeadModel},
};

struct OutputModel {
    shape: [usize; 4],
    value: f32,
}

impl TalkingHeadModel<CpuBackend> for OutputModel {
    fn forward_talking_head(
        &self,
        image: Tensor<CpuBackend, 4>,
        _audio: Tensor<CpuBackend, 4>,
    ) -> Tensor<CpuBackend, 4> {
        let device = image.device();
        let elements = self.shape.into_iter().product();
        Tensor::from_data(
            TensorData::new(vec![self.value; elements], self.shape),
            &device,
        )
    }
}

fn valid_inputs() -> (
    feathertalk_inference::UnetImageInput,
    feathertalk_inference::UnetAudioInput,
) {
    let crop = BgrFrame::new(168, 168, vec![64; 168 * 168 * 3]).unwrap();
    let image = build_unet_image_input(&crop, &RenderGeometry::standard()).unwrap();
    let features = FeatureMatrix::new(2, 1024, vec![0.0; 2 * 1024]).unwrap();
    let plan = InferenceFramePlan {
        output_index: 0,
        source_frame_index: 0,
        reference_frame_index: 0,
        audio_window: [None, None, None, None, Some(0), None, None, None],
    };
    let audio = build_unet_audio_input(&features, &plan).unwrap();
    (image, audio)
}

#[test]
fn prediction_returns_validated_channel_first_values() {
    let device = Default::default();
    let (image, audio) = valid_inputs();
    let values = run_unet_prediction::<CpuBackend, _>(
        &OutputModel {
            shape: [1, 3, 160, 160],
            value: 0.25,
        },
        &image,
        &audio,
        &device,
    )
    .unwrap();
    assert_eq!(values.len(), 3 * 160 * 160);
    assert!(values.iter().all(|value| *value == 0.25));
}

#[test]
fn prediction_rejects_wrong_shape_non_finite_and_out_of_range_outputs() {
    let device = Default::default();
    let (image, audio) = valid_inputs();
    for (model, expected) in [
        (
            OutputModel {
                shape: [1, 3, 80, 80],
                value: 0.5,
            },
            "shape",
        ),
        (
            OutputModel {
                shape: [1, 3, 160, 160],
                value: f32::NAN,
            },
            "finite",
        ),
        (
            OutputModel {
                shape: [1, 3, 160, 160],
                value: 1.01,
            },
            "range",
        ),
    ] {
        let error =
            run_unet_prediction::<CpuBackend, _>(&model, &image, &audio, &device).unwrap_err();
        match expected {
            "shape" => assert!(matches!(error, InferenceError::TensorShapeMismatch { .. })),
            "finite" => assert!(matches!(error, InferenceError::NonFiniteModelOutput { .. })),
            "range" => assert!(matches!(
                error,
                InferenceError::ModelOutputOutOfRange { .. }
            )),
            _ => unreachable!(),
        }
    }
}

#[test]
fn original_and_reparameterized_mobileone_run_through_the_same_adapter() {
    let device = Default::default();
    let (image, audio) = valid_inputs();
    let original = OriginalUnetConfig::parity_micro().init::<CpuBackend>(&device);
    let original_values =
        run_unet_prediction::<CpuBackend, _>(&original, &image, &audio, &device).unwrap();
    assert!(
        original_values
            .iter()
            .all(|value| (0.0..=1.0).contains(value))
    );

    let mobile = MobileOneUnetConfig::parity_micro()
        .init::<CpuBackend>(&device)
        .reparameterize();
    let mobile_values =
        run_unet_prediction::<CpuBackend, _>(&mobile, &image, &audio, &device).unwrap();
    assert!(
        mobile_values
            .iter()
            .all(|value| (0.0..=1.0).contains(value))
    );
}
