mod fixture;
mod support;

use feathertalk_training::{
    PREVIEW_TENSOR_ELEMENTS, PREVIEW_TENSOR_SHAPE, TrainingDataset, TrainingSample,
    read_preview_artifact, write_preview_artifact,
};
use feathertalk_training_data::TrainingItem;
use feathertalk_training_run::build_preview_artifact;
use fixture::{dataset, locked_project};
use support::{CpuAutodiffBackend, CpuDevice, model, on_step_stack};

const PLANE: usize = 160 * 160;

fn sha256() -> String {
    "a".repeat(64)
}

fn single_frame() -> TrainingSample {
    TrainingSample::SingleFrame {
        target_index: 1,
        reference_index: 0,
    }
}

#[test]
fn the_preview_masks_the_prediction_with_the_mouth_roi() {
    on_step_stack("preview-mask", || {
        let device = CpuDevice::default();
        let (_temp, project_dir) = locked_project(4);
        let data = dataset(&project_dir);
        let unet = model(&device);
        let sample = single_frame();

        let artifact = build_preview_artifact::<CpuAutodiffBackend, _, _>(
            &unet,
            &data,
            &device,
            &sample,
            3,
            12,
            "original-unet",
            &sha256(),
            "training",
        )
        .unwrap();

        assert_eq!(artifact.sample_index(), 1);
        assert_eq!(artifact.reference_index(), 0);
        assert_eq!(artifact.epoch(), 3);
        assert_eq!(artifact.global_step(), 12);
        assert_eq!(artifact.model_kind(), "original-unet");
        assert_eq!(artifact.model_config_sha256(), sha256());
        assert_eq!(artifact.worker_state(), "training");
        assert_eq!(artifact.shape(), PREVIEW_TENSOR_SHAPE);
        assert_eq!(artifact.prediction().len(), PREVIEW_TENSOR_ELEMENTS);
        assert_eq!(artifact.mouth_roi().len(), PREVIEW_TENSOR_ELEMENTS);

        let TrainingItem::SingleFrame(frame) = data.load_sample(&sample).unwrap() else {
            panic!("a single-frame sample must load a single frame");
        };
        assert_eq!(artifact.target(), frame.target());

        let mask = frame.mouth_mask();
        assert_eq!(mask.len(), PLANE);
        let mut inside = 0_usize;
        let mut outside = 0_usize;
        for channel in 0..3 {
            for (index, mask_value) in mask.iter().enumerate() {
                let masked = artifact.mouth_roi()[channel * PLANE + index];
                if *mask_value == 0.0 {
                    assert_eq!(masked, 0.0);
                    outside += 1;
                } else {
                    assert_eq!(masked, artifact.prediction()[channel * PLANE + index]);
                    inside += 1;
                }
            }
        }
        assert!(inside > 0, "the mouth mask must cover some pixels");
        assert!(outside > 0, "the mouth mask must exclude some pixels");
    });
}

#[test]
fn a_temporal_sample_has_no_preview() {
    on_step_stack("preview-temporal", || {
        let device = CpuDevice::default();
        let (_temp, project_dir) = locked_project(4);
        let data = dataset(&project_dir);
        let unet = model(&device);
        let sample = TrainingSample::TemporalPair {
            first_target_index: 1,
            second_target_index: 2,
            reference_index: 0,
        };

        let error = build_preview_artifact::<CpuAutodiffBackend, _, _>(
            &unet,
            &data,
            &device,
            &sample,
            0,
            1,
            "original-unet",
            &sha256(),
            "training",
        )
        .unwrap_err();

        assert_eq!(
            error.to_string(),
            "invalid training input: a preview needs a single-frame sample"
        );
    });
}

#[test]
fn the_preview_round_trips_through_disk() {
    on_step_stack("preview-round-trip", || {
        let device = CpuDevice::default();
        let (_temp, project_dir) = locked_project(4);
        let data = dataset(&project_dir);
        let unet = model(&device);

        let artifact = build_preview_artifact::<CpuAutodiffBackend, _, _>(
            &unet,
            &data,
            &device,
            &single_frame(),
            0,
            1,
            "original-unet",
            &sha256(),
            "training",
        )
        .unwrap();

        let root = tempfile::tempdir().unwrap();
        let destination = root.path().join("preview-000001");
        let manifest = write_preview_artifact(&destination, &artifact).unwrap();
        assert_eq!(manifest.shape, PREVIEW_TENSOR_SHAPE);

        let (loaded, loaded_manifest) =
            read_preview_artifact(&destination, "original-unet", &sha256()).unwrap();
        assert_eq!(loaded, artifact);
        assert_eq!(loaded_manifest, manifest);
    });
}
