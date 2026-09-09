mod support;

use std::fs;
use std::path::{Path, PathBuf};

use feathertalk_training::{TrainingDataset, TrainingError, TrainingSample};
use feathertalk_training_data::{ProjectTrainingDataset, TrainingItem};
use support::{FixtureSpec, INNER_SIZE, build_locked_project};

fn demo_frame() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../feathertalk-frame-adapters/tests/fixtures/demo_frame_v1/frame.jpg")
}

fn real_frame_spec(manifest_width: u32, manifest_height: u32) -> FixtureSpec {
    FixtureSpec {
        frame_count: 2,
        manifest_width,
        manifest_height,
        frame_bytes: fs::read(demo_frame()).unwrap(),
        face_xmin: 551,
        face_xmax: 710,
        face_ymin: 194,
        mouth_x: 600,
        mouth_y: 230,
    }
}

fn target_one() -> TrainingSample {
    TrainingSample::SingleFrame {
        target_index: 1,
        reference_index: 0,
    }
}

#[test]
fn opens_a_project_whose_frames_are_real_jpeg_files() {
    let (_temp, project_dir) = build_locked_project(&real_frame_spec(1280, 720));
    let dataset = ProjectTrainingDataset::open(&project_dir).unwrap();
    assert_eq!(dataset.frame_count(), 2);
    let item = dataset.load_sample(&target_one()).unwrap();
    let TrainingItem::SingleFrame(frame) = item else {
        panic!("expected a single-frame item");
    };
    assert_eq!(frame.image().len(), 6 * INNER_SIZE * INNER_SIZE);
    assert_eq!(frame.target().len(), 3 * INNER_SIZE * INNER_SIZE);
    assert_eq!(frame.mouth_mask().len(), INNER_SIZE * INNER_SIZE);
    assert!(
        frame
            .image()
            .iter()
            .all(|value| (0.0..=1.0).contains(value))
    );
}

#[test]
fn a_frame_that_contradicts_the_manifest_is_rejected() {
    let (_temp, project_dir) = build_locked_project(&real_frame_spec(640, 480));
    let dataset = ProjectTrainingDataset::open(&project_dir).unwrap();
    let error = dataset.load_sample(&target_one()).unwrap_err();
    let message = error.to_string();
    assert!(matches!(error, TrainingError::InvalidInput(_)));
    assert!(message.contains("frame is 1280x720 but the asset package declares 640x480"));
}
