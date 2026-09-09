use std::path::PathBuf;

use feathertalk_training::TrainingError;
use feathertalk_training_data::TrainingDataError;

fn cases() -> Vec<TrainingDataError> {
    vec![
        TrainingDataError::Project {
            path: PathBuf::from("project"),
            message: "asset package is not locked".to_owned(),
        },
        TrainingDataError::Features {
            path: PathBuf::from("project/assets/features/feather_hubert.f32"),
            message: "unexpected end of file".to_owned(),
        },
        TrainingDataError::FeatureShape {
            path: PathBuf::from("project/assets/features/feather_hubert.f32"),
            expected_tokens: 24,
            actual_tokens: 22,
            dims: 1024,
        },
        TrainingDataError::FrameIndexOutOfRange {
            index: 12,
            frame_count: 12,
        },
        TrainingDataError::Frame {
            index: 3,
            path: PathBuf::from("project/assets/frames/000003.jpg"),
            message: "not a file".to_owned(),
        },
        TrainingDataError::Landmarks {
            index: 3,
            path: PathBuf::from("project/assets/landmarks/000003.lms"),
            message: "wrong landmark count".to_owned(),
        },
        TrainingDataError::Sample {
            index: 3,
            message: "inner planes rejected the crop".to_owned(),
        },
        TrainingDataError::Batch {
            message: "batch is empty".to_owned(),
        },
    ]
}

#[test]
fn feature_shape_names_the_file_and_both_token_counts() {
    let message = cases()
        .into_iter()
        .map(|case| case.to_string())
        .find(|message| message.contains("feather_hubert.f32") && message.contains("token"))
        .unwrap();
    assert!(message.contains("24"), "{message}");
    assert!(message.contains("22"), "{message}");
    assert!(message.contains("1024"), "{message}");
}

#[test]
fn frame_and_landmark_errors_name_the_frame_index_and_path() {
    for case in cases() {
        let message = case.to_string();
        match case {
            TrainingDataError::Frame { index, path, .. }
            | TrainingDataError::Landmarks { index, path, .. } => {
                assert!(message.contains(&index.to_string()), "{message}");
                let name = path.file_name().unwrap().to_string_lossy().into_owned();
                assert!(message.contains(&name), "{message}");
            }
            _ => {}
        }
    }
}

#[test]
fn every_variant_maps_to_invalid_training_input() {
    let mut count = 0;
    for case in cases() {
        let expected = case.to_string();
        match TrainingError::from(case) {
            TrainingError::InvalidInput(message) => assert_eq!(message, expected),
            other => panic!("expected InvalidInput, got {other:?}"),
        }
        count += 1;
    }
    assert_eq!(count, 8);
}
