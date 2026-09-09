use feathertalk_face::{Detection, DetectionConfig, FaceError, non_max_suppression};

fn detection(bbox: [f32; 4], score: f32) -> Detection {
    Detection {
        bbox,
        score,
        keypoints: [[0.0; 2]; 5],
    }
}

#[test]
fn default_thresholds_are_schema_one_values() {
    let config = DetectionConfig::default();
    assert_eq!(config.confidence_threshold, 0.1);
    assert_eq!(config.nms_iou_threshold, 0.5);
}

#[test]
fn filters_low_scores_and_keeps_threshold_equality() {
    let detections = [
        detection([0.0, 0.0, 10.0, 10.0], 0.099),
        detection([20.0, 20.0, 10.0, 10.0], 0.1),
    ];
    assert_eq!(
        non_max_suppression(&detections, &DetectionConfig::default()).unwrap(),
        vec![1]
    );
}

#[test]
fn suppresses_overlap_above_threshold_and_retains_non_overlap() {
    let detections = [
        detection([0.0, 0.0, 10.0, 10.0], 0.9),
        detection([1.0, 1.0, 10.0, 10.0], 0.8),
        detection([20.0, 20.0, 10.0, 10.0], 0.7),
    ];
    assert_eq!(
        non_max_suppression(&detections, &DetectionConfig::default()).unwrap(),
        vec![0, 2]
    );
}

#[test]
fn equality_at_iou_threshold_is_retained() {
    let detections = [
        detection([0.0, 0.0, 2.0, 2.0], 0.9),
        detection([0.0, 0.0, 1.0, 2.0], 0.8),
    ];
    let config = DetectionConfig {
        confidence_threshold: 0.0,
        nms_iou_threshold: 0.5,
    };
    let kept = non_max_suppression(&detections, &config).unwrap();
    assert_eq!(kept, vec![0, 1]);
}

#[test]
fn equal_scores_use_original_index_as_tie_breaker() {
    let detections = [
        detection([20.0, 20.0, 5.0, 5.0], 0.8),
        detection([0.0, 0.0, 5.0, 5.0], 0.8),
    ];
    assert_eq!(
        non_max_suppression(&detections, &DetectionConfig::default()).unwrap(),
        vec![0, 1]
    );
}

#[test]
fn rejects_invalid_thresholds() {
    for config in [
        DetectionConfig {
            confidence_threshold: f32::NAN,
            nms_iou_threshold: 0.5,
        },
        DetectionConfig {
            confidence_threshold: 0.1,
            nms_iou_threshold: f32::INFINITY,
        },
        DetectionConfig {
            confidence_threshold: -0.1,
            nms_iou_threshold: 0.5,
        },
        DetectionConfig {
            confidence_threshold: 0.1,
            nms_iou_threshold: 1.1,
        },
    ] {
        assert!(matches!(
            non_max_suppression(&[], &config),
            Err(FaceError::InvalidConfiguration { .. })
        ));
    }
}

#[test]
fn rejects_non_finite_values_and_non_positive_geometry() {
    for detection in [
        detection([0.0, 0.0, 1.0, 1.0], f32::NAN),
        detection([f32::INFINITY, 0.0, 1.0, 1.0], 0.9),
        detection([0.0, 0.0, 0.0, 1.0], 0.9),
    ] {
        assert!(matches!(
            non_max_suppression(&[detection], &DetectionConfig::default()),
            Err(FaceError::NonFiniteValue { .. }) | Err(FaceError::InvalidDetectionGeometry { .. })
        ));
    }
}
