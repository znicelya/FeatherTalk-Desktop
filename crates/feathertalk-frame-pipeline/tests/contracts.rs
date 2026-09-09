use std::path::PathBuf;

use feathertalk_frame_pipeline::{
    AnomalyCode, FrameAnomaly, FramePipelineSpec, FrameQuality, PipelineError, QualityReport,
    RecoveryAction,
};

fn spec() -> FramePipelineSpec {
    FramePipelineSpec::new(
        PathBuf::from(r"C:\media\video_25fps.mp4"),
        PathBuf::from(r"C:\project\assets"),
        3,
        640,
        480,
    )
    .unwrap()
}

#[test]
fn valid_spec_exposes_fixed_six_digit_paths() {
    let value = spec();
    assert_eq!(value.frame_count(), 3);
    assert_eq!(
        value.frame_path(0),
        PathBuf::from(r"C:\project\assets\frames\000000.jpg")
    );
    assert_eq!(
        value.frame_path(12),
        PathBuf::from(r"C:\project\assets\frames\000012.jpg")
    );
    assert_eq!(
        value.landmark_path(2),
        PathBuf::from(r"C:\project\assets\landmarks\000002.lms")
    );
}

#[test]
fn spec_rejects_zero_or_overflowing_values() {
    for (frames, width, height, field) in [
        (0, 640, 480, "frame_count"),
        (100_000_001, 640, 480, "frame_count"),
        (3, 0, 480, "image_width"),
        (3, 640, 0, "image_height"),
        (3, 32_769, 480, "image_width"),
    ] {
        assert!(matches!(
            FramePipelineSpec::new(
                PathBuf::from(r"C:\media\video.mp4"),
                PathBuf::from(r"C:\project\assets"),
                frames,
                width,
                height,
            ),
            Err(PipelineError::InvalidField { field: actual, .. }) if actual == field
        ));
    }
}

#[test]
fn anomaly_and_report_round_trip_with_strict_schema() {
    let anomaly = FrameAnomaly::new(
        2,
        AnomalyCode::MultipleFaces,
        "More than one face was detected",
        "count=2",
        RecoveryAction::ExcludeFrame,
    )
    .unwrap();
    let frame = FrameQuality::new(
        0,
        "frames/000000.jpg",
        "landmarks/000000.lms",
        10,
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        0.9,
        [0.0, 0.0, 100.0, 100.0],
        30.0,
    )
    .unwrap();
    let report = QualityReport::new(3, vec![frame], vec![anomaly]).unwrap();
    let bytes = serde_json::to_vec(&report).unwrap();
    let decoded: QualityReport = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(decoded.frame_count(), 3);
    assert_eq!(decoded.anomalies().len(), 1);
    assert!(serde_json::from_slice::<QualityReport>(
        br#"{"schema_version":1,"frame_count":0,"accepted_count":0,"frames":[],"anomalies":[],"extra":true}"#
    )
    .is_err());
}

#[test]
fn a_video_directly_under_the_output_root_is_accepted() {
    // This is the real project layout: assets/video_25fps.mp4 next to the
    // frames/ and landmarks/ directories the pipeline writes.
    let value = FramePipelineSpec::new(
        PathBuf::from(r"C:\project\assets\video_25fps.mp4"),
        PathBuf::from(r"C:\project\assets"),
        3,
        640,
        480,
    )
    .unwrap();
    assert_eq!(
        value.frame_path(0),
        PathBuf::from(r"C:\project\assets\frames\000000.jpg")
    );
}

#[test]
fn output_root_rejects_only_the_paths_the_pipeline_owns() {
    for source in [
        r"C:\project\assets",
        r"C:\project\assets\frames\000000.jpg",
        r"C:\project\assets\landmarks\000000.lms",
        r"C:\project\assets\quality.json",
    ] {
        let result = FramePipelineSpec::new(
            PathBuf::from(source),
            PathBuf::from(r"C:\project\assets"),
            3,
            640,
            480,
        );
        assert!(
            matches!(
                result,
                Err(PipelineError::InvalidField {
                    field: "output_root",
                    ..
                })
            ),
            "{source} must be rejected"
        );
    }
}

#[test]
fn valid_spec_exposes_the_two_artifact_directories() {
    let value = spec();
    assert_eq!(
        value.frames_dir(),
        PathBuf::from(r"C:\project\assets\frames")
    );
    assert_eq!(
        value.landmarks_dir(),
        PathBuf::from(r"C:\project\assets\landmarks")
    );
    assert_eq!(value.frames_dir().join("000000.jpg"), value.frame_path(0));
    assert_eq!(
        value.landmarks_dir().join("000002.lms"),
        value.landmark_path(2)
    );
}
