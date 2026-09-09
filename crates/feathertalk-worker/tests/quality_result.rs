use std::path::PathBuf;

use feathertalk_frame_pipeline::{FramePipelineSpec, FrameQuality, QualityReport};
use feathertalk_worker::quality_to_json;

/// The layout the command actually produces: the video is the direct child of
/// the output root that Task 3 made legal.
fn spec() -> FramePipelineSpec {
    FramePipelineSpec::new(
        PathBuf::from(r"C:\project\assets\video_25fps.mp4"),
        PathBuf::from(r"C:\project\assets"),
        2,
        1280,
        720,
    )
    .unwrap()
}

fn frame(index: u64) -> FrameQuality {
    FrameQuality::new(
        index,
        format!("frames/{index:06}.jpg"),
        format!("landmarks/{index:06}.lms"),
        1024,
        "a".repeat(64),
        "b".repeat(64),
        0.9,
        [0.0, 0.0, 100.0, 100.0],
        30.0,
    )
    .unwrap()
}

fn report() -> QualityReport {
    QualityReport::new(2, vec![frame(0), frame(1)], Vec::new()).unwrap()
}

#[test]
fn the_payload_names_every_published_location() {
    let value = quality_to_json(&spec(), &report());

    assert_eq!(value["output_dir"], r"C:\project\assets");
    assert_eq!(value["frames_dir"], r"C:\project\assets\frames");
    assert_eq!(value["landmarks_dir"], r"C:\project\assets\landmarks");
    assert_eq!(value["quality_report"], r"C:\project\assets\quality.json");
    assert_eq!(value["frame_count"], 2);
    assert_eq!(value["frame_width"], 1280);
    assert_eq!(value["frame_height"], 720);
}

#[test]
fn the_payload_omits_the_per_frame_detail() {
    let value = quality_to_json(&spec(), &report());

    let object = value.as_object().expect("the payload is an object");
    assert_eq!(object.len(), 7);
    assert!(object.get("frames").is_none());
    assert!(object.get("anomalies").is_none());
    assert!(object.get("accepted_count").is_none());
}
