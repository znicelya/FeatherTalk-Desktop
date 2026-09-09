use std::fs;

use feathertalk_frame_pipeline::{
    FrameAnomaly, FrameQuality, PipelineError, QualityReport, RecoveryAction, read_quality_report,
};

fn valid_report() -> QualityReport {
    QualityReport::new(
        1,
        vec![
            FrameQuality::new(
                0,
                "frames/000000.jpg",
                "landmarks/000000.lms",
                3,
                "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
                0.9,
                [0.0, 0.0, 100.0, 100.0],
                30.0,
            )
            .unwrap(),
        ],
        Vec::new(),
    )
    .unwrap()
}

#[test]
fn reads_valid_report_and_rejects_unknown_fields() {
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("quality.json");
    fs::write(&path, serde_json::to_vec(&valid_report()).unwrap()).unwrap();
    assert_eq!(read_quality_report(&path).unwrap(), valid_report());

    let mut value: serde_json::Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
    value["unexpected"] = serde_json::json!(true);
    fs::write(&path, serde_json::to_vec(&value).unwrap()).unwrap();
    assert!(matches!(
        read_quality_report(&path),
        Err(PipelineError::ReportJson { .. })
    ));
}

#[test]
fn rejects_malformed_and_semantically_invalid_reports() {
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("quality.json");

    fs::write(&path, b"not json").unwrap();
    assert!(matches!(
        read_quality_report(&path),
        Err(PipelineError::ReportJson { .. })
    ));

    let mut value: serde_json::Value = serde_json::to_value(valid_report()).unwrap();
    value["accepted_count"] = serde_json::json!(99);
    fs::write(&path, serde_json::to_vec(&value).unwrap()).unwrap();
    assert!(matches!(
        read_quality_report(&path),
        Err(PipelineError::InvalidReport { .. })
    ));
}

#[test]
fn rejects_report_larger_than_bound_before_json_decode() {
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("quality.json");
    let bytes = vec![b' '; 16 * 1024 * 1024 + 1];
    fs::write(&path, bytes).unwrap();
    assert!(matches!(
        read_quality_report(&path),
        Err(PipelineError::ReportTooLarge { .. })
    ));
}

#[test]
fn rejects_symlink_report_without_following_it() {
    let root = tempfile::tempdir().unwrap();
    let target = root.path().join("target.json");
    let link = root.path().join("quality.json");
    fs::write(&target, serde_json::to_vec(&valid_report()).unwrap()).unwrap();
    #[cfg(windows)]
    match std::os::windows::fs::symlink_file(&target, &link) {
        Ok(()) => {}
        Err(error) if error.raw_os_error() == Some(1314) => {
            eprintln!("skipping symlink report test: Windows symlink privilege unavailable");
            return;
        }
        Err(error) => panic!("unable to create symlink fixture: {error}"),
    }
    #[cfg(unix)]
    std::os::unix::fs::symlink(&target, &link).unwrap();
    assert!(read_quality_report(&link).is_err());
}

#[test]
fn rejects_count_overflow_in_report_validation() {
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("quality.json");
    let value = serde_json::json!({
        "schema_version": 1,
        "frame_count": 100000000,
        "accepted_count": 18446744073709551615u64,
        "frames": [],
        "anomalies": [{
            "frame_index": 0,
            "code": "blurred_frame",
            "summary": "blurred",
            "technical_detail": "variance=1",
            "recovery_action": "exclude_frame"
        }]
    });
    fs::write(&path, serde_json::to_vec(&value).unwrap()).unwrap();
    assert!(matches!(
        read_quality_report(&path),
        Err(PipelineError::InvalidReport { .. })
    ));
}

#[test]
fn anomaly_schema_round_trips_with_stable_names() {
    let anomaly = FrameAnomaly::new(
        2,
        feathertalk_frame_pipeline::AnomalyCode::BlurredFrame,
        "Frame is too blurry",
        "laplacian_variance=1",
        RecoveryAction::ExcludeFrame,
    )
    .unwrap();
    let json = serde_json::to_string(&anomaly).unwrap();
    assert!(json.contains("blurred_frame"));
    assert!(json.contains("exclude_frame"));
    assert_eq!(
        serde_json::from_str::<FrameAnomaly>(&json).unwrap(),
        anomaly
    );
}
