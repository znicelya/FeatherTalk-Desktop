use feathertalk_domain::{DomainError, ErrorCode, Recovery, TaskError, TaskStage};

#[test]
fn every_error_code_has_the_wire_form_from_the_design() {
    let expected = [
        "MEDIA_INVALID",
        "FACE_NOT_FOUND",
        "LANDMARK_INVALID",
        "FEATURE_SHAPE_MISMATCH",
        "MODEL_INCOMPATIBLE",
        "GPU_OUT_OF_MEMORY",
        "GPU_DEVICE_LOST",
        "DISK_SPACE_LOW",
        "WORKER_CRASHED",
        "TASK_CANCELLED",
    ];
    assert_eq!(ErrorCode::ALL.len(), 10);
    for (code, wire) in ErrorCode::ALL.into_iter().zip(expected) {
        assert_eq!(code.as_wire(), wire);
        assert_eq!(serde_json::to_string(&code).unwrap(), format!("\"{wire}\""));
    }
}

#[test]
fn every_error_code_maps_to_an_actionable_recovery() {
    for code in ErrorCode::ALL {
        let recovery = code.default_recovery();
        if matches!(code, ErrorCode::TaskCancelled) {
            assert_eq!(recovery, Recovery::NotRecoverable);
        } else {
            assert_ne!(recovery, Recovery::NotRecoverable, "{code:?}");
        }
    }
}

#[test]
fn validate_rejects_an_empty_summary_and_oversized_fields() {
    let ok = TaskError::new(
        ErrorCode::MediaInvalid,
        "无法读取视频",
        "ffprobe exit 1",
        TaskStage::Preparing,
    );
    ok.validate().unwrap();

    let empty = TaskError::new(
        ErrorCode::MediaInvalid,
        "  ",
        "detail",
        TaskStage::Preparing,
    );
    assert!(matches!(
        empty.validate(),
        Err(DomainError::InvalidField {
            field: "summary",
            ..
        })
    ));

    let long_summary = "字".repeat(feathertalk_domain::MAX_SUMMARY_CHARS + 1);
    let too_long = TaskError::new(
        ErrorCode::MediaInvalid,
        &long_summary,
        "detail",
        TaskStage::Preparing,
    );
    assert!(matches!(
        too_long.validate(),
        Err(DomainError::InvalidField {
            field: "summary",
            ..
        })
    ));

    let long_detail = "x".repeat(feathertalk_domain::MAX_DETAIL_CHARS + 1);
    let too_long = TaskError::new(
        ErrorCode::MediaInvalid,
        "摘要",
        &long_detail,
        TaskStage::Preparing,
    );
    assert!(matches!(
        too_long.validate(),
        Err(DomainError::InvalidField {
            field: "detail",
            ..
        })
    ));
}

#[test]
fn task_error_round_trips_and_rejects_unknown_fields() {
    let error = TaskError::new(
        ErrorCode::GpuDeviceLost,
        "显卡连接中断",
        "device lost",
        TaskStage::Preparing,
    );
    let json = serde_json::to_string(&error).unwrap();
    assert_eq!(serde_json::from_str::<TaskError>(&json).unwrap(), error);
    assert!(serde_json::from_str::<TaskError>(r#"{"code":"GPU_DEVICE_LOST","summary":"a","detail":"b","stage":{"stage":"preparing"},"recovery":"resume_from_checkpoint","extra":1}"#).is_err());
}
