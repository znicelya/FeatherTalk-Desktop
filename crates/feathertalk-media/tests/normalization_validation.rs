#[path = "support/mod.rs"]
mod support;

use feathertalk_media::{MediaError, MediaInput, validate_input, validate_normalization};

#[test]
fn accepts_fixed_target_and_produces_fixed_output_names() {
    let (dir, input) = support::validated_source();
    let output = dir.path().join("assets");
    let layout =
        validate_normalization(&input, &support::normalization_spec(output.clone())).unwrap();
    assert_eq!(layout.output_dir(), std::fs::canonicalize(output).unwrap());
    assert_eq!(
        layout.video_path(),
        layout.output_dir().join("video_25fps.mp4")
    );
    assert_eq!(
        layout.audio_path(),
        layout.output_dir().join("audio_16k_mono.wav")
    );
}

#[test]
fn rejects_each_unsupported_target_value() {
    let (dir, input) = support::validated_source();
    let mut spec = support::normalization_spec(dir.path().join("assets"));
    spec.target_video_fps = 24;
    assert!(matches!(
        validate_normalization(&input, &spec),
        Err(MediaError::UnsupportedTarget {
            field: "target_video_fps",
            ..
        })
    ));
    spec.target_video_fps = 25;
    spec.target_audio_sample_rate = 48_000;
    assert!(matches!(
        validate_normalization(&input, &spec),
        Err(MediaError::UnsupportedTarget {
            field: "target_audio_sample_rate",
            ..
        })
    ));
    spec.target_audio_sample_rate = 16_000;
    spec.target_audio_channels = 2;
    assert!(matches!(
        validate_normalization(&input, &spec),
        Err(MediaError::UnsupportedTarget {
            field: "target_audio_channels",
            ..
        })
    ));
}

#[test]
fn creates_a_missing_output_directory() {
    let (dir, input) = support::validated_source();
    let output = dir.path().join("nested/assets");
    validate_normalization(&input, &support::normalization_spec(output.clone())).unwrap();
    assert!(output.is_dir());
}

#[test]
fn rejects_source_inside_output_directory() {
    let dir = tempfile::tempdir().unwrap();
    let output = dir.path().join("assets");
    std::fs::create_dir(&output).unwrap();
    let source = output.join("input.mp4");
    std::fs::write(&source, b"media").unwrap();
    let input = validate_input(&MediaInput { source }).unwrap();
    assert!(matches!(
        validate_normalization(&input, &support::normalization_spec(output)),
        Err(MediaError::OutputInsideInput { .. })
    ));
}

#[test]
fn rejects_output_directory_file() {
    let (dir, input) = support::validated_source();
    let output = dir.path().join("assets");
    std::fs::write(&output, b"not a directory").unwrap();
    assert!(matches!(
        validate_normalization(&input, &support::normalization_spec(output)),
        Err(MediaError::OutputDirectoryInvalid { .. })
    ));
}

#[test]
fn reports_source_conflict_with_fixed_output_name() {
    let dir = tempfile::tempdir().unwrap();
    let output = dir.path().join("assets");
    std::fs::create_dir(&output).unwrap();
    let source = output.join("video_25fps.mp4");
    std::fs::write(&source, b"media").unwrap();
    let input = validate_input(&MediaInput { source }).unwrap();
    assert!(matches!(
        validate_normalization(&input, &support::normalization_spec(output)),
        Err(MediaError::OutputConflictsWithInput { .. })
    ));
}

#[test]
fn rejects_symlinked_output_directory() {
    let (dir, input) = support::validated_source();
    let real = dir.path().join("real");
    let link = dir.path().join("assets");
    std::fs::create_dir(&real).unwrap();
    if support::create_dir_symlink(&real, &link).is_ok() {
        assert!(matches!(
            validate_normalization(&input, &support::normalization_spec(link)),
            Err(MediaError::SymlinkNotAllowed { .. })
        ));
    }
}

#[test]
fn permits_existing_regular_destinations() {
    let (dir, input) = support::validated_source();
    let output = dir.path().join("assets");
    std::fs::create_dir(&output).unwrap();
    std::fs::write(output.join("video_25fps.mp4"), b"old video").unwrap();
    std::fs::write(output.join("audio_16k_mono.wav"), b"old audio").unwrap();
    assert!(validate_normalization(&input, &support::normalization_spec(output)).is_ok());
}

#[test]
fn rejects_non_regular_existing_destination() {
    let (dir, input) = support::validated_source();
    let output = dir.path().join("assets");
    std::fs::create_dir(&output).unwrap();
    std::fs::create_dir(output.join("video_25fps.mp4")).unwrap();
    assert!(matches!(
        validate_normalization(&input, &support::normalization_spec(output)),
        Err(MediaError::OutputDestinationInvalid { .. })
    ));
}

#[test]
fn rejects_symlink_existing_destination() {
    let (dir, input) = support::validated_source();
    let output = dir.path().join("assets");
    let target = dir.path().join("existing.mp4");
    std::fs::create_dir(&output).unwrap();
    std::fs::write(&target, b"existing").unwrap();
    let destination = output.join("video_25fps.mp4");
    if support::create_file_symlink(&target, &destination).is_ok() {
        assert!(matches!(
            validate_normalization(&input, &support::normalization_spec(output)),
            Err(MediaError::OutputDestinationInvalid { .. })
        ));
    }
}
