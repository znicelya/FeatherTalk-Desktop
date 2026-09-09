#[path = "support/mod.rs"]
mod support;

use std::path::Path;

use feathertalk_inference::{
    InferenceError, RawFrameRenderSpec, RenderGeometry, staging_output_path,
    validate_output_destination,
};

#[test]
fn standard_geometry_matches_preprocess_contract() {
    let geometry = RenderGeometry::standard();
    assert_eq!(
        (
            geometry.crop_size(),
            geometry.inner_size(),
            geometry.border()
        ),
        (168, 160, 4)
    );
    assert_eq!(geometry.replacement_offset(), (4, 4));
    let crop = feathertalk_preprocess::default_crop_spec();
    assert_eq!(geometry.crop_size(), crop.crop_size);
    assert_eq!(geometry.inner_size(), crop.inner_size);
    assert_eq!(geometry.border(), crop.border);
}

#[test]
fn raw_spec_keeps_native_paths_and_fixed_fps() {
    let spec = RawFrameRenderSpec::new(
        1280,
        720,
        Path::new("drive audio.wav"),
        Path::new("result.mp4"),
    )
    .unwrap();
    assert_eq!(spec.width(), 1280);
    assert_eq!(spec.height(), 720);
    assert_eq!(spec.fps(), 25);
    assert_eq!(spec.audio_path(), Path::new("drive audio.wav"));
    assert_eq!(spec.output_path(), Path::new("result.mp4"));
}

#[test]
fn rejects_zero_dimensions_and_empty_paths() {
    assert!(matches!(
        RawFrameRenderSpec::new(0, 720, Path::new("a.wav"), Path::new("o.mp4")),
        Err(InferenceError::InvalidField { field: "width", .. })
    ));
    assert!(matches!(
        RawFrameRenderSpec::new(1280, 0, Path::new("a.wav"), Path::new("o.mp4")),
        Err(InferenceError::InvalidField {
            field: "height",
            ..
        })
    ));
    assert!(matches!(
        RawFrameRenderSpec::new(1, 1, Path::new(""), Path::new("o.mp4")),
        Err(InferenceError::InvalidField {
            field: "audio_path",
            ..
        })
    ));
    assert!(matches!(
        RawFrameRenderSpec::new(1, 1, Path::new("a.wav"), Path::new("")),
        Err(InferenceError::InvalidField {
            field: "output_path",
            ..
        })
    ));
}

#[test]
fn missing_output_with_existing_parent_is_valid_and_staging_preserves_extension() {
    let dir = tempfile::tempdir().unwrap();
    let output = dir.path().join("render result.mp4");
    validate_output_destination(&output).unwrap();
    let staging = staging_output_path(&output, "task-01").unwrap();
    assert_eq!(staging.parent(), output.parent());
    assert_eq!(staging.extension(), output.extension());
    assert!(
        staging
            .file_name()
            .unwrap()
            .to_string_lossy()
            .contains("task-01")
    );
    assert!(!staging.exists());
}

#[test]
fn existing_file_and_directory_are_rejected_without_mutation() {
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("result.mp4");
    std::fs::write(&file, b"sentinel").unwrap();
    assert!(matches!(
        validate_output_destination(&file),
        Err(InferenceError::OutputExists { .. })
    ));
    let directory = dir.path().join("directory");
    std::fs::create_dir(&directory).unwrap();
    assert!(matches!(
        validate_output_destination(&directory),
        Err(InferenceError::OutputNotRegular { .. })
    ));
    assert_eq!(std::fs::read(&file).unwrap(), b"sentinel");
}

#[test]
fn missing_parent_is_rejected_without_creating_directories() {
    let dir = tempfile::tempdir().unwrap();
    let parent = dir.path().join("missing");
    let output = parent.join("result.mp4");
    assert!(matches!(
        validate_output_destination(&output),
        Err(InferenceError::OutputParentInvalid { .. })
    ));
    assert!(!parent.exists());
}

#[test]
fn invalid_task_ids_are_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let output = dir.path().join("result.mp4");
    let long_id = "x".repeat(65);
    for id in ["", ".", "..", "task/id", "task id"] {
        assert!(matches!(
            staging_output_path(&output, id),
            Err(InferenceError::InvalidTaskId { .. })
        ));
    }
    assert!(matches!(
        staging_output_path(&output, &long_id),
        Err(InferenceError::InvalidTaskId { .. })
    ));
}

#[test]
fn symlinked_destination_and_parent_are_rejected_when_supported() {
    let dir = tempfile::tempdir().unwrap();
    let target = dir.path().join("target.mp4");
    std::fs::write(&target, b"sentinel").unwrap();
    let destination_link = dir.path().join("result.mp4");
    if support::create_file_symlink(&target, &destination_link).is_ok() {
        assert!(matches!(
            validate_output_destination(&destination_link),
            Err(InferenceError::OutputSymlink { .. })
        ));
        assert_eq!(std::fs::read(&target).unwrap(), b"sentinel");
    }

    let real_parent = dir.path().join("real-parent");
    let linked_parent = dir.path().join("linked-parent");
    std::fs::create_dir(&real_parent).unwrap();
    if support::create_dir_symlink(&real_parent, &linked_parent).is_ok() {
        let output = linked_parent.join("result.mp4");
        assert!(matches!(
            validate_output_destination(&output),
            Err(InferenceError::OutputSymlink { .. })
        ));
    }
}
