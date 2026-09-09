#[path = "support/mod.rs"]
mod support;

use std::path::PathBuf;

use feathertalk_media::{MediaError, MediaInput, validate_input};

#[test]
fn validates_and_canonicalizes_a_regular_source_file() {
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("input.mp4");
    std::fs::write(&source, b"media").unwrap();
    let validated = validate_input(&MediaInput {
        source: source.clone(),
    })
    .unwrap();
    assert_eq!(validated.source(), std::fs::canonicalize(source).unwrap());
}

#[test]
fn rejects_missing_input() {
    let dir = tempfile::tempdir().unwrap();
    let error = validate_input(&MediaInput {
        source: dir.path().join("missing.mp4"),
    })
    .unwrap_err();
    assert!(matches!(error, MediaError::InputMissing { .. }));
}

#[test]
fn rejects_directory_input() {
    let dir = tempfile::tempdir().unwrap();
    let error = validate_input(&MediaInput {
        source: dir.path().to_path_buf(),
    })
    .unwrap_err();
    assert!(matches!(error, MediaError::InputNotRegularFile { .. }));
}

#[test]
fn rejects_a_source_symlink() {
    let dir = tempfile::tempdir().unwrap();
    let target = dir.path().join("target.mp4");
    std::fs::write(&target, b"media").unwrap();
    let link = dir.path().join("link.mp4");
    if support::create_file_symlink(&target, &link).is_ok() {
        assert!(matches!(
            validate_input(&MediaInput { source: link }),
            Err(MediaError::SymlinkNotAllowed { .. })
        ));
    }
}

#[test]
fn rejects_a_symlinked_parent_component() {
    let dir = tempfile::tempdir().unwrap();
    let real = dir.path().join("real");
    let link = dir.path().join("link");
    std::fs::create_dir(&real).unwrap();
    let source = real.join("input.mp4");
    std::fs::write(&source, b"media").unwrap();
    if support::create_dir_symlink(&real, &link).is_ok() {
        let error = validate_input(&MediaInput {
            source: link.join("input.mp4"),
        })
        .unwrap_err();
        assert!(matches!(error, MediaError::SymlinkNotAllowed { .. }));
    }
}

#[test]
fn accepts_native_relative_path_only_after_it_resolves_to_a_file() {
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("input.mp4");
    std::fs::write(&source, b"media").unwrap();
    let input = MediaInput {
        source: PathBuf::from(&source),
    };
    assert_eq!(
        validate_input(&input).unwrap().source(),
        std::fs::canonicalize(source).unwrap()
    );
}
