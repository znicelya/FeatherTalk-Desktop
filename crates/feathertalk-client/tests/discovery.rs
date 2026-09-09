use std::path::PathBuf;

use feathertalk_client::{ClientError, WORKER_FILE_STEM, WorkerLocator, WorkerPathSource};
use tempfile::TempDir;

fn touch(dir: &TempDir, name: &str) -> PathBuf {
    let path = dir.path().join(name);
    std::fs::write(&path, b"stand-in for an executable").unwrap();
    path
}

#[test]
fn the_cli_option_outranks_the_environment_and_the_sibling() {
    let dir = TempDir::new().unwrap();
    let chosen = touch(&dir, "chosen");
    let other = touch(&dir, "other");
    let locator = WorkerLocator::from_parts(Some(chosen.clone()), Some(other.clone()), Some(other));
    assert_eq!(locator.resolve().unwrap(), chosen);
}

#[test]
fn the_environment_variable_outranks_the_sibling() {
    let dir = TempDir::new().unwrap();
    let chosen = touch(&dir, "chosen");
    let sibling = touch(&dir, "sibling");
    let locator = WorkerLocator::from_parts(None, Some(chosen.clone()), Some(sibling));
    assert_eq!(locator.resolve().unwrap(), chosen);
}

#[test]
fn the_sibling_is_used_when_nothing_is_configured() {
    let dir = TempDir::new().unwrap();
    let sibling = touch(&dir, "sibling");
    let locator = WorkerLocator::from_parts(None, None, Some(sibling.clone()));
    assert_eq!(locator.resolve().unwrap(), sibling);
}

#[test]
fn a_configured_path_that_is_missing_is_an_error_not_a_fallback() {
    let dir = TempDir::new().unwrap();
    let sibling = touch(&dir, "sibling");
    let missing = dir.path().join("missing-worker");
    let locator = WorkerLocator::from_parts(Some(missing.clone()), None, Some(sibling));
    let error = locator.resolve().unwrap_err();
    let ClientError::WorkerNotFound { probed } = error else {
        panic!("expected WorkerNotFound, got {error:?}");
    };
    assert_eq!(probed.len(), 3);
    assert_eq!(probed[0].source, WorkerPathSource::CliOption);
    assert_eq!(probed[0].path, Some(missing));
    assert_eq!(probed[1].source, WorkerPathSource::EnvVar);
    assert_eq!(probed[1].path, None);
    assert_eq!(probed[2].source, WorkerPathSource::SiblingOfCurrentExe);
    assert!(probed[2].path.is_some());
}

#[test]
fn every_probed_source_is_reported_when_none_is_set() {
    let locator = WorkerLocator::from_parts(None, None, None);
    let error = locator.resolve().unwrap_err();
    let ClientError::WorkerNotFound { probed } = error else {
        panic!("expected WorkerNotFound, got {error:?}");
    };
    let labels: Vec<&str> = probed
        .iter()
        .map(|candidate| candidate.source.as_label())
        .collect();
    assert_eq!(
        labels,
        vec![
            "--worker",
            "FEATHERTALK_WORKER_BIN",
            "sibling of the current executable",
        ]
    );
    assert!(probed.iter().all(|candidate| candidate.path.is_none()));
}

#[test]
fn the_sibling_name_carries_the_platform_executable_suffix() {
    let exe = PathBuf::from("some").join("dir").join("feathertalk.exe");
    let sibling = WorkerLocator::sibling_of(&exe).unwrap();
    assert_eq!(sibling.parent().unwrap(), exe.parent().unwrap());
    assert_eq!(
        sibling.file_name().unwrap().to_str().unwrap(),
        format!("{WORKER_FILE_STEM}{}", std::env::consts::EXE_SUFFIX)
    );
}
