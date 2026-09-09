use std::fs;

use feathertalk_app::worker_status::{WorkerStatus, source_key};
use feathertalk_client::{WorkerLocator, WorkerPathSource};

#[test]
fn an_unconfigured_locator_is_missing_and_lists_every_source() {
    let status = WorkerStatus::probe(&WorkerLocator::from_parts(None, None, None));
    assert!(!status.is_ready());
    assert_eq!(status.badge_key(), "shell.worker.missing");
    let probed = status.probed();
    assert_eq!(probed.len(), 3);
    assert!(probed.iter().all(|candidate| candidate.path.is_none()));
    assert_eq!(probed[0].source, WorkerPathSource::CliOption);
    assert_eq!(probed[1].source, WorkerPathSource::EnvVar);
    assert_eq!(probed[2].source, WorkerPathSource::SiblingOfCurrentExe);
}

#[test]
fn a_worker_next_to_the_shell_is_ready() {
    let root = tempfile::tempdir().expect("a temporary directory is available");
    let worker = root.path().join("feathertalk-worker");
    fs::write(&worker, b"binary").expect("the fake worker is written");
    let status = WorkerStatus::probe(&WorkerLocator::from_parts(None, None, Some(worker.clone())));
    assert!(status.is_ready());
    assert_eq!(status.badge_key(), "shell.worker.ready");
    assert!(matches!(&status, WorkerStatus::Ready { path } if path == &worker));
    assert!(
        status.probed().is_empty(),
        "a resolved worker needs no report"
    );
}

#[test]
fn a_configured_path_that_is_not_a_file_stays_missing() {
    let root = tempfile::tempdir().expect("a temporary directory is available");
    let configured = root.path().join("nowhere").join("feathertalk-worker");
    let sibling = root.path().join("feathertalk-worker");
    fs::write(&sibling, b"binary").expect("the fake worker is written");
    let status = WorkerStatus::probe(&WorkerLocator::from_parts(
        None,
        Some(configured.clone()),
        Some(sibling),
    ));
    assert!(!status.is_ready(), "a configured path never falls through");
    assert_eq!(
        status.probed()[1].path.as_deref(),
        Some(configured.as_path())
    );
}

#[test]
fn the_command_line_path_wins_over_the_environment() {
    let root = tempfile::tempdir().expect("a temporary directory is available");
    let chosen = root.path().join("chosen-worker");
    let other = root.path().join("other-worker");
    fs::write(&chosen, b"binary").expect("the chosen worker is written");
    fs::write(&other, b"binary").expect("the other worker is written");
    let status = WorkerStatus::probe(&WorkerLocator::from_parts(
        Some(chosen.clone()),
        Some(other.clone()),
        Some(other),
    ));
    assert!(matches!(&status, WorkerStatus::Ready { path } if path == &chosen));
}

#[test]
fn every_discovery_source_has_its_own_catalog_key() {
    let keys = [
        source_key(WorkerPathSource::CliOption),
        source_key(WorkerPathSource::EnvVar),
        source_key(WorkerPathSource::SiblingOfCurrentExe),
    ];
    assert_eq!(
        keys,
        [
            "shell.worker.source.cli",
            "shell.worker.source.env",
            "shell.worker.source.sibling"
        ]
    );
}
