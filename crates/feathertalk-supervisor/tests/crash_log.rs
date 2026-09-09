use std::path::PathBuf;

use feathertalk_domain::{TaskId, TaskKind};
use feathertalk_supervisor::crash_log::{
    CrashLogOutcome, CrashReport, MAX_LOG_BYTES, MAX_LOG_LINE_CHARS, write_crash_log,
};
use feathertalk_supervisor::policy::Disposition;
use tempfile::tempdir;
use time::macros::datetime;

fn report(stderr_tail: Vec<String>) -> CrashReport {
    CrashReport {
        task_id: TaskId::parse("1756000000000-0000abcd").expect("a well formed task id"),
        kind: TaskKind::Train,
        attempt: 2,
        worker_path: Some(PathBuf::from("C:/tools/feathertalk-worker.exe")),
        exit_status: Some(101),
        disposition: Disposition::Restart { resume: true },
        detail: "the worker exited without reporting a terminal stage".to_owned(),
        stderr_tail,
        observed_at: datetime!(2026-09-05 10:11:12 UTC),
    }
}

fn written_path(outcome: CrashLogOutcome) -> PathBuf {
    match outcome {
        CrashLogOutcome::Written(path) => path,
        CrashLogOutcome::Failed { reason } => panic!("the log should have been written: {reason}"),
    }
}

#[test]
fn a_crash_log_names_the_task_and_the_attempt() {
    let dir = tempdir().expect("a temporary directory");
    let outcome = write_crash_log(
        dir.path(),
        &report(vec!["thread 'main' panicked".to_owned()]),
    );
    let path = written_path(outcome);

    assert_eq!(
        path.file_name().and_then(|name| name.to_str()),
        Some("worker-crash-1756000000000-0000abcd-attempt-2.log")
    );
    let body = std::fs::read_to_string(&path).expect("the log is readable");
    for expected in [
        "2026-09-05T10:11:12Z",
        "1756000000000-0000abcd",
        "train",
        "feathertalk-worker.exe",
        "101",
        "the worker exited without reporting a terminal stage",
        "thread 'main' panicked",
    ] {
        assert!(body.contains(expected), "{expected} is missing from {body}");
    }
}

#[test]
fn the_log_directory_is_created_on_demand() {
    let dir = tempdir().expect("a temporary directory");
    let nested = dir.path().join("logs").join("worker");
    let path = written_path(write_crash_log(&nested, &report(Vec::new())));
    assert!(path.starts_with(&nested));
    assert!(path.is_file());
}

#[test]
fn a_long_stderr_line_is_truncated() {
    let dir = tempdir().expect("a temporary directory");
    let long = "x".repeat(MAX_LOG_LINE_CHARS * 3);
    let path = written_path(write_crash_log(dir.path(), &report(vec![long])));
    let body = std::fs::read_to_string(&path).expect("the log is readable");

    let longest = body
        .lines()
        .map(|line| line.chars().count())
        .max()
        .unwrap_or_default();
    assert!(
        longest <= MAX_LOG_LINE_CHARS,
        "a line of {longest} characters slipped through"
    );
    assert!(body.contains("truncated"));
}

#[test]
fn a_flood_of_stderr_stays_under_the_byte_cap() {
    let dir = tempdir().expect("a temporary directory");
    let flood = (0..500)
        .map(|index| format!("{index} {}", "y".repeat(1_000)))
        .collect();
    let path = written_path(write_crash_log(dir.path(), &report(flood)));
    let size = std::fs::metadata(&path)
        .expect("the log has metadata")
        .len();

    assert!(size <= MAX_LOG_BYTES as u64, "the log grew to {size} bytes");
    let body = std::fs::read_to_string(&path).expect("the log is readable");
    assert!(body.contains("stderr tail truncated"));
    assert!(body.contains("1756000000000-0000abcd"));
}

#[test]
fn an_unwritable_directory_reports_a_reason_instead_of_panicking() {
    let dir = tempdir().expect("a temporary directory");
    let blocker = dir.path().join("not-a-directory");
    std::fs::write(&blocker, b"in the way").expect("the blocking file is written");

    match write_crash_log(&blocker, &report(Vec::new())) {
        CrashLogOutcome::Failed { reason } => assert!(!reason.is_empty()),
        CrashLogOutcome::Written(path) => panic!("a file should not accept a log at {path:?}"),
    }
}
