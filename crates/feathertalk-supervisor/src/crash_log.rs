use std::path::{Path, PathBuf};

use feathertalk_domain::{TaskId, TaskKind};
use time::OffsetDateTime;
use time::format_description::well_known::Rfc3339;

use crate::policy::Disposition;

/// Longest line the log may hold, in characters. A worker that dies inside a
/// shader compiler can emit single lines of tens of kilobytes.
pub const MAX_LOG_LINE_CHARS: usize = 2_000;

/// Hard cap on one crash log. The point of the file is the last few lines before
/// the end, not a full transcript, and an unbounded write on a full disk is its
/// own failure.
pub const MAX_LOG_BYTES: usize = 65_536;

const LINE_MARKER: &str = " ... truncated";
const TAIL_MARKER: &str = "  ... stderr tail truncated";
const STDERR_PREFIX: &str = "  ";

/// Everything known about one attempt that ended badly.
///
/// The supervisor fills this in; the module only formats and writes. `observed_at`
/// is passed in rather than read from the clock here so tests get a fixed
/// timestamp without waiting for one.
#[derive(Debug, Clone)]
pub struct CrashReport {
    pub task_id: TaskId,
    pub kind: TaskKind,
    pub attempt: u32,
    pub worker_path: Option<PathBuf>,
    pub exit_status: Option<i32>,
    pub disposition: Disposition,
    pub detail: String,
    pub stderr_tail: Vec<String>,
    pub observed_at: OffsetDateTime,
}

/// Whether the last log made it to disk.
///
/// Losing the log must not change what the user is told about the task, so this
/// is a value in the report rather than an error the caller has to handle.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CrashLogOutcome {
    Written(PathBuf),
    Failed { reason: String },
}

/// Write one crash log into `dir`, creating the directory if it is missing.
///
/// The body is plain English text: this file is read by whoever is debugging an
/// installation, and it stays on the local disk — nothing here is sent anywhere.
pub fn write_crash_log(dir: &Path, report: &CrashReport) -> CrashLogOutcome {
    let observed_at = match report.observed_at.format(&Rfc3339) {
        Ok(text) => text,
        Err(error) => {
            return CrashLogOutcome::Failed {
                reason: format!("the timestamp could not be formatted: {error}"),
            };
        }
    };

    let body = render(report, &observed_at);
    let path = dir.join(format!(
        "worker-crash-{}-attempt-{}.log",
        report.task_id.as_str(),
        report.attempt
    ));

    if let Err(error) = std::fs::create_dir_all(dir) {
        return CrashLogOutcome::Failed {
            reason: format!("{} could not be created: {error}", dir.display()),
        };
    }
    match std::fs::write(&path, body) {
        Ok(()) => CrashLogOutcome::Written(path),
        Err(error) => CrashLogOutcome::Failed {
            reason: format!("{} could not be written: {error}", path.display()),
        },
    }
}

fn render(report: &CrashReport, observed_at: &str) -> String {
    let worker = report
        .worker_path
        .as_ref()
        .map(|path| path.display().to_string())
        .unwrap_or_else(|| "not resolved".to_owned());
    let exit_status = report
        .exit_status
        .map(|status| status.to_string())
        .unwrap_or_else(|| "unknown".to_owned());

    let mut body = String::new();
    for line in [
        format!("observed at: {observed_at}"),
        format!("task id: {}", report.task_id.as_str()),
        format!("command: {}", report.kind.as_slug()),
        format!("attempt: {}", report.attempt),
        format!("worker: {worker}"),
        format!("exit status: {exit_status}"),
        format!("decision: {}", describe(report.disposition)),
        format!("detail: {}", report.detail),
        "stderr tail:".to_owned(),
    ] {
        push_line(&mut body, &truncate_line(&line));
    }

    for line in &report.stderr_tail {
        let rendered = truncate_line(&format!("{STDERR_PREFIX}{line}"));
        if body.len() + rendered.len() + TAIL_MARKER.len() + 2 > MAX_LOG_BYTES {
            push_line(&mut body, TAIL_MARKER);
            return body;
        }
        push_line(&mut body, &rendered);
    }
    body
}

fn push_line(body: &mut String, line: &str) {
    body.push_str(line);
    body.push('\n');
}

/// Cut one line to [`MAX_LOG_LINE_CHARS`] characters, marking the cut.
///
/// Character-based rather than byte-based so a cut never lands inside a UTF-8
/// sequence: worker stderr carries Chinese summaries as well as English detail.
fn truncate_line(line: &str) -> String {
    if line.chars().count() <= MAX_LOG_LINE_CHARS {
        return line.to_owned();
    }
    let keep = MAX_LOG_LINE_CHARS.saturating_sub(LINE_MARKER.chars().count());
    let mut cut: String = line.chars().take(keep).collect();
    cut.push_str(LINE_MARKER);
    cut
}

fn describe(disposition: Disposition) -> &'static str {
    match disposition {
        Disposition::Finish => "reported to the caller as final",
        Disposition::Restart { resume: true } => "restart and resume from the last checkpoint",
        Disposition::Restart { resume: false } => "restart from the beginning",
        Disposition::GiveUp => "gave up after the attempt budget was spent",
    }
}
