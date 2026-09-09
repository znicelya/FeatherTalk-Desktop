#![allow(dead_code)]

use std::{
    ffi::OsStr,
    path::{Path, PathBuf},
    sync::Mutex,
};

use feathertalk_frame_pipeline::{CommandSpec, PipelineObserver, PipelinePhase};

/// The value ffmpeg receives after `flag`, as text.
pub fn flag_value(command: &CommandSpec, flag: &str) -> String {
    let arguments = command.arguments();
    let position = arguments
        .iter()
        .position(|argument| argument.as_os_str() == OsStr::new(flag))
        .unwrap_or_else(|| panic!("{flag} is missing from the frame command"));
    arguments
        .get(position + 1)
        .unwrap_or_else(|| panic!("{flag} carries no value"))
        .to_str()
        .unwrap_or_else(|| panic!("{flag} carries non-UTF-8 text"))
        .to_owned()
}

/// The same value parsed as a frame counter.
pub fn flag_number(command: &CommandSpec, flag: &str) -> u64 {
    flag_value(command, flag)
        .parse()
        .unwrap_or_else(|_| panic!("{flag} must carry a number"))
}

/// The frames one chunk command is expected to write, as `(index, path)` pairs.
///
/// Fake runners use this to stand in for ffmpeg's `image2` muxer: the command
/// ends with a `%06d.jpg` pattern, and `-start_number` plus `-frames:v` say
/// which indices that pattern expands to.
pub fn chunk_outputs(command: &CommandSpec) -> Vec<(u64, PathBuf)> {
    let pattern = Path::new(
        command
            .arguments()
            .last()
            .expect("the frame command ends with the output pattern"),
    );
    let directory = pattern
        .parent()
        .expect("the output pattern sits inside the frames directory")
        .to_owned();
    let first = flag_number(command, "-start_number");
    let count = flag_number(command, "-frames:v");
    (first..first + count)
        .map(|index| (index, directory.join(format!("{index:06}.jpg"))))
        .collect()
}

/// Records every phase a pipeline stage reports, and flips to cancelled once
/// `cancel_after` phases have arrived.
pub struct Recorder {
    phases: Mutex<Vec<PipelinePhase>>,
    cancel_after: Option<usize>,
}

impl Recorder {
    pub fn new(cancel_after: Option<usize>) -> Self {
        Self {
            phases: Mutex::new(Vec::new()),
            cancel_after,
        }
    }

    pub fn phases(&self) -> Vec<PipelinePhase> {
        self.phases.lock().unwrap().clone()
    }
}

impl PipelineObserver for Recorder {
    fn phase(&self, phase: PipelinePhase) {
        self.phases.lock().unwrap().push(phase);
    }

    fn is_cancelled(&self) -> bool {
        match self.cancel_after {
            Some(limit) => self.phases.lock().unwrap().len() >= limit,
            None => false,
        }
    }
}
