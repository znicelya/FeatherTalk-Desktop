//! The project directory and source video a session works on.
//!
//! Slice 5b took the project from `--project` and never changed it. The asset page
//! chooses it with a file dialog, so it lives in an entity the window can watch:
//! gpui repaints what a frame read, and a field on a global that nobody observes
//! would change without anything redrawing.

use std::path::{Path, PathBuf};

use crate::assets::AssetSurvey;
use crate::generate::{PickedCheckpoint, RenderSurvey};
use crate::tasks::Note;
use crate::training::TrainingSurvey;

/// What the asset page needs to know about the project in front of it.
#[derive(Debug, Default)]
pub struct ProjectState {
    dir: Option<PathBuf>,
    /// The source video normalisation will read.
    source_video: Option<PathBuf>,
    /// The checkpoint the generate page was pointed at by hand.
    checkpoint: Option<PickedCheckpoint>,
    /// The track a render would mix in, chosen by hand.
    audio: Option<PathBuf>,
    /// The output a render would write, chosen by hand.
    output: Option<PathBuf>,
    /// One photograph of the project's rendered videos.
    renders: RenderSurvey,
    survey: AssetSurvey,
    training: TrainingSurvey,
    notes: Vec<Note>,
}

impl ProjectState {
    /// The state a session starts in, with `dir` from `--project` when it was given.
    ///
    /// The flag is a convenience for launching into a known project; it is not
    /// required, and the picker overrides it.
    pub fn opened(dir: Option<PathBuf>) -> Self {
        let mut state = Self::default();
        if let Some(dir) = dir {
            state.select_dir(dir);
        }
        state
    }

    /// Adopt `dir` as the project, and hand back the form the rest of the shell uses.
    ///
    /// The source video is dropped on purpose: it belonged to the directory that
    /// was open a moment ago, and quietly normalising an old file into a new
    /// project is the kind of help nobody asks for.
    ///
    /// The three render choices go with it for the same reason. The preview
    /// switch is not one of them: how long a preview should be is a habit that
    /// outlives the directory, while a file belongs to the directory it was chosen
    /// in.
    pub fn select_dir(&mut self, dir: PathBuf) -> PathBuf {
        let (dir, failure) = absolute_dir(dir);
        if let Some(detail) = failure {
            self.notes
                .push(Note::new("assets.note.absolute_failed", Some(detail)));
        }
        self.source_video = None;
        self.checkpoint = None;
        self.audio = None;
        self.output = None;
        self.renders = RenderSurvey::inspect(&dir);
        self.survey = AssetSurvey::inspect(&dir);
        self.training = TrainingSurvey::inspect(&dir);
        self.dir = Some(dir.clone());
        dir
    }

    /// Remember the video normalisation will read.
    pub fn select_video(&mut self, video: PathBuf) {
        self.source_video = Some(video);
    }

    /// Adopt the checkpoint a dialog chose.
    pub fn select_checkpoint(&mut self, checkpoint: PickedCheckpoint) {
        self.checkpoint = Some(checkpoint);
    }

    /// Adopt the track a dialog chose.
    pub fn select_audio(&mut self, audio: PathBuf) {
        self.audio = Some(audio);
    }

    /// Adopt the output a dialog chose.
    pub fn select_output(&mut self, output: PathBuf) {
        self.output = Some(output);
    }

    /// Go back to the newest checkpoint, which is the default rather than a
    /// remembered choice.
    pub fn use_latest_checkpoint(&mut self) {
        self.checkpoint = None;
    }

    /// Photograph the project's directories again.
    ///
    /// Called after a task ends and when the user asks, because the disk can also
    /// be changed by the CLI or by hand.
    ///
    /// All three photographs are taken here rather than one each: this is already
    /// the "a task ended" call, and the next one costs four directory listings and
    /// two small JSON reads. Splitting them would mean deciding all over again who
    /// refreshes what and when.
    pub fn refresh(&mut self) {
        if let Some(dir) = self.dir.as_deref() {
            self.renders = RenderSurvey::inspect(dir);
            self.survey = AssetSurvey::inspect(dir);
            self.training = TrainingSurvey::inspect(dir);
        }
    }

    /// Record something the page has to say.
    pub fn note(&mut self, note: Note) {
        self.notes.push(note);
    }

    pub fn dir(&self) -> Option<&Path> {
        self.dir.as_deref()
    }

    pub fn source_video(&self) -> Option<&Path> {
        self.source_video.as_deref()
    }

    /// The checkpoint the generate page was pointed at by hand.
    pub fn checkpoint(&self) -> Option<&PickedCheckpoint> {
        self.checkpoint.as_ref()
    }

    /// The track a render would mix in, chosen by hand.
    pub fn audio(&self) -> Option<&Path> {
        self.audio.as_deref()
    }

    /// The output a render would write, chosen by hand.
    pub fn output(&self) -> Option<&Path> {
        self.output.as_deref()
    }

    /// What the project's renders look like right now.
    pub fn renders(&self) -> &RenderSurvey {
        &self.renders
    }

    pub fn survey(&self) -> &AssetSurvey {
        &self.survey
    }

    pub fn training(&self) -> &TrainingSurvey {
        &self.training
    }

    pub fn notes(&self) -> &[Note] {
        &self.notes
    }

    /// Where the supervisor writes crash logs, next to the project they belong to.
    pub fn log_dir(&self) -> Option<PathBuf> {
        self.dir.as_deref().map(|dir| dir.join("logs"))
    }
}

/// Make `dir` absolute, because worker admission rejects a relative project path.
///
/// `std::path::absolute` is lexical: it reads the process working directory and
/// touches nothing else. When it fails the path is kept as given and the reason is
/// returned -- the worker will say what it thinks of the path, which beats the
/// shell silently rewriting what the user chose.
pub fn absolute_dir(dir: PathBuf) -> (PathBuf, Option<String>) {
    match std::path::absolute(&dir) {
        Ok(absolute) => (absolute, None),
        Err(error) => (dir, Some(error.to_string())),
    }
}

/// Whether `dir` has a project manifest to read task history from.
///
/// A directory chosen for a brand new project has none, and asking `scan_project`
/// about it would produce a "the task history could not be read" note about a
/// project that has no history yet.
pub fn has_task_history(dir: &Path) -> bool {
    dir.join("project.json").is_file()
}
