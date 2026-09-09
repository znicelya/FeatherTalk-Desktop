//! What the generate page knows without a window.
//!
//! The command behind the page is `Render`; the page is called 生成, so the module
//! is named after the page the way `assets` and `training` are. What lives here is
//! everything that can be decided without a display: where the output goes, what
//! is already there, what the form holds, and whether it may be submitted.

use std::fs;
use std::path::{Path, PathBuf};

use feathertalk_domain::{RenderParams, Request};

use crate::assets::{self, AssetSurvey};
use crate::training::{self, CheckpointState, TrainingSurvey};

/// Where a render's output goes: `<project>/outputs/renders`.
///
/// `outputs/` belongs to the worker -- training writes `outputs/metrics` and
/// `outputs/preview` -- but nothing writes `renders`, so this layer both names it
/// and creates it. `TrainingPaths` in `feathertalk-worker` owns the other two
/// names and that crate depends on burn, which is why these strings are spelled
/// out here the way `training.rs` spells out the checkpoint layout.
pub fn renders_dir(project_dir: &Path) -> PathBuf {
    project_dir.join("outputs").join("renders")
}

/// The name prefix of a full render.
pub const RENDER_PREFIX: &str = "render-";

/// The name prefix of one that stopped at `max_output_frames`.
///
/// Two prefixes rather than one: the only difference between the two files is how
/// many frames are in them, and a directory listing cannot say that.
pub const PREVIEW_PREFIX: &str = "preview-";

/// The container the page derives, and the one it falls back to.
///
/// `raw_video_command` never passes `-f`, so FFmpeg picks the container from this
/// extension and from nothing else.
pub const VIDEO_EXTENSION: &str = "mp4";

/// `render-001.mp4`, for index 1.
///
/// Three digits is where the padding starts, not where it stops: `{:03}` pads and
/// never truncates, so index 1000 is `render-1000.mp4` and the next one after that
/// is larger again. Nothing wraps and nothing is overwritten.
pub fn render_name(prefix: &str, index: u32) -> String {
    format!("{prefix}{index:03}.{VIDEO_EXTENSION}")
}

/// The part of `name` before the video extension, matched without case.
///
/// `.MP4` is the same container as `.mp4`, and a file that came back from the
/// filesystem in capitals is still one of ours.
pub fn video_stem(name: &str) -> Option<&str> {
    let (stem, extension) = name.rsplit_once('.')?;
    if extension.eq_ignore_ascii_case(VIDEO_EXTENSION) {
        Some(stem)
    } else {
        None
    }
}

/// The index in `name`, when `name` is one this module would have written.
///
/// Everything after the prefix has to be ASCII digits and there has to be at
/// least one, so `render-.mp4` and a hand written `render-final.mp4` are not ours.
/// An index too large for a `u32` is not ours either: `parse` says so rather than
/// wrapping.
pub fn render_index(name: &str, prefix: &str) -> Option<u32> {
    let digits = video_stem(name)?.strip_prefix(prefix)?;
    if digits.is_empty() || !digits.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    digits.parse().ok()
}

/// Make sure `path` names a container FFmpeg can write.
pub fn with_video_extension(path: PathBuf) -> PathBuf {
    match path.extension() {
        // Somebody who typed `.mkv` meant `.mkv`.
        Some(_) => path,
        // FFmpeg picks the container from the extension and never from a flag, so
        // a path without one can only fail.
        None => path.with_extension(VIDEO_EXTENSION),
    }
}

/// One video `outputs/renders` already holds.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RenderedVideo {
    /// The file name, which is what the card shows.
    pub name: String,
    /// The absolute path, which is what the two system actions take.
    pub path: PathBuf,
    /// The index the name carries, when the name is one of ours.
    pub index: Option<u32>,
    /// The size on disk, so a zero byte file is visibly a zero byte file.
    pub bytes: u64,
}

/// One photograph of what a project's renders have produced.
///
/// Same rule as the other two surveys: rendering never reads the filesystem, so
/// the directory is photographed after a project is chosen, after a task ends and
/// when the user asks. What it answers is what the page needs -- which name the
/// next render takes, and which files are there to play.
#[derive(Debug, Default, Clone)]
pub struct RenderSurvey {
    /// Newest first: descending index, and descending name among equals.
    pub videos: Vec<RenderedVideo>,
    /// The index the next full render would take.
    pub next_render: u32,
    /// The index the next preview would take.
    pub next_preview: u32,
}

impl RenderSurvey {
    /// Photograph the render directory of `project_dir`.
    pub fn inspect(project_dir: &Path) -> Self {
        let dir = renders_dir(project_dir);
        let mut videos = Vec::new();
        let mut highest_render = 0;
        let mut highest_preview = 0;
        // An unreadable directory counts as empty, the way `assets::has_entries`
        // treats one: a project that has never rendered has no `outputs/renders`,
        // and both counters start at 1 either way.
        if let Ok(entries) = fs::read_dir(&dir) {
            for entry in entries.flatten() {
                let file_name = entry.file_name();
                let Some(name) = file_name.to_str() else {
                    // A name that is not valid Unicode cannot be matched against
                    // the prefixes and could not be shown; the file is still on
                    // disk, and the file manager can reach it.
                    continue;
                };
                if video_stem(name).is_none() {
                    continue;
                }
                // A directory wearing a video's name is not a render. The type is
                // asked rather than assumed, unlike the checkpoint scan, because
                // here the answer decides whether a play button appears.
                if !entry.file_type().is_ok_and(|kind| kind.is_file()) {
                    continue;
                }
                let render = render_index(name, RENDER_PREFIX);
                if let Some(index) = render {
                    highest_render = highest_render.max(index);
                }
                let preview = render_index(name, PREVIEW_PREFIX);
                if let Some(index) = preview {
                    highest_preview = highest_preview.max(index);
                }
                videos.push(RenderedVideo {
                    name: name.to_owned(),
                    path: dir.join(name),
                    // The two prefixes are disjoint, so at most one of them
                    // matched and `or` is a choice between a value and nothing.
                    index: render.or(preview),
                    bytes: entry.metadata().map_or(0, |data| data.len()),
                });
            }
        }
        videos.sort_by(|left, right| {
            right
                .index
                .cmp(&left.index)
                .then_with(|| right.name.cmp(&left.name))
        });
        Self {
            videos,
            // `saturating_add` rather than `+ 1`: the addition is unreachable at
            // `u32::MAX` and an overflow check is cheaper than reasoning about it.
            next_render: highest_render.saturating_add(1),
            next_preview: highest_preview.saturating_add(1),
        }
    }

    /// Whether the project has rendered anything yet.
    pub fn is_empty(&self) -> bool {
        self.videos.is_empty()
    }
}

/// The smallest preview the worker accepts.
///
/// Zero is not a shorter preview: `Some(0)` is a rejected request, so the control
/// never produces it.
pub const MIN_PREVIEW_FRAMES: u32 = 1;

/// The largest, which is `assets.json`'s own ceiling on `frame_count`.
pub const MAX_PREVIEW_FRAMES: u32 = 100_000_000;

/// What the switch starts at: four seconds at the render frame rate.
pub const DEFAULT_PREVIEW_FRAMES: u32 = 100;

/// The frame rate the worker writes into the container.
///
/// `feathertalk-worker::RENDER_FPS` is the same number, copied here for the reason
/// the four asset paths are copied: importing it would pull the worker's
/// dependency tree into the window process.
pub const RENDER_FPS: u32 = 25;

/// What the generate page's two controls say.
///
/// This lives in `AppState` rather than in `ProjectState`: how long a preview
/// should be is a habit, not a property of the directory that happens to be open.
/// The three paths a render needs are the other way around, so they live on the
/// project and are dropped with it.
#[derive(Debug, Clone)]
pub struct GenerateForm {
    preview: bool,
    preview_frames: u32,
}

impl Default for GenerateForm {
    fn default() -> Self {
        Self {
            preview: false,
            preview_frames: DEFAULT_PREVIEW_FRAMES,
        }
    }
}

impl GenerateForm {
    /// Whether only the first frames are rendered.
    pub fn preview(&self) -> bool {
        self.preview
    }

    /// Choose between a short preview and the whole sequence.
    pub fn set_preview(&mut self, preview: bool) {
        self.preview = preview;
    }

    /// How many frames a preview renders.
    pub fn preview_frames(&self) -> u32 {
        self.preview_frames
    }

    /// Take a frame count from the number input, which hands over an `f64`.
    ///
    /// The same shape as `TrainingForm::set_epochs`: a non-finite value keeps the
    /// previous one, and anything else is clamped into the protocol's range and
    /// rounded first, so the cast is exact -- a whole number between 1 and
    /// 100000000 is a `u32` -- and saturates rather than wrapping either way.
    pub fn set_preview_frames(&mut self, value: f64) {
        if !value.is_finite() {
            return;
        }
        let clamped = value
            .clamp(f64::from(MIN_PREVIEW_FRAMES), f64::from(MAX_PREVIEW_FRAMES))
            .round();
        self.preview_frames = clamped as u32;
    }

    /// The frame cap this form asks for, which is what `RenderParams` carries.
    pub fn max_output_frames(&self) -> Option<u64> {
        if self.preview {
            Some(u64::from(self.preview_frames))
        } else {
            None
        }
    }

    /// How long a preview lasts, at the frame rate the worker writes.
    ///
    /// Text rather than a number: this only ever goes beside the frame count on
    /// screen, and one decimal is the precision that sentence is read at.
    pub fn preview_seconds(&self) -> String {
        format!(
            "{:.1}",
            f64::from(self.preview_frames) / f64::from(RENDER_FPS)
        )
    }
}

/// A checkpoint the user chose by hand.
///
/// The state file is read once, when the dialog answers, and whatever came of it
/// is kept: rendering paints this and never the disk. `PartialEq` without `Eq`
/// because `CheckpointState` carries the loss weights, which are `f64`.
#[derive(Debug, Clone, PartialEq)]
pub struct PickedCheckpoint {
    /// The directory itself, which is what the request carries.
    pub path: PathBuf,
    /// The step its name declares, when the name is one of the worker's.
    pub step: Option<u64>,
    /// What its state file says, when it has one that reads.
    pub state: Option<CheckpointState>,
    /// Why it did not read, in English, for the notes card.
    pub error: Option<String>,
}

impl PickedCheckpoint {
    /// Adopt `path` as the checkpoint to render from, reading what it says.
    ///
    /// A directory with no state file is an error here, where `TrainingSurvey`
    /// reads the same absence as "not written yet". The difference is who chose
    /// the directory: the survey finds the newest one and may catch a publish mid
    /// flight, while this is a directory a user pointed at.
    ///
    /// The error does not stop a submission. Whether a directory holds a model
    /// this build can render is `read_training_checkpoint`'s judgement, and two
    /// judges disagreeing is worse than one judge speaking late.
    pub fn adopt(path: PathBuf) -> Self {
        let step = path
            .file_name()
            .and_then(|name| name.to_str())
            .and_then(training::checkpoint_step);
        let state_file = path.join(training::CHECKPOINT_STATE_FILE);
        let (state, error) = match training::read_checkpoint_state(&state_file) {
            Ok(state) => (Some(state), None),
            Err(error) => (None, Some(error.to_string())),
        };
        Self {
            path,
            step,
            state,
            error,
        }
    }
}

/// Which checkpoint a render would read: the chosen one, else the newest.
pub fn checkpoint_path<'a>(
    picked: Option<&'a PickedCheckpoint>,
    training: &'a TrainingSurvey,
) -> Option<&'a Path> {
    match picked {
        Some(picked) => Some(picked.path.as_path()),
        None => training
            .checkpoint
            .as_ref()
            .map(|checkpoint| checkpoint.path.as_path()),
    }
}

/// Which track a render would mix in: the chosen one, else the project's own.
///
/// The project's own is the track the features were taken from, which is the only
/// one whose sound matches the mouth. A project that was normalised and then had
/// its wav removed has neither, and the gate says so.
pub fn audio_path(
    picked: Option<&Path>,
    project_dir: &Path,
    survey: &AssetSurvey,
) -> Option<PathBuf> {
    match picked {
        Some(path) => Some(path.to_path_buf()),
        None => {
            if survey.audio {
                Some(assets::normalized_audio(project_dir))
            } else {
                None
            }
        }
    }
}

/// Where a render would write: the chosen path, else a derived name.
///
/// The two prefixes are what tells a preview from a finished render on disk, so
/// the switch decides which one the derivation uses. A chosen path is used as it
/// is: an explicit choice is not renamed behind the user's back, and after one the
/// switch stops moving the file name.
pub fn output_path(
    picked: Option<&Path>,
    project_dir: &Path,
    renders: &RenderSurvey,
    preview: bool,
) -> PathBuf {
    if let Some(path) = picked {
        return path.to_path_buf();
    }
    let (prefix, index) = if preview {
        (PREVIEW_PREFIX, renders.next_preview)
    } else {
        (RENDER_PREFIX, renders.next_render)
    };
    renders_dir(project_dir).join(render_name(prefix, index))
}

/// What the generate form may do right now.
///
/// The shell-wide gate -- no project, no worker, a task already running -- is
/// `tasks::blocked_key`, shared with the other two pages. This is only about
/// rendering's own prerequisites.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FormState {
    /// A prerequisite is missing; the key says which one.
    Blocked(&'static str),
    /// The worker would accept this.
    Ready,
}

impl FormState {
    /// Whether rendering's own prerequisites allow a submission.
    pub fn is_submittable(self) -> bool {
        match self {
            Self::Ready => true,
            Self::Blocked(_) => false,
        }
    }
}

/// Decide whether a render could be submitted.
///
/// The order is the worker's: `validate_project_dir` refuses an unlocked package
/// before it looks at anything else, and the two paths are what a request cannot be
/// built without. The same two functions the request uses are asked here, so the
/// button and the request can never disagree about which files are meant.
///
/// "The output already exists" is deliberately absent. A derived name cannot
/// collide, and a chosen one that does is `OutputExists` -- one verdict from one
/// judge beats two judges with two spellings of it.
pub fn form_state(
    assets: &AssetSurvey,
    training: &TrainingSurvey,
    picked_checkpoint: Option<&PickedCheckpoint>,
    picked_audio: Option<&Path>,
    project_dir: &Path,
) -> FormState {
    if !assets.is_locked() {
        return FormState::Blocked("generate.blocked.not_locked");
    }
    if checkpoint_path(picked_checkpoint, training).is_none() {
        return FormState::Blocked("generate.blocked.no_checkpoint");
    }
    if audio_path(picked_audio, project_dir, assets).is_none() {
        return FormState::Blocked("generate.blocked.no_audio");
    }
    FormState::Ready
}

/// The four paths and the frame cap one render is made of.
///
/// Borrowed rather than owned: every field is already sitting in the project state
/// or the form, and this lives for the length of one click handler. Naming the five
/// together is also what keeps a call site from swapping two paths of the same type.
pub struct RenderRequest<'a> {
    /// The project whose asset package the frames come from.
    pub project_dir: &'a Path,
    /// The checkpoint the weights come from.
    pub checkpoint: &'a Path,
    /// The track that goes into the container.
    pub audio: &'a Path,
    /// The file to write, which must not exist yet.
    pub output: &'a Path,
    /// `None` renders everything; `Some(n)` stops after n frames.
    pub max_output_frames: Option<u64>,
}

impl RenderRequest<'_> {
    /// The request this submits.
    pub fn build(&self) -> Request {
        Request::Render(RenderParams {
            project_dir: self.project_dir.to_path_buf(),
            checkpoint: self.checkpoint.to_path_buf(),
            audio: self.audio.to_path_buf(),
            output: self.output.to_path_buf(),
            max_output_frames: self.max_output_frames,
        })
    }
}

/// Create `outputs/renders`, because nothing else will.
///
/// `validate_output_destination` requires the parent directory to exist already,
/// and no code anywhere creates it: training creates `outputs/metrics` and
/// `outputs/preview`, inference creates nothing at all. So this runs on the click,
/// beside `manifest::ensure_manifest`, and a failure is a note rather than a
/// refusal -- the worker gets to give the one verdict about the path.
pub fn ensure_renders_dir(project_dir: &Path) -> std::io::Result<()> {
    fs::create_dir_all(renders_dir(project_dir))
}
