//! What a project directory already holds.
//!
//! Rendering never reads the filesystem: a frame that stats six paths is a frame
//! that can fail, and a render function has nowhere to report that. So the disk is
//! photographed once -- after a directory is chosen, after a task ends, and when
//! the user asks for it -- and the page draws the photograph.

use std::fs;
use std::path::{Path, PathBuf};

use feathertalk_domain::{
    ExtractFeaturesParams, ExtractFramesParams, NormalizeMediaParams, ProjectDirParams, Request,
    TaskKind,
};
use feathertalk_project::{AssetManifest, AssetPackageState, read_asset_manifest};

// The line type is shared with the training page, so it lives in `facts` and is
// re-exported here: `assets::Fact` is the path every caller already uses.
pub use crate::facts::{Fact, FactValue};

/// The asset directory of a project, where every artifact of the four commands
/// lands.
pub fn assets_dir(project_dir: &Path) -> PathBuf {
    project_dir.join("assets")
}

/// The 25fps video `normalize_media` writes and `extract_frames` reads.
pub fn normalized_video(project_dir: &Path) -> PathBuf {
    assets_dir(project_dir).join("video_25fps.mp4")
}

/// The 16kHz mono audio `normalize_media` writes and `extract_features` reads.
pub fn normalized_audio(project_dir: &Path) -> PathBuf {
    assets_dir(project_dir).join("audio_16k_mono.wav")
}

/// The feature stream `extract_features` writes and the lock commits.
pub fn feature_file(project_dir: &Path) -> PathBuf {
    assets_dir(project_dir)
        .join("features")
        .join("feather_hubert.f32")
}

/// One photograph of a project's asset directory.
///
/// The booleans are "the artifact is there", not "the step succeeded": the worker
/// reports success through the task pipeline. What this answers is the question
/// the page asks on every frame -- which step can run next.
#[derive(Debug, Default, Clone)]
pub struct AssetSurvey {
    /// `assets/video_25fps.mp4` exists.
    pub video: bool,
    /// `assets/audio_16k_mono.wav` exists.
    pub audio: bool,
    /// `assets/frames/` holds at least one entry.
    pub frames: bool,
    /// `assets/landmarks/` holds at least one entry.
    pub landmarks: bool,
    /// `assets/quality.json` exists.
    pub quality: bool,
    /// `assets/features/feather_hubert.f32` exists.
    pub features: bool,
    /// `assets/assets.json`, when it is there and readable.
    pub manifest: Option<AssetManifest>,
    /// Why the manifest could not be read, in the words of `feathertalk-project`.
    pub manifest_error: Option<String>,
}

impl AssetSurvey {
    /// Photograph the asset directory of `project_dir`.
    pub fn inspect(project_dir: &Path) -> Self {
        let assets = assets_dir(project_dir);
        let (manifest, manifest_error) = read_manifest(&assets.join("assets.json"));
        Self {
            video: normalized_video(project_dir).is_file(),
            audio: normalized_audio(project_dir).is_file(),
            frames: has_entries(&assets.join("frames")),
            landmarks: has_entries(&assets.join("landmarks")),
            quality: assets.join("quality.json").is_file(),
            features: feature_file(project_dir).is_file(),
            manifest,
            manifest_error,
        }
    }

    /// Whether the package is locked, which closes all four steps.
    pub fn is_locked(&self) -> bool {
        match &self.manifest {
            Some(manifest) => match manifest.state {
                AssetPackageState::Locked => true,
                AssetPackageState::Preparing => false,
            },
            None => false,
        }
    }
}

/// Read `assets.json`, keeping the reason when it cannot be read.
///
/// An absent manifest is not an error: a project that has not been normalised yet
/// has none. Anything else -- unreadable, oversized, invalid JSON, a manifest that
/// fails its own state's validation -- is reported verbatim, the way `ClientError`
/// is, because whoever has to fix it wants the original words.
fn read_manifest(path: &Path) -> (Option<AssetManifest>, Option<String>) {
    if !path.is_file() {
        return (None, None);
    }
    match read_asset_manifest(path) {
        Ok(manifest) => (Some(manifest), None),
        Err(error) => (None, Some(error.to_string())),
    }
}

/// Whether `dir` holds at least one entry.
///
/// A frame directory can hold a hundred thousand files, so counting them to answer
/// "is anything there" would be pure waste. An unreadable directory counts as
/// empty: the step that fills it can run again either way.
fn has_entries(dir: &Path) -> bool {
    match fs::read_dir(dir) {
        Ok(mut entries) => entries.next().is_some(),
        Err(_) => false,
    }
}

/// What the manifest says, as lines to paint.
///
/// An empty list means there is no manifest yet, and the page leaves the card out.
pub fn facts(survey: &AssetSurvey) -> Vec<Fact> {
    let Some(manifest) = &survey.manifest else {
        return Vec::new();
    };
    let mut facts = vec![
        Fact {
            label: "assets.package.state",
            value: FactValue::Key(state_key(&manifest.state)),
        },
        Fact {
            label: "assets.package.fps",
            value: FactValue::Text(manifest.video_fps.to_string()),
        },
        Fact {
            label: "assets.package.sample_rate",
            value: FactValue::Text(manifest.audio_sample_rate.to_string()),
        },
        Fact {
            label: "assets.package.channels",
            value: FactValue::Text(manifest.audio_channels.to_string()),
        },
        Fact {
            label: "assets.package.frames",
            value: FactValue::Text(manifest.frame_count.to_string()),
        },
        Fact {
            label: "assets.package.resolution",
            value: FactValue::Text(format!(
                "{}×{}",
                manifest.frame_width, manifest.frame_height
            )),
        },
    ];
    // A preparing manifest is allowed to carry a frame rate of zero, and dividing
    // by it would panic, so the duration is simply not among the facts yet.
    if let Some(seconds) = duration_seconds(manifest.frame_count, manifest.video_fps) {
        facts.push(Fact {
            label: "assets.package.duration",
            value: FactValue::Text(seconds),
        });
    }
    facts.push(Fact {
        label: "assets.package.tokens",
        value: FactValue::Text(manifest.feature_shape[0].to_string()),
    });
    facts.push(Fact {
        label: "assets.package.landmark_model",
        value: FactValue::Text(short_hash(&manifest.landmark_model_sha256)),
    });
    facts.push(Fact {
        label: "assets.package.feature_model",
        value: FactValue::Text(short_hash(&manifest.feature_model_sha256)),
    });
    facts
}

/// The catalog key of a package state.
fn state_key(state: &AssetPackageState) -> &'static str {
    match state {
        AssetPackageState::Preparing => "assets.package.preparing",
        AssetPackageState::Locked => "assets.package.locked",
    }
}

/// How long the frames run, to one decimal, or `None` when the frame rate is zero.
///
/// The division is done in floating point on purpose: `frame_count / video_fps` in
/// integers would report a two-second clip as "2" and a 3001-frame one as "120".
/// The frame count goes through `u32::try_from` so the conversion to `f64` is
/// lossless and cannot wrap; the manifest caps it at 100 million, so a value that
/// does not fit is a manifest no validator would have accepted, and the line is
/// left out rather than shown wrong.
fn duration_seconds(frames: u64, fps: u32) -> Option<String> {
    if fps == 0 {
        return None;
    }
    let frames = u32::try_from(frames).map(f64::from).ok()?;
    Some(format!("{:.1}", frames / f64::from(fps)))
}

/// The first eight characters of a hash, or `-` when it has not been filled in.
///
/// A preparing manifest carries empty hashes, and a full 64 character digest on a
/// summary card is noise: eight characters is enough to tell two models apart.
fn short_hash(hash: &str) -> String {
    if hash.is_empty() {
        return "-".to_owned();
    }
    hash.chars().take(8).collect()
}

/// One step of asset preparation.
///
/// The order is what the data forces, not a layout preference: frame extraction
/// reads the 25fps video normalisation writes, feature extraction reads the 16kHz
/// audio it writes, and the lock reads what both produced.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Step {
    Normalize,
    ExtractFrames,
    ExtractFeatures,
    Lock,
}

impl Step {
    /// The four steps in the order the page paints them.
    pub const ALL: [Step; 4] = [
        Step::Normalize,
        Step::ExtractFrames,
        Step::ExtractFeatures,
        Step::Lock,
    ];

    /// The command this step submits.
    pub fn kind(self) -> TaskKind {
        match self {
            Self::Normalize => TaskKind::NormalizeMedia,
            Self::ExtractFrames => TaskKind::ExtractFrames,
            Self::ExtractFeatures => TaskKind::ExtractFeatures,
            Self::Lock => TaskKind::LockAssetPackage,
        }
    }

    /// The catalog key of the step's name.
    pub fn label_key(self) -> &'static str {
        match self {
            Self::Normalize => "assets.step.normalize",
            Self::ExtractFrames => "assets.step.extract_frames",
            Self::ExtractFeatures => "assets.step.extract_features",
            Self::Lock => "assets.step.lock",
        }
    }

    /// The catalog key of the sentence describing what the step produces.
    pub fn hint_key(self) -> &'static str {
        match self {
            Self::Normalize => "assets.hint.normalize",
            Self::ExtractFrames => "assets.hint.extract_frames",
            Self::ExtractFeatures => "assets.hint.extract_features",
            Self::Lock => "assets.hint.lock",
        }
    }

    /// A stable element id prefix, so a re-render keeps each row's state.
    pub fn element_id(self) -> &'static str {
        match self {
            Self::Normalize => "normalize",
            Self::ExtractFrames => "extract-frames",
            Self::ExtractFeatures => "extract-features",
            Self::Lock => "lock",
        }
    }
}

/// What the page may do with one step right now.
///
/// The shell-wide gate -- no project, no worker, a task already running -- is
/// `tasks::blocked_key`. This is only about the step's own prerequisites.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StepState {
    /// A prerequisite is missing; the key says which one.
    Blocked(&'static str),
    /// Ready to run for the first time.
    Ready,
    /// The artifacts are there, so running it again redoes the work.
    Done,
    /// The package is locked, so no command would be accepted.
    Locked,
}

impl StepState {
    /// Whether the step's own prerequisites allow a submission.
    pub fn is_submittable(self) -> bool {
        match self {
            // Redoing a step is normal: the source video changed, a model
            // changed, or half the artifacts were deleted by hand.
            Self::Ready | Self::Done => true,
            Self::Blocked(_) | Self::Locked => false,
        }
    }
}

/// Decide what `step` can do, given what the project holds and whether a source
/// video has been chosen.
///
/// A locked package sweeps all four steps. There is no unlock command, the lock
/// refuses to run on a package that is already locked, and the feature commit
/// refuses to touch one, so saying it once on every button is more honest than
/// four Chinese rejections. Redoing a locked project means choosing another
/// directory, or deleting `assets/assets.json` by hand.
///
/// `Step::Lock` therefore never reports `Done`: "done" for the lock is exactly
/// `Locked`.
pub fn step_state(step: Step, survey: &AssetSurvey, has_source: bool) -> StepState {
    if survey.is_locked() {
        return StepState::Locked;
    }
    match step {
        Step::Normalize => match (survey.video && survey.audio, has_source) {
            (true, _) => StepState::Done,
            (false, true) => StepState::Ready,
            (false, false) => StepState::Blocked("assets.blocked.no_source"),
        },
        Step::ExtractFrames => {
            match (
                survey.frames && survey.landmarks && survey.quality,
                survey.video,
            ) {
                (true, _) => StepState::Done,
                (false, true) => StepState::Ready,
                (false, false) => StepState::Blocked("assets.blocked.no_video"),
            }
        }
        Step::ExtractFeatures => match (survey.features, survey.audio) {
            (true, _) => StepState::Done,
            (false, true) => StepState::Ready,
            (false, false) => StepState::Blocked("assets.blocked.no_audio"),
        },
        Step::Lock => match survey.features && survey.quality {
            true => StepState::Ready,
            false => StepState::Blocked("assets.blocked.no_features"),
        },
    }
}

/// The request `step` submits for the project at `project_dir`.
///
/// Only normalisation can fail to produce one, and only because the source video
/// is the one path that comes from the user rather than from the asset contract.
/// Nothing here is configurable: the worker fixes 25fps, 16kHz and mono, and
/// rejects anything else, so a form for those targets would only build an
/// interface the worker refuses.
pub fn request(step: Step, project_dir: &Path, source_video: Option<&Path>) -> Option<Request> {
    match step {
        Step::Normalize => source_video.map(|input| {
            Request::NormalizeMedia(NormalizeMediaParams {
                input: input.to_path_buf(),
                output_dir: assets_dir(project_dir),
            })
        }),
        Step::ExtractFrames => Some(Request::ExtractFrames(ExtractFramesParams {
            project_dir: project_dir.to_path_buf(),
            video: normalized_video(project_dir),
        })),
        Step::ExtractFeatures => Some(Request::ExtractFeatures(ExtractFeaturesParams {
            project_dir: project_dir.to_path_buf(),
            audio: normalized_audio(project_dir),
        })),
        Step::Lock => Some(Request::LockAssetPackage(ProjectDirParams {
            project_dir: project_dir.to_path_buf(),
        })),
    }
}
