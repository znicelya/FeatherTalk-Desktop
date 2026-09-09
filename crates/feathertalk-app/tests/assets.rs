//! What the asset page can tell about a project directory before it renders.

use std::fs;
use std::path::Path;

use feathertalk_app::assets::{
    AssetSurvey, FactValue, Step, StepState, facts, request, step_state,
};
use feathertalk_domain::Request;

/// A project directory with the asset tree the tests write into.
fn project() -> tempfile::TempDir {
    let dir = tempfile::tempdir().expect("a temporary directory");
    fs::create_dir_all(dir.path().join("assets")).expect("the asset directory");
    dir
}

/// Write `body` to `path`, creating whatever directories it needs.
fn write(path: &Path, body: &str) {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).expect("the parent directory");
    }
    fs::write(path, body).expect("the file writes");
}

/// A 64 character hex string, which is what the manifest calls a sha256.
fn sha256(seed: char) -> String {
    std::iter::repeat_n(seed, 64).collect()
}

/// A manifest in the preparing state: zeroes and empty hashes are allowed there.
fn preparing_json(fps: u32, frames: u64) -> String {
    format!(
        r#"{{
  "schema_version": 1,
  "state": "preparing",
  "video_fps": {fps},
  "audio_sample_rate": 0,
  "audio_channels": 0,
  "frame_count": {frames},
  "frame_width": 0,
  "frame_height": 0,
  "feature_type": "feather_hubert",
  "feature_shape": [0, 0, 0],
  "landmark_model_sha256": "",
  "feature_model_sha256": ""
}}"#
    )
}

/// A manifest in the locked state, where every field has to be filled in.
fn locked_json(frames: u64) -> String {
    let landmark = sha256('a');
    let feature = sha256('b');
    format!(
        r#"{{
  "schema_version": 1,
  "state": "locked",
  "video_fps": 25,
  "audio_sample_rate": 16000,
  "audio_channels": 1,
  "frame_count": {frames},
  "frame_width": 512,
  "frame_height": 512,
  "feature_type": "feather_hubert",
  "feature_shape": [{frames}, 2, 1024],
  "landmark_model_sha256": "{landmark}",
  "feature_model_sha256": "{feature}"
}}"#
    )
}

/// The value of the fact labelled `label`, or `None` when it is not shown.
fn fact(survey: &AssetSurvey, label: &str) -> Option<FactValue> {
    facts(survey)
        .into_iter()
        .find(|fact| fact.label == label)
        .map(|fact| fact.value)
}

#[test]
fn an_empty_project_has_nothing() {
    let project = project();

    let survey = AssetSurvey::inspect(project.path());

    assert!(!survey.video, "no normalised video");
    assert!(!survey.audio, "no normalised audio");
    assert!(!survey.frames, "no frames");
    assert!(!survey.landmarks, "no landmarks");
    assert!(!survey.quality, "no quality report");
    assert!(!survey.features, "no features");
    assert!(survey.manifest.is_none(), "no manifest");
    assert!(
        survey.manifest_error.is_none(),
        "an absent manifest is not an error"
    );
}

#[test]
fn each_artifact_shows_up_when_it_lands() {
    let project = project();
    let assets = project.path().join("assets");
    write(&assets.join("video_25fps.mp4"), "video");
    write(&assets.join("audio_16k_mono.wav"), "audio");
    write(&assets.join("frames/000000.jpg"), "frame");
    write(&assets.join("landmarks/000000.lms"), "0 0");
    write(&assets.join("quality.json"), "{}");
    write(&assets.join("features/feather_hubert.f32"), "features");

    let survey = AssetSurvey::inspect(project.path());

    assert!(survey.video);
    assert!(survey.audio);
    assert!(survey.frames);
    assert!(survey.landmarks);
    assert!(survey.quality);
    assert!(survey.features);
}

#[test]
fn an_empty_frame_directory_does_not_count() {
    let project = project();
    fs::create_dir_all(project.path().join("assets/frames")).expect("the frame directory");

    let survey = AssetSurvey::inspect(project.path());

    assert!(
        !survey.frames,
        "a directory with no frames in it is not done"
    );
}

#[test]
fn a_readable_manifest_is_parsed() {
    let project = project();
    write(
        &project.path().join("assets/assets.json"),
        &preparing_json(25, 100),
    );

    let survey = AssetSurvey::inspect(project.path());

    let manifest = survey.manifest.as_ref().expect("the manifest parses");
    assert_eq!(manifest.frame_count, 100);
    assert!(!survey.is_locked(), "preparing is not locked");
    assert!(survey.manifest_error.is_none());
}

#[test]
fn a_locked_manifest_is_reported_as_locked() {
    let project = project();
    write(
        &project.path().join("assets/assets.json"),
        &locked_json(3000),
    );

    let survey = AssetSurvey::inspect(project.path());

    assert!(survey.is_locked(), "a locked package reports itself locked");
}

#[test]
fn a_broken_manifest_becomes_an_error_line() {
    let project = project();
    write(&project.path().join("assets/assets.json"), "{");

    let survey = AssetSurvey::inspect(project.path());

    assert!(
        survey.manifest.is_none(),
        "nothing to show from a broken file"
    );
    assert!(
        survey.manifest_error.is_some(),
        "the reason the manifest could not be read is kept"
    );
}

#[test]
fn the_facts_list_the_manifest_numbers() {
    let locked = project();
    write(
        &locked.path().join("assets/assets.json"),
        &locked_json(3000),
    );

    let survey = AssetSurvey::inspect(locked.path());

    assert_eq!(
        fact(&survey, "assets.package.state"),
        Some(FactValue::Key("assets.package.locked"))
    );
    assert_eq!(
        fact(&survey, "assets.package.resolution"),
        Some(FactValue::Text("512×512".to_owned()))
    );
    assert_eq!(
        fact(&survey, "assets.package.duration"),
        Some(FactValue::Text("120.0".to_owned()))
    );

    let preparing = project();
    write(
        &preparing.path().join("assets/assets.json"),
        &preparing_json(0, 0),
    );
    let survey = AssetSurvey::inspect(preparing.path());

    assert!(
        fact(&survey, "assets.package.duration").is_none(),
        "a frame rate of zero has no duration to report"
    );
    assert_eq!(
        fact(&survey, "assets.package.landmark_model"),
        Some(FactValue::Text("-".to_owned())),
        "an empty hash is shown as absent rather than as an empty line"
    );
}

/// The state of every step, in the fixed order the page paints them.
fn states(survey: &AssetSurvey, has_source: bool) -> Vec<StepState> {
    Step::ALL
        .into_iter()
        .map(|step| step_state(step, survey, has_source))
        .collect()
}

#[test]
fn an_empty_project_blocks_every_step() {
    let survey = AssetSurvey::default();

    assert_eq!(
        states(&survey, false),
        vec![
            StepState::Blocked("assets.blocked.no_source"),
            StepState::Blocked("assets.blocked.no_video"),
            StepState::Blocked("assets.blocked.no_audio"),
            StepState::Blocked("assets.blocked.no_features"),
        ]
    );
}

#[test]
fn a_chosen_video_makes_normalisation_ready() {
    let survey = AssetSurvey::default();

    assert_eq!(step_state(Step::Normalize, &survey, true), StepState::Ready);
}

#[test]
fn normalised_media_finishes_the_first_step_and_opens_the_next_two() {
    let survey = AssetSurvey {
        video: true,
        audio: true,
        ..AssetSurvey::default()
    };

    assert_eq!(
        states(&survey, true),
        vec![
            StepState::Done,
            StepState::Ready,
            StepState::Ready,
            StepState::Blocked("assets.blocked.no_features"),
        ]
    );
}

#[test]
fn frames_without_a_report_are_not_done() {
    let survey = AssetSurvey {
        video: true,
        frames: true,
        landmarks: true,
        ..AssetSurvey::default()
    };

    assert_eq!(
        step_state(Step::ExtractFrames, &survey, false),
        StepState::Ready,
        "the quality report is part of what extraction publishes"
    );
}

#[test]
fn features_and_a_report_make_the_lock_ready() {
    let survey = AssetSurvey {
        video: true,
        audio: true,
        frames: true,
        landmarks: true,
        quality: true,
        features: true,
        ..AssetSurvey::default()
    };

    assert_eq!(
        states(&survey, true),
        vec![
            StepState::Done,
            StepState::Done,
            StepState::Done,
            StepState::Ready,
        ]
    );
}

#[test]
fn a_locked_package_locks_every_step() {
    let locked = project();
    write(
        &locked.path().join("assets/assets.json"),
        &locked_json(3000),
    );
    let survey = AssetSurvey::inspect(locked.path());

    for state in states(&survey, true) {
        assert_eq!(state, StepState::Locked);
        assert!(
            !state.is_submittable(),
            "a locked package accepts no command"
        );
    }
}

#[test]
fn a_done_step_can_still_be_submitted() {
    assert!(StepState::Ready.is_submittable());
    assert!(
        StepState::Done.is_submittable(),
        "redoing a step is a normal thing to want"
    );
    assert!(!StepState::Blocked("assets.blocked.no_source").is_submittable());
}

#[test]
fn each_step_builds_its_own_request() {
    let project = Path::new("/tmp/project");
    let assets = project.join("assets");
    let source = Path::new("/tmp/input.mp4");

    let Some(Request::NormalizeMedia(params)) = request(Step::Normalize, project, Some(source))
    else {
        panic!("normalisation carries the chosen video");
    };
    assert_eq!(params.input, source);
    assert_eq!(params.output_dir, assets);
    assert!(
        request(Step::Normalize, project, None).is_none(),
        "there is nothing to normalise without a video"
    );

    let Some(Request::ExtractFrames(params)) = request(Step::ExtractFrames, project, None) else {
        panic!("frame extraction only needs the project");
    };
    assert_eq!(params.project_dir, project);
    assert_eq!(params.video, assets.join("video_25fps.mp4"));

    let Some(Request::ExtractFeatures(params)) = request(Step::ExtractFeatures, project, None)
    else {
        panic!("feature extraction only needs the project");
    };
    assert_eq!(params.audio, assets.join("audio_16k_mono.wav"));

    let Some(Request::LockAssetPackage(params)) = request(Step::Lock, project, None) else {
        panic!("the lock only needs the project");
    };
    assert_eq!(params.project_dir, project);
}

#[test]
fn every_step_maps_to_the_command_it_submits() {
    let kinds: Vec<_> = Step::ALL.into_iter().map(Step::kind).collect();

    assert_eq!(
        kinds,
        vec![
            feathertalk_domain::TaskKind::NormalizeMedia,
            feathertalk_domain::TaskKind::ExtractFrames,
            feathertalk_domain::TaskKind::ExtractFeatures,
            feathertalk_domain::TaskKind::LockAssetPackage,
        ]
    );
}

/// The asset page and the training page paint the same kind of line, so the type
/// lives in one module and `assets` only re-exports it.
#[test]
fn the_fact_type_is_shared() {
    fn shared(fact: feathertalk_app::facts::Fact) -> &'static str {
        fact.label
    }

    let fact = feathertalk_app::assets::Fact {
        label: "assets.package.state",
        value: feathertalk_app::assets::FactValue::Key("assets.package.locked"),
    };

    assert_eq!(shared(fact), "assets.package.state");
}
