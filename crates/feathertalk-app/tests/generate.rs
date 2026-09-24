//! What the generate page knows without a window.

use std::fs;
use std::path::{Path, PathBuf};

use feathertalk_app::assets::AssetSurvey;
use feathertalk_app::generate::{
    audio_path, checkpoint_path, ensure_renders_dir, form_state, output_path, render_index,
    render_name, renders_dir, video_stem, with_video_extension, FormState, GenerateForm,
    PickedCheckpoint, RenderRequest, RenderSurvey, RenderedVideo, DEFAULT_PREVIEW_FRAMES,
    MAX_PREVIEW_FRAMES, MIN_PREVIEW_FRAMES, PREVIEW_PREFIX, RENDER_PREFIX,
};
use feathertalk_app::training::TrainingSurvey;
use feathertalk_domain::{RenderParams, Request};
use feathertalk_project::{AssetManifest, AssetPackageState, FeatureType};

/// A project directory with an empty `outputs/renders` in it.
fn project() -> tempfile::TempDir {
    let dir = tempfile::tempdir().expect("a temporary directory");
    fs::create_dir_all(dir.path().join("outputs").join("renders")).expect("the render directory");
    dir
}

/// Write `bytes` bytes of nothing to `outputs/renders/<name>`.
fn video(project_dir: &Path, name: &str, bytes: usize) {
    let path = project_dir.join("outputs").join("renders").join(name);
    fs::write(&path, vec![0u8; bytes]).expect("the file writes");
}

/// Write the checkpoint of `step` under `models/unet`, with `body` as its state.
///
/// An empty `body` writes no state file at all, which is the directory a user can
/// point the dialog at.
fn checkpoint(project_dir: &Path, step: u64, body: &str) -> PathBuf {
    let path = project_dir
        .join("models")
        .join("unet")
        .join(format!("checkpoint-{step:08}"));
    fs::create_dir_all(&path).expect("the checkpoint directory");
    if !body.is_empty() {
        fs::write(path.join("training-state.json"), body).expect("the state writes");
    }
    path
}

/// The `training-state.json` the worker writes beside a checkpoint's weights.
fn state_json() -> String {
    r#"{
  "schema_version": 1,
  "epoch": 12,
  "global_step": 2400,
  "random_seed": 1,
  "training_config": {
    "mode": "mouth_roi",
    "batch_size": 1,
    "learning_rate": 0.0001,
    "total_epochs": 200,
    "temporal_stride": 1,
    "mouth_weight": 4.0,
    "temporal_weight": 0.5,
    "temporal_mouth_weight": 4.0,
    "perceptual_weight": 0.01
  }
}"#
    .to_owned()
}

/// An asset survey whose package is in `state` and whose track is or is not there.
///
/// The generate page asks this photograph two questions -- is the package locked,
/// and is there a normalised track -- so the rest is the shape a locked package
/// has and is never read.
fn package(state: AssetPackageState, audio: bool) -> AssetSurvey {
    AssetSurvey {
        audio,
        manifest: Some(AssetManifest {
            schema_version: 1,
            state,
            video_fps: 25,
            audio_sample_rate: 16_000,
            audio_channels: 1,
            frame_count: 188,
            frame_width: 160,
            frame_height: 160,
            feature_type: FeatureType::FeatherHubert,
            feature_shape: [188, 2, 1024],
            landmark_model_sha256: std::iter::repeat_n('a', 64).collect(),
            feature_model_sha256: std::iter::repeat_n('b', 64).collect(),
        }),
        ..AssetSurvey::default()
    }
}

#[test]
fn the_render_directory_sits_under_outputs() {
    let dir = Path::new("C:/projects/demo");

    assert_eq!(renders_dir(dir), dir.join("outputs").join("renders"));
}

#[test]
fn a_derived_name_pads_the_index_to_three_digits() {
    assert_eq!(render_name(RENDER_PREFIX, 1), "render-001.mp4");
    assert_eq!(render_name(PREVIEW_PREFIX, 42), "preview-042.mp4");
}

#[test]
fn an_index_past_nine_hundred_and_ninety_nine_simply_grows() {
    assert_eq!(render_name(RENDER_PREFIX, 1000), "render-1000.mp4");
}

#[test]
fn the_extension_is_matched_without_case() {
    assert_eq!(video_stem("render-001.mp4"), Some("render-001"));
    assert_eq!(video_stem("render-001.MP4"), Some("render-001"));
    assert_eq!(video_stem("render-001.mkv"), None);
    assert_eq!(video_stem("render-001"), None);
}

#[test]
fn an_index_reads_back_out_of_a_name() {
    assert_eq!(render_index("render-001.mp4", RENDER_PREFIX), Some(1));
    assert_eq!(render_index("render-1000.mp4", RENDER_PREFIX), Some(1000));
    assert_eq!(render_index("preview-007.MP4", PREVIEW_PREFIX), Some(7));
}

#[test]
fn a_name_that_is_not_ours_has_no_index() {
    for name in [
        "preview-001.mp4",
        "render-abc.mp4",
        "render-.mp4",
        "render-001.mkv",
        "render-001",
        "render-99999999999.mp4",
        "clip.mp4",
    ] {
        assert_eq!(render_index(name, RENDER_PREFIX), None, "{name}");
    }
}

#[test]
fn a_path_without_an_extension_gets_the_video_one() {
    assert_eq!(
        with_video_extension(PathBuf::from("C:/out/clip")),
        PathBuf::from("C:/out/clip.mp4")
    );
}

#[test]
fn a_path_that_already_names_a_container_is_left_alone() {
    for name in ["C:/out/clip.mkv", "C:/out/clip.mp4", "C:/out/clip.MP4"] {
        assert_eq!(
            with_video_extension(PathBuf::from(name)),
            PathBuf::from(name)
        );
    }
}

#[test]
fn a_project_that_never_rendered_starts_both_counters_at_one() {
    let dir = tempfile::tempdir().expect("a temporary directory");

    let survey = RenderSurvey::inspect(dir.path());

    assert!(survey.is_empty(), "there is no outputs/renders at all");
    assert_eq!(survey.next_render, 1);
    assert_eq!(survey.next_preview, 1);
}

#[test]
fn an_empty_render_directory_reads_the_same_way() {
    let dir = project();

    let survey = RenderSurvey::inspect(dir.path());

    assert!(survey.is_empty());
    assert_eq!(survey.next_render, 1);
    assert_eq!(survey.next_preview, 1);
}

#[test]
fn each_prefix_counts_on_its_own_and_holes_do_not_close() {
    let dir = project();
    video(dir.path(), "render-001.mp4", 3);
    video(dir.path(), "render-003.mp4", 5);
    video(dir.path(), "preview-002.mp4", 7);

    let survey = RenderSurvey::inspect(dir.path());

    assert_eq!(
        survey.next_render, 4,
        "one past the largest, not one past the count"
    );
    assert_eq!(survey.next_preview, 3);
}

#[test]
fn the_newest_index_comes_first() {
    let dir = project();
    video(dir.path(), "render-001.mp4", 1);
    video(dir.path(), "render-012.mp4", 1);
    video(dir.path(), "preview-007.mp4", 1);

    let survey = RenderSurvey::inspect(dir.path());

    let names: Vec<&str> = survey
        .videos
        .iter()
        .map(|video| video.name.as_str())
        .collect();
    assert_eq!(
        names,
        ["render-012.mp4", "preview-007.mp4", "render-001.mp4"]
    );
}

#[test]
fn a_video_records_its_path_and_its_size() {
    let dir = project();
    video(dir.path(), "render-001.mp4", 11);

    let survey = RenderSurvey::inspect(dir.path());

    assert_eq!(
        survey.videos,
        vec![RenderedVideo {
            name: "render-001.mp4".to_owned(),
            path: dir
                .path()
                .join("outputs")
                .join("renders")
                .join("render-001.mp4"),
            index: Some(1),
            bytes: 11,
        }]
    );
}

#[test]
fn a_video_nobody_here_named_is_still_playable() {
    let dir = project();
    video(dir.path(), "clip.mp4", 2);

    let survey = RenderSurvey::inspect(dir.path());

    assert_eq!(
        survey.videos.len(),
        1,
        "it is a video in the render directory"
    );
    assert_eq!(survey.videos[0].index, None, "but it is not one of ours");
    assert_eq!(survey.next_render, 1, "so it moves neither counter");
    assert_eq!(survey.next_preview, 1);
}

#[test]
fn what_is_not_a_video_file_is_not_listed() {
    let dir = project();
    video(dir.path(), "notes.txt", 4);
    fs::create_dir_all(
        dir.path()
            .join("outputs")
            .join("renders")
            .join("render-005.mp4"),
    )
    .expect("a directory wearing a video name");

    let survey = RenderSurvey::inspect(dir.path());

    assert!(survey.videos.is_empty());
    assert_eq!(survey.next_render, 1, "a directory is not a render");
}

#[test]
fn a_new_form_renders_the_whole_sequence() {
    let form = GenerateForm::default();

    assert!(
        !form.preview(),
        "a render is a full render until asked otherwise"
    );
    assert_eq!(form.preview_frames(), DEFAULT_PREVIEW_FRAMES);
    assert_eq!(
        form.max_output_frames(),
        None,
        "no cap is the whole sequence"
    );
}

#[test]
fn turning_the_preview_on_caps_the_output() {
    let mut form = GenerateForm::default();

    form.set_preview(true);

    assert_eq!(
        form.max_output_frames(),
        Some(u64::from(DEFAULT_PREVIEW_FRAMES))
    );
}

#[test]
fn a_frame_count_under_one_is_lifted_to_one() {
    let mut form = GenerateForm::default();

    form.set_preview_frames(0.0);

    // `Some(0)` is refused by the worker, so the control never produces it.
    assert_eq!(form.preview_frames(), MIN_PREVIEW_FRAMES);
}

#[test]
fn a_frame_count_over_the_ceiling_is_held_there() {
    let mut form = GenerateForm::default();

    form.set_preview_frames(1e12);

    assert_eq!(form.preview_frames(), MAX_PREVIEW_FRAMES);
}

#[test]
fn a_fractional_frame_count_is_rounded() {
    let mut form = GenerateForm::default();

    form.set_preview_frames(12.6);

    assert_eq!(form.preview_frames(), 13);
}

#[test]
fn a_frame_count_that_is_not_a_number_changes_nothing() {
    let mut form = GenerateForm::default();
    form.set_preview_frames(64.0);

    form.set_preview_frames(f64::NAN);
    form.set_preview_frames(f64::INFINITY);

    assert_eq!(form.preview_frames(), 64);
}

#[test]
fn the_preview_length_is_said_in_seconds() {
    let mut form = GenerateForm::default();

    form.set_preview_frames(100.0);
    assert_eq!(form.preview_seconds(), "4.0");

    form.set_preview_frames(25.0);
    assert_eq!(form.preview_seconds(), "1.0");
}

#[test]
fn adopting_a_checkpoint_reads_the_step_out_of_its_name() {
    let dir = project();
    let path = checkpoint(dir.path(), 2400, &state_json());

    let picked = PickedCheckpoint::adopt(path.clone());

    assert_eq!(picked.path, path);
    assert_eq!(picked.step, Some(2400));
}

#[test]
fn adopting_a_checkpoint_reads_the_state_beside_the_weights() {
    let dir = project();
    let path = checkpoint(dir.path(), 2400, &state_json());

    let picked = PickedCheckpoint::adopt(path);

    let state = picked.state.expect("the state file reads");
    assert_eq!(state.epoch, 12);
    assert_eq!(state.training_config.mode, "mouth_roi");
    assert_eq!(picked.error, None);
}

#[test]
fn a_chosen_directory_without_a_state_file_records_the_reason() {
    let dir = project();
    let path = checkpoint(dir.path(), 2400, "");

    let picked = PickedCheckpoint::adopt(path);

    assert_eq!(picked.step, Some(2400), "the name still says which step");
    assert!(picked.state.is_none());
    // The survey reads the same absence as "not written yet". This directory is
    // one a user pointed at, and saying nothing about it is the worse answer.
    assert!(
        picked.error.is_some(),
        "an explicit choice reports its failure"
    );
}

#[test]
fn a_directory_that_is_not_a_checkpoint_is_adopted_anyway() {
    let dir = project();
    let path = dir.path().join("pictures");
    fs::create_dir_all(&path).expect("the directory");

    let picked = PickedCheckpoint::adopt(path);

    // Whether a directory holds a model this build can render is the worker's
    // judgement; the interface passes it on and `ModelIncompatible` comes back.
    assert_eq!(picked.step, None);
    assert!(picked.error.is_some());
}

#[test]
fn the_latest_checkpoint_stands_in_until_one_is_chosen() {
    let dir = project();
    checkpoint(dir.path(), 1200, &state_json());
    let newest = checkpoint(dir.path(), 2400, &state_json());
    let training = TrainingSurvey::inspect(dir.path());

    assert_eq!(checkpoint_path(None, &training), Some(newest.as_path()));
}

#[test]
fn a_chosen_checkpoint_wins_over_the_latest() {
    let dir = project();
    let older = checkpoint(dir.path(), 1200, &state_json());
    checkpoint(dir.path(), 2400, &state_json());
    let training = TrainingSurvey::inspect(dir.path());
    let picked = PickedCheckpoint::adopt(older.clone());

    assert_eq!(
        checkpoint_path(Some(&picked), &training),
        Some(older.as_path())
    );
}

#[test]
fn nothing_trained_and_nothing_chosen_is_no_checkpoint() {
    let dir = project();
    let training = TrainingSurvey::inspect(dir.path());

    assert_eq!(checkpoint_path(None, &training), None);
}

#[test]
fn the_project_track_stands_in_until_one_is_chosen() {
    let dir = project();
    let survey = package(AssetPackageState::Locked, true);

    assert_eq!(
        audio_path(None, dir.path(), &survey),
        Some(dir.path().join("assets").join("audio_16k_mono.wav")),
        "the track the features were taken from"
    );
}

#[test]
fn a_chosen_track_is_used_as_given() {
    let dir = project();
    let survey = package(AssetPackageState::Locked, true);
    let elsewhere = dir.path().join("narration.wav");

    let resolved = audio_path(Some(elsewhere.as_path()), dir.path(), &survey);

    assert_eq!(resolved, Some(elsewhere));
}

#[test]
fn a_project_without_a_normalised_track_has_none() {
    let dir = project();
    let survey = package(AssetPackageState::Locked, false);

    // Locking checks frames, landmarks and features; nobody promises the wav is
    // still where normalisation left it.
    assert_eq!(audio_path(None, dir.path(), &survey), None);
}

#[test]
fn the_derived_output_follows_the_preview_switch() {
    let dir = project();
    let renders = RenderSurvey::inspect(dir.path());

    let full = output_path(None, dir.path(), &renders, false);
    let preview = output_path(None, dir.path(), &renders, true);

    assert_eq!(full, renders_dir(dir.path()).join("render-001.mp4"));
    assert_eq!(preview, renders_dir(dir.path()).join("preview-001.mp4"));
}

#[test]
fn a_chosen_output_path_is_used_as_given() {
    let dir = project();
    let renders = RenderSurvey::inspect(dir.path());
    let elsewhere = dir.path().join("elsewhere").join("mine.mkv");

    let resolved = output_path(Some(elsewhere.as_path()), dir.path(), &renders, true);

    // An explicit choice beats the derivation, extension included: after one the
    // preview switch stops renaming the file.
    assert_eq!(resolved, elsewhere);
}

#[test]
fn an_unlocked_package_is_refused_first() {
    let dir = project();

    // Nothing else is in place either; the lock is still the answer, because
    // `validate_project_dir` is the first thing the worker does.
    let state = form_state(
        &package(AssetPackageState::Preparing, false),
        &TrainingSurvey::default(),
        None,
        None,
        dir.path(),
    );

    assert_eq!(state, FormState::Blocked("generate.blocked.not_locked"));
    assert!(!state.is_submittable());
}

#[test]
fn a_locked_package_with_nothing_trained_asks_for_a_checkpoint() {
    let dir = project();

    let state = form_state(
        &package(AssetPackageState::Locked, true),
        &TrainingSurvey::default(),
        None,
        None,
        dir.path(),
    );

    assert_eq!(state, FormState::Blocked("generate.blocked.no_checkpoint"));
}

#[test]
fn a_chosen_checkpoint_answers_that_condition() {
    let dir = project();
    let picked = PickedCheckpoint::adopt(checkpoint(dir.path(), 2400, &state_json()));

    // The survey is empty on purpose: the choice is what the request carries, and
    // a choice that would not read is a note rather than a block.
    let state = form_state(
        &package(AssetPackageState::Locked, true),
        &TrainingSurvey::default(),
        Some(&picked),
        None,
        dir.path(),
    );

    assert_eq!(state, FormState::Ready);
}

#[test]
fn a_locked_package_without_a_track_asks_for_one() {
    let dir = project();
    checkpoint(dir.path(), 2400, &state_json());
    let training = TrainingSurvey::inspect(dir.path());

    let state = form_state(
        &package(AssetPackageState::Locked, false),
        &training,
        None,
        None,
        dir.path(),
    );

    assert_eq!(state, FormState::Blocked("generate.blocked.no_audio"));
}

#[test]
fn a_chosen_track_answers_that_condition() {
    let dir = project();
    checkpoint(dir.path(), 2400, &state_json());
    let training = TrainingSurvey::inspect(dir.path());
    let track = dir.path().join("narration.wav");

    let state = form_state(
        &package(AssetPackageState::Locked, false),
        &training,
        None,
        Some(track.as_path()),
        dir.path(),
    );

    assert_eq!(state, FormState::Ready);
}

#[test]
fn everything_in_place_is_ready() {
    let dir = project();
    checkpoint(dir.path(), 2400, &state_json());
    let training = TrainingSurvey::inspect(dir.path());

    let state = form_state(
        &package(AssetPackageState::Locked, true),
        &training,
        None,
        None,
        dir.path(),
    );

    assert_eq!(state, FormState::Ready);
    assert!(state.is_submittable());
}

#[test]
fn a_full_render_carries_no_frame_cap() {
    let dir = project();
    let weights = checkpoint(dir.path(), 2400, &state_json());
    let audio = dir.path().join("assets").join("audio_16k_mono.wav");
    let output = renders_dir(dir.path()).join("render-001.mp4");

    let request = RenderRequest {
        project_dir: dir.path(),
        checkpoint: weights.as_path(),
        audio: audio.as_path(),
        output: output.as_path(),
        max_output_frames: None,
    }
    .build();

    assert_eq!(
        request,
        Request::Render(RenderParams {
            project_dir: dir.path().to_path_buf(),
            checkpoint: weights,
            audio,
            output,
            max_output_frames: None,
        })
    );
}

#[test]
fn a_preview_carries_its_frame_cap() {
    let dir = project();
    let weights = checkpoint(dir.path(), 2400, &state_json());
    let audio = dir.path().join("narration.wav");
    let output = renders_dir(dir.path()).join("preview-001.mp4");

    let request = RenderRequest {
        project_dir: dir.path(),
        checkpoint: weights.as_path(),
        audio: audio.as_path(),
        output: output.as_path(),
        max_output_frames: Some(100),
    }
    .build();

    match request {
        Request::Render(params) => assert_eq!(params.max_output_frames, Some(100)),
        other => panic!("a render request is a render request: {other:?}"),
    }
}

#[test]
fn the_render_directory_is_created_and_creating_it_twice_is_fine() {
    let dir = tempfile::tempdir().expect("a temporary directory");

    ensure_renders_dir(dir.path()).expect("the directory is created");
    ensure_renders_dir(dir.path()).expect("a second call is not a failure");

    assert!(renders_dir(dir.path()).is_dir());
}
