//! The project directory and source video a session works on.

use std::fs;
use std::path::{Path, PathBuf};

use feathertalk_app::generate::PickedCheckpoint;
use feathertalk_app::project::{ProjectState, absolute_dir, has_task_history};
use feathertalk_app::tasks::Note;

/// A project directory with an asset tree to write artifacts into.
fn project() -> tempfile::TempDir {
    let dir = tempfile::tempdir().expect("a temporary directory");
    fs::create_dir_all(dir.path().join("assets")).expect("the asset directory");
    dir
}

fn write(path: &Path, body: &str) {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).expect("the parent directory");
    }
    fs::write(path, body).expect("the file writes");
}

/// Publish a checkpoint of `step`, which is what makes a project look trained.
fn checkpoint(project_dir: &Path, step: u64) {
    let dir = project_dir
        .join("models")
        .join("unet")
        .join(format!("checkpoint-{step:08}"));
    fs::create_dir_all(&dir).expect("the checkpoint directory");
}

/// Write a rendered video, which is what makes a project look generated.
fn render(project_dir: &Path, name: &str) {
    write(
        &project_dir.join("outputs").join("renders").join(name),
        "video",
    );
}

#[test]
fn a_session_without_a_flag_has_no_project() {
    let state = ProjectState::opened(None);

    assert!(state.dir().is_none());
    assert!(state.source_video().is_none());
    assert!(state.log_dir().is_none());
    assert!(!state.survey().video, "nothing has been surveyed");
    assert!(state.notes().is_empty());
}

#[test]
fn the_flag_is_adopted_and_surveyed() {
    let project = project();
    write(&project.path().join("assets/video_25fps.mp4"), "video");

    let state = ProjectState::opened(Some(project.path().to_path_buf()));

    assert_eq!(state.dir(), Some(project.path()));
    assert!(state.survey().video, "the flag's directory was surveyed");
    assert_eq!(state.log_dir(), Some(project.path().join("logs")));
}

#[test]
fn a_relative_directory_becomes_absolute() {
    let (absolute, failure) = absolute_dir(PathBuf::from("project"));

    assert!(
        absolute.is_absolute(),
        "worker admission rejects a relative project path"
    );
    assert!(failure.is_none());
}

#[test]
fn choosing_another_project_drops_the_previous_video() {
    let first = project();
    let second = project();
    let mut state = ProjectState::opened(Some(first.path().to_path_buf()));
    state.select_video(PathBuf::from("D:/media/kanghui.mp4"));

    let adopted = state.select_dir(second.path().to_path_buf());

    assert_eq!(adopted, second.path());
    assert_eq!(state.dir(), Some(second.path()));
    assert!(
        state.source_video().is_none(),
        "the video belonged to the directory that was open a moment ago"
    );
}

#[test]
fn a_refresh_picks_up_a_new_artifact() {
    let project = project();
    let mut state = ProjectState::opened(Some(project.path().to_path_buf()));
    assert!(!state.survey().audio);

    write(&project.path().join("assets/audio_16k_mono.wav"), "audio");
    state.refresh();

    assert!(state.survey().audio, "the second photograph sees the file");
}

#[test]
fn a_note_is_kept_with_its_detail() {
    let mut state = ProjectState::opened(None);

    state.note(Note::new(
        "assets.note.pick_failed",
        Some("the dialog closed".to_owned()),
    ));

    assert_eq!(
        state.notes(),
        [Note::new(
            "assets.note.pick_failed",
            Some("the dialog closed".to_owned())
        )]
    );
}

#[test]
fn a_directory_without_a_manifest_has_no_history() {
    let project = project();

    assert!(
        !has_task_history(project.path()),
        "a new project has no history to fail to read"
    );

    write(&project.path().join("project.json"), "{}");

    assert!(has_task_history(project.path()));
}

#[test]
fn a_selected_video_is_remembered() {
    let mut state = ProjectState::opened(None);

    state.select_video(PathBuf::from("D:/media/kanghui.mp4"));

    assert_eq!(
        state.source_video(),
        Some(Path::new("D:/media/kanghui.mp4"))
    );
}

#[test]
fn opening_a_trained_project_photographs_the_training() {
    let project = project();
    checkpoint(project.path(), 376);

    let state = ProjectState::opened(Some(project.path().to_path_buf()));

    let checkpoint = state.training().checkpoint.as_ref().expect("a checkpoint");
    assert_eq!(checkpoint.step, 376);
}

#[test]
fn refresh_rephotographs_the_training() {
    let project = project();
    let mut state = ProjectState::opened(Some(project.path().to_path_buf()));
    assert!(state.training().checkpoint.is_none());

    checkpoint(project.path(), 188);
    state.refresh();

    assert!(
        state.training().checkpoint.is_some(),
        "the second photograph sees the checkpoint"
    );
}

#[test]
fn choosing_another_directory_forgets_the_training() {
    let trained = project();
    checkpoint(trained.path(), 376);
    let empty = project();
    let mut state = ProjectState::opened(Some(trained.path().to_path_buf()));

    state.select_dir(empty.path().to_path_buf());

    assert!(state.training().checkpoint.is_none());
    assert!(!state.training().has_history());
}

#[test]
fn opening_a_rendered_project_photographs_its_renders() {
    let project = project();
    render(project.path(), "render-001.mp4");

    let state = ProjectState::opened(Some(project.path().to_path_buf()));

    assert_eq!(state.renders().videos.len(), 1);
    assert_eq!(state.renders().next_render, 2);
    assert!(
        state.checkpoint().is_none(),
        "the latest one is the default"
    );
    assert!(
        state.audio().is_none(),
        "the project's own track is the default"
    );
    assert!(state.output().is_none(), "the derived name is the default");
}

#[test]
fn refresh_rephotographs_the_renders() {
    let project = project();
    let mut state = ProjectState::opened(Some(project.path().to_path_buf()));
    assert!(state.renders().is_empty());

    render(project.path(), "preview-001.mp4");
    state.refresh();

    assert_eq!(state.renders().next_preview, 2);
}

#[test]
fn choosing_another_directory_forgets_the_render_choices() {
    let first = project();
    render(first.path(), "render-001.mp4");
    let mut state = ProjectState::opened(Some(first.path().to_path_buf()));
    state.select_checkpoint(PickedCheckpoint::adopt(first.path().join("models")));
    state.select_audio(first.path().join("narration.wav"));
    state.select_output(first.path().join("mine.mp4"));

    let second = project();
    state.select_dir(second.path().to_path_buf());

    // The three choices belonged to the directory that was open a moment ago,
    // which is why `source_video` is dropped in the same place.
    assert!(state.checkpoint().is_none());
    assert!(state.audio().is_none());
    assert!(state.output().is_none());
    assert!(state.renders().is_empty(), "and the photograph is retaken");
}

#[test]
fn going_back_to_the_latest_checkpoint_clears_the_chosen_one() {
    let project = project();
    let mut state = ProjectState::opened(Some(project.path().to_path_buf()));
    state.select_checkpoint(PickedCheckpoint::adopt(project.path().join("models")));
    assert!(state.checkpoint().is_some());

    state.use_latest_checkpoint();

    assert!(state.checkpoint().is_none());
}
