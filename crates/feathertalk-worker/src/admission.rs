use std::{fs, path::Path};

use feathertalk_domain::{ErrorCode, TaskError, TaskStage};

use crate::error_map::clamp;

/// The manifest every project directory carries. `feathertalk-project` owns the
/// name but exports no constant for it (`src/package.rs:66`), so the literal is
/// duplicated the way `cli/src/render.rs` duplicates the worker's environment
/// variable names.
const PROJECT_MANIFEST: &str = "project.json";

/// `feathertalk_project::validate_project_dir` cannot be reused here: it
/// requires the finished asset set, including directories the commands that
/// call this are about to create. What has to hold before a command runs is
/// narrower -- a real directory carrying a manifest.
///
/// Two commands need this answer, and the answer is four user-facing summaries.
/// Keeping one copy is what stops the wording from drifting per command.
pub(crate) fn check_project_dir(project_dir: &Path) -> Result<(), TaskError> {
    if !project_dir.is_absolute() {
        return Err(invalid_request(
            "工程目录必须是绝对路径",
            format!("project_dir {} is not absolute", project_dir.display()),
        ));
    }
    // `symlink_metadata` does not follow links, so a symlinked directory is
    // rejected here the way `feathertalk-project` rejects one.
    let metadata = fs::symlink_metadata(project_dir).map_err(|error| {
        invalid_request(
            "工程目录不可用",
            format!("{}: {error}", project_dir.display()),
        )
    })?;
    if !metadata.is_dir() {
        return Err(invalid_request(
            "工程目录不可用",
            format!("{} is not a directory", project_dir.display()),
        ));
    }
    let manifest = project_dir.join(PROJECT_MANIFEST);
    let found = fs::symlink_metadata(&manifest)
        .map(|metadata| metadata.is_file())
        .unwrap_or(false);
    if !found {
        return Err(invalid_request(
            "工程目录缺少 project.json",
            format!("{} is missing or not a regular file", manifest.display()),
        ));
    }
    Ok(())
}

/// What has to hold before a model directory is read: a real directory at an
/// absolute path. Which of the two layouts it is is `inspecting`'s question, and
/// whether that layout is complete is the readers' question.
pub(crate) fn check_model_source(source: &Path) -> Result<(), TaskError> {
    if !source.is_absolute() {
        return Err(invalid_request(
            "模型目录必须是绝对路径",
            format!("source {} is not absolute", source.display()),
        ));
    }
    let metadata = fs::symlink_metadata(source).map_err(|error| {
        invalid_request("模型目录不可用", format!("{}: {error}", source.display()))
    })?;
    if !metadata.is_dir() {
        return Err(invalid_request(
            "模型目录不可用",
            format!("{} is not a directory", source.display()),
        ));
    }
    Ok(())
}

/// Every admission failure reports `MediaInvalid`: the request named a
/// directory or an input file the worker cannot work with.
pub(crate) fn invalid_request(summary: &'static str, detail: String) -> TaskError {
    TaskError::new(
        ErrorCode::MediaInvalid,
        summary,
        &clamp(&detail),
        TaskStage::Preparing,
    )
}
