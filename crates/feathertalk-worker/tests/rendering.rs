use std::path::{Component, Path};

use feathertalk_domain::{ErrorCode, RenderParams, TaskStage};
use feathertalk_export::ModelConfiguration;
use feathertalk_models::unet::OriginalUnetConfig;
use feathertalk_worker::{
    RENDER_BACKEND_NAME, RENDER_FPS, RenderVariant, check_max_output_frames, check_render_paths,
    checkpoint_descriptor, progress_total, project_assets, render_job, render_variant,
    staging_task_id,
};
use tempfile::tempdir;

/// A `TempDir` root keeps the fixture absolute on every platform without
/// hard-coding a drive letter.
fn params(root: &Path) -> RenderParams {
    RenderParams {
        project_dir: root.join("project"),
        checkpoint: root.join("checkpoint"),
        audio: root.join("voice.wav"),
        output: root.join("preview.mp4"),
        max_output_frames: None,
    }
}

/// The trailing names of a path, so an assertion can check the separator was
/// native rather than embedded in one component.
fn tail(path: &Path, count: usize) -> Vec<String> {
    let mut names: Vec<String> = path
        .components()
        .filter_map(|component| match component {
            Component::Normal(name) => Some(name.to_string_lossy().into_owned()),
            _ => None,
        })
        .collect();
    if names.len() > count {
        names.drain(..names.len() - count);
    }
    names
}

#[test]
fn the_project_assets_sit_where_the_lock_wrote_them() {
    let root = tempdir().unwrap();
    let project = root.path().join("project");
    let assets = project_assets(&project);

    assert_eq!(tail(&assets.frame_dir, 3), ["project", "assets", "frames"]);
    assert_eq!(
        tail(&assets.landmark_dir, 3),
        ["project", "assets", "landmarks"]
    );
    assert_eq!(
        tail(&assets.feature_path, 3),
        ["assets", "features", "feather_hubert.f32"]
    );
    // Joined one component at a time, so nothing carries a foreign separator.
    assert!(
        assets.feature_path.starts_with(&project),
        "{}",
        assets.feature_path.display()
    );
}

#[test]
fn a_known_model_kind_resolves_to_the_configuration_that_named_it() {
    for kind in ["original_unet", "mobileone_unet"] {
        let variant = render_variant(kind).expect("a known kind resolves");
        assert_eq!(variant.configuration().model_type(), kind);
    }
    // The production channels, which is what training wrote the checkpoint with.
    let variant = render_variant("original_unet").expect("original_unet resolves");
    let RenderVariant::OriginalUnet(config) = &variant else {
        panic!("original_unet is the original variant");
    };
    assert_eq!(config.channels, OriginalUnetConfig::production().channels);
}

#[test]
fn an_unknown_model_kind_resolves_to_nothing() {
    assert!(render_variant("unet").is_none());
    assert!(render_variant("").is_none());
    assert!(render_variant("ORIGINAL_UNET").is_none());
}

#[test]
fn a_relative_path_in_a_render_request_is_refused() {
    let root = tempdir().unwrap();
    check_render_paths(&params(root.path())).expect("an absolute request is admitted");

    for (field, mutate) in [
        ("检查点目录", 0usize),
        ("音频文件", 1usize),
        ("输出文件", 2usize),
    ] {
        let mut relative = params(root.path());
        match mutate {
            0 => relative.checkpoint = Path::new("checkpoint").to_path_buf(),
            1 => relative.audio = Path::new("voice.wav").to_path_buf(),
            _ => relative.output = Path::new("preview.mp4").to_path_buf(),
        }
        let error = check_render_paths(&relative).expect_err("a relative path is refused");
        assert_eq!(error.code, ErrorCode::MediaInvalid, "{field}");
        assert_eq!(error.stage, TaskStage::Preparing, "{field}");
        assert!(error.summary.contains(field), "{}", error.summary);
        assert!(error.detail.contains("not absolute"), "{}", error.detail);
        error.validate().unwrap();
    }
}

#[test]
fn a_frame_cap_is_carried_or_refused_but_never_truncated() {
    assert_eq!(check_max_output_frames(None).unwrap(), None);
    assert_eq!(check_max_output_frames(Some(3)).unwrap(), Some(3));

    let error = check_max_output_frames(Some(0)).expect_err("zero frames is not a render");
    assert_eq!(error.code, ErrorCode::MediaInvalid);
    assert!(error.detail.contains("zero"), "{}", error.detail);

    // On a 64-bit host every `u64` fits, so the conversion is the assertion
    // rather than the rejection: what must never happen is a silent truncation.
    match usize::try_from(u64::MAX) {
        Ok(_) => assert_eq!(
            check_max_output_frames(Some(u64::MAX)).unwrap(),
            Some(usize::MAX)
        ),
        Err(_) => {
            let error = check_max_output_frames(Some(u64::MAX))
                .expect_err("a cap that does not fit is refused");
            assert_eq!(error.code, ErrorCode::MediaInvalid);
        }
    }
}

#[test]
fn the_progress_total_is_the_smaller_of_the_manifest_and_the_cap() {
    assert_eq!(progress_total(4, None), 4);
    assert_eq!(progress_total(4, Some(2)), 2);
    assert_eq!(progress_total(4, Some(9)), 4);
}

#[test]
fn two_staging_task_ids_never_collide() {
    let first = staging_task_id();
    let second = staging_task_id();

    assert_ne!(first, second);
    for id in [&first, &second] {
        assert!(id.starts_with("render-"), "{id}");
        assert!(!id.contains('/') && !id.contains('\\'), "{id}");
    }
}

#[test]
fn the_render_backend_is_the_one_the_payload_reports() {
    assert_eq!(RENDER_BACKEND_NAME, "ndarray-cpu");
    assert_eq!(RENDER_FPS, 25);
}

#[test]
fn a_render_job_carries_the_project_layout_and_the_checkpoint_identity() {
    let root = tempdir().unwrap();
    let params = params(root.path());
    let variant = render_variant("original_unet").expect("original_unet is a known kind");
    let descriptor = checkpoint_descriptor(&variant.configuration()).unwrap();
    let ffmpeg = root.path().join("ffmpeg.exe");

    let job = render_job(&params, 4, &ffmpeg, descriptor.clone(), 2, 8)
        .expect("an absolute request with four frames is a job");

    let assets = project_assets(&params.project_dir);
    assert_eq!(job.request.frame_dir(), assets.frame_dir);
    assert_eq!(job.request.landmark_dir(), assets.landmark_dir);
    assert_eq!(job.request.feature_path(), assets.feature_path);
    assert_eq!(job.request.audio_path(), params.audio);
    assert_eq!(job.request.ffmpeg_path(), ffmpeg);
    assert_eq!(job.request.output_path(), params.output);
    assert!(job.request.task_id().starts_with("render-"));
    // The total comes from the locked manifest, never from the feature file.
    assert_eq!(job.progress_total, 4);
    assert_eq!(job.source_frame_count, 4);
    assert_eq!(job.max_output_frames, None);
    assert_eq!(job.checkpoint_dir, params.checkpoint);
    assert_eq!(job.checkpoint_epoch, 2);
    assert_eq!(job.checkpoint_global_step, 8);
    assert_eq!(job.descriptor, descriptor);
}

#[test]
fn a_capped_render_job_keeps_the_source_count_and_lowers_the_total() {
    let root = tempdir().unwrap();
    let mut params = params(root.path());
    params.max_output_frames = Some(2);
    let descriptor = checkpoint_descriptor(&ModelConfiguration::original_unet(
        &OriginalUnetConfig::production(),
    ))
    .unwrap();

    let job = render_job(
        &params,
        4,
        &root.path().join("ffmpeg.exe"),
        descriptor,
        0,
        0,
    )
    .expect("a capped request is a job");

    assert_eq!(job.progress_total, 2);
    assert_eq!(job.source_frame_count, 4);
    assert_eq!(job.max_output_frames, Some(2));
}

#[test]
fn a_project_with_one_frame_cannot_be_rendered() {
    let root = tempdir().unwrap();
    let params = params(root.path());
    let descriptor = checkpoint_descriptor(&ModelConfiguration::original_unet(
        &OriginalUnetConfig::production(),
    ))
    .unwrap();

    let error = render_job(
        &params,
        1,
        &root.path().join("ffmpeg.exe"),
        descriptor,
        0,
        0,
    )
    .expect_err("one frame is not enough to walk forwards and back");

    assert_eq!(error.code, ErrorCode::MediaInvalid);
    assert_eq!(error.stage, TaskStage::Preparing);
    assert!(error.detail.contains("minimum is 2"), "{}", error.detail);
}
