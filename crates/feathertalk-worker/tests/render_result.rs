use std::path::{Path, PathBuf};

use feathertalk_export::ModelConfiguration;
use feathertalk_inference::{OfflineRenderRequest, OfflineRenderResult, execute_offline_render};
use feathertalk_models::unet::OriginalUnetConfig;
use feathertalk_training::CheckpointDescriptor;
use feathertalk_worker::{
    RenderBackend, RenderDevice, RenderSummary, checkpoint_descriptor, project_assets,
    render_to_json,
};
use tempfile::TempDir;

#[path = "support/mod.rs"]
mod support;

use support::{MemorySinkFactory, StubFrameReader, render_audio, render_model, render_tree};

/// The identity of the production configuration, which is what a real training
/// checkpoint records.
fn descriptor() -> CheckpointDescriptor {
    let configuration = ModelConfiguration::original_unet(&OriginalUnetConfig::production());
    checkpoint_descriptor(&configuration).expect("the configuration serialises")
}

/// Renders a project once, because `OfflineRenderResult` has private fields and
/// no constructor: the only way to get one is to render.
///
/// The temporary root travels back with the result -- dropping it would delete
/// the video the payload names.
fn rendered(frames: usize, max_output_frames: Option<usize>) -> (TempDir, OfflineRenderResult) {
    let (root, project) = render_tree(frames, frames);
    let assets = project_assets(&project);
    let request = OfflineRenderRequest::new(
        assets.frame_dir,
        assets.landmark_dir,
        assets.feature_path,
        render_audio(&project),
        std::env::current_exe().expect("the test binary knows its own path"),
        root.path().join("render.mp4"),
        "task-render-result",
        frames,
        max_output_frames,
    )
    .expect("the request is valid");
    let device = RenderDevice::default();
    let result = execute_offline_render::<RenderBackend, _, _, _>(
        &render_model(&device),
        &device,
        &request,
        &StubFrameReader::default(),
        &MemorySinkFactory::default(),
    )
    .expect("the render finishes");
    (root, result)
}

#[test]
fn a_render_payload_names_the_weights_the_video_came_from() {
    let descriptor = descriptor();
    let checkpoint_dir = PathBuf::from("C:/tmp/project/models/unet/checkpoint-00000004");
    let (_root, result) = rendered(2, None);

    let payload = render_to_json(&RenderSummary {
        result: &result,
        descriptor: &descriptor,
        checkpoint_dir: &checkpoint_dir,
        checkpoint_epoch: 1,
        checkpoint_global_step: 4,
        source_frame_count: 2,
        max_output_frames: None,
    });

    let object = payload.as_object().expect("the payload is an object");
    assert_eq!(object.len(), 14, "{payload}");
    assert_eq!(payload["frame_count"], 2);
    // The stub reader hands back 168x168 frames, and inference reports the size
    // it actually wrote rather than the size the manifest advertises.
    assert_eq!(payload["width"], 168);
    assert_eq!(payload["height"], 168);
    // The container's frame rate is fixed by inference, not by the request.
    assert_eq!(payload["fps"], 25);
    assert_eq!(payload["backend"], "ndarray-cpu");
    assert_eq!(payload["model_kind"], "original_unet");
    assert_eq!(payload["checkpoint_epoch"], 1);
    assert_eq!(payload["checkpoint_global_step"], 4);
    assert_eq!(payload["source_frame_count"], 2);
    // A request without a cap reports the absence, rather than repeating the
    // frame count and pretending the client asked for it.
    assert!(payload["max_output_frames"].is_null(), "{payload}");
    assert_eq!(
        payload["architecture_version"],
        descriptor.architecture_version.as_str()
    );
    assert_eq!(
        payload["model_config_sha256"],
        descriptor.model_config_sha256.as_str()
    );
}

#[test]
fn a_capped_render_reports_the_cap_next_to_what_it_wrote() {
    let descriptor = descriptor();
    let checkpoint_dir = PathBuf::from("C:/tmp/project/models/unet/checkpoint-00000006");
    // Three source frames, two of them rendered: the cap has to be visible next
    // to the project's own frame count, or a truncated video would look complete.
    let (_root, result) = rendered(3, Some(2));

    let payload = render_to_json(&RenderSummary {
        result: &result,
        descriptor: &descriptor,
        checkpoint_dir: &checkpoint_dir,
        checkpoint_epoch: 2,
        checkpoint_global_step: 6,
        source_frame_count: 3,
        max_output_frames: Some(2),
    });

    assert_eq!(payload["frame_count"], 2);
    assert_eq!(payload["source_frame_count"], 3);
    assert_eq!(payload["max_output_frames"], 2);
}

#[test]
fn a_render_payload_keeps_the_paths_as_the_host_spells_them() {
    let descriptor = descriptor();
    let checkpoint_dir = Path::new("C:/tmp/project")
        .join("models")
        .join("unet")
        .join("checkpoint-00000004");
    let (_root, result) = rendered(2, None);

    let payload = render_to_json(&RenderSummary {
        result: &result,
        descriptor: &descriptor,
        checkpoint_dir: &checkpoint_dir,
        checkpoint_epoch: 1,
        checkpoint_global_step: 4,
        source_frame_count: 2,
        max_output_frames: None,
    });

    let output = payload["output_path"]
        .as_str()
        .expect("the output path is a string");
    let checkpoint = payload["checkpoint_dir"]
        .as_str()
        .expect("the checkpoint path is a string");
    assert_eq!(output, result.output_path().display().to_string());
    assert_eq!(checkpoint, checkpoint_dir.display().to_string());
    // `display` keeps the host's separator, so a Windows payload keeps its
    // backslashes instead of being normalised behind the client's back.
    assert!(output.contains(std::path::MAIN_SEPARATOR), "{output}");
    assert!(
        checkpoint.contains(std::path::MAIN_SEPARATOR),
        "{checkpoint}"
    );
}
