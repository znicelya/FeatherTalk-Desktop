use std::path::{Path, PathBuf};

use feathertalk_frame_pipeline::PipelineError;
use feathertalk_worker::{FrameModels, WorkerConfig};

/// The artifact directories committed one and two crates over. `FrameModels`
/// names `manifest.json` and `model.safetensors` inside the SCRFD one, and
/// hands the PFLD one to `PfldRuntime::load` whole.
fn scrfd_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../feathertalk-scrfd/artifacts/scrfd_2_5g")
}

fn pfld_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../feathertalk-pfld/artifacts/pfld_ghost_one")
}

fn config(scrfd: &Path, pfld: &Path) -> WorkerConfig {
    WorkerConfig::from_values_with_models(
        None,
        None,
        None,
        Some(scrfd.display().to_string()),
        Some(pfld.display().to_string()),
    )
}

#[test]
fn the_committed_artifacts_load_into_three_live_adapters() {
    let config = config(&scrfd_dir(), &pfld_dir());
    let models = FrameModels::load(config.models().expect("both directories are absolute"))
        .expect("the committed artifacts load");

    // The three accessors hand out live trait objects. Decoding a path that
    // does not exist proves the decoder is wired without needing pixels.
    let error = models
        .decoder()
        .decode(0, Path::new(r"C:\missing\000000.jpg"))
        .expect_err("a missing frame cannot decode");
    assert!(matches!(error, PipelineError::Io { .. }), "{error:?}");
}

#[test]
fn a_directory_without_scrfd_artifacts_reports_an_adapter_failure() {
    let empty = tempfile::tempdir().unwrap();
    let config = config(empty.path(), &pfld_dir());

    // `expect_err` would need `FrameModels: Debug`, and the adapters it holds
    // deliberately do not implement it, so the error is taken by `err()`.
    let error = FrameModels::load(config.models().expect("both directories are absolute"))
        .err()
        .expect("an empty directory has no manifest");

    match error {
        PipelineError::Adapter { component, message } => {
            assert_eq!(component, "scrfd");
            assert!(!message.is_empty());
        }
        other => panic!("expected an adapter failure, got {other:?}"),
    }
}

#[test]
fn a_directory_without_pfld_artifacts_reports_an_adapter_failure() {
    let empty = tempfile::tempdir().unwrap();
    let config = config(&scrfd_dir(), empty.path());

    let error = FrameModels::load(config.models().expect("both directories are absolute"))
        .err()
        .expect("an empty directory has no manifest");

    match error {
        PipelineError::Adapter { component, .. } => assert_eq!(component, "pfld"),
        other => panic!("expected an adapter failure, got {other:?}"),
    }
}
