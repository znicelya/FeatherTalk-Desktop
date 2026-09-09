//! Frame geometry read from the checked-in JPEG fixtures.

use std::fs;
use std::path::{Path, PathBuf};

use feathertalk_frame_adapters::probe_jpeg_geometry;
use feathertalk_frame_pipeline::PipelineError;

fn fixture_frame() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/demo_frame_v1/frame.jpg")
}

#[test]
fn a_real_frame_reports_its_pixel_dimensions() {
    let path = fixture_frame();
    let bytes = fs::read(&path).expect("the demo frame fixture must be readable");
    assert_eq!(probe_jpeg_geometry(&path, &bytes).unwrap(), (1280, 720));
}

#[test]
fn garbage_bytes_name_the_frame_that_is_broken() {
    let path = Path::new("assets/frames/000007.jpg");
    let error = probe_jpeg_geometry(path, b"not a jpeg at all").unwrap_err();
    let PipelineError::FrameUndecodable {
        path: reported,
        message,
    } = error
    else {
        panic!("garbage must be an undecodable frame: {error:?}");
    };
    assert_eq!(reported, path);
    assert!(
        !message.is_empty(),
        "the decoder's own message must survive"
    );
}

#[test]
fn a_truncated_frame_is_undecodable() {
    let path = fixture_frame();
    let bytes = fs::read(&path).expect("the demo frame fixture must be readable");
    let error = probe_jpeg_geometry(&path, &bytes[..4]).unwrap_err();
    assert!(
        matches!(error, PipelineError::FrameUndecodable { .. }),
        "{error:?}"
    );
}
