mod support;

use std::{
    path::{Path, PathBuf},
    sync::{Arc, OnceLock},
};

use feathertalk_face::{ImageSize, compute_face_crop_geometry};
use feathertalk_frame_adapters::{FrameImageCache, JpegFrameDecoder, PfldLandmarkPredictor};
use feathertalk_frame_pipeline::{
    DecodedFrame, FaceDetection, FrameDecoder, LandmarkPredictor, PipelineError,
};
use feathertalk_models::backend::CpuBackend;

/// The committed PFLD artifact directory, one crate over. `PfldRuntime::load`
/// names the manifest and the weights inside it.
fn artifact_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../feathertalk-pfld/artifacts/pfld_ghost_one")
}

/// One cache for the whole binary, so the decoder and the predictor share the
/// decoded pixels exactly as they do in production.
fn shared_cache() -> Arc<FrameImageCache> {
    static CACHE: OnceLock<Arc<FrameImageCache>> = OnceLock::new();
    Arc::clone(CACHE.get_or_init(|| Arc::new(FrameImageCache::new())))
}

/// Loading the GhostOne graph moves a 125 768-byte module struct through
/// several frames, which overruns the default test-thread stack on Windows.
/// `feathertalk-weights` solves the same problem for its detached clone with a
/// dedicated 64 MiB stack and a boxed return slot; this mirrors it.
const PREDICTOR_LOAD_STACK_BYTES: usize = 64 * 1024 * 1024;

/// Reading the weights costs more than the 192x192 forward pass, so the shared
/// predictor is loaded once. `&'static` is sound because `LandmarkPredictor` is
/// `Send + Sync`.
fn shared_predictor() -> &'static PfldLandmarkPredictor<CpuBackend> {
    static PREDICTOR: OnceLock<Box<PfldLandmarkPredictor<CpuBackend>>> = OnceLock::new();
    PREDICTOR.get_or_init(|| {
        std::thread::Builder::new()
            .name("pfld-predictor-load".to_owned())
            .stack_size(PREDICTOR_LOAD_STACK_BYTES)
            .spawn(|| {
                Box::new(
                    PfldLandmarkPredictor::load(
                        &artifact_dir(),
                        Default::default(),
                        shared_cache(),
                    )
                    .expect("the committed artifact loads"),
                )
            })
            .expect("the loader thread starts")
            .join()
            .expect("the loader thread does not panic")
    })
}

fn decoded(path: &Path) -> DecodedFrame {
    JpegFrameDecoder::new(shared_cache())
        .decode(0, path)
        .unwrap()
}

/// SCRFD is deliberately absent from this file: the landmark path reads only
/// `bbox`, so the tests wrap a recorded box and leave the rest at defaults. A
/// detection regression can therefore never fail a landmark test.
fn face(bbox: [f32; 4]) -> FaceDetection {
    FaceDetection {
        bbox,
        score: 1.0,
        keypoints: [[0.0; 2]; 5],
    }
}

#[test]
fn the_reference_crop_reproduces_the_python_landmarks() {
    let fixture = support::load_and_verify_fixture().unwrap();
    let expected = support::expected_landmarks(&fixture);
    let frame = decoded(&fixture.root.join("frame.jpg"));

    let landmarks = shared_predictor()
        .predict(&frame, &face(expected.bbox))
        .unwrap();

    assert_eq!(landmarks.points().len(), support::PFLD_LANDMARK_COUNT);
    assert_eq!(expected.points.len(), support::PFLD_LANDMARK_COUNT);

    // `landmarks.json` was cut from cv2's decode of this JPEG while this path
    // runs `jpeg-decoder`. The crop geometry is identical, so the only source of
    // disagreement is a level or two per channel in the source pixels.
    for (index, (point, wanted)) in landmarks.points().iter().zip(&expected.points).enumerate() {
        assert!(
            (point.x - wanted[0]).abs() <= 1 && (point.y - wanted[1]).abs() <= 1,
            "point {index}: ({}, {}) vs ({}, {})",
            point.x,
            point.y,
            wanted[0],
            wanted[1]
        );
    }
}

#[test]
fn the_demo_frame_landmarks_match_the_fixture() {
    let fixture = support::load_and_verify_demo_fixture().unwrap();
    let expected = support::demo_frame(&fixture, "sharp");
    let frame = decoded(&expected.path);

    // Nothing else checks the Rust crop geometry against a 1280x720 frame; Task
    // 13 covers only the 640x640 synthetic one. Pinning it here means a landmark
    // drift localises to the model rather than to the crop.
    let geometry = compute_face_crop_geometry(
        ImageSize {
            width: frame.width(),
            height: frame.height(),
        },
        expected.detection.bbox,
    )
    .unwrap();
    assert_eq!(geometry.size, expected.crop.size);
    assert_eq!(
        (geometry.origin_x, geometry.origin_y),
        (expected.crop.origin_x, expected.crop.origin_y)
    );

    let landmarks = shared_predictor()
        .predict(&frame, &face(expected.detection.bbox))
        .unwrap();

    assert_eq!(landmarks.points().len(), support::PFLD_LANDMARK_COUNT);
    assert_eq!(expected.landmarks.len(), support::PFLD_LANDMARK_COUNT);

    // The measured residual is 0 px on every point, because the demo fixture's
    // truth was generated from the JPEG decode rather than from a PNG. 2 px is
    // headroom for a future crop that lands one source pixel differently.
    for (index, (point, wanted)) in landmarks
        .points()
        .iter()
        .zip(&expected.landmarks)
        .enumerate()
    {
        assert!(
            (point.x - wanted[0]).abs() <= 2 && (point.y - wanted[1]).abs() <= 2,
            "point {index}: ({}, {}) vs ({}, {})",
            point.x,
            point.y,
            wanted[0],
            wanted[1]
        );
    }
}

#[test]
fn a_degenerate_bbox_is_reported_as_an_adapter_error() {
    let fixture = support::load_and_verify_fixture().unwrap();
    let frame = decoded(&fixture.root.join("frame.jpg"));

    // A sub-pixel box rounds to a zero-area rectangle, which
    // `compute_face_crop_geometry` rejects. SCRFD cannot produce one, but the
    // adapter must report it rather than panic if one ever arrives.
    match shared_predictor()
        .predict(&frame, &face([10.0, 10.0, 0.4, 0.4]))
        .unwrap_err()
    {
        PipelineError::Adapter { component, message } => {
            assert_eq!(component, "pfld");
            assert_eq!(
                message,
                "invalid crop geometry for bbox: integer edges must define a positive rectangle"
            );
        }
        other => panic!("expected an adapter error, got {other}"),
    }
}

#[test]
fn an_out_of_frame_bbox_still_decodes_through_the_padded_branch() {
    let fixture = support::load_and_verify_fixture().unwrap();
    let frame = decoded(&fixture.root.join("frame.jpg"));

    // Task 8's `padded` case: a 945x945 square whose origin is (-122, -122).
    // Task 13 pins its blob byte for byte, so this asserts only that the
    // adapter drives the negative-origin path to a decoded result.
    let landmarks = shared_predictor()
        .predict(&frame, &face([-100.0, -80.0, 900.0, 860.0]))
        .unwrap();

    assert_eq!(landmarks.points().len(), support::PFLD_LANDMARK_COUNT);
}
