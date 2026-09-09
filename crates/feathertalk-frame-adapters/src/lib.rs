//! Production implementations of the `feathertalk-frame-pipeline` traits.
//!
//! The numerical work lives in free functions so that parity can be tested
//! without loading weights; the adapters themselves only build tensors, call
//! `forward`, and copy results back to the host.

mod cache;
mod decoder;
mod geometry;
mod pfld;
mod scrfd;

pub use cache::FrameImageCache;
pub use decoder::JpegFrameDecoder;
/// Re-exported because it appears in `ScrfdFaceDetector::load`'s signature.
/// Without it the function is public but uncallable from another crate.
pub use feathertalk_scrfd::ScrfdArtifactPaths;
pub use geometry::probe_jpeg_geometry;
pub use pfld::{PfldLandmarkPredictor, pfld_input};
pub use scrfd::{LevelHostData, ScrfdFaceDetector, ScrfdInput, scrfd_detections, scrfd_input};

/// Per-frame pixel budget, the same value as `feathertalk_inference`'s frame
/// reader uses.
///
/// Duplicated rather than imported: `feathertalk-inference` is not a dependency
/// of this crate, and making it one would pull its whole dependency chain into
/// the adapter layer.
pub const DEFAULT_MAX_FRAME_PIXELS: u64 = 64 * 1024 * 1024;
