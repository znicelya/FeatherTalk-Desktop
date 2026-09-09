//! Frame geometry read from a JPEG header.

use std::path::Path;

use feathertalk_frame_pipeline::PipelineError;
use feathertalk_image::jpeg_dimensions;

/// Read a frame's pixel dimensions from its JPEG header.
///
/// Pure over `bytes`: the caller owns the file read and its size cap, which is
/// what lets the tests here run without a temporary directory. `path` is
/// carried only so the error names the frame that is broken.
pub fn probe_jpeg_geometry(path: &Path, bytes: &[u8]) -> Result<(u32, u32), PipelineError> {
    jpeg_dimensions(bytes).map_err(|error| PipelineError::FrameUndecodable {
        path: path.to_path_buf(),
        message: error.to_string(),
    })
}
