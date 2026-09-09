use std::{path::Path, sync::Arc};

use feathertalk_frame_pipeline::{DecodedFrame, FrameDecoder, PipelineError};
use feathertalk_image::{laplacian_variance, to_gray};

use crate::cache::FrameImageCache;

/// `FrameDecoder` built on the JPEG decoder and the Laplacian blur measure.
#[derive(Debug)]
pub struct JpegFrameDecoder {
    cache: Arc<FrameImageCache>,
}

impl JpegFrameDecoder {
    /// Share `cache` with the detector and the landmark predictor so that a
    /// frame is decoded once per evaluation rather than three times.
    pub fn new(cache: Arc<FrameImageCache>) -> Self {
        Self { cache }
    }
}

impl FrameDecoder for JpegFrameDecoder {
    /// `index` is unused: no `PipelineError` variant carries a frame number, and
    /// the pipeline attaches it when it builds a `FrameAnomaly`.
    ///
    /// `DecodedFrame::new` validates the dimensions and the variance itself, so
    /// this does not re-check them.
    fn decode(&self, _index: u64, path: &Path) -> Result<DecodedFrame, PipelineError> {
        let image = self.cache.load(path)?;
        let variance = laplacian_variance(&to_gray(&image));
        DecodedFrame::new(path.to_path_buf(), image.width(), image.height(), variance)
    }
}
