use std::{
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};

use feathertalk_frame_pipeline::PipelineError;
use feathertalk_image::{BgrImage, decode_jpeg};

use crate::DEFAULT_MAX_FRAME_PIXELS;

/// Single-entry decode cache shared by the three adapters.
///
/// `DecodedFrame` carries only the path, the dimensions and the blur variance,
/// but `detect` and `predict` both need the pixels. `evaluate_frames_with_models`
/// walks frames strictly in order (decode, detect, choose_primary, gate,
/// predict), so one slot reaches a 100 percent hit rate and each frame is
/// decoded exactly once.
///
/// Known limitation: the entry is keyed by path alone. If something rewrites a
/// frame file while an evaluation is running, `load` returns the stale pixels.
/// The pipeline extracts every frame before evaluating any of them, so it cannot
/// hit this.
#[derive(Debug)]
pub struct FrameImageCache {
    max_pixels: u64,
    entry: Mutex<Option<(PathBuf, Arc<BgrImage>)>>,
}

impl FrameImageCache {
    /// A cache with the default per-frame pixel budget.
    pub fn new() -> Self {
        Self::with_max_pixels(DEFAULT_MAX_FRAME_PIXELS)
    }

    /// A cache that rejects any frame declaring more than `max_pixels` pixels.
    pub fn with_max_pixels(max_pixels: u64) -> Self {
        Self {
            max_pixels,
            entry: Mutex::new(None),
        }
    }

    /// Decoded pixels for `path`, reusing the cached image when the path matches.
    pub fn load(&self, path: &Path) -> Result<Arc<BgrImage>, PipelineError> {
        let mut entry = self.entry.lock().map_err(|_| PipelineError::Adapter {
            component: "jpeg",
            message: "frame cache mutex is poisoned".to_owned(),
        })?;
        if let Some((cached_path, image)) = entry.as_ref()
            && cached_path == path
        {
            return Ok(Arc::clone(image));
        }

        let bytes = std::fs::read(path).map_err(|source| PipelineError::Io {
            operation: "decode_frame",
            path: path.to_path_buf(),
            source,
        })?;
        let image = Arc::new(decode_jpeg(&bytes, self.max_pixels).map_err(|error| {
            PipelineError::Adapter {
                component: "jpeg",
                message: error.to_string(),
            }
        })?);
        *entry = Some((path.to_path_buf(), Arc::clone(&image)));
        Ok(image)
    }
}

impl Default for FrameImageCache {
    fn default() -> Self {
        Self::new()
    }
}
