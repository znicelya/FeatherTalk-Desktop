use std::{
    path::{Path, PathBuf},
    sync::Arc,
};

use burn::tensor::{Device, backend::Backend};

use feathertalk_frame_adapters::{
    FrameImageCache, JpegFrameDecoder, PfldLandmarkPredictor, ScrfdArtifactPaths, ScrfdFaceDetector,
};
use feathertalk_frame_pipeline::{FaceDetector, FrameDecoder, LandmarkPredictor, PipelineError};
use feathertalk_models::backend::CpuBackend;

use crate::{ModelToolchain, WgpuContext};

/// Loading the GhostOne graph moves a 125 768-byte module struct through
/// several frames, which overruns a default thread stack. The precedents are
/// `feathertalk-frame-adapters/tests/pfld_model.rs` and
/// `feathertalk-weights/src/pfld/mod.rs`: a dedicated big-stack thread and a
/// boxed return slot.
const PREDICTOR_LOAD_STACK_BYTES: usize = 64 * 1024 * 1024;

/// The three adapters one `extract_frames` job needs, loaded together.
///
/// They share one image cache so the detector and the predictor reuse the
/// pixels the decoder already produced, which is the arrangement the adapter
/// parity tests certify. Both model adapters use the same selected device.
pub struct FrameModels<B: Backend = CpuBackend> {
    decoder: JpegFrameDecoder,
    detector: ScrfdFaceDetector<B>,
    predictor: Box<PfldLandmarkPredictor<B>>,
}

impl FrameModels<CpuBackend> {
    pub fn load(models: &ModelToolchain) -> Result<Self, PipelineError> {
        Self::load_on(models, Default::default())
    }
}

impl<B: Backend> FrameModels<B> {
    pub fn load_on(models: &ModelToolchain, device: Device<B>) -> Result<Self, PipelineError> {
        Self::load_checked(models, device, None)
    }

    pub(crate) fn load_checked(
        models: &ModelToolchain,
        device: Device<B>,
        context: Option<WgpuContext>,
    ) -> Result<Self, PipelineError> {
        let cache = Arc::new(FrameImageCache::new());
        let decoder = JpegFrameDecoder::new(Arc::clone(&cache));
        let detector = ScrfdFaceDetector::<B>::load(
            &scrfd_paths(models.scrfd_dir()),
            device.clone(),
            Arc::clone(&cache),
        )?;
        let predictor = load_predictor::<B>(models.pfld_dir().to_owned(), cache, device, context)?;
        Ok(Self {
            decoder,
            detector,
            predictor,
        })
    }

    pub fn decoder(&self) -> &dyn FrameDecoder {
        &self.decoder
    }

    pub fn detector(&self) -> &dyn FaceDetector {
        &self.detector
    }

    pub fn predictor(&self) -> &dyn LandmarkPredictor {
        self.predictor.as_ref()
    }
}

/// SCRFD takes the manifest and the weights separately; PFLD takes the
/// directory. The two file names are fixed by the importer that wrote them.
fn scrfd_paths(dir: &Path) -> ScrfdArtifactPaths {
    ScrfdArtifactPaths {
        manifest: dir.join("manifest.json"),
        weights: dir.join("model.safetensors"),
    }
}

fn load_predictor<B: Backend>(
    artifacts: PathBuf,
    cache: Arc<FrameImageCache>,
    device: Device<B>,
    context: Option<WgpuContext>,
) -> Result<Box<PfldLandmarkPredictor<B>>, PipelineError> {
    std::thread::Builder::new()
        .name("pfld-predictor-load".to_owned())
        .stack_size(PREDICTOR_LOAD_STACK_BYTES)
        .spawn(move || {
            let predictor = PfldLandmarkPredictor::<B>::load(&artifacts, device.clone(), cache)
                .map(Box::new)?;
            // CubeCL records errors per thread stream. Drain the loader's
            // stream here; the execution thread cannot observe it afterwards.
            if let Some(context) = context {
                context
                    .check()
                    .map_err(|error| adapter_failure(error.to_string()))?;
            } else {
                B::sync(&device).map_err(|error| adapter_failure(error.to_string()))?;
            }
            Ok(predictor)
        })
        .map_err(|error| adapter_failure(format!("spawning the loader thread failed: {error}")))?
        .join()
        .map_err(|_| adapter_failure("the loader thread panicked".to_owned()))?
}

fn adapter_failure(message: String) -> PipelineError {
    PipelineError::Adapter {
        component: "pfld",
        message,
    }
}
