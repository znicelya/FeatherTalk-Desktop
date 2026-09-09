use std::{fmt, path::Path, sync::Arc};

use burn::tensor::{Tensor, TensorData, backend::Backend};
use feathertalk_face::{FaceCropGeometry, ImageSize, compute_face_crop_geometry};
use feathertalk_frame_pipeline::{DecodedFrame, FaceDetection, LandmarkPredictor, PipelineError};
use feathertalk_image::{BgrImage, resize_linear};
use feathertalk_pfld::{
    CropGeometry, PFLD_INPUT_SHAPE, PFLDLandmarks, PfldRuntime,
    decode_landmarks_with_default_mean_face,
};

use crate::cache::FrameImageCache;

/// PFLD consumes a 192x192 crop.
const PFLD_EDGE: u32 = 192;

/// Build the PFLD input blob for one face.
///
/// The port of the crop, `cv2.copyMakeBorder`, `cv2.resize` and
/// `cv2.dnn.blobFromImage` sequence in `data_utils/detect_face.py`. It loads no
/// weights, so its parity is testable on its own.
pub fn pfld_input(
    image: &BgrImage,
    geometry: &FaceCropGeometry,
) -> Result<Vec<f32>, PipelineError> {
    let size = geometry.size as usize;
    if size == 0 {
        return Err(PipelineError::Adapter {
            component: "pfld",
            message: "crop size must be non-zero".to_owned(),
        });
    }

    // `source` is already clipped to the image, so both coordinates are
    // non-negative; `RectI` stores them signed because `requested` is not.
    let (Ok(source_x), Ok(source_y)) = (
        usize::try_from(geometry.source.x),
        usize::try_from(geometry.source.y),
    ) else {
        return Err(PipelineError::Adapter {
            component: "pfld",
            message: format!(
                "clipped crop origin is negative: ({}, {})",
                geometry.source.x, geometry.source.y
            ),
        });
    };

    let source_width = geometry.source.width as usize;
    let source_height = geometry.source.height as usize;
    let pad_left = geometry.padding.left as usize;
    let pad_top = geometry.padding.top as usize;
    let image_width = image.width() as usize;
    let image_height = image.height() as usize;

    // Neither check can fire for a geometry `compute_face_crop_geometry`
    // produced: it clips `source` to the image and derives `padding` from the
    // same integers. They stay because the copy below indexes `Vec`s at
    // computed offsets, and an out-of-bounds panic in an adapter kills the
    // worker instead of producing a frame anomaly.
    if source_x + source_width > image_width || source_y + source_height > image_height {
        return Err(PipelineError::Adapter {
            component: "pfld",
            message: format!(
                "source rectangle {source_width}x{source_height} at ({source_x}, {source_y}) exceeds the {image_width}x{image_height} frame"
            ),
        });
    }
    if pad_left + source_width > size || pad_top + source_height > size {
        return Err(PipelineError::Adapter {
            component: "pfld",
            message: format!(
                "source rectangle {source_width}x{source_height} at ({pad_left}, {pad_top}) does not fit the {size}x{size} canvas"
            ),
        });
    }

    // `copyMakeBorder(..., BORDER_CONSTANT, 0)`: the canvas starts black and
    // the clipped rectangle lands at the padding offset.
    let mut canvas = vec![0_u8; size * size * 3];
    let source_bytes = image.as_bytes();
    let row_bytes = source_width * 3;
    for row in 0..source_height {
        let from = ((source_y + row) * image_width + source_x) * 3;
        let to = ((pad_top + row) * size + pad_left) * 3;
        canvas[to..to + row_bytes].copy_from_slice(&source_bytes[from..from + row_bytes]);
    }

    let square = BgrImage::new(geometry.size, geometry.size, canvas).map_err(|error| {
        PipelineError::Adapter {
            component: "pfld",
            message: format!("crop canvas {size}x{size} is invalid: {error}"),
        }
    })?;
    let resized =
        resize_linear(&square, PFLD_EDGE, PFLD_EDGE).map_err(|error| PipelineError::Adapter {
            component: "pfld",
            message: format!("resize to {PFLD_EDGE}x{PFLD_EDGE} failed: {error}"),
        })?;

    let plane = PFLD_EDGE as usize * PFLD_EDGE as usize;
    let resized_bytes = resized.as_bytes();
    let mut data = vec![0.0_f32; 3 * plane];
    for pixel in 0..plane {
        // PFLD was trained on BGR, so channel c reads byte c. `scrfd_input`
        // reads 2 - c; the two are not interchangeable. The division must stay
        // a division: `1.0 / 255.0` is inexact in f32, and the reference
        // divides.
        for channel in 0..3 {
            data[channel * plane + pixel] = f32::from(resized_bytes[pixel * 3 + channel]) / 255.0;
        }
    }

    Ok(data)
}

/// `LandmarkPredictor` backed by the PFLD GhostOne model.
///
/// Holds the weights, the device and the shared decode cache. There is nothing
/// to configure: the crop square is derived from the detection and the mean face
/// is compiled into `feathertalk-pfld`.
pub struct PfldLandmarkPredictor<B: Backend> {
    runtime: PfldRuntime<B>,
    device: B::Device,
    cache: Arc<FrameImageCache>,
}

impl<B: Backend> PfldLandmarkPredictor<B> {
    /// Load the artifact directory and share `cache` with the decoder.
    ///
    /// The parameter is a directory rather than a manifest and weights pair
    /// because that is `PfldRuntime::load`'s shape; `ScrfdFaceDetector::load`
    /// takes the pair for the same reason.
    pub fn load(
        artifacts: &Path,
        device: B::Device,
        cache: Arc<FrameImageCache>,
    ) -> Result<Self, PipelineError> {
        match PfldRuntime::load(artifacts, &device) {
            Ok(runtime) => Ok(Self::from_runtime(runtime, device, cache)),
            Err(error) => Err(PipelineError::Adapter {
                component: "pfld",
                message: error.to_string(),
            }),
        }
    }

    /// Wrap weights that are already in memory.
    pub fn from_runtime(
        runtime: PfldRuntime<B>,
        device: B::Device,
        cache: Arc<FrameImageCache>,
    ) -> Self {
        Self {
            runtime,
            device,
            cache,
        }
    }
}

/// `PfldRuntime` does not implement `Debug` and design §10 freezes the public
/// surface of `feathertalk-pfld`. There is no configuration to print either, so
/// this reports the type name and stops.
impl<B: Backend> fmt::Debug for PfldLandmarkPredictor<B> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PfldLandmarkPredictor")
            .finish_non_exhaustive()
    }
}

impl<B: Backend> LandmarkPredictor for PfldLandmarkPredictor<B> {
    fn predict(
        &self,
        frame: &DecodedFrame,
        face: &FaceDetection,
    ) -> Result<PFLDLandmarks, PipelineError> {
        let image = self.cache.load(frame.path())?;

        // The only step that can fail on a well-formed frame: a box whose
        // integer edges collapse is rejected instead of silently expanded.
        let geometry: FaceCropGeometry = compute_face_crop_geometry(
            ImageSize {
                width: frame.width(),
                height: frame.height(),
            },
            face.bbox,
        )
        .map_err(|error| PipelineError::Adapter {
            component: "pfld",
            message: error.to_string(),
        })?;

        let data = pfld_input(&image, &geometry)?;
        let input = Tensor::<B, 4>::from_data(
            TensorData::new(data, PFLD_INPUT_SHAPE.to_vec()),
            &self.device,
        );
        let output = self
            .runtime
            .forward(input)
            .map_err(|error| PipelineError::Adapter {
                component: "pfld",
                message: error.to_string(),
            })?;
        let values =
            output
                .into_data()
                .into_vec::<f32>()
                .map_err(|error| PipelineError::Adapter {
                    component: "pfld",
                    message: format!("landmark output: {error}"),
                })?;

        // The decode maps normalised model space back onto source pixels, so it
        // needs the padded square the crop came from, not the frame. `size` is
        // both edges by construction, and the origin may be negative.
        decode_landmarks_with_default_mean_face(
            &values,
            CropGeometry {
                width: geometry.size,
                height: geometry.size,
                offset_x: geometry.origin_x,
                offset_y: geometry.origin_y,
            },
        )
        .map_err(|error| PipelineError::Adapter {
            component: "pfld",
            message: error.to_string(),
        })
    }
}
