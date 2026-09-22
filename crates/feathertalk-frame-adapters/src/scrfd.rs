use std::{fmt, sync::Arc};

use burn::tensor::{Tensor, TensorData, Transaction};
use feathertalk_face::{
    Detection, DetectionConfig, FaceError, ImageSize, ResizeTransform, decode_level,
    generate_anchor_centers, non_max_suppression, resize_with_padding};
use feathertalk_frame_pipeline::{
    DecodedFrame, FACE_CONFIDENCE_THRESHOLD, FaceDetection, FaceDetector, NMS_IOU_THRESHOLD,
    PipelineError};
use feathertalk_image::{BgrImage, resize_area};
use feathertalk_scrfd::{SCRFD_INPUT_SHAPE, ScrfdArtifactPaths, ScrfdLevelOutput, ScrfdModel};

use crate::cache::FrameImageCache;

/// SCRFD's fixed square input edge, the canvas `resize_with_padding` targets.
const SCRFD_EDGE: u32 = 640;

/// `(0.0 - 127.5) / 128.0`: what the zero-filled letterbox border normalizes
/// to. `blobFromImage` applies the mean and scale to the padding as well, so
/// the border is not zero in the blob.
const PADDED_VALUE: f32 = -0.99609375;

/// A ready-to-upload SCRFD input plus the letterbox that produced it.
#[derive(Debug, Clone)]
pub struct ScrfdInput {
    /// The mapping Task 11 inverts to return detections in source pixels.
    pub transform: ResizeTransform,
    /// NCHW `[1, 3, 640, 640]`, RGB, normalized to `(v - 127.5) / 128`.
    pub data: Vec<f32>}

/// Letterbox `image` into SCRFD's 640x640 canvas and build the input blob.
///
/// This is the port of `resize_image` followed by `cv2.dnn.blobFromImage` in
/// `data_utils/detect_face.py`. It performs no inference and loads no weights,
/// so its parity is testable on its own.
pub fn scrfd_input(image: &BgrImage) -> Result<ScrfdInput, PipelineError> {
    let transform = resize_with_padding(ImageSize {
        width: image.width(),
        height: image.height()})
    .map_err(|error| PipelineError::Adapter {
        component: "scrfd",
        message: format!("letterbox failed: {error}")})?;

    let resized =
        resize_area(image, transform.new_width, transform.new_height).map_err(|error| {
            PipelineError::Adapter {
                component: "scrfd",
                message: format!(
                    "resize to {}x{} failed: {error}",
                    transform.new_width, transform.new_height
                )}
        })?;

    let edge = SCRFD_EDGE as usize;
    let plane = edge * edge;
    let width = resized.width() as usize;
    let height = resized.height() as usize;
    let pad_x = transform.pad_x as usize;
    let pad_y = transform.pad_y as usize;

    // Both branches of `resize_with_padding` cap the resized edge at 640, so
    // this cannot fire today. It stays because the loop below indexes a `Vec`
    // at a computed offset, and an out-of-bounds panic in an adapter kills the
    // worker instead of producing a frame anomaly.
    if pad_x + width > edge || pad_y + height > edge {
        return Err(PipelineError::Adapter {
            component: "scrfd",
            message: format!(
                "letterbox does not fit the {edge}x{edge} canvas: {width}x{height} at ({pad_x}, {pad_y})"
            )});
    }

    let mut data = vec![PADDED_VALUE; 3 * plane];
    let bytes = resized.as_bytes();
    for y in 0..height {
        for x in 0..width {
            let source = (y * width + x) * 3;
            let target = (y + pad_y) * edge + x + pad_x;
            // SCRFD consumes RGB and `BgrImage` stores B, G, R, so channel c
            // reads byte 2 - c. Task 13's `pfld_input` keeps BGR; the two are
            // not interchangeable.
            for channel in 0..3 {
                data[channel * plane + target] =
                    (f32::from(bytes[source + 2 - channel]) - 127.5) * (1.0 / 128.0);
            }
        }
    }

    Ok(ScrfdInput { transform, data })
}

/// SCRFD emits two anchors per feature-map location at every stride.
const ANCHORS_PER_LOCATION: u32 = 2;

/// One SCRFD output level copied back to host memory.
///
/// `bbox_distances` and `keypoint_distances` are the raw regression outputs, in
/// stride units; `decode_level` applies the stride and the letterbox.
#[derive(Debug, Clone)]
pub struct LevelHostData {
    /// Index into `SCRFD_STRIDES`, used only in error messages.
    pub level: usize,
    pub stride: u32,
    /// One score per anchor: 12 800, 3 200 and 800 for strides 8, 16 and 32.
    pub scores: Vec<f32>,
    pub bbox_distances: Vec<[f32; 4]>,
    pub keypoint_distances: Vec<[f32; 10]>}

/// Decode three SCRFD levels and reduce them with non-maximum suppression.
///
/// The port of `detect_face.py`'s postprocessing loop. Anchors below
/// `config.confidence_threshold` are skipped before decoding, exactly as the
/// reference does, and boxes that clamp to nothing are dropped rather than
/// failing the frame.
pub fn scrfd_detections(
    levels: &[LevelHostData; 3],
    transform: &ResizeTransform,
    config: &DetectionConfig,
) -> Result<Vec<FaceDetection>, PipelineError> {
    let mut candidates: Vec<Detection> = Vec::new();

    for level in levels {
        let anchors = generate_anchor_centers(transform.model, level.stride, ANCHORS_PER_LOCATION)
            .map_err(|error| level_error(level.level, None, &error))?;

        // `decode_level` checks these too, but it is called with one-anchor
        // slices below, so its check can never see a truncated buffer.
        for (field, actual) in [
            ("scores", level.scores.len()),
            ("bbox_distances", level.bbox_distances.len()),
            ("keypoint_distances", level.keypoint_distances.len()),
        ] {
            if actual != anchors.len() {
                return Err(PipelineError::Adapter {
                    component: "scrfd",
                    message: format!(
                        "level {} {field} holds {actual} entries, expected {}",
                        level.level,
                        anchors.len()
                    )});
            }
        }

        for (index, score) in level.scores.iter().enumerate() {
            // The reference filters before decoding, so a sub-threshold anchor
            // is allowed to hold geometry that would be rejected below.
            if *score < config.confidence_threshold {
                continue;
            }
            let window = index..index + 1;
            match decode_level(
                level.level,
                level.stride,
                &anchors[window.clone()],
                &level.scores[window.clone()],
                &level.bbox_distances[window.clone()],
                &level.keypoint_distances[window],
                transform,
            ) {
                Ok(decoded) => candidates.extend(decoded),
                // A box that clamps to zero area is not an error: upstream
                // never emits it, and the frame may still hold a real face.
                Err(FaceError::InvalidDetectionGeometry { .. }) => continue,
                Err(error) => return Err(level_error(level.level, Some(index), &error))}
        }
    }

    let kept =
        non_max_suppression(&candidates, config).map_err(|error| PipelineError::Adapter {
            component: "scrfd",
            message: error.to_string()})?;

    Ok(kept
        .into_iter()
        .map(|index| {
            let candidate = candidates[index];
            FaceDetection {
                bbox: candidate.bbox,
                score: candidate.score,
                keypoints: candidate.keypoints}
        })
        .collect())
}

/// Attach the level, and where known the anchor, to a `feathertalk-face`
/// failure. `decode_level` reports index 0 for a one-anchor slice, so its own
/// message cannot identify the anchor.
fn level_error(level: usize, anchor: Option<usize>, error: &FaceError) -> PipelineError {
    let message = match anchor {
        Some(anchor) => format!("level {level} anchor {anchor}: {error}"),
        None => format!("level {level}: {error}")};
    PipelineError::Adapter {
        component: "scrfd",
        message}
}

/// `FaceDetector` backed by the SCRFD 2.5G model.
///
/// Holds the weights, the device and the shared decode cache. `detect` takes
/// `&self` and allocates only the input blob and the host copies of the three
/// levels, so a single detector can serve every worker.
pub struct ScrfdFaceDetector {
    model: ScrfdModel,
    device: burn::tensor::Device,
    cache: Arc<FrameImageCache>,
    config: DetectionConfig}

impl ScrfdFaceDetector {
    /// Load the artifact pair and share `cache` with the decoder.
    ///
    /// `ScrfdError` is flattened into an adapter message: the pipeline reports
    /// artifact problems as `ModelFailed`, and the path is already in the text.
    pub fn load(
        paths: &ScrfdArtifactPaths,
        device: burn::tensor::Device,
        cache: Arc<FrameImageCache>,
    ) -> Result<Self, PipelineError> {
        let model = ScrfdModel::load(paths, &device).map_err(|error| PipelineError::Adapter {
            component: "scrfd",
            message: error.to_string()})?;
        Ok(Self::from_model(model, device, cache))
    }

    /// Wrap weights that are already in memory, at the production thresholds.
    pub fn from_model(
        model: ScrfdModel,
        device: burn::tensor::Device,
        cache: Arc<FrameImageCache>,
    ) -> Self {
        Self {
            model,
            device,
            cache,
            config: DetectionConfig {
                confidence_threshold: FACE_CONFIDENCE_THRESHOLD,
                nms_iou_threshold: NMS_IOU_THRESHOLD}}
    }

    /// Override the thresholds. The pipeline never calls this; it exists so a
    /// caller sweeping thresholds does not have to re-read the weights.
    #[must_use]
    pub fn with_detection_config(mut self, config: DetectionConfig) -> Self {
        self.config = config;
        self
    }
}

/// `ScrfdModel` does not implement `Debug` and design §10 freezes the public
/// surface of `feathertalk-scrfd`, so this prints the thresholds and stops.
impl fmt::Debug for ScrfdFaceDetector {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ScrfdFaceDetector")
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

impl FaceDetector for ScrfdFaceDetector {
    fn detect(&self, frame: &DecodedFrame) -> Result<Vec<FaceDetection>, PipelineError> {
        let image = self.cache.load(frame.path())?;
        let ScrfdInput { transform, data } = scrfd_input(&image)?;
        let input = Tensor::<4>::from_data(
            TensorData::new(data, SCRFD_INPUT_SHAPE.to_vec()),
            &self.device,
        );
        let output = self
            .model
            .forward(input)
            .map_err(|error| PipelineError::Adapter {
                component: "scrfd",
                message: error.to_string()})?;

        let levels = host_levels(output.levels)?;

        scrfd_detections(&levels, &transform, &self.config)
    }
}

/// SCRFD returns nine tensors. Submit their readbacks together so decoding
/// waits once per frame instead of once per field at each pyramid level.
fn host_levels(
    outputs: [ScrfdLevelOutput; 3],
) -> Result<[LevelHostData; 3], PipelineError> {
    let strides = outputs.each_ref().map(|output| output.stride);
    let mut transaction = Transaction::default();
    for output in outputs {
        transaction = transaction
            .register(output.scores)
            .register(output.bbox_deltas)
            .register(output.keypoint_deltas);
    }
    let mut data = transaction
        .try_execute()
        .map_err(|error| PipelineError::Adapter {
            component: "scrfd",
            message: format!("reading detection outputs: {error}")})?
        .into_iter();
    let mut next_level = || -> [TensorData; 3] {
        std::array::from_fn(|_| data.next().expect("three readbacks per SCRFD level"))
    };
    Ok([
        host_level(0, strides[0], next_level())?,
        host_level(1, strides[1], next_level())?,
        host_level(2, strides[2], next_level())?,
    ])
}

/// Reshape one downloaded level for `scrfd_detections`.
///
/// `chunks_exact` silently drops a trailing partial group, which would produce
/// a short `Vec` rather than a panic; `scrfd_detections` compares every length
/// against `anchors.len()`, so a truncated buffer still surfaces as an adapter
/// error naming the level and the field.
fn host_level(
    level: usize,
    stride: u32,
    [scores, bbox_deltas, keypoint_deltas]: [TensorData; 3],
) -> Result<LevelHostData, PipelineError> {
    let scores = host_floats(level, "scores", scores)?;
    let bbox_distances = host_floats(level, "bbox_deltas", bbox_deltas)?
        .chunks_exact(4)
        .map(|chunk| [chunk[0], chunk[1], chunk[2], chunk[3]])
        .collect();
    let keypoint_distances = host_floats(level, "keypoint_deltas", keypoint_deltas)?
        .chunks_exact(10)
        .map(|chunk| {
            let mut values = [0.0_f32; 10];
            values.copy_from_slice(chunk);
            values
        })
        .collect();

    Ok(LevelHostData {
        level,
        stride,
        scores,
        bbox_distances,
        keypoint_distances})
}

/// `into_vec::<f32>` requires the backend's float element type to be exactly
/// `f32`, which both `CpuBackend` and `GpuBackend` satisfy. A backend built on
/// `f16` would fail here rather than silently losing precision, and the
/// mismatch arrives as an adapter error naming the level and the field.
fn host_floats(
    level: usize,
    field: &'static str,
    data: TensorData,
) -> Result<Vec<f32>, PipelineError> {
    data.try_into_vec::<f32>()
        .map_err(|error| PipelineError::Adapter {
            component: "scrfd",
            message: format!("level {level} {field}: {error}")})
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::Flex;

    #[test]
    fn batched_readback_keeps_all_levels_fields_and_anchor_coordinates() {
        let device = Default::default();
        let outputs: [ScrfdLevelOutput; 3] = std::array::from_fn(|level| {
            let base = level as f32 * 100.0;
            ScrfdLevelOutput {
                stride: 8 << level,
                scores: Tensor::from_floats([[base, base + 1.0]], &device),
                bbox_deltas: Tensor::from_data(
                    TensorData::new(
                        (0..8).map(|i| base + 10.0 + i as f32).collect::<Vec<_>>(),
                        [1, 2, 4],
                    ),
                    &device,
                ),
                keypoint_deltas: Tensor::from_data(
                    TensorData::new(
                        (0..20).map(|i| base + 20.0 + i as f32).collect::<Vec<_>>(),
                        [1, 2, 10],
                    ),
                    &device,
                )}
        });

        for (index, level) in host_levels(outputs).unwrap().into_iter().enumerate() {
            let base = index as f32 * 100.0;
            assert_eq!(level.level, index);
            assert_eq!(level.stride, 8 << index);
            assert_eq!(level.scores, [base, base + 1.0]);
            assert_eq!(
                level.bbox_distances,
                [
                    [base + 10.0, base + 11.0, base + 12.0, base + 13.0],
                    [base + 14.0, base + 15.0, base + 16.0, base + 17.0],
                ]
            );
            assert_eq!(
                level.keypoint_distances,
                [
                    std::array::from_fn(|i| base + 20.0 + i as f32),
                    std::array::from_fn(|i| base + 30.0 + i as f32),
                ]
            );
        }
    }
}
