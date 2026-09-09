use feathertalk_preprocess::audio_window_indices;

use crate::{InferenceError, sequence::index_at};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InferenceFramePlan {
    pub output_index: usize,
    pub source_frame_index: usize,
    pub reference_frame_index: usize,
    pub audio_window: [Option<usize>; 8],
}

#[derive(Debug, Clone)]
pub struct RenderPlan {
    source_frame_count: usize,
    feature_frame_count: usize,
    output_frame_count: usize,
}

impl RenderPlan {
    pub fn new(
        source_frame_count: usize,
        feature_frame_count: usize,
        max_output_frames: Option<usize>,
    ) -> Result<Self, InferenceError> {
        crate::sequence::period_for(source_frame_count)?;
        if feature_frame_count == 0 {
            return Err(InferenceError::EmptyFeatures);
        }
        if max_output_frames == Some(0) {
            return Err(InferenceError::InvalidField {
                field: "max_output_frames",
                message: "must be greater than zero when provided".into(),
            });
        }
        let output_frame_count = max_output_frames
            .unwrap_or(feature_frame_count)
            .min(feature_frame_count);
        Ok(Self {
            source_frame_count,
            feature_frame_count,
            output_frame_count,
        })
    }

    pub fn output_frame_count(&self) -> usize {
        self.output_frame_count
    }

    pub fn frame(&self, output_index: usize) -> Result<InferenceFramePlan, InferenceError> {
        if output_index >= self.output_frame_count {
            return Err(InferenceError::OutputFrameOutOfRange {
                index: output_index,
                count: self.output_frame_count,
            });
        }
        let source_frame_index = index_at(output_index, self.source_frame_count)?;
        let audio_window =
            audio_window_indices(output_index, self.feature_frame_count).map_err(|error| {
                InferenceError::InvalidField {
                    field: "audio_window",
                    message: error.to_string(),
                }
            })?;
        Ok(InferenceFramePlan {
            output_index,
            source_frame_index,
            reference_frame_index: source_frame_index,
            audio_window,
        })
    }
}
