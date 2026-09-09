use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::PipelineError;

pub const QUALITY_SCHEMA_VERSION: u32 = 1;
pub const MAX_FRAME_COUNT: u64 = 100_000_000;
pub const MAX_IMAGE_DIMENSION: u32 = 32_768;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FramePipelineSpec {
    video_path: PathBuf,
    output_root: PathBuf,
    frame_count: u64,
    image_width: u32,
    image_height: u32,
}

impl FramePipelineSpec {
    pub fn new(
        video_path: PathBuf,
        output_root: PathBuf,
        frame_count: u64,
        image_width: u32,
        image_height: u32,
    ) -> Result<Self, PipelineError> {
        validate_absolute_path("video_path", &video_path)?;
        validate_absolute_path("output_root", &output_root)?;
        validate_positive_limit("frame_count", frame_count, MAX_FRAME_COUNT)?;
        validate_dimension("image_width", image_width)?;
        validate_dimension("image_height", image_height)?;
        // The source video may live beside the outputs -- that is the project
        // layout -- but it must not be one of the three paths extraction and
        // publication write.
        if video_path == output_root {
            return Err(invalid(
                "output_root",
                "must not equal the source video path",
            ));
        }
        if video_path.starts_with(&output_root)
            && video_path.parent() != Some(output_root.as_path())
        {
            return Err(invalid(
                "output_root",
                "must not contain the source video path in a nested directory",
            ));
        }
        if video_path == output_root.join("quality.json") {
            return Err(invalid(
                "output_root",
                "must not equal the quality report path",
            ));
        }
        Ok(Self {
            video_path,
            output_root,
            frame_count,
            image_width,
            image_height,
        })
    }

    pub fn video_path(&self) -> &Path {
        &self.video_path
    }

    pub fn output_root(&self) -> &Path {
        &self.output_root
    }

    pub fn frame_count(&self) -> u64 {
        self.frame_count
    }

    pub fn image_width(&self) -> u32 {
        self.image_width
    }

    pub fn image_height(&self) -> u32 {
        self.image_height
    }

    pub fn frames_dir(&self) -> PathBuf {
        self.output_root.join("frames")
    }

    pub fn landmarks_dir(&self) -> PathBuf {
        self.output_root.join("landmarks")
    }

    pub fn frame_path(&self, index: u64) -> PathBuf {
        self.frames_dir().join(format!("{index:06}.jpg"))
    }

    pub fn landmark_path(&self, index: u64) -> PathBuf {
        self.landmarks_dir().join(format!("{index:06}.lms"))
    }

    pub fn quality_path(&self) -> PathBuf {
        self.output_root.join("quality.json")
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FaceDetection {
    pub bbox: [f32; 4],
    pub score: f32,
    pub keypoints: [[f32; 2]; 5],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AnomalyCode {
    FaceNotFound,
    MultipleFaces,
    BboxOutOfBounds,
    LandmarkInvalid,
    BlurredFrame,
    FrameDecodeFailed,
    FrameWriteFailed,
    ModelFailed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RecoveryAction {
    ExcludeFrame,
    RerunFrame,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FrameAnomaly {
    frame_index: u64,
    code: AnomalyCode,
    summary: String,
    technical_detail: String,
    recovery_action: RecoveryAction,
}

impl FrameAnomaly {
    pub fn new(
        frame_index: u64,
        code: AnomalyCode,
        summary: impl Into<String>,
        technical_detail: impl Into<String>,
        recovery_action: RecoveryAction,
    ) -> Result<Self, PipelineError> {
        let summary = summary.into();
        let technical_detail = technical_detail.into();
        validate_text("summary", &summary, 512)?;
        validate_text("technical_detail", &technical_detail, 4096)?;
        Ok(Self {
            frame_index,
            code,
            summary,
            technical_detail,
            recovery_action,
        })
    }

    pub fn frame_index(&self) -> u64 {
        self.frame_index
    }

    pub fn code(&self) -> AnomalyCode {
        self.code
    }

    pub fn summary(&self) -> &str {
        &self.summary
    }

    pub fn technical_detail(&self) -> &str {
        &self.technical_detail
    }

    pub fn recovery_action(&self) -> RecoveryAction {
        self.recovery_action
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FrameQuality {
    index: u64,
    frame_file: String,
    landmark_file: String,
    frame_bytes: u64,
    frame_sha256: String,
    landmark_sha256: String,
    face_score: f32,
    bbox: [f32; 4],
    blur_variance: f64,
}

impl FrameQuality {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        index: u64,
        frame_file: impl Into<String>,
        landmark_file: impl Into<String>,
        frame_bytes: u64,
        frame_sha256: impl Into<String>,
        landmark_sha256: impl Into<String>,
        face_score: f32,
        bbox: [f32; 4],
        blur_variance: f64,
    ) -> Result<Self, PipelineError> {
        let frame_file = frame_file.into();
        let landmark_file = landmark_file.into();
        let frame_sha256 = frame_sha256.into();
        let landmark_sha256 = landmark_sha256.into();
        validate_artifact_path("frame_file", &frame_file, "frames", index, "jpg")?;
        validate_artifact_path("landmark_file", &landmark_file, "landmarks", index, "lms")?;
        if frame_bytes == 0 {
            return Err(report_invalid("frame_bytes", "must be greater than zero"));
        }
        validate_sha256("frame_sha256", &frame_sha256)?;
        validate_sha256("landmark_sha256", &landmark_sha256)?;
        if !face_score.is_finite() || !(0.0..=1.0).contains(&face_score) {
            return Err(report_invalid(
                "face_score",
                "must be finite and within [0,1]",
            ));
        }
        if bbox.iter().any(|value| !value.is_finite()) || bbox[2] <= 0.0 || bbox[3] <= 0.0 {
            return Err(report_invalid("bbox", "must be finite with positive size"));
        }
        if !blur_variance.is_finite() || blur_variance < 0.0 {
            return Err(report_invalid(
                "blur_variance",
                "must be finite and non-negative",
            ));
        }
        Ok(Self {
            index,
            frame_file,
            landmark_file,
            frame_bytes,
            frame_sha256,
            landmark_sha256,
            face_score,
            bbox,
            blur_variance,
        })
    }

    pub fn index(&self) -> u64 {
        self.index
    }

    pub fn frame_file(&self) -> &str {
        &self.frame_file
    }

    pub fn landmark_file(&self) -> &str {
        &self.landmark_file
    }

    pub fn frame_bytes(&self) -> u64 {
        self.frame_bytes
    }

    pub fn frame_sha256(&self) -> &str {
        &self.frame_sha256
    }

    pub fn landmark_sha256(&self) -> &str {
        &self.landmark_sha256
    }

    pub fn face_score(&self) -> f32 {
        self.face_score
    }

    pub fn bbox(&self) -> [f32; 4] {
        self.bbox
    }

    pub fn blur_variance(&self) -> f64 {
        self.blur_variance
    }

    pub fn validate(&self) -> Result<(), PipelineError> {
        Self::new(
            self.index,
            self.frame_file.clone(),
            self.landmark_file.clone(),
            self.frame_bytes,
            self.frame_sha256.clone(),
            self.landmark_sha256.clone(),
            self.face_score,
            self.bbox,
            self.blur_variance,
        )
        .map(|_| ())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QualityReport {
    schema_version: u32,
    frame_count: u64,
    accepted_count: u64,
    frames: Vec<FrameQuality>,
    anomalies: Vec<FrameAnomaly>,
}

impl QualityReport {
    pub fn new(
        frame_count: u64,
        frames: Vec<FrameQuality>,
        anomalies: Vec<FrameAnomaly>,
    ) -> Result<Self, PipelineError> {
        validate_positive_limit("frame_count", frame_count, MAX_FRAME_COUNT)?;
        let accepted_count = u64::try_from(frames.len())
            .map_err(|_| report_invalid("frames", "length exceeds u64"))?;
        let report = Self {
            schema_version: QUALITY_SCHEMA_VERSION,
            frame_count,
            accepted_count,
            frames,
            anomalies,
        };
        report.validate()?;
        Ok(report)
    }

    pub fn schema_version(&self) -> u32 {
        self.schema_version
    }

    pub fn frame_count(&self) -> u64 {
        self.frame_count
    }

    pub fn accepted_count(&self) -> u64 {
        self.accepted_count
    }

    pub fn frames(&self) -> &[FrameQuality] {
        &self.frames
    }

    pub fn anomalies(&self) -> &[FrameAnomaly] {
        &self.anomalies
    }

    pub fn validate(&self) -> Result<(), PipelineError> {
        if self.schema_version != QUALITY_SCHEMA_VERSION {
            return Err(report_invalid(
                "schema_version",
                format!("expected {QUALITY_SCHEMA_VERSION}"),
            ));
        }
        validate_positive_limit("frame_count", self.frame_count, MAX_FRAME_COUNT)?;
        if self.accepted_count != self.frames.len() as u64 {
            return Err(report_invalid("accepted_count", "must equal frames length"));
        }
        let total_classified = self
            .accepted_count
            .checked_add(self.anomalies.len() as u64)
            .ok_or_else(|| report_invalid("frames", "accepted frames and anomalies overflow"))?;
        if total_classified > self.frame_count {
            return Err(report_invalid(
                "frames",
                "accepted frames and anomalies exceed frame_count",
            ));
        }
        let mut seen = std::collections::HashSet::new();
        for frame in &self.frames {
            frame.validate()?;
            if frame.index >= self.frame_count || !seen.insert(frame.index) {
                return Err(report_invalid(
                    "frames.index",
                    "must be unique and within frame_count",
                ));
            }
        }
        for anomaly in &self.anomalies {
            if anomaly.frame_index >= self.frame_count || !seen.insert(anomaly.frame_index) {
                return Err(report_invalid(
                    "anomalies.frame_index",
                    "must be unique and within frame_count",
                ));
            }
            validate_text("summary", &anomaly.summary, 512)?;
            validate_text("technical_detail", &anomaly.technical_detail, 4096)?;
        }
        Ok(())
    }
}

fn validate_absolute_path(field: &'static str, path: &Path) -> Result<(), PipelineError> {
    if !path.is_absolute() || path.as_os_str().is_empty() {
        Err(invalid(field, "must be a non-empty absolute path"))
    } else {
        Ok(())
    }
}

fn validate_positive_limit(
    field: &'static str,
    value: u64,
    maximum: u64,
) -> Result<(), PipelineError> {
    if value == 0 || value > maximum {
        Err(invalid(field, format!("must be within 1..={maximum}")))
    } else {
        Ok(())
    }
}

fn validate_dimension(field: &'static str, value: u32) -> Result<(), PipelineError> {
    if value == 0 || value > MAX_IMAGE_DIMENSION {
        Err(invalid(
            field,
            format!("must be within 1..={MAX_IMAGE_DIMENSION}"),
        ))
    } else {
        Ok(())
    }
}

fn validate_text(field: &'static str, value: &str, maximum: usize) -> Result<(), PipelineError> {
    if value.trim() != value || value.is_empty() || value.chars().count() > maximum {
        Err(invalid(
            field,
            format!("must be trimmed and 1..={maximum} characters"),
        ))
    } else {
        Ok(())
    }
}

fn validate_sha256(field: &str, value: &str) -> Result<(), PipelineError> {
    if value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        Ok(())
    } else {
        Err(report_invalid(
            field,
            "must be 64 lowercase hexadecimal characters",
        ))
    }
}

fn validate_artifact_path(
    field: &str,
    value: &str,
    directory: &str,
    index: u64,
    extension: &str,
) -> Result<(), PipelineError> {
    let expected = format!("{directory}/{index:06}.{extension}");
    if value == expected {
        Ok(())
    } else {
        Err(report_invalid(field, format!("expected {expected}")))
    }
}

fn invalid(field: &'static str, message: impl Into<String>) -> PipelineError {
    PipelineError::InvalidField {
        field,
        message: message.into(),
    }
}

fn report_invalid(field: impl Into<String>, message: impl Into<String>) -> PipelineError {
    PipelineError::InvalidReport {
        field: field.into(),
        message: message.into(),
    }
}
