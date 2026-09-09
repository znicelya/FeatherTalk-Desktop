#![allow(dead_code)]

use std::ffi::OsStr;
use std::fs;
use std::path::{Path, PathBuf};

use feathertalk_audio::{FeatureMatrix, write_feature_file};
use feathertalk_inference::{
    BgrFrame, FrameReader, InferenceError, MouthMasking, RenderGeometry, build_face_crop,
    build_inner_image_planes,
};
use feathertalk_preprocess::{
    Landmarks, MaskRect, PFLD_LANDMARK_COUNT, compute_face_bbox, default_crop_spec,
    default_mouth_roi_spec, mouth_roi_rect, read_landmarks,
};
use feathertalk_project::{
    AssetManifest, AssetPackageState, FeatureType, ModelSelection, ProjectManifest,
    TaskHistoryEntry, TaskHistoryStatus, lock_asset_package, write_asset_manifest_atomic,
    write_project_manifest_atomic,
};
use tempfile::TempDir;

pub const FRAME_WIDTH: u32 = 256;
pub const FRAME_HEIGHT: u32 = 256;
pub const INNER_SIZE: usize = 160;
pub const FEATURE_DIMS: usize = 1024;

/// A frame reader that synthesises a deterministic gradient instead of decoding JPEG bytes.
///
/// It still rejects a wrong file name and a missing file, so the dataset's frame-level error
/// paths stay testable while the fixture frames on disk remain five-byte placeholders.
#[derive(Debug, Clone, Copy, Default)]
pub struct GradientFrameReader;

impl FrameReader for GradientFrameReader {
    fn read(&self, index: usize, path: &Path) -> Result<BgrFrame, InferenceError> {
        let expected = format!("{index:06}.jpg");
        if path.file_name() != Some(OsStr::new(&expected)) {
            return Err(InferenceError::FrameReader {
                index,
                path: path.to_path_buf(),
                message: format!("expected a file named {expected}"),
            });
        }
        if !path.is_file() {
            return Err(InferenceError::FrameReader {
                index,
                path: path.to_path_buf(),
                message: "not a file".to_owned(),
            });
        }
        let width = FRAME_WIDTH as usize;
        let height = FRAME_HEIGHT as usize;
        let mut bgr = Vec::with_capacity(width * height * 3);
        for y in 0..height {
            for x in 0..width {
                for channel in 0..3 {
                    let seed = x + y * 2 + channel * 7 + index * 3;
                    bgr.push((seed % 251) as u8);
                }
            }
        }
        BgrFrame::new(FRAME_WIDTH, FRAME_HEIGHT, bgr)
    }
}

/// Everything the fixtures vary between tests.
pub struct FixtureSpec {
    pub frame_count: usize,
    pub manifest_width: u32,
    pub manifest_height: u32,
    pub frame_bytes: Vec<u8>,
    pub face_xmin: u32,
    pub face_xmax: u32,
    pub face_ymin: u32,
    pub mouth_x: u32,
    pub mouth_y: u32,
}

impl FixtureSpec {
    /// The 256x256 gradient project the dataset tests start from.
    pub fn gradient(frame_count: usize) -> Self {
        Self {
            frame_count,
            manifest_width: FRAME_WIDTH,
            manifest_height: FRAME_HEIGHT,
            frame_bytes: b"stub\n".to_vec(),
            face_xmin: 40,
            face_xmax: 200,
            face_ymin: 60,
            mouth_x: 100,
            mouth_y: 160,
        }
    }

    pub fn manifest(&self) -> AssetManifest {
        AssetManifest {
            schema_version: 1,
            state: AssetPackageState::Locked,
            video_fps: 25,
            audio_sample_rate: 16_000,
            audio_channels: 1,
            frame_count: self.frame_count as u64,
            frame_width: self.manifest_width,
            frame_height: self.manifest_height,
            feature_type: FeatureType::FeatherHubert,
            feature_shape: [self.frame_count as u64, 2, 1024],
            landmark_model_sha256: "a".repeat(64),
            feature_model_sha256: "b".repeat(64),
        }
    }
}

pub fn locked_project(frame_count: usize) -> (TempDir, PathBuf) {
    build_locked_project(&FixtureSpec::gradient(frame_count))
}

/// Writes every required artifact and then locks the asset package.
///
/// The order matters: `write_asset_manifest_atomic` refuses to overwrite a manifest that already
/// validates as locked, so `lock_asset_package` has to run last.
pub fn build_locked_project(spec: &FixtureSpec) -> (TempDir, PathBuf) {
    let temp = tempfile::tempdir().unwrap();
    let project_dir = temp.path().join("project");
    fs::create_dir_all(project_dir.join("assets/frames")).unwrap();
    fs::create_dir_all(project_dir.join("assets/landmarks")).unwrap();
    fs::create_dir_all(project_dir.join("assets/features")).unwrap();
    fs::write(project_dir.join("assets/video_25fps.mp4"), b"video").unwrap();
    fs::write(project_dir.join("assets/audio_16k_mono.wav"), b"audio").unwrap();
    write_features(&project_dir, spec.frame_count);
    for index in 0..spec.frame_count {
        let frame_path = project_dir.join(format!("assets/frames/{index:06}.jpg"));
        fs::write(&frame_path, &spec.frame_bytes).unwrap();
        write_landmarks(&project_dir, spec, index);
    }
    let project_path = project_dir.join("project.json");
    write_project_manifest_atomic(&project_path, &valid_project()).unwrap();
    lock_asset_package(&project_dir, spec.manifest()).unwrap();
    (temp, project_dir)
}

/// Overwrites the feature file with `2 * frame_count` tokens of deterministic values.
pub fn write_features(project_dir: &Path, frame_count: usize) {
    let tokens = 2 * frame_count;
    let mut values = Vec::with_capacity(tokens * FEATURE_DIMS);
    for offset in 0..tokens * FEATURE_DIMS {
        values.push((offset % 97) as f32 / 97.0);
    }
    let matrix = FeatureMatrix::new(tokens, FEATURE_DIMS, values).unwrap();
    let path = project_dir.join("assets/features/feather_hubert.f32");
    write_feature_file(&path, &matrix).unwrap();
}

/// Writes 110 landmark lines: three of them fix the face box, twenty carry the mouth.
pub fn write_landmarks(project_dir: &Path, spec: &FixtureSpec, index: usize) {
    let mut lines = vec![String::from("0 0"); PFLD_LANDMARK_COUNT];
    lines[1] = format!("{} 0", spec.face_xmin);
    lines[31] = format!("{} 0", spec.face_xmax);
    lines[52] = format!("0 {}", spec.face_ymin);
    for (offset, line) in lines.iter_mut().skip(90).take(20).enumerate() {
        let x = spec.mouth_x + offset as u32;
        let y = spec.mouth_y + (offset + index) as u32;
        *line = format!("{x} {y}");
    }
    let path = project_dir.join(format!("assets/landmarks/{index:06}.lms"));
    fs::write(path, lines.join("\n")).unwrap();
}

/// Replaces the locked asset manifest with a preparing one.
pub fn downgrade_to_preparing(project_dir: &Path) {
    let manifest_path = project_dir.join("assets/assets.json");
    fs::remove_file(&manifest_path).unwrap();
    write_asset_manifest_atomic(&manifest_path, &preparing_manifest()).unwrap();
}

pub fn landmarks_for(project_dir: &Path, index: usize) -> Landmarks {
    let path = project_dir.join(format!("assets/landmarks/{index:06}.lms"));
    read_landmarks(&path).unwrap()
}

pub fn face_crop(project_dir: &Path, index: usize) -> BgrFrame {
    let frame_path = project_dir.join(format!("assets/frames/{index:06}.jpg"));
    let frame = GradientFrameReader.read(index, &frame_path).unwrap();
    let landmarks = landmarks_for(project_dir, index);
    let bbox = compute_face_bbox(&landmarks).unwrap();
    let geometry = RenderGeometry::standard();
    build_face_crop(&frame, &bbox, &geometry).unwrap()
}

pub fn inner_planes(project_dir: &Path, index: usize, masking: MouthMasking) -> Vec<f32> {
    let crop = face_crop(project_dir, index);
    let geometry = RenderGeometry::standard();
    let planes = build_inner_image_planes(&crop, &geometry, masking).unwrap();
    planes.into_values()
}

pub fn mouth_rect(project_dir: &Path, index: usize) -> MaskRect {
    let landmarks = landmarks_for(project_dir, index);
    let crop = default_crop_spec();
    let spec = default_mouth_roi_spec();
    mouth_roi_rect(&landmarks, &crop, &spec).unwrap()
}

pub fn preparing_manifest() -> AssetManifest {
    AssetManifest {
        schema_version: 1,
        state: AssetPackageState::Preparing,
        video_fps: 0,
        audio_sample_rate: 0,
        audio_channels: 0,
        frame_count: 0,
        frame_width: 0,
        frame_height: 0,
        feature_type: FeatureType::FeatherHubert,
        feature_shape: [0, 0, 0],
        landmark_model_sha256: String::new(),
        feature_model_sha256: String::new(),
    }
}

pub fn valid_project() -> ProjectManifest {
    ProjectManifest {
        schema_version: 1,
        project_id: "demo".into(),
        display_name: "Demo".into(),
        asset_package: "assets/assets.json".into(),
        default_model: ModelSelection::OriginalUnet,
        task_history: vec![TaskHistoryEntry {
            task_id: "task-1".into(),
            kind: "preprocess".into(),
            status: TaskHistoryStatus::Completed,
            updated_at: "2026-08-20T10:00:00Z".into(),
        }],
    }
}
