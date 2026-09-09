use crate::ScrfdError;

pub const SCRFD_SCHEMA_VERSION: u32 = 1;
pub const SCRFD_ARCHITECTURE_VERSION: u32 = 1;
pub const SCRFD_MODEL_KIND: &str = "scrfd_2.5g_kps";
pub const SCRFD_SOURCE_ONNX_BYTES: u64 = 3_291_017;
pub const SCRFD_SOURCE_ONNX_SHA256: &str =
    "32d20c77b9e2dc1d07e94c2ab9d25bdd5cd05eddbe0b46e7b38e7a1eca22e99a";
pub const SCRFD_SOURCE_OPSET: u64 = 12;
pub const SCRFD_INPUT_SHAPE: [usize; 4] = [1, 3, 640, 640];
pub const SCRFD_STRIDES: [u32; 3] = [8, 16, 32];
pub const SCRFD_ANCHORS: [usize; 3] = [12_800, 3_200, 800];

const OUTPUT_NAMES: [&str; 9] = [
    "out0", "out1", "out2", "out3", "out4", "out5", "out6", "out7", "out8",
];

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScrfdArtifactManifest {
    pub schema_version: u32,
    pub model_kind: String,
    pub architecture_version: u32,
    pub source: ScrfdSourceManifest,
    pub generator: ScrfdGeneratorManifest,
    pub input: ScrfdInputManifest,
    pub levels: [ScrfdLevelManifest; 3],
    pub generated_source: ScrfdFileManifest,
    pub weights: ScrfdWeightManifest,
    pub license: ScrfdLicenseManifest,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScrfdSourceManifest {
    pub format: String,
    pub file_name: String,
    pub file_bytes: u64,
    pub sha256: String,
    pub opset: u64,
    pub input_name: String,
    pub output_names: [String; 9],
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScrfdGeneratorManifest {
    pub burn: String,
    pub burn_onnx: String,
    pub burn_store: String,
    pub simplify: bool,
    pub load_strategy: String,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScrfdInputManifest {
    pub dtype: String,
    pub shape: [usize; 4],
    pub scale: f32,
    pub mean: [f32; 3],
    pub swap_rb: bool,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScrfdLevelManifest {
    pub stride: u32,
    pub anchors: usize,
    pub score: ScrfdOutputManifest,
    pub bbox: ScrfdOutputManifest,
    pub keypoints: ScrfdOutputManifest,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScrfdOutputManifest {
    pub onnx_name: String,
    pub source_shape: Vec<usize>,
    pub public_shape: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScrfdFileManifest {
    pub file_name: String,
    pub file_bytes: u64,
    pub sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScrfdWeightManifest {
    pub format: String,
    pub file_name: String,
    pub file_bytes: u64,
    pub sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScrfdLicenseManifest {
    pub license_id: String,
    pub redistribution_approved: bool,
    pub evidence: String,
}

fn invalid(field: impl Into<String>, message: impl Into<String>) -> ScrfdError {
    ScrfdError::InvalidManifest {
        field: field.into(),
        message: message.into(),
    }
}

fn is_lower_hex_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn require_nonempty(field: &str, value: &str) -> Result<(), ScrfdError> {
    if value.is_empty() {
        Err(invalid(field, "must not be empty"))
    } else {
        Ok(())
    }
}

fn require_hash(field: &str, value: &str) -> Result<(), ScrfdError> {
    if is_lower_hex_sha256(value) {
        Ok(())
    } else {
        Err(invalid(
            field,
            "must be 64 lowercase hexadecimal characters",
        ))
    }
}

fn require_positive(field: &str, value: u64) -> Result<(), ScrfdError> {
    if value == 0 {
        Err(invalid(field, "must be greater than zero"))
    } else {
        Ok(())
    }
}

fn require_shape(field: &str, actual: &[usize], expected: &[usize]) -> Result<(), ScrfdError> {
    if actual.contains(&0) {
        return Err(invalid(field, "dimensions must be greater than zero"));
    }
    if actual != expected {
        return Err(invalid(
            field,
            format!("expected {expected:?}, got {actual:?}"),
        ));
    }
    Ok(())
}

impl ScrfdArtifactManifest {
    pub fn validate(&self) -> Result<(), ScrfdError> {
        if self.schema_version != SCRFD_SCHEMA_VERSION {
            return Err(ScrfdError::UnsupportedSchemaVersion {
                expected: SCRFD_SCHEMA_VERSION,
                actual: self.schema_version,
            });
        }
        if self.architecture_version != SCRFD_ARCHITECTURE_VERSION {
            return Err(ScrfdError::UnsupportedArchitectureVersion {
                expected: SCRFD_ARCHITECTURE_VERSION,
                actual: self.architecture_version,
            });
        }

        if self.model_kind != SCRFD_MODEL_KIND {
            return Err(invalid(
                "model_kind",
                format!("expected {SCRFD_MODEL_KIND}"),
            ));
        }
        if self.source.format != "onnx" {
            return Err(invalid("source.format", "expected onnx"));
        }
        if self.source.file_name != "scrfd_2.5g_kps.onnx" {
            return Err(invalid("source.file_name", "unexpected source file name"));
        }
        require_positive("source.file_bytes", self.source.file_bytes)?;
        if self.source.file_bytes != SCRFD_SOURCE_ONNX_BYTES {
            return Err(invalid(
                "source.file_bytes",
                format!(
                    "expected {SCRFD_SOURCE_ONNX_BYTES}, got {}",
                    self.source.file_bytes
                ),
            ));
        }
        require_hash("source.sha256", &self.source.sha256)?;
        if self.source.sha256 != SCRFD_SOURCE_ONNX_SHA256 {
            return Err(invalid(
                "source.sha256",
                "does not match approved ONNX source",
            ));
        }
        if self.source.opset == 0 {
            return Err(invalid("source.opset", "must be greater than zero"));
        }
        if self.source.opset != SCRFD_SOURCE_OPSET {
            return Err(invalid(
                "source.opset",
                "does not match approved ONNX opset",
            ));
        }
        require_nonempty("source.input_name", &self.source.input_name)?;
        if self.source.input_name != "images" {
            return Err(invalid("source.input_name", "expected images"));
        }
        for (index, (actual, expected)) in self
            .source
            .output_names
            .iter()
            .zip(OUTPUT_NAMES)
            .enumerate()
        {
            require_nonempty(&format!("source.output_names[{index}]"), actual)?;
            if actual != expected {
                return Err(invalid(
                    format!("source.output_names[{index}]"),
                    format!("expected {expected}"),
                ));
            }
        }

        for (field, value) in [
            ("generator.burn", &self.generator.burn),
            ("generator.burn_onnx", &self.generator.burn_onnx),
            ("generator.burn_store", &self.generator.burn_store),
            ("generator.load_strategy", &self.generator.load_strategy),
        ] {
            require_nonempty(field, value)?;
        }
        for (field, value) in [
            ("generator.burn", self.generator.burn.as_str()),
            ("generator.burn_onnx", self.generator.burn_onnx.as_str()),
            ("generator.burn_store", self.generator.burn_store.as_str()),
        ] {
            if value != "0.21.0" {
                return Err(invalid(field, "expected 0.21.0"));
            }
        }
        if !self.generator.simplify {
            return Err(invalid("generator.simplify", "must be true"));
        }
        if self.generator.load_strategy != "none" {
            return Err(invalid("generator.load_strategy", "expected none"));
        }

        if self.input.dtype != "float32" {
            return Err(invalid("input.dtype", "expected float32"));
        }
        if self.input.shape != SCRFD_INPUT_SHAPE {
            return Err(invalid(
                "input.shape",
                format!("expected {SCRFD_INPUT_SHAPE:?}, got {:?}", self.input.shape),
            ));
        }
        if self.input.scale.to_bits() != (1.0_f32 / 128.0).to_bits() {
            return Err(invalid("input.scale", "must equal 1/128 as binary32"));
        }
        for (index, value) in self.input.mean.iter().enumerate() {
            if value.to_bits() != 127.5_f32.to_bits() {
                return Err(invalid(format!("input.mean[{index}]"), "must equal 127.5"));
            }
        }
        if !self.input.swap_rb {
            return Err(invalid("input.swap_rb", "must be true"));
        }

        let expected_levels = [
            (8_u32, 12_800_usize, "out0", "out3", "out6"),
            (16, 3_200, "out1", "out4", "out7"),
            (32, 800, "out2", "out5", "out8"),
        ];
        for (level_index, (level, (stride, anchors, score_name, bbox_name, keypoints_name))) in
            self.levels.iter().zip(expected_levels).enumerate()
        {
            let prefix = format!("levels[{level_index}]");
            if level.stride != stride {
                return Err(invalid(
                    format!("{prefix}.stride"),
                    format!("expected {stride}"),
                ));
            }
            if level.anchors != anchors {
                return Err(invalid(
                    format!("{prefix}.anchors"),
                    format!("expected {anchors}"),
                ));
            }
            validate_output(
                &format!("{prefix}.score"),
                &level.score,
                score_name,
                &[1, anchors, 1],
                &[1, anchors],
            )?;
            validate_output(
                &format!("{prefix}.bbox"),
                &level.bbox,
                bbox_name,
                &[1, anchors, 4],
                &[1, anchors, 4],
            )?;
            validate_output(
                &format!("{prefix}.keypoints"),
                &level.keypoints,
                keypoints_name,
                &[1, anchors, 10],
                &[1, anchors, 10],
            )?;
        }

        validate_file("generated_source", &self.generated_source, "scrfd_2_5g.rs")?;
        if self.weights.format != "safetensors" {
            return Err(invalid("weights.format", "expected safetensors"));
        }
        if self.weights.file_name != "model.safetensors" {
            return Err(invalid("weights.file_name", "unexpected weights file name"));
        }
        require_positive("weights.file_bytes", self.weights.file_bytes)?;
        require_hash("weights.sha256", &self.weights.sha256)?;

        require_nonempty("license.license_id", &self.license.license_id)?;
        if self.license.license_id != "NOASSERTION" {
            return Err(invalid("license.license_id", "expected NOASSERTION"));
        }
        if self.license.redistribution_approved {
            return Err(invalid(
                "license.redistribution_approved",
                "must remain false until provenance is verified",
            ));
        }
        require_nonempty("license.evidence", &self.license.evidence)?;
        Ok(())
    }
}

fn validate_file(
    prefix: &str,
    file: &ScrfdFileManifest,
    expected_name: &str,
) -> Result<(), ScrfdError> {
    if file.file_name != expected_name {
        return Err(invalid(
            format!("{prefix}.file_name"),
            "unexpected file name",
        ));
    }
    require_positive(&format!("{prefix}.file_bytes"), file.file_bytes)?;
    require_hash(&format!("{prefix}.sha256"), &file.sha256)
}

fn validate_output(
    prefix: &str,
    output: &ScrfdOutputManifest,
    expected_name: &str,
    expected_source: &[usize],
    expected_public: &[usize],
) -> Result<(), ScrfdError> {
    if output.onnx_name != expected_name {
        return Err(invalid(
            format!("{prefix}.onnx_name"),
            format!("expected {expected_name}"),
        ));
    }
    require_shape(
        &format!("{prefix}.source_shape"),
        &output.source_shape,
        expected_source,
    )?;
    require_shape(
        &format!("{prefix}.public_shape"),
        &output.public_shape,
        expected_public,
    )
}
