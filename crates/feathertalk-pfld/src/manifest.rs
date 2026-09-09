use crate::PfldRuntimeError;

pub const PFLD_RUNTIME_SCHEMA_VERSION: u32 = 1;
pub const PFLD_ARCHITECTURE_VERSION: &str = "burn-pfld-inference-v1";
pub const PFLD_MODEL_TYPE: &str = "pfld_ghost_one";
pub const PFLD_CHECKPOINT_EPOCH: u64 = 335;
pub const PFLD_INPUT_SHAPE: [usize; 4] = [1, 3, 192, 192];
pub const PFLD_OUTPUT_SHAPE: [usize; 2] = [1, 220];
pub const PFLD_EXPECTED_TENSOR_COUNT: usize = 1735;
pub const PFLD_EXPECTED_TOTAL_ELEMENTS: u64 = 910_902;
pub const PFLD_SOURCE_SHA256: &str =
    "bada866661ad5fa1080a085f51fe9c016c69958c406951afa4afc7840f856de0";
pub const PFLD_MODEL_SHA256: &str =
    "e131dd764236fde54a27b2f7084906119f06c28b140bf127b459ec967e92915b";
pub const PFLD_MODEL_BYTES: u64 = 3_825_080;
pub const MAX_MANIFEST_BYTES: u64 = 1024 * 1024;
pub const MAX_WEIGHT_BYTES: u64 = 32 * 1024 * 1024;

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PfldSourceManifest {
    pub file_name: String,
    pub sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PfldTensorSpec {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: String,
}

impl PfldTensorSpec {
    pub fn new<const N: usize>(name: &str, shape: [usize; N]) -> Self {
        Self {
            name: name.to_owned(),
            shape: shape.to_vec(),
            dtype: "f32".to_owned(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PfldModelManifest {
    pub format: String,
    pub file_name: String,
    pub sha256: String,
    pub tensor_count: usize,
    pub total_elements: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PfldLicenseManifest {
    pub spdx: String,
    pub redistribution_approved: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PfldRuntimeManifest {
    pub schema_version: u32,
    pub model_type: String,
    pub architecture_version: String,
    pub source: PfldSourceManifest,
    pub epoch: u64,
    pub input: PfldTensorSpec,
    pub output: PfldTensorSpec,
    pub model: PfldModelManifest,
    pub license: PfldLicenseManifest,
}

impl PfldRuntimeManifest {
    pub fn approved(
        source_file_name: String,
        source_sha256: String,
        model_sha256: String,
        tensor_count: usize,
        total_elements: u64,
    ) -> Self {
        Self {
            schema_version: PFLD_RUNTIME_SCHEMA_VERSION,
            model_type: PFLD_MODEL_TYPE.to_owned(),
            architecture_version: PFLD_ARCHITECTURE_VERSION.to_owned(),
            source: PfldSourceManifest {
                file_name: source_file_name,
                sha256: source_sha256,
            },
            epoch: PFLD_CHECKPOINT_EPOCH,
            input: PfldTensorSpec::new("input", PFLD_INPUT_SHAPE),
            output: PfldTensorSpec::new("landmarks", PFLD_OUTPUT_SHAPE),
            model: PfldModelManifest {
                format: "safetensors".to_owned(),
                file_name: "model.safetensors".to_owned(),
                sha256: model_sha256,
                tensor_count,
                total_elements,
            },
            license: PfldLicenseManifest {
                spdx: "NOASSERTION".to_owned(),
                redistribution_approved: false,
            },
        }
    }

    pub fn validate(&self) -> Result<(), PfldRuntimeError> {
        if self.schema_version != PFLD_RUNTIME_SCHEMA_VERSION {
            return Err(PfldRuntimeError::UnsupportedSchemaVersion {
                expected: PFLD_RUNTIME_SCHEMA_VERSION,
                actual: self.schema_version,
            });
        }
        if self.model_type != PFLD_MODEL_TYPE {
            return Err(invalid("model_type", format!("expected {PFLD_MODEL_TYPE}")));
        }
        if self.architecture_version != PFLD_ARCHITECTURE_VERSION {
            return Err(PfldRuntimeError::UnsupportedArchitectureVersion {
                expected: PFLD_ARCHITECTURE_VERSION.to_owned(),
                actual: self.architecture_version.clone(),
            });
        }
        if self.epoch != PFLD_CHECKPOINT_EPOCH {
            return Err(invalid(
                "epoch",
                format!("expected {PFLD_CHECKPOINT_EPOCH}, got {}", self.epoch),
            ));
        }
        if self.source.file_name != "checkpoint_epoch_335.pth.tar" {
            return Err(invalid("source.file_name", "unexpected source file name"));
        }
        require_hash("source.sha256", &self.source.sha256)?;
        if self.source.sha256 != PFLD_SOURCE_SHA256 {
            return Err(invalid(
                "source.sha256",
                "does not match approved checkpoint",
            ));
        }
        if self.input != PfldTensorSpec::new("input", PFLD_INPUT_SHAPE) {
            return Err(invalid(
                "input",
                "does not match [1,3,192,192] f32 contract",
            ));
        }
        if self.output != PfldTensorSpec::new("landmarks", PFLD_OUTPUT_SHAPE) {
            return Err(invalid("output", "does not match [1,220] f32 contract"));
        }
        if self.model.format != "safetensors" {
            return Err(invalid("model.format", "expected safetensors"));
        }
        if self.model.file_name != "model.safetensors" {
            return Err(invalid("model.file_name", "unexpected model file name"));
        }
        if self.model.tensor_count != PFLD_EXPECTED_TENSOR_COUNT {
            return Err(invalid(
                "model.tensor_count",
                format!("expected {PFLD_EXPECTED_TENSOR_COUNT}"),
            ));
        }
        if self.model.total_elements != PFLD_EXPECTED_TOTAL_ELEMENTS {
            return Err(invalid(
                "model.total_elements",
                format!("expected {PFLD_EXPECTED_TOTAL_ELEMENTS}"),
            ));
        }
        require_hash("model.sha256", &self.model.sha256)?;
        if self.model.sha256 != PFLD_MODEL_SHA256 {
            return Err(invalid("model.sha256", "does not match approved artifact"));
        }
        if self.license.spdx != "NOASSERTION" {
            return Err(invalid("license.spdx", "expected NOASSERTION"));
        }
        if self.license.redistribution_approved {
            return Err(invalid(
                "license.redistribution_approved",
                "must remain false",
            ));
        }
        Ok(())
    }
}

fn invalid(field: impl Into<String>, message: impl Into<String>) -> PfldRuntimeError {
    PfldRuntimeError::InvalidManifest {
        field: field.into(),
        message: message.into(),
    }
}

fn require_hash(field: &str, value: &str) -> Result<(), PfldRuntimeError> {
    if value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        Ok(())
    } else {
        Err(invalid(
            field,
            "must be 64 lowercase hexadecimal characters",
        ))
    }
}
