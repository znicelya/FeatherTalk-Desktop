//! Model tool form, request validation and measured result summaries.
//!
//! The request builder never touches the filesystem. `preflight` is called by
//! the submit handler, leaving the worker authoritative for model formats.

use std::fs;
use std::path::{Component, Path, PathBuf};

use feathertalk_domain::{
    ExportModelPackageParams, ExportOnnxParams, ImportLegacyModelParams, InspectModelParams,
    LegacyModelKind, MigrateLegacyFeaturesParams, OnnxExportKind, Request, TaskKind,
};
use serde_json::Value;

use crate::facts::{Fact, FactValue};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ModelOperation {
    #[default]
    Inspect,
    ImportLegacy,
    ExportPackage,
    ExportOnnx,
    MigrateFeatures,
}

impl ModelOperation {
    pub const ALL: [Self; 5] = [
        Self::Inspect,
        Self::ImportLegacy,
        Self::ExportPackage,
        Self::ExportOnnx,
        Self::MigrateFeatures,
    ];

    pub fn task_kind(self) -> TaskKind {
        match self {
            Self::Inspect => TaskKind::InspectModel,
            Self::ImportLegacy => TaskKind::ImportLegacyModel,
            Self::ExportPackage => TaskKind::ExportModelPackage,
            Self::ExportOnnx => TaskKind::ExportOnnx,
            Self::MigrateFeatures => TaskKind::MigrateLegacyFeatures,
        }
    }

    pub fn from_task_kind(kind: TaskKind) -> Option<Self> {
        Self::ALL
            .into_iter()
            .find(|operation| operation.task_kind() == kind)
    }

    pub fn element_id(self) -> &'static str {
        match self {
            Self::Inspect => "models-operation-inspect",
            Self::ImportLegacy => "models-operation-import",
            Self::ExportPackage => "models-operation-package",
            Self::ExportOnnx => "models-operation-onnx",
            Self::MigrateFeatures => "models-operation-features",
        }
    }

    pub fn label_key(self) -> &'static str {
        match self {
            Self::Inspect => "models.operation.inspect",
            Self::ImportLegacy => "models.operation.import",
            Self::ExportPackage => "models.operation.package",
            Self::ExportOnnx => "models.operation.onnx",
            Self::MigrateFeatures => "models.operation.features",
        }
    }

    pub fn description_key(self) -> &'static str {
        match self {
            Self::Inspect => "models.description.inspect",
            Self::ImportLegacy => "models.description.import",
            Self::ExportPackage => "models.description.package",
            Self::ExportOnnx => "models.description.onnx",
            Self::MigrateFeatures => "models.description.features",
        }
    }

    pub fn source_label_key(self) -> &'static str {
        match self {
            Self::Inspect => "models.source.model",
            Self::ImportLegacy => "models.source.legacy",
            Self::ExportPackage => "models.source.checkpoint",
            Self::ExportOnnx => "models.source.package",
            Self::MigrateFeatures => "models.source.features",
        }
    }

    pub fn destination_label_key(self) -> &'static str {
        match self {
            Self::Inspect | Self::ImportLegacy | Self::ExportPackage => {
                "models.destination.package"
            }
            Self::ExportOnnx => "models.destination.onnx",
            Self::MigrateFeatures => "models.destination.features",
        }
    }

    pub fn source_is_file(self) -> bool {
        matches!(self, Self::ImportLegacy | Self::MigrateFeatures)
    }

    pub fn needs_destination(self) -> bool {
        self != Self::Inspect
    }

    pub fn destination_is_directory(self) -> bool {
        matches!(self, Self::ImportLegacy | Self::ExportPackage)
    }

    pub fn model_kinds(self) -> &'static [ModelKind] {
        match self {
            // The protocol enum also names PFLD and MobileOne, but importing.rs
            // explicitly rejects those kinds in the standard package writer.
            Self::ImportLegacy => &[ModelKind::FeatherHubert, ModelKind::OriginalUnet],
            Self::ExportOnnx => &[
                ModelKind::FeatherHubert,
                ModelKind::OriginalUnet,
                ModelKind::MobileOneUnet,
            ],
            _ => &[],
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ModelKind {
    #[default]
    FeatherHubert,
    OriginalUnet,
    MobileOneUnet,
}

impl ModelKind {
    pub fn label_key(self) -> &'static str {
        match self {
            Self::FeatherHubert => "models.kind.feather_hubert",
            Self::OriginalUnet => "models.kind.original_unet",
            Self::MobileOneUnet => "models.kind.mobileone_unet",
        }
    }

    pub fn element_id(self) -> &'static str {
        match self {
            Self::FeatherHubert => "models-kind-feather-hubert",
            Self::OriginalUnet => "models-kind-original-unet",
            Self::MobileOneUnet => "models-kind-mobileone-unet",
        }
    }

    fn legacy_kind(self) -> Option<LegacyModelKind> {
        match self {
            Self::FeatherHubert => Some(LegacyModelKind::FeatherHubert),
            Self::OriginalUnet => Some(LegacyModelKind::OriginalUnet),
            Self::MobileOneUnet => None,
        }
    }

    fn onnx_kind(self) -> OnnxExportKind {
        match self {
            Self::FeatherHubert => OnnxExportKind::FeatherHubert,
            Self::OriginalUnet => OnnxExportKind::OriginalUnet,
            Self::MobileOneUnet => OnnxExportKind::MobileOneUnet,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelIssue {
    pub key: &'static str,
    pub detail: Option<String>,
}

impl ModelIssue {
    pub fn new(key: &'static str) -> Self {
        Self { key, detail: None }
    }

    pub fn with_detail(key: &'static str, detail: impl Into<String>) -> Self {
        Self {
            key,
            detail: Some(detail.into()),
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct ModelForm {
    operation: ModelOperation,
    kind: ModelKind,
    source: Option<PathBuf>,
    destination: Option<PathBuf>,
    pub issue: Option<ModelIssue>,
    pub result_details_open: bool,
}

impl ModelForm {
    pub fn operation(&self) -> ModelOperation {
        self.operation
    }
    pub fn kind(&self) -> ModelKind {
        self.kind
    }
    pub fn source(&self) -> Option<&Path> {
        self.source.as_deref()
    }
    pub fn destination(&self) -> Option<&Path> {
        self.destination.as_deref()
    }

    pub fn set_operation(&mut self, operation: ModelOperation) -> bool {
        if self.operation == operation {
            return false;
        }
        self.operation = operation;
        self.kind = ModelKind::default();
        self.source = None;
        self.destination = None;
        self.issue = None;
        true
    }

    pub fn set_kind(&mut self, kind: ModelKind) {
        self.kind = kind;
        self.issue = None;
    }

    pub fn select_source(&mut self, path: PathBuf) {
        self.source = Some(path);
        self.issue = None;
    }

    pub fn select_destination(&mut self, path: PathBuf) {
        self.destination = Some(path);
        self.issue = None;
    }

    /// A folder picker supplies the parent, since the published package itself
    /// must not exist yet. Selecting it does not create any directory.
    pub fn select_destination_parent(&mut self, parent: &Path) {
        if self.operation.destination_is_directory() {
            self.select_destination(parent.join(self.suggested_destination_name()));
        }
    }

    pub fn suggested_destination_name(&self) -> String {
        let source_name = self
            .source()
            .and_then(Path::file_name)
            .and_then(|name| name.to_str());
        match self.operation {
            ModelOperation::Inspect => String::new(),
            ModelOperation::ImportLegacy => {
                let name = source_name.unwrap_or("model");
                let stem = name
                    .strip_suffix(".pth.tar")
                    .or_else(|| name.strip_suffix(".pth"))
                    .unwrap_or(name);
                format!("{stem}-package")
            }
            ModelOperation::ExportPackage => format!("{}-package", source_name.unwrap_or("model")),
            ModelOperation::ExportOnnx => format!("{}.onnx", source_name.unwrap_or("model")),
            ModelOperation::MigrateFeatures => {
                let name = source_name.unwrap_or("feather_hubert.npy");
                format!("{}.f32", name.strip_suffix(".npy").unwrap_or(name))
            }
        }
    }

    /// A cancelled or outdated dialog leaves the current selection intact.
    pub fn apply_source_pick(
        &mut self,
        operation: ModelOperation,
        picked: Option<PathBuf>,
    ) -> bool {
        if operation != self.operation {
            return false;
        }
        let Some(path) = picked else {
            return false;
        };
        self.select_source(path);
        true
    }

    pub fn apply_destination_pick(
        &mut self,
        operation: ModelOperation,
        picked: Option<PathBuf>,
    ) -> bool {
        if operation != self.operation {
            return false;
        }
        let Some(path) = picked else {
            return false;
        };
        self.select_destination(path);
        true
    }

    /// Build the exact domain request from the current form, with no disk I/O.
    pub fn request(&self) -> Result<Request, ModelIssue> {
        let source = self
            .source()
            .filter(|path| !path.as_os_str().is_empty())
            .ok_or_else(|| ModelIssue::new("models.error.no_source"))?;
        if !source.is_absolute() {
            return Err(ModelIssue::new("models.error.source_absolute"));
        }
        let name = source
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("");
        match self.operation {
            ModelOperation::ImportLegacy
                if !name.ends_with(".pth") && !name.ends_with(".pth.tar") =>
            {
                return Err(ModelIssue::new("models.error.legacy_extension"));
            }
            ModelOperation::MigrateFeatures if !name.ends_with(".npy") => {
                return Err(ModelIssue::new("models.error.features_extension"));
            }
            _ => {}
        }
        if self.operation == ModelOperation::Inspect {
            return Ok(Request::InspectModel(InspectModelParams {
                source: source.to_path_buf(),
            }));
        }
        let destination = self
            .destination()
            .filter(|path| !path.as_os_str().is_empty())
            .ok_or_else(|| ModelIssue::new("models.error.no_destination"))?;
        if !destination.is_absolute() {
            return Err(ModelIssue::new("models.error.destination_absolute"));
        }
        let source_location = normalized(source);
        let destination_location = normalized(destination);
        if source_location == destination_location {
            return Err(ModelIssue::new("models.error.same_path"));
        }
        // Packages and checkpoints require an exact set of directory entries.
        // Publishing inside one would make the source invalid for its next use.
        if !self.operation.source_is_file()
            && Path::new(&destination_location).starts_with(Path::new(&source_location))
        {
            return Err(ModelIssue::new("models.error.destination_inside_source"));
        }
        if destination.file_name().is_none() {
            return Err(ModelIssue::new("models.error.destination_name"));
        }
        let source = source.to_path_buf();
        let destination = destination.to_path_buf();
        Ok(match self.operation {
            ModelOperation::Inspect => {
                unreachable!("inspect returned before destination validation")
            }
            ModelOperation::ImportLegacy => Request::ImportLegacyModel(ImportLegacyModelParams {
                source,
                kind: self
                    .kind
                    .legacy_kind()
                    .ok_or_else(|| ModelIssue::new("models.error.unsupported_kind"))?,
                destination,
            }),
            ModelOperation::ExportPackage => {
                Request::ExportModelPackage(ExportModelPackageParams {
                    source,
                    destination,
                })
            }
            ModelOperation::ExportOnnx => Request::ExportOnnx(ExportOnnxParams {
                source,
                kind: self.kind.onnx_kind(),
                destination,
            }),
            ModelOperation::MigrateFeatures => {
                Request::MigrateLegacyFeatures(MigrateLegacyFeaturesParams {
                    source,
                    destination,
                })
            }
        })
    }

    /// Check the selected paths when submitting, without loading model weights
    /// or creating output. Publication and format validation remain worker work.
    pub fn preflight(&self) -> Result<Request, ModelIssue> {
        let request = self.request()?;
        let source = self.source().expect("request validated the source");
        let metadata = fs::symlink_metadata(source).map_err(|error| {
            ModelIssue::with_detail(
                "models.error.source_unavailable",
                format!("{}: {error}", source.display()),
            )
        })?;
        let valid_source = if self.operation.source_is_file() {
            metadata.is_file()
        } else {
            metadata.is_dir()
        };
        if metadata.file_type().is_symlink() || !valid_source {
            let key = if self.operation.source_is_file() {
                "models.error.source_file"
            } else {
                "models.error.source_directory"
            };
            return Err(ModelIssue::with_detail(key, source.display().to_string()));
        }
        if self.operation.needs_destination() {
            let destination = self
                .destination()
                .expect("request validated the destination");
            match fs::symlink_metadata(destination) {
                Ok(_) => {
                    return Err(ModelIssue::with_detail(
                        "models.error.destination_exists",
                        destination.display().to_string(),
                    ));
                }
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                Err(error) => {
                    return Err(ModelIssue::with_detail(
                        "models.error.destination_unavailable",
                        format!("{}: {error}", destination.display()),
                    ));
                }
            }
            let parent = destination
                .parent()
                .ok_or_else(|| ModelIssue::new("models.error.destination_parent"))?;
            let metadata = fs::symlink_metadata(parent).map_err(|error| {
                ModelIssue::with_detail(
                    "models.error.destination_parent",
                    format!("{}: {error}", parent.display()),
                )
            })?;
            if metadata.file_type().is_symlink() || !metadata.is_dir() {
                return Err(ModelIssue::with_detail(
                    "models.error.destination_parent",
                    parent.display().to_string(),
                ));
            }
        }
        Ok(request)
    }
}

fn normalized(path: &Path) -> String {
    let mut normalized = PathBuf::new();
    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                normalized.pop();
            }
            other => normalized.push(other.as_os_str()),
        }
    }
    let value = normalized.to_string_lossy().into_owned();
    #[cfg(windows)]
    {
        value.to_lowercase()
    }
    #[cfg(not(windows))]
    {
        value
    }
}

/// Readable fields from a terminal worker result. Missing/null values stay
/// absent, particularly the parameter counts an inspected checkpoint cannot know.
pub fn result_facts(kind: TaskKind, result: &Value) -> Vec<Fact> {
    if ModelOperation::from_task_kind(kind).is_none() {
        return Vec::new();
    }
    let mut facts = Vec::new();
    if let Some(value) = result.get("model_kind").and_then(Value::as_str) {
        let value = match value {
            "feather_hubert" => FactValue::Key("models.kind.feather_hubert"),
            "original_unet" => FactValue::Key("models.kind.original_unet"),
            "mobileone_unet" | "mobile_one_unet" => FactValue::Key("models.kind.mobileone_unet"),
            other => FactValue::Text(other.to_owned()),
        };
        facts.push(Fact {
            label: "models.result.model_kind",
            value,
        });
    }
    if kind == TaskKind::InspectModel {
        if let Some(compatible) = result.get("compatible").and_then(Value::as_bool) {
            facts.push(Fact {
                label: "models.result.compatibility",
                value: FactValue::Key(if compatible {
                    "models.result.compatible"
                } else {
                    "models.result.incompatible"
                }),
            });
        }
        if let Some(reasons) = result.get("incompatibilities").and_then(Value::as_array) {
            for reason in reasons.iter().filter_map(Value::as_str) {
                let value = match reason {
                    "minimum_app_version" => FactValue::Key("models.reason.minimum_app_version"),
                    "model_kind" => FactValue::Key("models.reason.model_kind"),
                    "architecture_version" => FactValue::Key("models.reason.architecture_version"),
                    "model_config_sha256" => FactValue::Key("models.reason.model_config_sha256"),
                    "file_size" => FactValue::Key("models.reason.file_size"),
                    other => FactValue::Text(other.to_owned()),
                };
                facts.push(Fact {
                    label: "models.result.reason",
                    value,
                });
            }
        }
    }
    for (field, label) in [
        ("source_path", "models.result.source"),
        ("source", "models.result.source"),
        ("destination", "models.result.destination"),
        ("architecture_version", "models.result.architecture"),
        ("parameter_count", "models.result.parameters"),
        ("total_elements", "models.result.parameters"),
        ("tensor_count", "models.result.tensors"),
        ("epoch", "models.result.epoch"),
        ("global_step", "models.result.step"),
        ("opset", "models.result.opset"),
        ("tokens", "models.result.tokens"),
        ("dims", "models.result.dimensions"),
        ("bytes", "models.result.bytes"),
    ] {
        let value = match result.get(field) {
            Some(Value::String(value)) => Some(value.clone()),
            Some(Value::Number(value)) => Some(value.to_string()),
            _ => None,
        };
        if let Some(value) = value {
            facts.push(Fact {
                label,
                value: FactValue::Text(value),
            });
        }
    }
    facts
}
