use std::{
    collections::{BTreeMap, BTreeSet},
    path::Path,
};

use burn::tensor::{DType, backend::Backend};
use burn_store::pytorch::{PytorchReader, reader::PickleValue};
use feathertalk_models::feather_hubert::{FeatherHubertConfig, FeatherHubertEncoder};

use crate::{
    LegacyImportRequest, LegacyModelKind, WeightImportError, import_into,
    legacy::select_top_level_key,
    source::{
        DEFAULT_MAX_FILE_BYTES, DEFAULT_MAX_TENSOR_COUNT, DEFAULT_MAX_TOTAL_ELEMENTS, SnapshotFile,
        tensor_elements,
    },
};

#[derive(Debug, Clone)]
pub struct FeatherHubertCheckpoint {
    config: FeatherHubertConfig,
    source_sha256: String,
    tensor_count: usize,
    total_elements: u64,
}

impl FeatherHubertCheckpoint {
    pub fn config(&self) -> &FeatherHubertConfig {
        &self.config
    }

    pub fn source_sha256(&self) -> &str {
        &self.source_sha256
    }

    pub fn tensor_count(&self) -> usize {
        self.tensor_count
    }

    pub fn total_elements(&self) -> u64 {
        self.total_elements
    }
}

#[derive(Debug, Clone)]
struct TensorFact {
    dtype: DType,
    shape: Vec<usize>,
}

impl TensorFact {
    #[cfg(test)]
    fn f32(shape: Vec<usize>) -> Self {
        Self {
            dtype: DType::F32,
            shape,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct MetadataConfig {
    channels: usize,
    expansion: usize,
    num_blocks: usize,
    output_dim: usize,
    dropout: f64,
}

#[derive(Debug, Clone, Copy)]
struct SafetyLimits {
    max_tensor_count: usize,
    max_total_elements: u64,
}

pub fn inspect_feather_hubert_checkpoint(
    path: impl AsRef<Path>,
) -> Result<FeatherHubertCheckpoint, WeightImportError> {
    let snapshot = SnapshotFile::copy_from(path.as_ref(), DEFAULT_MAX_FILE_BYTES)?;
    inspect_snapshot(
        &snapshot,
        SafetyLimits {
            max_tensor_count: DEFAULT_MAX_TENSOR_COUNT,
            max_total_elements: DEFAULT_MAX_TOTAL_ELEMENTS,
        },
    )
}

pub fn load_feather_hubert_checkpoint<B: Backend>(
    path: impl AsRef<Path>,
    device: &B::Device,
) -> Result<(FeatherHubertEncoder<B>, FeatherHubertCheckpoint), WeightImportError> {
    let path = path.as_ref();
    let checkpoint = inspect_feather_hubert_checkpoint(path)?;
    let mut model = checkpoint.config.clone().init::<B>(device);
    let report = import_into::<B, _>(
        &mut model,
        &LegacyImportRequest {
            path: path.to_owned(),
            kind: LegacyModelKind::FeatherHubert,
            top_level_key: None,
            max_file_bytes: DEFAULT_MAX_FILE_BYTES,
            max_tensor_count: DEFAULT_MAX_TENSOR_COUNT,
            max_total_elements: DEFAULT_MAX_TOTAL_ELEMENTS,
        },
    )?;
    if report.source_sha256 != checkpoint.source_sha256
        || report.tensor_count != checkpoint.tensor_count
        || report.total_elements != checkpoint.total_elements
    {
        return Err(WeightImportError::UnsupportedStructure(format!(
            "checkpoint changed between inspection and import: inspected hash/count/elements {}/{}/{}, imported {}/{}/{}",
            checkpoint.source_sha256,
            checkpoint.tensor_count,
            checkpoint.total_elements,
            report.source_sha256,
            report.tensor_count,
            report.total_elements,
        )));
    }
    Ok((model, checkpoint))
}

fn inspect_snapshot(
    snapshot: &SnapshotFile,
    limits: SafetyLimits,
) -> Result<FeatherHubertCheckpoint, WeightImportError> {
    let selection_request = LegacyImportRequest {
        path: snapshot.path().to_owned(),
        kind: LegacyModelKind::FeatherHubert,
        top_level_key: None,
        max_file_bytes: DEFAULT_MAX_FILE_BYTES,
        max_tensor_count: limits.max_tensor_count,
        max_total_elements: limits.max_total_elements,
    };
    let top_level_key = select_top_level_key(snapshot.path(), &selection_request)?;
    let reader = match top_level_key.as_deref() {
        Some(key) => PytorchReader::with_top_level_key(snapshot.path(), key),
        None => PytorchReader::new(snapshot.path()),
    }
    .map_err(store_error)?;

    let mut facts = BTreeMap::new();
    let mut total_elements = 0u64;
    for (key, tensor) in reader.into_tensors() {
        total_elements = total_elements
            .checked_add(tensor_elements(&tensor)?)
            .ok_or_else(|| {
                WeightImportError::UnsafeLimit("total tensor elements overflowed u64".to_owned())
            })?;
        if facts
            .insert(
                key.clone(),
                TensorFact {
                    dtype: tensor.dtype,
                    shape: tensor.shape.to_vec(),
                },
            )
            .is_some()
        {
            return Err(WeightImportError::DuplicateKey(key));
        }
    }
    let tensor_count = facts.len();
    validate_limits(tensor_count, total_elements, limits)?;

    let metadata = read_metadata(snapshot.path())?;
    let config = infer_config(&facts, metadata)?;
    Ok(FeatherHubertCheckpoint {
        config,
        source_sha256: snapshot.sha256().to_owned(),
        tensor_count,
        total_elements,
    })
}

fn read_metadata(path: &Path) -> Result<Option<MetadataConfig>, WeightImportError> {
    let root = PytorchReader::read_pickle_data(path, None).map_err(store_error)?;
    let PickleValue::Dict(root) = root else {
        return Err(WeightImportError::UnsupportedStructure(
            "checkpoint root must be a dictionary".to_owned(),
        ));
    };
    let config = root
        .get("config")
        .map(|value| parse_metadata_config("config", value))
        .transpose()?;
    let args = root
        .get("args")
        .map(|value| parse_metadata_config("args", value))
        .transpose()?;
    match (config, args) {
        (Some(config), Some(args)) if config != args => {
            Err(WeightImportError::UnsupportedStructure(
                "checkpoint config and args FeatherHuBERT values differ".to_owned(),
            ))
        }
        (Some(config), _) | (_, Some(config)) => Ok(Some(config)),
        (None, None) => Ok(None),
    }
}

fn parse_metadata_config(
    name: &str,
    value: &PickleValue,
) -> Result<MetadataConfig, WeightImportError> {
    let PickleValue::Dict(values) = value else {
        return Err(WeightImportError::UnsupportedStructure(format!(
            "checkpoint {name} must be a dictionary"
        )));
    };
    Ok(MetadataConfig {
        channels: positive_usize(name, values, "channels")?,
        expansion: positive_usize(name, values, "expansion")?,
        num_blocks: positive_usize(name, values, "num_blocks")?,
        output_dim: positive_usize(name, values, "output_dim")?,
        dropout: dropout_value(name, values)?,
    })
}

fn positive_usize(
    name: &str,
    values: &std::collections::HashMap<String, PickleValue>,
    field: &str,
) -> Result<usize, WeightImportError> {
    let Some(PickleValue::Int(value)) = values.get(field) else {
        return Err(WeightImportError::UnsupportedStructure(format!(
            "checkpoint {name}.{field} must be a positive integer"
        )));
    };
    usize::try_from(*value)
        .ok()
        .filter(|value| *value > 0)
        .ok_or_else(|| {
            WeightImportError::UnsupportedStructure(format!(
                "checkpoint {name}.{field} must be a positive integer fitting usize"
            ))
        })
}

fn dropout_value(
    name: &str,
    values: &std::collections::HashMap<String, PickleValue>,
) -> Result<f64, WeightImportError> {
    let value = match values.get("dropout") {
        Some(PickleValue::Int(value)) => *value as f64,
        Some(PickleValue::Float(value)) => *value,
        _ => {
            return Err(WeightImportError::UnsupportedStructure(format!(
                "checkpoint {name}.dropout must be numeric"
            )));
        }
    };
    if !value.is_finite() || !(0.0..1.0).contains(&value) {
        return Err(WeightImportError::UnsupportedStructure(format!(
            "checkpoint {name}.dropout must be finite and in [0,1)"
        )));
    }
    Ok(value)
}

fn infer_config(
    facts: &BTreeMap<String, TensorFact>,
    metadata: Option<MetadataConfig>,
) -> Result<FeatherHubertConfig, WeightImportError> {
    let proj = required_fact(facts, "proj.weight")?;
    require_f32("proj.weight", proj)?;
    if proj.shape.len() != 3 || proj.shape[2] != 1 || proj.shape[0] == 0 || proj.shape[1] == 0 {
        return Err(WeightImportError::ShapeMismatch("proj.weight".to_owned()));
    }
    let output_dim = proj.shape[0];
    let channels = proj.shape[1];

    let expand = required_fact(facts, "encoder.0.pw_expand.weight")?;
    require_f32("encoder.0.pw_expand.weight", expand)?;
    if expand.shape.len() != 3
        || expand.shape[1] != channels
        || expand.shape[2] != 1
        || expand.shape[0] == 0
        || !expand.shape[0].is_multiple_of(channels)
    {
        return Err(WeightImportError::ShapeMismatch(
            "encoder.0.pw_expand.weight".to_owned(),
        ));
    }
    let expansion = expand.shape[0] / channels;
    let num_blocks = contiguous_block_count(facts)?;

    let expected = expected_facts(channels, expansion, num_blocks, output_dim)?;
    if let Some(key) = expected.keys().find(|key| !facts.contains_key(*key)) {
        return Err(WeightImportError::MissingTensor(key.clone()));
    }
    if let Some(key) = facts.keys().find(|key| !expected.contains_key(*key)) {
        return Err(WeightImportError::UnexpectedTensor(key.clone()));
    }
    for (key, expected) in expected {
        let actual = facts.get(&key).expect("key sets were checked above");
        require_f32(&key, actual)?;
        if actual.shape != expected.shape {
            return Err(WeightImportError::ShapeMismatch(key));
        }
    }

    if let Some(metadata) = metadata
        && (metadata.channels != channels
            || metadata.expansion != expansion
            || metadata.num_blocks != num_blocks
            || metadata.output_dim != output_dim)
    {
        return Err(WeightImportError::UnsupportedStructure(format!(
            "checkpoint metadata config {:?} differs from tensor-derived {channels}/{expansion}/{num_blocks}/{output_dim}",
            metadata,
        )));
    }

    Ok(FeatherHubertConfig {
        channels,
        expansion,
        num_blocks,
        output_dim,
        dropout: 0.0,
    })
}

fn contiguous_block_count(
    facts: &BTreeMap<String, TensorFact>,
) -> Result<usize, WeightImportError> {
    let mut indices = BTreeSet::new();
    for key in facts.keys() {
        let Some(rest) = key.strip_prefix("encoder.") else {
            continue;
        };
        let Some((index, _)) = rest.split_once('.') else {
            return Err(WeightImportError::UnsupportedStructure(format!(
                "invalid encoder tensor key {key}"
            )));
        };
        let index = index.parse::<usize>().map_err(|_| {
            WeightImportError::UnsupportedStructure(format!("invalid encoder index in {key}"))
        })?;
        indices.insert(index);
    }
    if indices.is_empty() {
        return Err(WeightImportError::MissingTensor(
            "encoder.0.pw_expand.weight".to_owned(),
        ));
    }
    for (expected, actual) in indices.iter().copied().enumerate() {
        if actual != expected {
            return Err(WeightImportError::UnsupportedStructure(format!(
                "encoder block indices must be contiguous from zero, got {indices:?}"
            )));
        }
    }
    Ok(indices.len())
}

fn expected_facts(
    channels: usize,
    expansion: usize,
    num_blocks: usize,
    output_dim: usize,
) -> Result<BTreeMap<String, TensorFact>, WeightImportError> {
    let hidden = channels.checked_mul(expansion).ok_or_else(|| {
        WeightImportError::UnsafeLimit("hidden channel count overflowed usize".to_owned())
    })?;
    let frontend = [
        (1, 64, 10),
        (64, 128, 3),
        (128, 256, 3),
        (256, 384, 3),
        (384, channels, 3),
        (channels, channels, 2),
        (channels, channels, 2),
    ];
    let mut expected = BTreeMap::new();
    for (index, (input, output, kernel)) in frontend.into_iter().enumerate() {
        expected.insert(
            format!("frontend.layers.{index}.conv.weight"),
            TensorFact::f32_for_runtime(vec![output, input, kernel]),
        );
        for parameter in ["weight", "bias"] {
            expected.insert(
                format!("frontend.layers.{index}.norm.{parameter}"),
                TensorFact::f32_for_runtime(vec![output]),
            );
        }
    }
    for index in 0..num_blocks {
        for parameter in ["weight", "bias"] {
            expected.insert(
                format!("encoder.{index}.norm.{parameter}"),
                TensorFact::f32_for_runtime(vec![channels]),
            );
        }
        expected.insert(
            format!("encoder.{index}.pw_expand.weight"),
            TensorFact::f32_for_runtime(vec![hidden, channels, 1]),
        );
        expected.insert(
            format!("encoder.{index}.dw_conv.weight"),
            TensorFact::f32_for_runtime(vec![hidden, 1, 5]),
        );
        expected.insert(
            format!("encoder.{index}.pw_project.weight"),
            TensorFact::f32_for_runtime(vec![channels, hidden, 1]),
        );
    }
    for parameter in ["weight", "bias"] {
        expected.insert(
            format!("final_norm.{parameter}"),
            TensorFact::f32_for_runtime(vec![channels]),
        );
    }
    expected.insert(
        "proj.weight".to_owned(),
        TensorFact::f32_for_runtime(vec![output_dim, channels, 1]),
    );
    expected.insert(
        "proj.bias".to_owned(),
        TensorFact::f32_for_runtime(vec![output_dim]),
    );
    Ok(expected)
}

impl TensorFact {
    fn f32_for_runtime(shape: Vec<usize>) -> Self {
        Self {
            dtype: DType::F32,
            shape,
        }
    }
}

fn required_fact<'a>(
    facts: &'a BTreeMap<String, TensorFact>,
    key: &str,
) -> Result<&'a TensorFact, WeightImportError> {
    facts
        .get(key)
        .ok_or_else(|| WeightImportError::MissingTensor(key.to_owned()))
}

fn require_f32(key: &str, fact: &TensorFact) -> Result<(), WeightImportError> {
    if fact.dtype != DType::F32 {
        return Err(WeightImportError::DTypeMismatch(key.to_owned()));
    }
    Ok(())
}

fn validate_limits(
    tensor_count: usize,
    total_elements: u64,
    limits: SafetyLimits,
) -> Result<(), WeightImportError> {
    if tensor_count > limits.max_tensor_count {
        return Err(WeightImportError::UnsafeLimit(format!(
            "tensor count {tensor_count} exceeds {}",
            limits.max_tensor_count
        )));
    }
    if total_elements > limits.max_total_elements {
        return Err(WeightImportError::UnsafeLimit(format!(
            "total tensor elements {total_elements} exceeds {}",
            limits.max_total_elements
        )));
    }
    Ok(())
}

fn store_error(error: impl std::fmt::Display) -> WeightImportError {
    WeightImportError::Store(error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_facts(
        channels: usize,
        expansion: usize,
        num_blocks: usize,
        output_dim: usize,
    ) -> BTreeMap<String, TensorFact> {
        expected_facts(channels, expansion, num_blocks, output_dim).unwrap()
    }

    fn metadata(
        channels: usize,
        expansion: usize,
        num_blocks: usize,
        output_dim: usize,
        dropout: f64,
    ) -> MetadataConfig {
        MetadataConfig {
            channels,
            expansion,
            num_blocks,
            output_dim,
            dropout,
        }
    }

    #[test]
    fn facts_infer_the_micro_config() {
        let facts = valid_facts(32, 2, 2, 64);
        let config = infer_config(&facts, None).unwrap();
        assert_eq!((config.channels, config.expansion), (32, 2));
        assert_eq!((config.num_blocks, config.output_dim), (2, 64));
        assert_eq!(config.dropout, 0.0);
    }

    #[test]
    fn facts_reject_missing_block_tensor_wrong_shape_dtype_and_extra_key() {
        let mut missing = valid_facts(32, 2, 2, 64);
        missing.remove("encoder.1.dw_conv.weight");
        assert!(matches!(
            infer_config(&missing, None),
            Err(WeightImportError::MissingTensor(_))
        ));

        let mut shape = valid_facts(32, 2, 2, 64);
        shape.get_mut("proj.weight").unwrap().shape = vec![64, 31, 1];
        assert!(matches!(
            infer_config(&shape, None),
            Err(WeightImportError::ShapeMismatch(_))
        ));

        let mut dtype = valid_facts(32, 2, 2, 64);
        dtype.get_mut("final_norm.weight").unwrap().dtype = DType::I64;
        assert!(matches!(
            infer_config(&dtype, None),
            Err(WeightImportError::DTypeMismatch(_))
        ));

        let mut extra = valid_facts(32, 2, 2, 64);
        extra.insert("unexpected.weight".into(), TensorFact::f32(vec![1]));
        assert!(matches!(
            infer_config(&extra, None),
            Err(WeightImportError::UnexpectedTensor(_))
        ));
    }

    #[test]
    fn facts_reject_gapped_blocks_and_non_integral_expansion() {
        let mut gapped = valid_facts(32, 2, 3, 64);
        gapped.retain(|key, _| !key.starts_with("encoder.1."));
        assert!(matches!(
            infer_config(&gapped, None),
            Err(WeightImportError::UnsupportedStructure(_))
        ));

        let mut expansion = valid_facts(32, 2, 2, 64);
        expansion
            .get_mut("encoder.0.pw_expand.weight")
            .unwrap()
            .shape = vec![63, 32, 1];
        assert!(matches!(
            infer_config(&expansion, None),
            Err(WeightImportError::ShapeMismatch(_))
        ));
    }

    #[test]
    fn metadata_must_match_structure_and_dropout_is_disabled_for_inference() {
        let facts = valid_facts(32, 2, 2, 64);
        assert!(matches!(
            infer_config(&facts, Some(metadata(64, 2, 2, 64, 0.05))),
            Err(WeightImportError::UnsupportedStructure(_))
        ));
        let config = infer_config(&facts, Some(metadata(32, 2, 2, 64, 0.05))).unwrap();
        assert_eq!(config.dropout, 0.0);
    }

    #[test]
    fn metadata_dropout_validation_rejects_invalid_values() {
        for value in [f64::NAN, -0.1, 1.0] {
            let mut values = std::collections::HashMap::new();
            values.insert("channels".to_owned(), PickleValue::Int(32));
            values.insert("expansion".to_owned(), PickleValue::Int(2));
            values.insert("num_blocks".to_owned(), PickleValue::Int(2));
            values.insert("output_dim".to_owned(), PickleValue::Int(64));
            values.insert("dropout".to_owned(), PickleValue::Float(value));
            assert!(matches!(
                parse_metadata_config("config", &PickleValue::Dict(values)),
                Err(WeightImportError::UnsupportedStructure(_))
            ));
        }
    }

    #[test]
    fn safety_limits_reject_counts_and_elements() {
        assert!(matches!(
            validate_limits(
                2,
                10,
                SafetyLimits {
                    max_tensor_count: 1,
                    max_total_elements: 10,
                }
            ),
            Err(WeightImportError::UnsafeLimit(_))
        ));
        assert!(matches!(
            validate_limits(
                1,
                11,
                SafetyLimits {
                    max_tensor_count: 1,
                    max_total_elements: 10,
                }
            ),
            Err(WeightImportError::UnsafeLimit(_))
        ));
    }
}
