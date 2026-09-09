use std::{collections::BTreeSet, path::Path};

use burn::tensor::DType;
use burn_store::pytorch::{PytorchReader, reader::PickleValue};

use crate::{
    PfldIgnoredTensors, PfldImportRequest, TensorAudit, TensorSummary, WeightImportError,
    source::tensor_elements,
};

use super::{
    PFLD_CHECKPOINT_EPOCH,
    key_map::{
        LOCALIZATION_KEYS, is_valid_batch_norm_counter, map_pfld_key, reject_duplicate_destinations,
    },
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct PfldEnvelope {
    pub(super) has_auxiliarynet: bool,
}

#[derive(Debug, Clone, Copy)]
struct SafetyLimits {
    max_tensor_count: usize,
    max_total_elements: u64,
}

#[cfg(test)]
impl SafetyLimits {
    fn unbounded() -> Self {
        Self {
            max_tensor_count: usize::MAX,
            max_total_elements: u64::MAX,
        }
    }
}

#[derive(Debug, Clone)]
struct TensorFact {
    key: String,
    dtype: DType,
    elements: u64,
}

#[derive(Debug, Clone)]
pub(super) struct PfldInspection {
    pub(super) backbone: TensorSummary,
    pub(super) applied: TensorSummary,
    pub(super) ignored: PfldIgnoredTensors,
    pub(super) expected_applied: BTreeSet<String>,
    pub(super) expected_unused: BTreeSet<String>,
}

pub(super) fn validate_envelope(value: PickleValue) -> Result<PfldEnvelope, WeightImportError> {
    let PickleValue::Dict(root) = value else {
        return Err(WeightImportError::InvalidPfldEnvelope(
            "checkpoint root must be a dictionary".to_owned(),
        ));
    };

    let mut root_keys = root.keys().map(String::as_str).collect::<Vec<_>>();
    root_keys.sort_unstable();
    let has_auxiliarynet = match root_keys.as_slice() {
        ["epoch", "pfld_backbone"] => false,
        ["auxiliarynet", "epoch", "pfld_backbone"] => true,
        _ => {
            return Err(WeightImportError::InvalidPfldEnvelope(format!(
                "checkpoint root keys must be exactly epoch/pfld_backbone with optional auxiliarynet, got {root_keys:?}"
            )));
        }
    };

    match root.get("epoch") {
        Some(PickleValue::Int(epoch)) if *epoch == PFLD_CHECKPOINT_EPOCH as i64 => {}
        Some(PickleValue::Int(epoch)) => {
            return Err(WeightImportError::InvalidPfldEpoch {
                expected: PFLD_CHECKPOINT_EPOCH,
                actual: epoch.to_string(),
            });
        }
        Some(actual) => {
            return Err(WeightImportError::InvalidPfldEpoch {
                expected: PFLD_CHECKPOINT_EPOCH,
                actual: format!("{actual:?}"),
            });
        }
        None => {
            return Err(WeightImportError::InvalidPfldEnvelope(
                "checkpoint root is missing epoch".to_owned(),
            ));
        }
    }

    validate_tensor_mapping(root.get("pfld_backbone"), "pfld_backbone", true)?;
    if has_auxiliarynet {
        validate_tensor_mapping(root.get("auxiliarynet"), "auxiliarynet", false)?;
    }

    Ok(PfldEnvelope { has_auxiliarynet })
}

fn validate_tensor_mapping(
    value: Option<&PickleValue>,
    name: &str,
    require_non_empty: bool,
) -> Result<(), WeightImportError> {
    let Some(PickleValue::Dict(tensors)) = value else {
        return Err(WeightImportError::InvalidPfldEnvelope(format!(
            "{name} must be a tensor dictionary"
        )));
    };
    let tensor_count = tensors
        .iter()
        .filter(|(key, value)| key.as_str() != "_metadata" && matches!(value, PickleValue::None))
        .count();
    if require_non_empty && tensor_count == 0 {
        return Err(WeightImportError::InvalidPfldEnvelope(format!(
            "{name} must not be empty"
        )));
    }
    for (key, value) in tensors {
        if key == "_metadata" {
            validate_state_dict_metadata(value, name)?;
        } else if !matches!(value, PickleValue::None) {
            return Err(WeightImportError::InvalidPfldEnvelope(format!(
                "{name} must contain only tensors and standard _metadata"
            )));
        }
    }
    Ok(())
}

fn validate_state_dict_metadata(value: &PickleValue, name: &str) -> Result<(), WeightImportError> {
    let PickleValue::Dict(metadata) = value else {
        return Err(WeightImportError::InvalidPfldEnvelope(format!(
            "{name}._metadata must be a dictionary"
        )));
    };
    if metadata.values().any(|value| {
        !matches!(
            value,
            PickleValue::Dict(entry)
                if entry.len() == 1
                    && matches!(entry.get("version"), Some(PickleValue::Int(_)))
        )
    }) {
        return Err(WeightImportError::InvalidPfldEnvelope(format!(
            "{name}._metadata entries must contain only an integer version"
        )));
    }
    Ok(())
}

pub(super) fn inspect_checkpoint(
    path: &Path,
    envelope: PfldEnvelope,
    request: &PfldImportRequest,
) -> Result<PfldInspection, WeightImportError> {
    let backbone = read_tensor_facts(path, "pfld_backbone")?;
    let auxiliary = envelope
        .has_auxiliarynet
        .then(|| read_tensor_facts(path, "auxiliarynet"))
        .transpose()?;

    audit_tensor_facts(
        backbone,
        auxiliary,
        SafetyLimits {
            max_tensor_count: request.max_tensor_count,
            max_total_elements: request.max_total_elements,
        },
    )
}

fn read_tensor_facts(
    path: &Path,
    top_level_key: &str,
) -> Result<Vec<TensorFact>, WeightImportError> {
    let reader = PytorchReader::with_top_level_key(path, top_level_key)
        .map_err(|error| WeightImportError::Store(error.to_string()))?;
    reader
        .into_tensors()
        .into_iter()
        .map(|(key, snapshot)| {
            Ok(TensorFact {
                key,
                dtype: snapshot.dtype,
                elements: tensor_elements(&snapshot)?,
            })
        })
        .collect()
}

fn audit_tensor_facts(
    backbone: Vec<TensorFact>,
    auxiliary: Option<Vec<TensorFact>>,
    limits: SafetyLimits,
) -> Result<PfldInspection, WeightImportError> {
    let backbone_summary = summarize_facts(&backbone)?;
    let auxiliary_summary = auxiliary
        .as_deref()
        .map(summarize_facts)
        .transpose()?
        .unwrap_or(TensorSummary {
            tensor_count: 0,
            total_elements: 0,
        });
    let global_tensor_count = backbone_summary
        .tensor_count
        .checked_add(auxiliary_summary.tensor_count)
        .ok_or_else(|| {
            WeightImportError::UnsafeLimit("tensor count overflowed usize".to_owned())
        })?;
    let global_total_elements = backbone_summary
        .total_elements
        .checked_add(auxiliary_summary.total_elements)
        .ok_or_else(|| {
            WeightImportError::UnsafeLimit("total tensor elements overflowed u64".to_owned())
        })?;
    if global_tensor_count > limits.max_tensor_count {
        return Err(WeightImportError::UnsafeLimit(format!(
            "tensor count {global_tensor_count} exceeds {}",
            limits.max_tensor_count
        )));
    }
    if global_total_elements > limits.max_total_elements {
        return Err(WeightImportError::UnsafeLimit(format!(
            "total tensor elements {global_total_elements} exceeds {}",
            limits.max_total_elements
        )));
    }

    let source_keys = backbone
        .iter()
        .map(|fact| fact.key.clone())
        .collect::<BTreeSet<_>>();
    let mut localization_keys = backbone
        .iter()
        .filter(|fact| fact.key.starts_with("localization."))
        .map(|fact| fact.key.clone())
        .collect::<Vec<_>>();
    localization_keys.sort();
    let expected_localization = LOCALIZATION_KEYS.map(str::to_owned).to_vec();
    if localization_keys != expected_localization {
        return Err(WeightImportError::InvalidPfldIgnoredSet(format!(
            "localization tensors must be exactly {expected_localization:?}, got {localization_keys:?}"
        )));
    }

    let mut applied_source_keys = Vec::new();
    let mut applied_count = 0usize;
    let mut applied_elements = 0u64;
    let mut batch_norm_keys = Vec::new();
    let mut batch_norm_elements = 0u64;
    let mut localization_elements = 0u64;
    let mut expected_applied = BTreeSet::new();
    let mut expected_unused = BTreeSet::new();

    for fact in &backbone {
        if fact.key.starts_with("localization.") {
            localization_elements = checked_add_elements(localization_elements, fact.elements)?;
            expected_unused.insert(fact.key.clone());
            continue;
        }
        if is_valid_batch_norm_counter(&fact.key, &source_keys) {
            batch_norm_elements = checked_add_elements(batch_norm_elements, fact.elements)?;
            batch_norm_keys.push(fact.key.clone());
            expected_unused.insert(map_pfld_key(&fact.key));
            continue;
        }

        let mapped = map_pfld_key(&fact.key);
        if fact.dtype != DType::F32 {
            return Err(WeightImportError::DTypeMismatch(mapped));
        }
        applied_count = applied_count.checked_add(1).ok_or_else(|| {
            WeightImportError::UnsafeLimit("tensor count overflowed usize".to_owned())
        })?;
        applied_elements = checked_add_elements(applied_elements, fact.elements)?;
        applied_source_keys.push(fact.key.clone());
        expected_applied.insert(mapped);
    }

    reject_duplicate_destinations(applied_source_keys)?;
    batch_norm_keys.sort();

    let auxiliary_audit = auxiliary.map(|facts| {
        let mut keys = facts
            .into_iter()
            .map(|fact| format!("auxiliarynet.{}", fact.key))
            .collect::<Vec<_>>();
        keys.sort();
        TensorAudit {
            tensor_count: auxiliary_summary.tensor_count,
            total_elements: auxiliary_summary.total_elements,
            keys,
        }
    });

    Ok(PfldInspection {
        backbone: backbone_summary,
        applied: TensorSummary {
            tensor_count: applied_count,
            total_elements: applied_elements,
        },
        ignored: PfldIgnoredTensors {
            batch_norm_counters: TensorAudit {
                tensor_count: batch_norm_keys.len(),
                total_elements: batch_norm_elements,
                keys: batch_norm_keys,
            },
            localization: TensorAudit {
                tensor_count: localization_keys.len(),
                total_elements: localization_elements,
                keys: localization_keys,
            },
            auxiliarynet: auxiliary_audit,
        },
        expected_applied,
        expected_unused,
    })
}

fn summarize_facts(facts: &[TensorFact]) -> Result<TensorSummary, WeightImportError> {
    let tensor_count = facts.iter().try_fold(0usize, |count, _| {
        count.checked_add(1).ok_or_else(|| {
            WeightImportError::UnsafeLimit("tensor count overflowed usize".to_owned())
        })
    })?;
    let total_elements = facts.iter().try_fold(0u64, |total, fact| {
        checked_add_elements(total, fact.elements)
    })?;
    Ok(TensorSummary {
        tensor_count,
        total_elements,
    })
}

fn checked_add_elements(total: u64, elements: u64) -> Result<u64, WeightImportError> {
    total.checked_add(elements).ok_or_else(|| {
        WeightImportError::UnsafeLimit("total tensor elements overflowed u64".to_owned())
    })
}

#[cfg(test)]
mod tests {
    use burn::tensor::DType;
    use burn_store::pytorch::reader::PickleValue;

    use crate::WeightImportError;

    use super::*;

    fn tensor_dict(keys: &[&str]) -> PickleValue {
        PickleValue::Dict(
            keys.iter()
                .map(|key| ((*key).to_owned(), PickleValue::None))
                .collect(),
        )
    }

    fn root(entries: impl IntoIterator<Item = (&'static str, PickleValue)>) -> PickleValue {
        PickleValue::Dict(
            entries
                .into_iter()
                .map(|(key, value)| (key.to_owned(), value))
                .collect(),
        )
    }

    #[test]
    fn accepts_standard_pytorch_state_dict_metadata() {
        let backbone = PickleValue::Dict(
            [
                ("conv8.0.weight".to_owned(), PickleValue::None),
                (
                    "_metadata".to_owned(),
                    PickleValue::Dict(
                        [
                            (
                                "".to_owned(),
                                PickleValue::Dict(
                                    [("version".to_owned(), PickleValue::Int(1))]
                                        .into_iter()
                                        .collect(),
                                ),
                            ),
                            (
                                "conv8.0".to_owned(),
                                PickleValue::Dict(
                                    [("version".to_owned(), PickleValue::Int(2))]
                                        .into_iter()
                                        .collect(),
                                ),
                            ),
                        ]
                        .into_iter()
                        .collect(),
                    ),
                ),
            ]
            .into_iter()
            .collect(),
        );

        assert_eq!(
            validate_envelope(root([
                ("epoch", PickleValue::Int(335)),
                ("pfld_backbone", backbone),
            ]))
            .unwrap(),
            PfldEnvelope {
                has_auxiliarynet: false
            }
        );
    }

    #[test]
    fn accepts_only_the_two_approved_envelopes() {
        let minimal = root([
            ("epoch", PickleValue::Int(335)),
            ("pfld_backbone", tensor_dict(&["conv8.0.weight"])),
        ]);
        assert_eq!(
            validate_envelope(minimal).unwrap(),
            PfldEnvelope {
                has_auxiliarynet: false
            }
        );

        let with_auxiliary = root([
            ("epoch", PickleValue::Int(335)),
            ("pfld_backbone", tensor_dict(&["conv8.0.weight"])),
            ("auxiliarynet", tensor_dict(&["conv1.weight"])),
        ]);
        assert_eq!(
            validate_envelope(with_auxiliary).unwrap(),
            PfldEnvelope {
                has_auxiliarynet: true
            }
        );
    }

    #[test]
    fn rejects_bad_roots_keys_epoch_and_tensor_mappings() {
        assert!(matches!(
            validate_envelope(PickleValue::List(Vec::new())),
            Err(WeightImportError::InvalidPfldEnvelope(_))
        ));
        assert!(matches!(
            validate_envelope(root([("pfld_backbone", tensor_dict(&["conv8.0.weight"]))])),
            Err(WeightImportError::InvalidPfldEnvelope(_))
        ));
        assert!(matches!(
            validate_envelope(root([
                ("epoch", PickleValue::String("335".to_owned())),
                ("pfld_backbone", tensor_dict(&["conv8.0.weight"])),
            ])),
            Err(WeightImportError::InvalidPfldEpoch { expected: 335, .. })
        ));
        assert!(matches!(
            validate_envelope(root([
                ("epoch", PickleValue::Int(334)),
                ("pfld_backbone", tensor_dict(&["conv8.0.weight"])),
            ])),
            Err(WeightImportError::InvalidPfldEpoch {
                expected: 335,
                actual
            }) if actual == "334"
        ));
        assert!(matches!(
            validate_envelope(root([
                ("epoch", PickleValue::Int(335)),
                ("pfld_backbone", tensor_dict(&[])),
            ])),
            Err(WeightImportError::InvalidPfldEnvelope(_))
        ));
        assert!(matches!(
            validate_envelope(root([
                ("epoch", PickleValue::Int(335)),
                ("pfld_backbone", tensor_dict(&["conv8.0.weight"])),
                ("state_dict", tensor_dict(&["conv8.0.weight"])),
            ])),
            Err(WeightImportError::InvalidPfldEnvelope(_))
        ));
    }

    fn fact(key: &str, dtype: DType, elements: u64) -> TensorFact {
        TensorFact {
            key: key.to_owned(),
            dtype,
            elements,
        }
    }

    #[test]
    fn audit_requires_exact_localization_and_counts_auxiliary_tensors() {
        let backbone = vec![
            fact("conv8.0.weight", DType::F32, 64),
            fact("localization.0.weight", DType::F32, 2_304),
            fact("localization.0.bias", DType::F32, 32),
            fact("localization.3.weight", DType::F32, 64),
            fact("localization.3.bias", DType::F32, 10),
        ];
        let auxiliary = vec![fact("conv1.weight", DType::F32, 7)];
        let inspection = audit_tensor_facts(
            backbone,
            Some(auxiliary),
            SafetyLimits {
                max_tensor_count: 6,
                max_total_elements: 2_481,
            },
        )
        .unwrap();

        assert_eq!(
            inspection.backbone,
            TensorSummary {
                tensor_count: 5,
                total_elements: 2_474
            }
        );
        assert_eq!(
            inspection.applied,
            TensorSummary {
                tensor_count: 1,
                total_elements: 64
            }
        );
        assert_eq!(
            inspection.ignored.localization.keys,
            LOCALIZATION_KEYS.map(str::to_owned)
        );
        assert_eq!(
            inspection.ignored.auxiliarynet.unwrap(),
            TensorAudit {
                tensor_count: 1,
                total_elements: 7,
                keys: vec!["auxiliarynet.conv1.weight".to_owned()],
            }
        );
    }

    #[test]
    fn audit_rejects_partial_or_extra_localization_sets() {
        let partial = vec![
            fact("localization.0.weight", DType::F32, 1),
            fact("localization.0.bias", DType::F32, 1),
            fact("localization.3.weight", DType::F32, 1),
        ];
        assert!(matches!(
            audit_tensor_facts(partial, None, SafetyLimits::unbounded()),
            Err(WeightImportError::InvalidPfldIgnoredSet(_))
        ));

        let extra = vec![
            fact("localization.0.weight", DType::F32, 1),
            fact("localization.0.bias", DType::F32, 1),
            fact("localization.3.weight", DType::F32, 1),
            fact("localization.3.bias", DType::F32, 1),
            fact("localization.6.weight", DType::F32, 1),
        ];
        assert!(matches!(
            audit_tensor_facts(extra, None, SafetyLimits::unbounded()),
            Err(WeightImportError::InvalidPfldIgnoredSet(_))
        ));
    }

    #[test]
    fn global_limits_include_ignored_backbone_and_auxiliary_tensors() {
        let backbone = LOCALIZATION_KEYS
            .into_iter()
            .map(|key| fact(key, DType::F32, 1))
            .collect::<Vec<_>>();
        let auxiliary = vec![fact("weight", DType::F32, 1)];
        assert!(matches!(
            audit_tensor_facts(
                backbone.clone(),
                Some(auxiliary.clone()),
                SafetyLimits {
                    max_tensor_count: 4,
                    max_total_elements: 10,
                }
            ),
            Err(WeightImportError::UnsafeLimit(_))
        ));
        assert!(matches!(
            audit_tensor_facts(
                backbone,
                Some(auxiliary),
                SafetyLimits {
                    max_tensor_count: 10,
                    max_total_elements: 4,
                }
            ),
            Err(WeightImportError::UnsafeLimit(_))
        ));
    }

    #[test]
    fn only_non_ignored_float32_tensors_can_be_applied() {
        let mut backbone = LOCALIZATION_KEYS
            .into_iter()
            .map(|key| fact(key, DType::F32, 1))
            .collect::<Vec<_>>();
        backbone.push(fact("conv8.0.weight", DType::F64, 64));
        assert!(matches!(
            audit_tensor_facts(backbone, None, SafetyLimits::unbounded()),
            Err(WeightImportError::DTypeMismatch(path)) if path == "conv8.weight"
        ));
    }
}
