use std::collections::BTreeSet;

use burn_store::KeyRemapper;

use crate::WeightImportError;

pub(super) const LOCALIZATION_KEYS: [&str; 4] = [
    "localization.0.bias",
    "localization.0.weight",
    "localization.3.bias",
    "localization.3.weight",
];

pub(super) fn pfld_remapper() -> KeyRemapper {
    KeyRemapper::new()
        .add_pattern(r"(^|\.)rbr_conv\.([0-9]+)\.conv\.", "${1}branches.${2}.0.")
        .expect("reviewed literal regex")
        .add_pattern(r"(^|\.)rbr_conv\.([0-9]+)\.bn\.", "${1}branches.${2}.1.")
        .expect("reviewed literal regex")
        .add_pattern(r"(^|\.)rbr_scale\.conv\.", "${1}scale.0.")
        .expect("reviewed literal regex")
        .add_pattern(r"(^|\.)rbr_scale\.bn\.", "${1}scale.1.")
        .expect("reviewed literal regex")
        .add_pattern(r"(^|\.)rbr_skip\.", "${1}skip.")
        .expect("reviewed literal regex")
        .add_pattern(r"(^|\.)ghost_conv\.0\.", "${1}ghost.")
        .expect("reviewed literal regex")
        .add_pattern(r"(^|\.)ghost_conv\.1\.", "${1}depthwise.")
        .expect("reviewed literal regex")
        .add_pattern(r"(^|\.)ghost_conv\.2\.", "${1}linear.")
        .expect("reviewed literal regex")
        .add_pattern(r"(^|\.)(ghost|linear)\.primary_conv\.", "${1}${2}.primary.")
        .expect("reviewed literal regex")
        .add_pattern(
            r"(^|\.)(ghost|linear)\.cheap_operation\.",
            "${1}${2}.cheap.",
        )
        .expect("reviewed literal regex")
        .add_pattern(r"^conv8\.0\.", "conv8.")
        .expect("reviewed literal regex")
        .add_pattern(r"^conv_out\.", "head.")
        .expect("reviewed literal regex")
        .add_pattern(
            r"^(.+\.(?:branches\.[0-9]+\.1|scale\.1|skip))\.weight$",
            "${1}.gamma",
        )
        .expect("reviewed literal regex")
        .add_pattern(
            r"^(.+\.(?:branches\.[0-9]+\.1|scale\.1|skip))\.bias$",
            "${1}.beta",
        )
        .expect("reviewed literal regex")
}

pub(super) fn map_pfld_key(key: &str) -> String {
    let mut mapped = key.to_owned();
    for (pattern, replacement) in pfld_remapper().patterns {
        mapped = pattern
            .replace_all(&mapped, replacement.as_str())
            .into_owned();
    }
    mapped
}

pub(super) fn reject_duplicate_destinations(
    source_keys: impl IntoIterator<Item = String>,
) -> Result<(), WeightImportError> {
    let mut destinations = BTreeSet::new();
    for source_key in source_keys {
        let destination = map_pfld_key(&source_key);
        if !destinations.insert(destination.clone()) {
            return Err(WeightImportError::DuplicateKey(destination));
        }
    }
    Ok(())
}

pub(super) fn is_valid_batch_norm_counter(
    source_key: &str,
    source_keys: &BTreeSet<String>,
) -> bool {
    let Some(source_parent) = source_key.strip_suffix(".num_batches_tracked") else {
        return false;
    };
    if !is_reviewed_source_batch_norm_parent(source_parent) {
        return false;
    }
    let running_mean = format!("{source_parent}.running_mean");
    let running_var = format!("{source_parent}.running_var");
    if !source_keys.contains(&running_mean) || !source_keys.contains(&running_var) {
        return false;
    }
    let mapped_mean = map_pfld_key(&running_mean);
    let mapped_var = map_pfld_key(&running_var);
    let Some(mapped_parent) = mapped_mean.strip_suffix(".running_mean") else {
        return false;
    };
    is_mapped_batch_norm_parent(mapped_parent)
        && mapped_var == format!("{mapped_parent}.running_var")
}

fn is_reviewed_source_batch_norm_parent(parent: &str) -> bool {
    let segments = parent.split('.').collect::<Vec<_>>();
    match segments.as_slice() {
        [.., "rbr_conv", index, "bn"] => is_decimal_index(index),
        [.., "rbr_scale", "bn"] | [.., "rbr_skip"] => true,
        _ => false,
    }
}

fn is_mapped_batch_norm_parent(parent: &str) -> bool {
    let segments = parent.split('.').collect::<Vec<_>>();
    match segments.as_slice() {
        [.., "branches", index, "1"] => is_decimal_index(index),
        [.., "scale", "1"] | [.., "skip"] => true,
        _ => false,
    }
}

fn is_decimal_index(segment: &str) -> bool {
    !segment.is_empty() && segment.bytes().all(|byte| byte.is_ascii_digit())
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use crate::WeightImportError;

    use super::*;

    #[test]
    fn reviewed_pfld_paths_map_exactly() {
        let cases = [
            ("conv1.rbr_conv.0.conv.weight", "conv1.branches.0.0.weight"),
            ("conv1.rbr_conv.0.bn.weight", "conv1.branches.0.1.gamma"),
            (
                "conv3_1.ghost_conv.0.primary_conv.rbr_conv.0.bn.weight",
                "conv3_1.ghost.primary.branches.0.1.gamma",
            ),
            (
                "conv3_1.ghost_conv.1.rbr_conv.0.conv.weight",
                "conv3_1.depthwise.branches.0.0.weight",
            ),
            (
                "conv3_1.ghost_conv.2.cheap_operation.rbr_scale.bn.bias",
                "conv3_1.linear.cheap.scale.1.beta",
            ),
            ("conv8.0.weight", "conv8.weight"),
            ("conv_out.bias", "head.bias"),
        ];

        for (source, expected) in cases {
            assert_eq!(map_pfld_key(source), expected);
        }
    }

    #[test]
    fn unapproved_near_matches_are_unchanged() {
        for key in [
            "prefix.conv8.0.weight",
            "conv80.weight",
            "conv1.rbr_conv.named.conv.weight",
            "conv1.rbr_scale.extra.bn.weight",
            "conv3_1.ghost_conv.3.primary_conv.weight",
            "conv3_1.primary_convolution.weight",
            "conv1.rbr_skip_extra.weight",
        ] {
            assert_eq!(map_pfld_key(key), key);
        }
    }

    #[test]
    fn colliding_pfld_destinations_are_rejected() {
        let error =
            reject_duplicate_destinations(["conv8.0.weight".to_owned(), "conv8.weight".to_owned()])
                .unwrap_err();
        assert!(matches!(
            error,
            WeightImportError::DuplicateKey(key) if key == "conv8.weight"
        ));
    }

    #[test]
    fn batch_norm_counter_requires_reviewed_sibling_buffers() {
        let keys = [
            "conv1.rbr_conv.0.bn.running_mean",
            "conv1.rbr_conv.0.bn.running_var",
            "conv1.rbr_conv.0.bn.num_batches_tracked",
        ]
        .into_iter()
        .map(str::to_owned)
        .collect::<BTreeSet<_>>();
        assert!(is_valid_batch_norm_counter(
            "conv1.rbr_conv.0.bn.num_batches_tracked",
            &keys
        ));

        let missing_var = [
            "conv1.rbr_conv.0.bn.running_mean",
            "conv1.rbr_conv.0.bn.num_batches_tracked",
        ]
        .into_iter()
        .map(str::to_owned)
        .collect::<BTreeSet<_>>();
        assert!(!is_valid_batch_norm_counter(
            "conv1.rbr_conv.0.bn.num_batches_tracked",
            &missing_var
        ));

        let arbitrary = [
            "unknown.running_mean",
            "unknown.running_var",
            "unknown.num_batches_tracked",
        ]
        .into_iter()
        .map(str::to_owned)
        .collect::<BTreeSet<_>>();
        assert!(!is_valid_batch_norm_counter(
            "unknown.num_batches_tracked",
            &arbitrary
        ));

        let burn_looking_counterfeit = [
            "unknown.branches.0.1.running_mean",
            "unknown.branches.0.1.running_var",
            "unknown.branches.0.1.num_batches_tracked",
        ]
        .into_iter()
        .map(str::to_owned)
        .collect::<BTreeSet<_>>();
        assert!(!is_valid_batch_norm_counter(
            "unknown.branches.0.1.num_batches_tracked",
            &burn_looking_counterfeit
        ));
    }
}
