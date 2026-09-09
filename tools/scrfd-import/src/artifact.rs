use std::{collections::BTreeMap, path::Path};

use crate::ToolError;

#[cfg(any(scrfd_generated, test))]
pub(crate) fn canonicalize_safetensors_bytes(bytes: Vec<u8>) -> Result<Vec<u8>, ToolError> {
    const HEADER_LENGTH_BYTES: usize = 8;

    let header_length = bytes
        .get(..HEADER_LENGTH_BYTES)
        .ok_or_else(|| ToolError::Store("safetensors header is shorter than 8 bytes".to_owned()))?;
    let header_length = u64::from_le_bytes(
        header_length
            .try_into()
            .expect("the slice length is checked above"),
    );
    let header_length = usize::try_from(header_length)
        .map_err(|_| ToolError::Store("safetensors header length exceeds usize".to_owned()))?;
    let data_start = HEADER_LENGTH_BYTES
        .checked_add(header_length)
        .filter(|end| *end <= bytes.len())
        .ok_or_else(|| ToolError::Store("safetensors header length is invalid".to_owned()))?;

    let mut header: serde_json::Value =
        serde_json::from_slice(&bytes[HEADER_LENGTH_BYTES..data_start]).map_err(|error| {
            ToolError::Store(format!("invalid safetensors header JSON: {error}"))
        })?;
    if !header.is_object() {
        return Err(ToolError::Store(
            "safetensors header must be a JSON object".to_owned(),
        ));
    }
    sort_json_keys(&mut header);
    let mut canonical_header = serde_json::to_vec(&header)
        .map_err(|error| ToolError::Store(format!("serialize safetensors header: {error}")))?;
    canonical_header.resize(canonical_header.len().next_multiple_of(8), b' ');
    let canonical_length = u64::try_from(canonical_header.len())
        .map_err(|_| ToolError::Store("canonical safetensors header is too large".to_owned()))?;

    let mut canonical = Vec::with_capacity(
        HEADER_LENGTH_BYTES + canonical_header.len() + bytes.len().saturating_sub(data_start),
    );
    canonical.extend(canonical_length.to_le_bytes());
    canonical.extend(canonical_header);
    canonical.extend_from_slice(&bytes[data_start..]);
    Ok(canonical)
}

#[cfg(any(scrfd_generated, test))]
fn sort_json_keys(value: &mut serde_json::Value) {
    match value {
        serde_json::Value::Array(values) => {
            for value in values {
                sort_json_keys(value);
            }
        }
        serde_json::Value::Object(values) => {
            for value in values.values_mut() {
                sort_json_keys(value);
            }
            values.sort_keys();
        }
        _ => {}
    }
}

pub fn ensure_destination_absent(path: &Path) -> Result<(), ToolError> {
    match std::fs::symlink_metadata(path) {
        Ok(_) => Err(ToolError::DestinationExists(path.to_owned())),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(source) => Err(ToolError::Io {
            operation: "inspect destination",
            path: path.to_owned(),
            source,
        }),
    }
}

pub fn validate_apply_result(result: &burn_store::ApplyResult) -> Result<(), ToolError> {
    if let Some(path) = result.missing.iter().map(|(path, _)| path).min() {
        return Err(ToolError::Store(format!("missing tensor: {path}")));
    }
    if let Some(error) = result.errors.first() {
        return Err(ToolError::Store(error.to_string()));
    }
    if let Some(path) = result.skipped.iter().min() {
        return Err(ToolError::Store(format!("skipped tensor: {path}")));
    }
    if let Some(path) = result.unused.iter().min() {
        return Err(ToolError::Store(format!("unexpected tensor: {path}")));
    }
    Ok(())
}

pub fn snapshot_map<B, M>(
    module: &M,
) -> Result<BTreeMap<String, burn_store::TensorSnapshot>, ToolError>
where
    B: burn::tensor::backend::Backend,
    M: burn_store::ModuleSnapshot<B>,
{
    let mut snapshots = BTreeMap::new();
    for snapshot in module.collect(None, None, false) {
        let path = snapshot.full_path();
        if snapshots.insert(path.clone(), snapshot).is_some() {
            return Err(ToolError::Snapshot(format!(
                "duplicate tensor path: {path}"
            )));
        }
    }
    Ok(snapshots)
}

pub fn compare_snapshots<B, M>(expected: &M, actual: &M) -> Result<(), ToolError>
where
    B: burn::tensor::backend::Backend,
    M: burn_store::ModuleSnapshot<B>,
{
    let expected = snapshot_map::<B, M>(expected)?;
    let actual = snapshot_map::<B, M>(actual)?;
    let expected_keys = expected.keys().collect::<Vec<_>>();
    let actual_keys = actual.keys().collect::<Vec<_>>();
    if expected_keys != actual_keys {
        return Err(ToolError::Snapshot(format!(
            "tensor keys differ: expected {expected_keys:?}, got {actual_keys:?}"
        )));
    }
    for key in expected_keys {
        let left = &expected[key];
        let right = &actual[key];
        if left.shape != right.shape {
            return Err(ToolError::Snapshot(format!(
                "shape differs for {key}: {:?} vs {:?}",
                left.shape, right.shape
            )));
        }
        if left.dtype != right.dtype {
            return Err(ToolError::Snapshot(format!(
                "dtype differs for {key}: {:?} vs {:?}",
                left.dtype, right.dtype
            )));
        }
        let left_data = left
            .to_data()
            .map_err(|error| ToolError::Snapshot(format!("{key}: {error}")))?;
        let right_data = right
            .to_data()
            .map_err(|error| ToolError::Snapshot(format!("{key}: {error}")))?;
        if left_data != right_data {
            return Err(ToolError::Snapshot(format!(
                "tensor data differs for {key}"
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{DType, Shape};
    use burn_store::{ApplyError, ApplyResult};

    #[test]
    fn strict_apply_rejects_every_non_applied_entry() {
        let empty = || ApplyResult {
            applied: Vec::new(),
            skipped: Vec::new(),
            missing: Vec::new(),
            unused: Vec::new(),
            errors: Vec::new(),
        };
        let mut missing = empty();
        missing
            .missing
            .push(("conv.weight".to_owned(), "Struct:Model".to_owned()));
        let mut unused = empty();
        unused.unused.push("extra.weight".to_owned());
        let mut skipped = empty();
        skipped.skipped.push("head.bias".to_owned());
        let mut shape = empty();
        shape.errors.push(ApplyError::ShapeMismatch {
            path: "neck.weight".to_owned(),
            expected: Shape::new([1, 2]),
            found: Shape::new([2, 1]),
        });
        let mut dtype = empty();
        dtype.errors.push(ApplyError::DTypeMismatch {
            path: "score.bias".to_owned(),
            expected: DType::F32,
            found: DType::I32,
        });
        let mut adapter = empty();
        adapter.errors.push(ApplyError::AdapterError {
            path: "neck.weight".to_owned(),
            message: "adapter failed".to_owned(),
        });
        let mut load = empty();
        load.errors.push(ApplyError::LoadError {
            path: "head.weight".to_owned(),
            message: "load failed".to_owned(),
        });

        for result in [missing, unused, skipped, shape, dtype, adapter, load] {
            assert!(validate_apply_result(&result).is_err());
        }
    }

    #[test]
    fn publishing_rejects_an_existing_file_or_directory() {
        let temp = tempfile::tempdir().unwrap();
        for name in ["file", "directory"] {
            let path = temp.path().join(name);
            if name == "file" {
                std::fs::write(&path, b"occupied").unwrap();
            } else {
                std::fs::create_dir(&path).unwrap();
            }
            assert!(ensure_destination_absent(&path).is_err());
        }
    }

    #[test]
    fn safetensors_header_canonicalization_removes_map_iteration_order() {
        fn fixture(header: &str) -> Vec<u8> {
            let mut header = header.as_bytes().to_vec();
            header.resize(header.len().next_multiple_of(8), b' ');
            let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
            bytes.extend(header);
            bytes.extend(1.25_f32.to_le_bytes());
            bytes
        }

        let first = fixture(
            r#"{"__metadata__":{"format":"safetensors","version":"0.21.0","producer":"burn"},"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#,
        );
        let second = fixture(
            r#"{"__metadata__":{"producer":"burn","format":"safetensors","version":"0.21.0"},"x":{"data_offsets":[0,4],"shape":[1],"dtype":"F32"}}"#,
        );

        let first = canonicalize_safetensors_bytes(first).unwrap();
        let second = canonicalize_safetensors_bytes(second).unwrap();
        assert_eq!(first, second);
        assert_eq!(&first[first.len() - 4..], &1.25_f32.to_le_bytes());
    }
}
