#[cfg(scrfd_generated)]
mod generated {
    include!(concat!(env!("OUT_DIR"), "/scrfd_generated.rs"));
}

#[cfg(scrfd_generated)]
pub fn convert_burnpack<B: burn::tensor::backend::Backend>(
    burnpack: &std::path::Path,
    safetensors: &std::path::Path,
) -> Result<(), crate::ToolError> {
    use burn_store::{BurnpackStore, ModuleSnapshot, SafetensorsStore};
    use std::io::Write;

    let device = Default::default();
    let mut model = generated::Model::<B>::new(&device);
    let mut burnpack_store = BurnpackStore::from_file(burnpack);
    let result = model
        .load_from(&mut burnpack_store)
        .map_err(|error| crate::ToolError::Store(error.to_string()))?;
    crate::validate_apply_result(&result)?;

    let mut save_store = SafetensorsStore::from_file(safetensors).overwrite(false);
    model
        .save_into(&mut save_store)
        .map_err(|error| crate::ToolError::Store(error.to_string()))?;

    let bytes = std::fs::read(safetensors).map_err(|source| crate::ToolError::Io {
        operation: "read generated safetensors",
        path: safetensors.to_owned(),
        source,
    })?;
    let bytes = crate::artifact::canonicalize_safetensors_bytes(bytes)?;
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .truncate(true)
        .open(safetensors)
        .map_err(|source| crate::ToolError::Io {
            operation: "open generated safetensors for canonicalization",
            path: safetensors.to_owned(),
            source,
        })?;
    file.write_all(&bytes)
        .map_err(|source| crate::ToolError::Io {
            operation: "write canonical safetensors",
            path: safetensors.to_owned(),
            source,
        })?;
    file.sync_all().map_err(|source| crate::ToolError::Io {
        operation: "sync canonical safetensors",
        path: safetensors.to_owned(),
        source,
    })?;
    let mut reloaded = generated::Model::<B>::new(&device);
    let mut memory_store = SafetensorsStore::from_bytes(Some(bytes))
        .allow_partial(true)
        .validate(false);
    let reload_result = reloaded
        .load_from(&mut memory_store)
        .map_err(|error| crate::ToolError::Store(error.to_string()))?;
    crate::validate_apply_result(&reload_result)?;
    crate::compare_snapshots::<B, _>(&model, &reloaded)
}
