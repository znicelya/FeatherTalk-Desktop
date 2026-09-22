use std::path::PathBuf;

use burn_store::{ModuleSnapshot, SafetensorsStore};

use crate::WeightImportError;

pub fn save_safetensors<M>(module: &M, path: impl Into<PathBuf>) -> Result<(), WeightImportError>
where
    M: ModuleSnapshot,
{
    let path = path.into();
    let mut store = SafetensorsStore::from_file(&path).overwrite(true);
    module
        .save_into(&mut store)
        .map_err(|error| WeightImportError::Store(error.to_string()))?;
    Ok(())
}
