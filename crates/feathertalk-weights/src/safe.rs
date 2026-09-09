use std::path::PathBuf;

use burn::tensor::backend::Backend;
use burn_store::{ModuleSnapshot, SafetensorsStore};

use crate::WeightImportError;

pub fn save_safetensors<B, M>(module: &M, path: impl Into<PathBuf>) -> Result<(), WeightImportError>
where
    B: Backend,
    M: ModuleSnapshot<B>,
{
    let path = path.into();
    let mut store = SafetensorsStore::from_file(&path).overwrite(true);
    module
        .save_into(&mut store)
        .map_err(|error| WeightImportError::Store(error.to_string()))?;
    Ok(())
}
