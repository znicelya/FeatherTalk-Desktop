use std::{
    fs::{self, File, OpenOptions},
    io::{Read, Write},
    path::{Path, PathBuf}};

use sha2::{Digest, Sha256};

use crate::WeightImportError;

pub(crate) const DEFAULT_MAX_FILE_BYTES: u64 = 4 * 1024 * 1024 * 1024;
pub(crate) const DEFAULT_MAX_TENSOR_COUNT: usize = 10_000;
pub(crate) const DEFAULT_MAX_TOTAL_ELEMENTS: u64 = 2_000_000_000;

pub(crate) struct SnapshotFile {
    _directory: tempfile::TempDir,
    _handle: File,
    path: PathBuf,
    sha256: String}

impl SnapshotFile {
    pub(crate) fn copy_from(path: &Path, max_file_bytes: u64) -> Result<Self, WeightImportError> {
        let source_length = fs::metadata(path)?.len();
        if source_length > max_file_bytes {
            return Err(WeightImportError::UnsafeLimit(format!(
                "source file length {source_length} exceeds {max_file_bytes}"
            )));
        }

        let mut source = File::open(path)?;
        let directory = tempfile::Builder::new()
            .prefix("feathertalk-weights-")
            .tempdir()?;
        let snapshot_path = directory.path().join("checkpoint.pth");
        let mut destination = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&snapshot_path)?;
        let mut hasher = Sha256::new();
        let mut copied = 0u64;
        let mut buffer = [0u8; 64 * 1024];

        loop {
            let read = source.read(&mut buffer)?;
            if read == 0 {
                break;
            }
            copied = copied
                .checked_add(u64::try_from(read).expect("buffer length fits u64"))
                .ok_or_else(|| {
                    WeightImportError::UnsafeLimit("source file length overflowed u64".to_owned())
                })?;
            if copied > max_file_bytes {
                return Err(WeightImportError::UnsafeLimit(format!(
                    "source file length exceeds {max_file_bytes} while copying"
                )));
            }
            destination.write_all(&buffer[..read])?;
            hasher.update(&buffer[..read]);
        }
        destination.sync_all()?;

        Ok(Self {
            _directory: directory,
            _handle: destination,
            path: snapshot_path,
            sha256: hex::encode(hasher.finalize())})
    }

    pub(crate) fn path(&self) -> &Path {
        &self.path
    }

    pub(crate) fn sha256(&self) -> &str {
        &self.sha256
    }
}

pub(crate) fn sha256_file(path: &Path) -> Result<String, WeightImportError> {
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 64 * 1024];

    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }

    Ok(hex::encode(hasher.finalize()))
}

/// Convert the pytorch reader dtype enum to burn's. burn 0.22.0-pre.4 gives the
/// reader its own `DType` (no `Flex32`/`QFloat`), so bridge it explicitly.
pub(crate) fn reader_dtype_to_burn(dtype: burn_store::pytorch_reader::DType) -> burn::tensor::DType {
    use burn::tensor::DType as B;
    use burn_store::pytorch_reader::DType as R;
    match dtype {
        R::F64 => B::F64,
        R::F32 => B::F32,
        R::F16 => B::F16,
        R::BF16 => B::BF16,
        R::I64 => B::I64,
        R::I32 => B::I32,
        R::I16 => B::I16,
        R::I8 => B::I8,
        R::U64 => B::U64,
        R::U32 => B::U32,
        R::U16 => B::U16,
        R::U8 => B::U8,
        R::Bool => B::Bool(burn::tensor::BoolStore::Native),
    }
}

/// Element count from raw dimensions, sharing the overflow guards of
/// [`tensor_elements`]. burn 0.22.0-pre.4 split the pytorch reader tensor out of
/// `burn_pack::Tensor`, so reader-side callers pass `Tensor::shape()` here.
pub(crate) fn shape_elements<'a>(dims: impl IntoIterator<Item = &'a usize>) -> Result<u64, WeightImportError> {
    dims.into_iter().try_fold(1u64, |total, dimension| {
        let dimension = u64::try_from(*dimension).map_err(|_| {
            WeightImportError::UnsafeLimit("tensor dimension exceeds u64".to_owned())
        })?;
        total.checked_mul(dimension).ok_or_else(|| {
            WeightImportError::UnsafeLimit("tensor element count overflowed u64".to_owned())
        })
    })
}
pub(crate) fn tensor_elements(snapshot: &burn_store::burn_pack::Tensor) -> Result<u64, WeightImportError> {
    shape_elements(snapshot.shape.iter())
}
