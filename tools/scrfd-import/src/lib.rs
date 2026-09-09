mod artifact;
#[cfg(scrfd_generated)]
pub mod convert;
mod onnx;

use std::{
    fs::File,
    io::Read,
    panic::{AssertUnwindSafe, catch_unwind},
    path::{Path, PathBuf},
};

use feathertalk_scrfd::{SCRFD_SOURCE_ONNX_BYTES, SCRFD_SOURCE_ONNX_SHA256};
use sha2::{Digest, Sha256};

pub use artifact::{
    compare_snapshots, ensure_destination_absent, snapshot_map, validate_apply_result,
};

pub const SOURCE_RELATIVE_PATH: &str = "data_utils/scrfd_2.5g_kps.onnx";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OnnxContract {
    pub opset: u64,
    pub input_name: String,
    pub input_elem_type: i32,
    pub input_shape: Vec<usize>,
    pub output_names: [String; 9],
    pub output_shapes: Vec<Vec<usize>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GeneratedBurnFiles {
    pub source: PathBuf,
    pub burnpack: PathBuf,
}

#[derive(Debug, thiserror::Error)]
pub enum ToolError {
    #[error("I/O error during {operation} at {}: {source}", path.display())]
    Io {
        operation: &'static str,
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("source contract error: {0}")]
    SourceContract(String),
    #[error("ONNX protobuf decode error: {0}")]
    OnnxDecode(String),
    #[error("destination already exists: {}", .0.display())]
    DestinationExists(PathBuf),
    #[error("path is not valid UTF-8: {}", .0.display())]
    NonUtf8Path(PathBuf),
    #[error("Burn ONNX generation failed: {0}")]
    Generation(String),
    #[error("Burn store error: {0}")]
    Store(String),
    #[error("snapshot comparison failed: {0}")]
    Snapshot(String),
    #[error("manifest error: {0}")]
    Manifest(String),
    #[error("converter process failed with status {status:?}: {stderr}")]
    ConversionProcess { status: Option<i32>, stderr: String },
    #[error("generated tree differs at {}", .0.display())]
    TreeMismatch(PathBuf),
}

pub fn inspect_source(repo_root: &Path) -> Result<OnnxContract, ToolError> {
    let path = repo_root.join(SOURCE_RELATIVE_PATH);
    let mut file =
        File::open(&path).map_err(|source| io_error("open source ONNX", &path, source))?;
    let before = file
        .metadata()
        .map_err(|source| io_error("read source ONNX metadata", &path, source))?
        .len();
    if before != SCRFD_SOURCE_ONNX_BYTES {
        return Err(ToolError::SourceContract(format!(
            "expected {SCRFD_SOURCE_ONNX_BYTES} source bytes, got {before}"
        )));
    }
    let capacity = usize::try_from(before)
        .map_err(|_| ToolError::SourceContract("source byte count exceeds usize".to_owned()))?;
    let mut bytes = Vec::with_capacity(capacity);
    file.read_to_end(&mut bytes)
        .map_err(|source| io_error("read source ONNX", &path, source))?;
    let after = file
        .metadata()
        .map_err(|source| io_error("re-read source ONNX metadata", &path, source))?
        .len();
    if before != after || bytes.len() as u64 != before {
        return Err(ToolError::SourceContract(
            "source ONNX changed while being read".to_owned(),
        ));
    }
    let actual_hash = hex::encode(Sha256::digest(&bytes));
    if actual_hash != SCRFD_SOURCE_ONNX_SHA256 {
        return Err(ToolError::SourceContract(format!(
            "expected SHA-256 {SCRFD_SOURCE_ONNX_SHA256}, got {actual_hash}"
        )));
    }
    onnx::parse_contract(&bytes)
}

pub fn generate_burn_files(
    repo_root: &Path,
    destination: &Path,
) -> Result<GeneratedBurnFiles, ToolError> {
    inspect_source(repo_root)?;
    let destination = absolute_path(destination)?;
    if std::fs::symlink_metadata(&destination).is_ok() {
        return Err(ToolError::DestinationExists(destination));
    }
    let parent = destination.parent().ok_or_else(|| {
        ToolError::Generation(format!(
            "destination has no parent: {}",
            destination.display()
        ))
    })?;
    std::fs::create_dir_all(parent)
        .map_err(|source| io_error("create destination parent", parent, source))?;
    let staging_root = tempfile::Builder::new()
        .prefix("scrfd-burn-")
        .tempdir_in(parent)
        .map_err(|source| io_error("create staging directory", parent, source))?;
    let staged_output = staging_root.path().join("generated");
    let staged_utf8 = staged_output
        .to_str()
        .ok_or_else(|| ToolError::NonUtf8Path(staged_output.clone()))?;

    let _guard = CurrentDirGuard::change_to(repo_root)?;
    let result = catch_unwind(AssertUnwindSafe(|| {
        let mut generator = burn_onnx::ModelGen::new();
        generator
            .input(SOURCE_RELATIVE_PATH)
            .out_dir(staged_utf8)
            .development(false)
            .simplify(true)
            .partition(true)
            .load_strategy(burn_onnx::LoadStrategy::None);
        generator.run_from_cli();
    }));
    if let Err(payload) = result {
        return Err(ToolError::Generation(panic_message(payload)));
    }
    drop(_guard);

    require_names(&staged_output, &["scrfd_2.bpk", "scrfd_2.rs"])?;
    let old_source = staged_output.join("scrfd_2.rs");
    let normalized_source = staged_output.join("scrfd_2_5g.rs");
    std::fs::rename(&old_source, &normalized_source)
        .map_err(|source| io_error("normalize generated source name", &old_source, source))?;
    let old_burnpack = staged_output.join("scrfd_2.bpk");
    let normalized_burnpack = staged_output.join("scrfd_2.5g_kps.bpk");
    std::fs::rename(&old_burnpack, &normalized_burnpack)
        .map_err(|source| io_error("normalize generated burnpack name", &old_burnpack, source))?;
    require_names(&staged_output, &["scrfd_2.5g_kps.bpk", "scrfd_2_5g.rs"])?;
    std::fs::rename(&staged_output, &destination)
        .map_err(|source| io_error("publish generated files", &destination, source))?;
    drop(staging_root);

    Ok(GeneratedBurnFiles {
        source: destination.join("scrfd_2_5g.rs"),
        burnpack: destination.join("scrfd_2.5g_kps.bpk"),
    })
}

fn absolute_path(path: &Path) -> Result<PathBuf, ToolError> {
    if path.is_absolute() {
        Ok(path.to_path_buf())
    } else {
        std::env::current_dir()
            .map(|current| current.join(path))
            .map_err(|source| io_error("read current directory", path, source))
    }
}

fn require_names(directory: &Path, expected: &[&str]) -> Result<(), ToolError> {
    let entries = std::fs::read_dir(directory)
        .map_err(|source| io_error("read generated directory", directory, source))?;
    let mut actual = entries
        .map(|entry| {
            entry
                .map(|entry| entry.file_name().to_string_lossy().into_owned())
                .map_err(|source| io_error("read generated entry", directory, source))
        })
        .collect::<Result<Vec<_>, _>>()?;
    actual.sort();
    let mut expected = expected
        .iter()
        .map(|name| (*name).to_owned())
        .collect::<Vec<_>>();
    expected.sort();
    if actual != expected {
        return Err(ToolError::Generation(format!(
            "expected generated files {expected:?}, got {actual:?}"
        )));
    }
    Ok(())
}

fn io_error(operation: &'static str, path: &Path, source: std::io::Error) -> ToolError {
    ToolError::Io {
        operation,
        path: path.to_path_buf(),
        source,
    }
}

fn panic_message(payload: Box<dyn std::any::Any + Send>) -> String {
    if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else if let Some(message) = payload.downcast_ref::<&'static str>() {
        (*message).to_owned()
    } else {
        "unknown panic".to_owned()
    }
}

struct CurrentDirGuard(PathBuf);

impl CurrentDirGuard {
    fn change_to(path: &Path) -> Result<Self, ToolError> {
        let original = std::env::current_dir()
            .map_err(|source| io_error("read current directory", path, source))?;
        std::env::set_current_dir(path)
            .map_err(|source| io_error("change current directory", path, source))?;
        Ok(Self(original))
    }
}

impl Drop for CurrentDirGuard {
    fn drop(&mut self) {
        let _ = std::env::set_current_dir(&self.0);
    }
}
