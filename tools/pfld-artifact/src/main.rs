use std::{
    fs::{self, OpenOptions},
    io::Write,
    path::{Path, PathBuf},
};

use burn::backend::NdArray;
use feathertalk_models::{PFLD_GhostOne, PfldConfig};
use feathertalk_pfld::{
    PFLD_EXPECTED_TENSOR_COUNT, PFLD_EXPECTED_TOTAL_ELEMENTS, PfldRuntimeManifest,
};
use feathertalk_weights::{PfldImportRequest, import_pfld_checkpoint};
use sha2::{Digest, Sha256};

type CpuBackend = NdArray<f32>;

const SOURCE_SHA256: &str = "bada866661ad5fa1080a085f51fe9c016c69958c406951afa4afc7840f856de0";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let repository = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("..");
    let checkpoint = repository.join("data_utils/checkpoint_epoch_335.pth.tar");
    let arguments = std::env::args_os().skip(1).collect::<Vec<_>>();
    let verify = arguments.iter().any(|argument| argument == "--verify");
    let destination = arguments
        .iter()
        .find(|argument| argument.as_os_str() != "--verify")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            repository.join("rust/crates/feathertalk-pfld/artifacts/pfld_ghost_one")
        });
    generate(&checkpoint, &destination, verify)
}

fn generate(
    checkpoint: &Path,
    destination: &Path,
    verify: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let checkpoint = fs::canonicalize(checkpoint)?;
    let destination_parent = destination
        .parent()
        .ok_or_else(|| "artifact destination has no parent".to_owned())?;
    fs::create_dir_all(destination_parent)?;
    let destination_parent = fs::canonicalize(destination_parent)?;
    let destination = destination_parent.join(
        destination
            .file_name()
            .ok_or_else(|| "artifact destination has no file name".to_owned())?,
    );
    let source_hash = sha256_file(&checkpoint)?;
    if source_hash != SOURCE_SHA256 {
        return Err(format!("source checkpoint hash changed: {source_hash}").into());
    }
    let parent = destination
        .parent()
        .ok_or_else(|| "artifact destination has no parent".to_owned())?;
    let staging_parent = tempfile::tempdir_in(parent)?;
    let imported_dir = staging_parent.path().join("imported");
    let device = Default::default();
    let mut model = PFLD_GhostOne::<CpuBackend>::new(PfldConfig::production(), &device);
    let report = import_pfld_checkpoint::<CpuBackend, _>(
        &mut model,
        &PfldImportRequest {
            checkpoint: checkpoint.to_owned(),
            destination_dir: imported_dir.clone(),
            ..PfldImportRequest::default()
        },
    )?;
    if report.applied.len() != PFLD_EXPECTED_TENSOR_COUNT
        || report.manifest.model.total_elements != PFLD_EXPECTED_TOTAL_ELEMENTS
    {
        return Err("imported PFLD tensor summary does not match runtime contract".into());
    }
    let imported_model_bytes = fs::read(imported_dir.join("model.safetensors"))
        .map_err(|error| format!("read imported model {}: {error}", imported_dir.display()))?;
    let model_bytes = canonicalize_safetensors(&imported_model_bytes)?;
    let model_hash = hex::encode(Sha256::digest(&model_bytes));
    let manifest = PfldRuntimeManifest::approved(
        "checkpoint_epoch_335.pth.tar".to_owned(),
        source_hash,
        model_hash,
        report.manifest.model.tensor_count,
        report.manifest.model.total_elements,
    );
    manifest.validate()?;

    let staged_dir = tempfile::Builder::new()
        .prefix(".pfld-artifact-")
        .tempdir_in(parent)?;
    let staged = staged_dir.path().to_owned();
    fs::write(staged.join("model.safetensors"), model_bytes)
        .map_err(|error| format!("write staged model {}: {error}", staged.display()))?;
    let mut manifest_bytes = serde_json::to_vec_pretty(&manifest)?;
    manifest_bytes.push(b'\n');
    let mut manifest_file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(staged.join("manifest.json"))
        .map_err(|error| format!("create staged manifest {}: {error}", staged.display()))?;
    manifest_file
        .write_all(&manifest_bytes)
        .map_err(|error| format!("write staged manifest {}: {error}", staged.display()))?;
    manifest_file
        .sync_all()
        .map_err(|error| format!("sync staged manifest {}: {error}", staged.display()))?;

    if destination.exists() {
        let existing_manifest = fs::read(destination.join("manifest.json")).map_err(|error| {
            format!("read existing manifest {}: {error}", destination.display())
        })?;
        let existing_model = fs::read(destination.join("model.safetensors"))
            .map_err(|error| format!("read existing model {}: {error}", destination.display()))?;
        let staged_model = fs::read(staged.join("model.safetensors"))?;
        if existing_manifest == manifest_bytes && existing_model == staged_model {
            return Ok(());
        }
        let existing_manifest_hash = hex::encode(Sha256::digest(&existing_manifest));
        let staged_manifest_hash = hex::encode(Sha256::digest(&manifest_bytes));
        let existing_model_hash = hex::encode(Sha256::digest(&existing_model));
        let staged_model_hash = hex::encode(Sha256::digest(&staged_model));
        return Err(format!(
            "artifact destination exists with different bytes: manifest {existing_manifest_hash} != {staged_manifest_hash}; model {existing_model_hash} != {staged_model_hash}"
        )
        .into());
    }
    if verify {
        return Err("artifact destination is missing during verification".into());
    }
    publish_staged_directory(
        staged_dir,
        &destination,
        |source, destination| fs::rename(source, destination),
        |source, destination| fs::copy(source, destination),
    )?;
    Ok(())
}

fn publish_staged_directory<R, C>(
    staged_dir: tempfile::TempDir,
    destination: &Path,
    rename: R,
    mut copy: C,
) -> Result<(), Box<dyn std::error::Error>>
where
    R: FnOnce(&Path, &Path) -> std::io::Result<()>,
    C: FnMut(&Path, &Path) -> std::io::Result<u64>,
{
    let staging_path = staged_dir.path();
    match rename(staging_path, destination) {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::PermissionDenied => {
            fs::create_dir(destination).map_err(|create_error| {
                format!(
                    "publish fallback create {} after rename error {error}: {create_error}",
                    destination.display()
                )
            })?;
            for name in ["manifest.json", "model.safetensors"] {
                if let Err(copy_error) = copy(&staging_path.join(name), &destination.join(name)) {
                    let cleanup_error = fs::remove_dir_all(destination).err();
                    let cleanup_message = cleanup_error
                        .map(|error| format!("; cleanup failed: {error}"))
                        .unwrap_or_default();
                    return Err(format!(
                        "publish fallback copy {name}: {copy_error}{cleanup_message}"
                    )
                    .into());
                }
            }
        }
        Err(error) => {
            return Err(format!(
                "publish {} -> {}: {error}",
                staging_path.display(),
                destination.display()
            )
            .into());
        }
    }
    Ok(())
}

fn sha256_file(path: &Path) -> Result<String, std::io::Error> {
    let bytes = fs::read(path)?;
    Ok(hex::encode(Sha256::digest(bytes)))
}

fn canonicalize_safetensors(bytes: &[u8]) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let parsed = safetensors::SafeTensors::deserialize(bytes)?;
    let mut tensors = parsed.tensors().into_iter().collect::<Vec<_>>();
    tensors.sort_by(|left, right| left.0.cmp(&right.0));
    Ok(safetensors::serialize(tensors, None)?)
}

#[allow(dead_code)]
fn _path_buf(path: &Path) -> PathBuf {
    path.to_owned()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io;

    #[test]
    fn permission_denied_publish_cleans_up_staging_directory() {
        let parent = tempfile::tempdir().unwrap();
        let staged = tempfile::Builder::new()
            .prefix(".pfld-artifact-")
            .tempdir_in(parent.path())
            .unwrap();
        let staged_path = staged.path().to_owned();
        fs::write(staged.path().join("manifest.json"), b"manifest").unwrap();
        fs::write(staged.path().join("model.safetensors"), b"model").unwrap();
        let destination = parent.path().join("published");

        publish_staged_directory(
            staged,
            &destination,
            |_source, _destination| {
                Err(io::Error::new(
                    io::ErrorKind::PermissionDenied,
                    "simulated cross-volume rename failure",
                ))
            },
            |source, destination| fs::copy(source, destination),
        )
        .unwrap();

        assert!(!staged_path.exists());
        assert_eq!(
            fs::read(destination.join("manifest.json")).unwrap(),
            b"manifest"
        );
        assert_eq!(
            fs::read(destination.join("model.safetensors")).unwrap(),
            b"model"
        );
    }

    #[test]
    fn failed_fallback_copy_removes_partial_destination_and_staging() {
        let parent = tempfile::tempdir().unwrap();
        let staged = tempfile::Builder::new()
            .prefix(".pfld-artifact-")
            .tempdir_in(parent.path())
            .unwrap();
        let staged_path = staged.path().to_owned();
        fs::write(staged.path().join("manifest.json"), b"manifest").unwrap();
        fs::write(staged.path().join("model.safetensors"), b"model").unwrap();
        let destination = parent.path().join("published");

        let result = publish_staged_directory(
            staged,
            &destination,
            |_source, _destination| {
                Err(io::Error::new(
                    io::ErrorKind::PermissionDenied,
                    "simulated cross-volume rename failure",
                ))
            },
            |source: &Path, destination: &Path| {
                if source.file_name().and_then(|name| name.to_str()) == Some("model.safetensors") {
                    return Err(io::Error::other("simulated model copy failure"));
                }
                fs::copy(source, destination)
            },
        );

        assert!(result.is_err());
        assert!(!staged_path.exists());
        assert!(!destination.exists());
    }
}
