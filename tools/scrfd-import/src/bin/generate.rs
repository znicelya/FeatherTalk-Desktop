use std::{
    fs::{self, OpenOptions},
    io::Write,
    path::{Path, PathBuf},
    process::Command,
};

use clap::Parser;
use feathertalk_scrfd::{
    ScrfdArtifactManifest, ScrfdFileManifest, ScrfdGeneratorManifest, ScrfdInputManifest,
    ScrfdLevelManifest, ScrfdLicenseManifest, ScrfdOutputManifest, ScrfdSourceManifest,
    ScrfdWeightManifest,
};
use feathertalk_scrfd_import::{
    GeneratedBurnFiles, ToolError, ensure_destination_absent, generate_burn_files, inspect_source,
};
use sha2::{Digest, Sha256};

#[derive(Debug, Parser)]
struct Args {
    #[arg(long)]
    repo_root: PathBuf,
    #[arg(
        long,
        conflicts_with = "verify_against",
        required_unless_present = "verify_against"
    )]
    destination: Option<PathBuf>,
    #[arg(
        long,
        conflicts_with = "destination",
        required_unless_present = "destination"
    )]
    verify_against: Option<PathBuf>,
}

fn main() {
    if let Err(error) = run(Args::parse()) {
        eprintln!("SCRFD generation failed: {error}");
        std::process::exit(1);
    }
}

fn run(args: Args) -> Result<(), ToolError> {
    let repo_root = absolute(&args.repo_root)?;
    inspect_source(&repo_root)?;
    let destination = args.destination.map(|path| absolute(&path)).transpose()?;
    let verify_against = args
        .verify_against
        .map(|path| absolute(&path))
        .transpose()?;
    if let Some(path) = &destination {
        ensure_destination_absent(path)?;
    }

    let temp_parent = destination
        .as_ref()
        .and_then(|path| path.parent())
        .map(Path::to_path_buf)
        .unwrap_or_else(std::env::temp_dir);
    fs::create_dir_all(&temp_parent)
        .map_err(|source| io_error("create generation parent", &temp_parent, source))?;
    let temp = tempfile::Builder::new()
        .prefix("scrfd-final-")
        .tempdir_in(&temp_parent)
        .map_err(|source| io_error("create generation staging", &temp_parent, source))?;
    let raw_dir = temp.path().join("raw");
    let generated = generate_burn_files(&repo_root, &raw_dir)?;
    let tree = temp.path().join("tree");
    let generated_source = tree.join("src/generated/scrfd_2_5g.rs");
    let artifact_contract = tree.join("src/generated/artifact_contract.rs");
    let staged_safetensors = tree.join("artifacts/scrfd_2_5g/model.safetensors");
    let manifest_path = tree.join("artifacts/scrfd_2_5g/manifest.json");
    for path in [generated_source.parent(), staged_safetensors.parent()] {
        let path = path.expect("generated paths have parents");
        fs::create_dir_all(path)
            .map_err(|source| io_error("create generated tree", path, source))?;
    }
    copy_create_new(&generated.source, &generated_source)?;
    run_converter(&generated, &staged_safetensors, temp.path())?;

    let source_bytes = file_bytes(&generated_source)?;
    let weight_bytes = file_bytes(&staged_safetensors)?;
    let source_hash = sha256(&source_bytes);
    let weight_hash = sha256(&weight_bytes);
    write_constants(
        &artifact_contract,
        source_bytes.len() as u64,
        &source_hash,
        weight_bytes.len() as u64,
        &weight_hash,
    )?;
    let manifest = build_manifest(
        source_bytes.len() as u64,
        &source_hash,
        weight_bytes.len() as u64,
        &weight_hash,
    );
    manifest
        .validate()
        .map_err(|error| ToolError::Manifest(error.to_string()))?;
    let mut manifest_bytes = serde_json::to_vec_pretty(&manifest)
        .map_err(|error| ToolError::Manifest(error.to_string()))?;
    manifest_bytes.push(b'\n');
    write_create_new(&manifest_path, &manifest_bytes)?;
    require_tree(&tree)?;

    if let Some(destination) = destination {
        fs::rename(&tree, &destination)
            .map_err(|source| io_error("publish generated artifact tree", &destination, source))?;
    } else {
        let committed = verify_against.expect("one mode is required");
        verify_tree(&tree, &committed)?;
    }
    Ok(())
}

fn run_converter(
    generated: &GeneratedBurnFiles,
    safetensors: &Path,
    temp_root: &Path,
) -> Result<(), ToolError> {
    let cargo = std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into());
    let output = Command::new(cargo)
        .arg("run")
        .arg("--manifest-path")
        .arg(Path::new(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml"))
        .arg("--bin")
        .arg("convert")
        .arg("--")
        .arg("--burnpack")
        .arg(&generated.burnpack)
        .arg("--safetensors")
        .arg(safetensors)
        .env("SCRFD_GENERATED_RS", &generated.source)
        .env("CARGO_TARGET_DIR", temp_root.join("converter-target"))
        .output()
        .map_err(|source| io_error("spawn SCRFD converter", Path::new("cargo"), source))?;
    if !output.status.success() {
        return Err(ToolError::ConversionProcess {
            status: output.status.code(),
            stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
        });
    }
    Ok(())
}

fn build_manifest(
    source_bytes: u64,
    source_hash: &str,
    weight_bytes: u64,
    weight_hash: &str,
) -> ScrfdArtifactManifest {
    let output = |name: &str, source: &[usize], public: &[usize]| ScrfdOutputManifest {
        onnx_name: name.to_owned(),
        source_shape: source.to_vec(),
        public_shape: public.to_vec(),
    };
    ScrfdArtifactManifest {
        schema_version: 1,
        model_kind: "scrfd_2.5g_kps".to_owned(),
        architecture_version: 1,
        source: ScrfdSourceManifest {
            format: "onnx".to_owned(),
            file_name: "scrfd_2.5g_kps.onnx".to_owned(),
            file_bytes: feathertalk_scrfd::SCRFD_SOURCE_ONNX_BYTES,
            sha256: feathertalk_scrfd::SCRFD_SOURCE_ONNX_SHA256.to_owned(),
            opset: 12,
            input_name: "images".to_owned(),
            output_names: [
                "out0", "out1", "out2", "out3", "out4", "out5", "out6", "out7", "out8",
            ]
            .map(str::to_owned),
        },
        generator: ScrfdGeneratorManifest {
            burn: "0.21.0".to_owned(),
            burn_onnx: "0.21.0".to_owned(),
            burn_store: "0.21.0".to_owned(),
            simplify: true,
            load_strategy: "none".to_owned(),
        },
        input: ScrfdInputManifest {
            dtype: "float32".to_owned(),
            shape: [1, 3, 640, 640],
            scale: 1.0 / 128.0,
            mean: [127.5; 3],
            swap_rb: true,
        },
        levels: [
            ScrfdLevelManifest {
                stride: 8,
                anchors: 12_800,
                score: output("out0", &[1, 12_800, 1], &[1, 12_800]),
                bbox: output("out3", &[1, 12_800, 4], &[1, 12_800, 4]),
                keypoints: output("out6", &[1, 12_800, 10], &[1, 12_800, 10]),
            },
            ScrfdLevelManifest {
                stride: 16,
                anchors: 3_200,
                score: output("out1", &[1, 3_200, 1], &[1, 3_200]),
                bbox: output("out4", &[1, 3_200, 4], &[1, 3_200, 4]),
                keypoints: output("out7", &[1, 3_200, 10], &[1, 3_200, 10]),
            },
            ScrfdLevelManifest {
                stride: 32,
                anchors: 800,
                score: output("out2", &[1, 800, 1], &[1, 800]),
                bbox: output("out5", &[1, 800, 4], &[1, 800, 4]),
                keypoints: output("out8", &[1, 800, 10], &[1, 800, 10]),
            },
        ],
        generated_source: ScrfdFileManifest {
            file_name: "scrfd_2_5g.rs".to_owned(),
            file_bytes: source_bytes,
            sha256: source_hash.to_owned(),
        },
        weights: ScrfdWeightManifest {
            format: "safetensors".to_owned(),
            file_name: "model.safetensors".to_owned(),
            file_bytes: weight_bytes,
            sha256: weight_hash.to_owned(),
        },
        license: ScrfdLicenseManifest {
            license_id: "NOASSERTION".to_owned(),
            redistribution_approved: false,
            evidence: "repository does not provide a verifiable model-weight license".to_owned(),
        },
    }
}

fn write_constants(
    path: &Path,
    source_bytes: u64,
    source_hash: &str,
    weight_bytes: u64,
    weight_hash: &str,
) -> Result<(), ToolError> {
    let constants = format!(
        "pub(crate) const GENERATED_SOURCE_BYTES: u64 = {source_bytes};\n\
         pub(crate) const GENERATED_SOURCE_SHA256: &str = \"{source_hash}\";\n\
         pub(crate) const MODEL_SAFETENSORS_BYTES: u64 = {weight_bytes};\n\
         pub(crate) const MODEL_SAFETENSORS_SHA256: &str = \"{weight_hash}\";\n"
    );
    write_create_new(path, constants.as_bytes())
}

fn require_tree(root: &Path) -> Result<(), ToolError> {
    let expected = [
        "src/generated/scrfd_2_5g.rs",
        "src/generated/artifact_contract.rs",
        "artifacts/scrfd_2_5g/model.safetensors",
        "artifacts/scrfd_2_5g/manifest.json",
    ];
    let mut actual = Vec::new();
    collect_files(root, root, &mut actual)?;
    actual.sort();
    let mut expected = expected.to_vec();
    expected.sort();
    if actual != expected {
        return Err(ToolError::Generation(format!(
            "generated tree differs: expected {expected:?}, got {actual:?}"
        )));
    }
    Ok(())
}

fn verify_tree(generated: &Path, committed: &Path) -> Result<(), ToolError> {
    let files = [
        "src/generated/scrfd_2_5g.rs",
        "src/generated/artifact_contract.rs",
        "artifacts/scrfd_2_5g/model.safetensors",
        "artifacts/scrfd_2_5g/manifest.json",
    ];
    for relative in files {
        let actual = generated.join(relative);
        let expected = committed.join(relative);
        if file_bytes(&actual)? != file_bytes(&expected)? {
            return Err(ToolError::TreeMismatch(expected));
        }
    }
    Ok(())
}

fn collect_files(root: &Path, directory: &Path, output: &mut Vec<String>) -> Result<(), ToolError> {
    for entry in fs::read_dir(directory)
        .map_err(|source| io_error("read generated tree", directory, source))?
    {
        let entry =
            entry.map_err(|source| io_error("read generated tree entry", directory, source))?;
        let path = entry.path();
        if path.is_dir() {
            collect_files(root, &path, output)?;
        } else {
            output.push(
                path.strip_prefix(root)
                    .unwrap()
                    .to_string_lossy()
                    .replace('\\', "/"),
            );
        }
    }
    Ok(())
}

fn copy_create_new(source: &Path, destination: &Path) -> Result<(), ToolError> {
    let bytes = file_bytes(source)?;
    write_create_new(destination, &bytes)
}

fn write_create_new(path: &Path, bytes: &[u8]) -> Result<(), ToolError> {
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|source| io_error("create generated file", path, source))?;
    file.write_all(bytes)
        .map_err(|source| io_error("write generated file", path, source))?;
    file.sync_all()
        .map_err(|source| io_error("sync generated file", path, source))?;
    Ok(())
}

fn file_bytes(path: &Path) -> Result<Vec<u8>, ToolError> {
    fs::read(path).map_err(|source| io_error("read generated file", path, source))
}

fn sha256(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

fn absolute(path: &Path) -> Result<PathBuf, ToolError> {
    if path.is_absolute() {
        Ok(path.to_owned())
    } else {
        std::env::current_dir()
            .map(|current| current.join(path))
            .map_err(|source| io_error("read current directory", path, source))
    }
}

fn io_error(operation: &'static str, path: &Path, source: std::io::Error) -> ToolError {
    ToolError::Io {
        operation,
        path: path.to_owned(),
        source,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn verification_allows_normal_files_outside_the_four_artifacts() {
        let temp = tempfile::tempdir().unwrap();
        let generated = temp.path().join("generated");
        let committed = temp.path().join("committed");
        for root in [&generated, &committed] {
            fs::create_dir_all(root.join("src/generated")).unwrap();
            fs::create_dir_all(root.join("artifacts/scrfd_2_5g")).unwrap();
            for relative in [
                "src/generated/scrfd_2_5g.rs",
                "src/generated/artifact_contract.rs",
                "artifacts/scrfd_2_5g/model.safetensors",
                "artifacts/scrfd_2_5g/manifest.json",
            ] {
                fs::write(root.join(relative), relative.as_bytes()).unwrap();
            }
        }
        fs::write(committed.join("Cargo.toml"), b"[package]\n").unwrap();

        verify_tree(&generated, &committed).unwrap();
    }
}
