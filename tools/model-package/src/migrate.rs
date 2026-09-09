use std::{fs, path::PathBuf};

use feathertalk_audio::{
    FeatureArtifact, FeatureMatrix, MAX_FEATURE_FILE_BYTES, write_feature_file_no_clobber,
};
use feathertalk_export::{
    FeatherHubertPackageRequest, ModelDescription, ModelPackageManifest, PackageBuildRequest,
    SourceManifest, TrainingManifest, build_feather_hubert_package, write_model_package,
};
use feathertalk_models::{
    backend::CpuBackend,
    unet::{OriginalUnet, OriginalUnetConfig},
};
use feathertalk_weights::{LegacyImportRequest, LegacyModelKind, import_into};
use ndarray::{ArrayD, Ix3};
use ndarray_npy::ReadNpyExt;

use crate::CliResult;

const SOURCE_FORMAT: &str = "pytorch-pickle-restricted";

#[derive(Debug, Clone, Copy)]
pub enum ModelMigrationKind {
    FeatherHubert,
    OriginalUnet,
}

#[derive(Debug)]
pub struct ModelMigrationRequest {
    pub kind: ModelMigrationKind,
    pub source: PathBuf,
    pub licenses: PathBuf,
    pub destination: PathBuf,
    pub created_at: String,
    pub minimum_app_version: String,
}

#[derive(Debug)]
pub struct FeatureMigrationReport {
    pub source_shape: [usize; 3],
    pub artifact: FeatureArtifact,
}

pub fn migrate_model(request: &ModelMigrationRequest) -> CliResult<ModelPackageManifest> {
    validate_legacy_extension(&request.source)?;
    match request.kind {
        ModelMigrationKind::FeatherHubert => {
            Ok(build_feather_hubert_package(&FeatherHubertPackageRequest {
                source: request.source.clone(),
                licenses: request.licenses.clone(),
                destination: request.destination.clone(),
                created_at: request.created_at.clone(),
                minimum_app_version: request.minimum_app_version.clone(),
            })?
            .manifest)
        }
        ModelMigrationKind::OriginalUnet => migrate_original_unet(request),
    }
}

pub fn migrate_features(
    source: &std::path::Path,
    destination: &std::path::Path,
) -> CliResult<FeatureMigrationReport> {
    if destination.exists() {
        return Err(format!("destination already exists: {}", destination.display()).into());
    }
    let metadata = fs::symlink_metadata(source)?;
    if metadata.file_type().is_symlink() || !metadata.file_type().is_file() {
        return Err(format!("NPY source must be a regular file: {}", source.display()).into());
    }
    if metadata.len() > MAX_FEATURE_FILE_BYTES {
        return Err(format!(
            "NPY source exceeds {MAX_FEATURE_FILE_BYTES} bytes: {}",
            source.display()
        )
        .into());
    }
    let file = fs::File::open(source)?;
    let array = ArrayD::<f32>::read_npy(std::io::BufReader::new(file)).map_err(|error| {
        format!(
            "invalid NPY {}; expected an f32 array: {error}",
            source.display()
        )
    })?;
    let rank = array.ndim();
    if rank != 3 {
        return Err(
            format!("invalid NPY rank {rank}; expected rank 3 [video_frames, 2, 1024]").into(),
        );
    }
    let array = array
        .into_dimensionality::<Ix3>()
        .expect("rank was validated");
    let shape = array.shape();
    if shape[0] == 0 || shape[1] != 2 || shape[2] != 1024 {
        return Err(format!(
            "invalid NPY shape {:?}; expected [video_frames, 2, 1024] with video_frames > 0",
            shape
        )
        .into());
    }
    let source_shape = [shape[0], shape[1], shape[2]];
    let tokens = shape[0]
        .checked_mul(shape[1])
        .ok_or("feature token count overflowed usize")?;
    let values = array.iter().copied().collect();
    let matrix = FeatureMatrix::new(tokens, shape[2], values)?;
    let artifact = write_feature_file_no_clobber(destination, &matrix)?;
    Ok(FeatureMigrationReport {
        source_shape,
        artifact,
    })
}

fn migrate_original_unet(request: &ModelMigrationRequest) -> CliResult<ModelPackageManifest> {
    let device = Default::default();
    let config = OriginalUnetConfig::production();
    let mut model = config.clone().init::<CpuBackend>(&device);
    let import = import_into::<CpuBackend, OriginalUnet<CpuBackend>>(
        &mut model,
        &LegacyImportRequest {
            path: request.source.clone(),
            kind: LegacyModelKind::OriginalUnet,
            ..Default::default()
        },
    )?;
    let file_name = source_file_name(&request.source)?;
    let package_request = PackageBuildRequest {
        destination: request.destination.clone(),
        description: ModelDescription::original_unet(config.clone()),
        source_path: request.source.clone(),
        source: SourceManifest {
            format: SOURCE_FORMAT.to_owned(),
            identifier: "feathertalk-original-unet".to_owned(),
            version: legacy_version(&file_name)?.to_owned(),
            file_name,
            sha256: import.source_sha256,
            url: None,
        },
        licenses_path: request.licenses.clone(),
        created_at: request.created_at.clone(),
        minimum_app_version: request.minimum_app_version.clone(),
        training: TrainingManifest::default(),
    };
    Ok(
        write_model_package::<CpuBackend, _, _>(
            &package_request,
            &model,
            &device,
            move |device| config.clone().init::<CpuBackend>(device),
        )?
        .manifest,
    )
}

fn validate_legacy_extension(path: &std::path::Path) -> CliResult<()> {
    let file_name = source_file_name(path)?;
    let lowercase = file_name.to_ascii_lowercase();
    if !lowercase.ends_with(".pth") && !lowercase.ends_with(".pth.tar") {
        return Err("legacy model source must end in .pth or .pth.tar".into());
    }
    Ok(())
}

fn source_file_name(path: &std::path::Path) -> CliResult<String> {
    path.file_name()
        .and_then(|value| value.to_str())
        .filter(|value| !value.is_empty())
        .map(str::to_owned)
        .ok_or_else(|| "source file name must be valid UTF-8".into())
}

fn legacy_version(file_name: &str) -> CliResult<&str> {
    let version = file_name
        .strip_suffix(".pth.tar")
        .or_else(|| file_name.strip_suffix(".pth"))
        .filter(|value| !value.is_empty())
        .ok_or("legacy source must have a non-empty version stem")?;
    Ok(version)
}
