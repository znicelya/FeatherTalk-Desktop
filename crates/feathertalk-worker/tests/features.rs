use std::{
    fs,
    path::{Path, PathBuf},
};

use feathertalk_audio::ChunkEncoder;
use feathertalk_export::{
    LicenseBundle, LicenseEntry, ModelConfiguration, ModelDescription, ModelPackageManifest,
    PackageBuildRequest, PackageError, SourceManifest, TrainingManifest, write_model_package,
};
use feathertalk_models::{
    backend::CpuBackend,
    feather_hubert::{FeatherHubertConfig, FeatherHubertEncoder},
};
use feathertalk_worker::{FeatureModel, WorkerConfig};
use sha2::{Digest, Sha256};
use tempfile::TempDir;

/// Publishes a micro FeatherHuBERT package under `root` and returns its directory.
///
/// The configuration is `parity_micro()`, 32/2/2/64/0.0, so an assertion on
/// `output_dim` can tell a manifest-driven load from one that silently fell back
/// to `FeatherHubertConfig::default()`, which is 512/2/12/1024/0.05.
fn published_package(root: &Path) -> PathBuf {
    let source_path = root.join("source.pth");
    fs::write(&source_path, b"source-fixture").unwrap();
    let source_sha256 = hex::encode(Sha256::digest(b"source-fixture"));
    let licenses_path = root.join("LICENSES.input.json");
    let licenses = LicenseBundle {
        schema_version: 1,
        entries: vec![LicenseEntry {
            component: "synthetic FeatherHuBERT fixture".to_owned(),
            license_id: "LicenseRef-Test".to_owned(),
            source_url: "https://example.invalid/feather-hubert".to_owned(),
            notice: "test-only local record".to_owned(),
        }],
    };
    fs::write(&licenses_path, serde_json::to_vec(&licenses).unwrap()).unwrap();
    let config = FeatherHubertConfig::parity_micro();
    let device = Default::default();
    let model = config.init::<CpuBackend>(&device);
    let request = PackageBuildRequest {
        destination: root.join("hubert"),
        description: ModelDescription::feather_hubert(config.clone()),
        source_path,
        source: SourceManifest {
            format: "test".to_owned(),
            identifier: "feather-hubert-fixture".to_owned(),
            version: "1".to_owned(),
            file_name: "source.pth".to_owned(),
            sha256: source_sha256,
            url: None,
        },
        licenses_path,
        created_at: "2026-08-27T00:00:00Z".to_owned(),
        minimum_app_version: "0.1.0".to_owned(),
        training: TrainingManifest::default(),
    };
    write_model_package::<CpuBackend, FeatherHubertEncoder<CpuBackend>, _>(
        &request,
        &model,
        &device,
        |device| config.init::<CpuBackend>(device),
    )
    .unwrap();
    request.destination
}

fn config_for(hubert_dir: &Path) -> WorkerConfig {
    WorkerConfig::from_values_with_toolchains(
        None,
        None,
        None,
        None,
        None,
        Some(hubert_dir.display().to_string()),
    )
}

#[test]
fn a_published_package_loads_with_the_configuration_the_manifest_declares() {
    let root = TempDir::new().unwrap();
    let directory = published_package(root.path());
    let config = config_for(&directory);

    let (encoder, model_sha256) = FeatureModel::load(config.features().unwrap())
        .unwrap()
        .into_parts();

    assert_eq!(encoder.output_dim(), 64);
    let manifest: ModelPackageManifest =
        serde_json::from_slice(&fs::read(directory.join("manifest.json")).unwrap()).unwrap();
    assert_eq!(model_sha256, manifest.model.sha256);
    assert_eq!(
        manifest.configuration,
        ModelConfiguration::FeatherHubert {
            channels: 32,
            expansion: 2,
            num_blocks: 2,
            output_dim: 64,
            dropout: 0.0,
        }
    );
}

#[test]
fn a_directory_without_a_package_is_refused_before_any_weight_is_read() {
    let root = TempDir::new().unwrap();
    let config = config_for(root.path());

    let error = FeatureModel::load(config.features().unwrap()).unwrap_err();

    match error {
        PackageError::InvalidRequest(message) => assert!(
            message.contains("package directory entries must be exactly"),
            "unexpected message: {message}"
        ),
        other => panic!("expected an invalid request, got {other:?}"),
    }
}

#[test]
fn a_package_of_another_kind_is_refused_by_name() {
    let root = TempDir::new().unwrap();
    let directory = published_package(root.path());
    let manifest_path = directory.join("manifest.json");
    let mut manifest: ModelPackageManifest =
        serde_json::from_slice(&fs::read(&manifest_path).unwrap()).unwrap();
    let other = ModelDescription::from_configuration(ModelConfiguration::MobileOneUnet {
        channels: [2, 4, 8, 16, 32],
        num_conv_branches: 1,
        reparameterized: false,
    });
    manifest.model_type = other.model_type;
    manifest.architecture_version = other.architecture_version;
    manifest.configuration = other.configuration;
    manifest.inputs = other.inputs;
    manifest.outputs = other.outputs;
    fs::write(&manifest_path, serde_json::to_vec(&manifest).unwrap()).unwrap();
    let config = config_for(&directory);

    let error = FeatureModel::load(config.features().unwrap()).unwrap_err();

    match error {
        PackageError::InvalidManifest(message) => assert_eq!(
            message,
            "expected a feather_hubert configuration, got mobileone_unet"
        ),
        other => panic!("expected an invalid manifest, got {other:?}"),
    }
}
