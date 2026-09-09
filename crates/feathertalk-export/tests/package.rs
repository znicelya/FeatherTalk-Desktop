use std::{collections::BTreeMap, fs};

use burn::{
    nn::{Linear, LinearConfig},
    tensor::backend::Backend,
};
use burn_store::ModuleSnapshot;
use feathertalk_export::{
    LicenseBundle, LicenseEntry, MAX_MANIFEST_BYTES, ModelConfiguration, ModelDescription,
    ModelPackageManifest, PackageBuildRequest, PackageError, SourceManifest, TrainingManifest,
    load_model_package, read_package_manifest, write_model_package,
};
use feathertalk_models::backend::CpuBackend;
use sha2::{Digest, Sha256};

fn description() -> ModelDescription {
    ModelDescription::from_configuration(ModelConfiguration::OriginalUnet {
        channels: [2, 4, 8, 16, 32],
    })
}

fn fixture() -> (tempfile::TempDir, PackageBuildRequest, Linear<CpuBackend>) {
    let root = tempfile::tempdir().unwrap();
    let source_path = root.path().join("source.pth");
    fs::write(&source_path, b"source-fixture").unwrap();
    let source_sha256 = hex::encode(Sha256::digest(b"source-fixture"));
    let licenses_path = root.path().join("LICENSES.input.json");
    let licenses = LicenseBundle {
        schema_version: 1,
        entries: vec![LicenseEntry {
            component: "test component".to_owned(),
            license_id: "LicenseRef-Test".to_owned(),
            source_url: "https://example.invalid/test".to_owned(),
            notice: "test-only local record".to_owned(),
        }],
    };
    fs::write(&licenses_path, serde_json::to_vec(&licenses).unwrap()).unwrap();
    let device = Default::default();
    let model = LinearConfig::new(2, 2).init::<CpuBackend>(&device);
    let request = PackageBuildRequest {
        destination: root.path().join("published"),
        description: description(),
        source_path,
        source: SourceManifest {
            format: "test".to_owned(),
            identifier: "linear-fixture".to_owned(),
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
    (root, request, model)
}

#[test]
fn package_round_trip_contains_exact_three_files_and_preserves_tensor_data() {
    let (root, request, model) = fixture();
    let device = Default::default();
    let report = write_model_package::<CpuBackend, _, _>(&request, &model, &device, |device| {
        LinearConfig::new(2, 2).init::<CpuBackend>(device)
    })
    .unwrap();

    let mut names = fs::read_dir(&request.destination)
        .unwrap()
        .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
        .collect::<Vec<_>>();
    names.sort();
    assert_eq!(
        names,
        ["LICENSES.json", "manifest.json", "model.safetensors"]
    );
    assert_eq!(report.manifest.model.file_name, "model.safetensors");
    assert_eq!(report.manifest.licenses.file_name, "LICENSES.json");

    let (loaded, manifest) = load_model_package::<CpuBackend, _, _>(
        &request.destination,
        &request.description,
        &device,
        |device| LinearConfig::new(2, 2).init::<CpuBackend>(device),
    )
    .unwrap();
    assert_eq!(manifest, report.manifest);
    assert_module_data_equal(&model, &loaded);
    assert!(root.path().join("published").is_dir());
}

#[test]
fn manifest_reader_returns_the_published_manifest_and_enforces_the_directory_contract() {
    let (_root, request, model) = fixture();
    let device = Default::default();
    let report = write_model_package::<CpuBackend, _, _>(&request, &model, &device, |device| {
        LinearConfig::new(2, 2).init::<CpuBackend>(device)
    })
    .unwrap();

    let manifest = read_package_manifest(&request.destination).unwrap();
    assert_eq!(manifest, report.manifest);
    assert_eq!(manifest.configuration, request.description.configuration);

    fs::write(request.destination.join("notes.txt"), b"unexpected").unwrap();
    let error = read_package_manifest(&request.destination).unwrap_err();
    assert!(matches!(error, PackageError::InvalidRequest(_)));
    fs::remove_file(request.destination.join("notes.txt")).unwrap();

    let oversized = vec![b' '; usize::try_from(MAX_MANIFEST_BYTES).unwrap() + 1];
    fs::write(request.destination.join("manifest.json"), oversized).unwrap();
    let error = read_package_manifest(&request.destination).unwrap_err();
    assert!(matches!(error, PackageError::InvalidRequest(_)));
    assert!(error.to_string().contains("manifest exceeds 65536 bytes"));
}

#[test]
fn existing_destination_is_rejected_without_clobbering_it() {
    let (_root, request, model) = fixture();
    fs::create_dir(&request.destination).unwrap();
    fs::write(request.destination.join("sentinel"), b"keep").unwrap();
    let device = Default::default();

    let error = write_model_package::<CpuBackend, _, _>(&request, &model, &device, |device| {
        LinearConfig::new(2, 2).init::<CpuBackend>(device)
    })
    .unwrap_err();
    assert!(matches!(error, PackageError::InvalidRequest(_)));
    assert_eq!(
        fs::read(request.destination.join("sentinel")).unwrap(),
        b"keep"
    );
}

#[test]
fn invalid_license_fails_before_publication() {
    let (_root, mut request, model) = fixture();
    fs::write(&request.licenses_path, b"{not-json}").unwrap();
    let device = Default::default();

    let error = write_model_package::<CpuBackend, _, _>(&request, &model, &device, |device| {
        LinearConfig::new(2, 2).init::<CpuBackend>(device)
    })
    .unwrap_err();
    assert!(matches!(error, PackageError::InvalidLicense(_)));
    assert!(!request.destination.exists());
    request.destination.set_file_name("other");
}

#[test]
fn loader_rejects_tampered_model_before_burn_store_access() {
    let (_root, request, model) = fixture();
    let device = Default::default();
    write_model_package::<CpuBackend, _, _>(&request, &model, &device, |device| {
        LinearConfig::new(2, 2).init::<CpuBackend>(device)
    })
    .unwrap();
    fs::write(request.destination.join("model.safetensors"), b"broken").unwrap();

    let error = load_model_package::<CpuBackend, _, _>(
        &request.destination,
        &request.description,
        &device,
        |device| LinearConfig::new(2, 2).init::<CpuBackend>(device),
    )
    .unwrap_err();
    assert!(matches!(error, PackageError::HashMismatch { .. }));
}

#[test]
fn loader_rejects_corrupt_safetensors_even_when_declared_hash_matches() {
    let (_root, request, model) = fixture();
    let device = Default::default();
    write_model_package::<CpuBackend, _, _>(&request, &model, &device, |device| {
        LinearConfig::new(2, 2).init::<CpuBackend>(device)
    })
    .unwrap();

    let corrupt = b"not-a-safetensors-file";
    fs::write(request.destination.join("model.safetensors"), corrupt).unwrap();
    let manifest_path = request.destination.join("manifest.json");
    let mut manifest: ModelPackageManifest =
        serde_json::from_slice(&fs::read(&manifest_path).unwrap()).unwrap();
    manifest.model.bytes = corrupt.len() as u64;
    manifest.model.sha256 = hex::encode(Sha256::digest(corrupt));
    fs::write(&manifest_path, serde_json::to_vec(&manifest).unwrap()).unwrap();

    let error = load_model_package::<CpuBackend, _, _>(
        &request.destination,
        &request.description,
        &device,
        |device| LinearConfig::new(2, 2).init::<CpuBackend>(device),
    )
    .unwrap_err();
    assert!(matches!(error, PackageError::Store(_)));
}

#[test]
fn loader_rejects_description_mismatch_before_corrupt_weight_decode() {
    let (_root, request, model) = fixture();
    let device = Default::default();
    write_model_package::<CpuBackend, _, _>(&request, &model, &device, |device| {
        LinearConfig::new(2, 2).init::<CpuBackend>(device)
    })
    .unwrap();
    fs::write(request.destination.join("model.safetensors"), b"broken").unwrap();
    let wrong = ModelDescription::from_configuration(ModelConfiguration::FeatherHubert {
        channels: 32,
        expansion: 2,
        num_blocks: 2,
        output_dim: 64,
        dropout: 0.0,
    });

    let error =
        load_model_package::<CpuBackend, _, _>(&request.destination, &wrong, &device, |device| {
            LinearConfig::new(2, 2).init::<CpuBackend>(device)
        })
        .unwrap_err();
    assert!(matches!(error, PackageError::InvalidRequest(_)));
}

#[test]
fn source_change_during_staged_validation_aborts_and_cleans_staging() {
    let (root, request, model) = fixture();
    let device = Default::default();
    let source_path = request.source_path.clone();

    let error = write_model_package::<CpuBackend, _, _>(&request, &model, &device, move |device| {
        fs::write(&source_path, b"changed-during-validation").unwrap();
        LinearConfig::new(2, 2).init::<CpuBackend>(device)
    })
    .unwrap_err();

    assert!(matches!(error, PackageError::HashMismatch { .. }));
    assert!(!request.destination.exists());
    assert!(fs::read_dir(root.path()).unwrap().all(|entry| {
        !entry
            .unwrap()
            .file_name()
            .to_string_lossy()
            .starts_with(".feathertalk-model-")
    }));
}

#[test]
fn loader_rejects_extra_directory_entry_and_unknown_manifest_field() {
    let (_root, request, model) = fixture();
    let device = Default::default();
    write_model_package::<CpuBackend, _, _>(&request, &model, &device, |device| {
        LinearConfig::new(2, 2).init::<CpuBackend>(device)
    })
    .unwrap();

    fs::write(request.destination.join("notes.txt"), b"unexpected").unwrap();
    let error = load_model_package::<CpuBackend, _, _>(
        &request.destination,
        &request.description,
        &device,
        |device| LinearConfig::new(2, 2).init::<CpuBackend>(device),
    )
    .unwrap_err();
    assert!(matches!(error, PackageError::InvalidRequest(_)));
    fs::remove_file(request.destination.join("notes.txt")).unwrap();

    let mut manifest: serde_json::Value =
        serde_json::from_slice(&fs::read(request.destination.join("manifest.json")).unwrap())
            .unwrap();
    manifest["unexpected"] = true.into();
    fs::write(
        request.destination.join("manifest.json"),
        serde_json::to_vec(&manifest).unwrap(),
    )
    .unwrap();
    let error = load_model_package::<CpuBackend, _, _>(
        &request.destination,
        &request.description,
        &device,
        |device| LinearConfig::new(2, 2).init::<CpuBackend>(device),
    )
    .unwrap_err();
    assert!(matches!(error, PackageError::InvalidManifest(_)));
}

#[test]
fn loader_rejects_symlinked_model_when_supported() {
    let (root, request, model) = fixture();
    let device = Default::default();
    write_model_package::<CpuBackend, _, _>(&request, &model, &device, |device| {
        LinearConfig::new(2, 2).init::<CpuBackend>(device)
    })
    .unwrap();
    let original = fs::read(request.destination.join("model.safetensors")).unwrap();
    let backup = root.path().join("model.backup");
    fs::write(&backup, original).unwrap();
    fs::remove_file(request.destination.join("model.safetensors")).unwrap();
    if std::os::windows::fs::symlink_file(&backup, request.destination.join("model.safetensors"))
        .is_err()
    {
        return;
    }

    let error = load_model_package::<CpuBackend, _, _>(
        &request.destination,
        &request.description,
        &device,
        |device| LinearConfig::new(2, 2).init::<CpuBackend>(device),
    )
    .unwrap_err();
    assert!(matches!(error, PackageError::InvalidRequest(_)));
}

fn assert_module_data_equal<B: Backend, M: ModuleSnapshot<B>>(left: &M, right: &M) {
    let left = left
        .collect(None, None, false)
        .into_iter()
        .map(|snapshot| (snapshot.full_path(), snapshot.to_data().unwrap()))
        .collect::<BTreeMap<_, _>>();
    let right = right
        .collect(None, None, false)
        .into_iter()
        .map(|snapshot| (snapshot.full_path(), snapshot.to_data().unwrap()))
        .collect::<BTreeMap<_, _>>();
    assert_eq!(left, right);
}
