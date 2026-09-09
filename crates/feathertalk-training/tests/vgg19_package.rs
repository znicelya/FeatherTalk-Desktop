use std::{fs, path::PathBuf};

use burn::{
    module::Module,
    nn::{Linear, LinearConfig, conv::Conv2d},
    tensor::backend::Backend,
};
use burn_store::ModuleSnapshot;
use feathertalk_training::{
    TrainingError, VGG19_ARCHITECTURE_VERSION, VGG19_MODEL_KIND, VGG19_PACKAGE_SCHEMA_VERSION,
    VGG19_SOURCE_URL, Vgg19Conv3_3, Vgg19FileManifest, Vgg19InputManifest, Vgg19LicenseBundle,
    Vgg19LicenseEntry, Vgg19PackageManifest, Vgg19SourceManifest, load_vgg19_package,
    read_vgg19_manifest,
};
use feathertalk_weights::save_safetensors;
use sha2::{Digest, Sha256};

type CpuBackend = burn::backend::NdArray<f32>;

const MODEL_FILE_NAME: &str = "model.safetensors";
const LICENSE_FILE_NAME: &str = "LICENSES.json";
const MANIFEST_FILE_NAME: &str = "manifest.json";

#[test]
fn valid_three_file_package_loads_all_fourteen_tensors() {
    let fixture = valid_package();

    let manifest = read_vgg19_manifest(&fixture.directory).unwrap();
    let loaded = load_vgg19_package::<CpuBackend>(&fixture.directory, &Default::default()).unwrap();

    assert_eq!(manifest, fixture.manifest);
    assert_module_snapshots_equal(&fixture.original, &loaded);
}

#[test]
fn loaded_vgg_parameters_are_marked_no_grad() {
    let fixture = valid_package();
    let loaded = load_vgg19_package::<burn::backend::Autodiff<CpuBackend>>(
        &fixture.directory,
        &Default::default(),
    )
    .unwrap();

    assert!(!loaded.conv1_1.weight.val().is_require_grad());
    assert!(
        !loaded
            .conv3_3
            .bias
            .as_ref()
            .unwrap()
            .val()
            .is_require_grad()
    );
}

#[test]
fn unknown_manifest_field_is_rejected() {
    let fixture = valid_package();
    let path = fixture.directory.join(MANIFEST_FILE_NAME);
    let mut manifest: serde_json::Value =
        serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
    manifest["unexpected"] = serde_json::json!(true);
    fs::write(path, serde_json::to_vec_pretty(&manifest).unwrap()).unwrap();

    assert_invalid_package_contains(&fixture.directory, "unknown field");
}

#[test]
fn wrong_model_hash_is_rejected() {
    let mut fixture = valid_package();
    fixture.manifest.model.sha256 = "0".repeat(64);
    write_manifest(&fixture.directory, &fixture.manifest);

    let error = load_error(&fixture.directory);
    assert!(matches!(
        error,
        TrainingError::HashMismatch { file, .. } if file == MODEL_FILE_NAME
    ));
}

#[test]
fn wrong_declared_model_length_is_rejected() {
    let mut fixture = valid_package();
    fixture.manifest.model.bytes += 1;
    write_manifest(&fixture.directory, &fixture.manifest);

    assert_invalid_package_contains(&fixture.directory, "declared byte length");
}

#[test]
fn missing_directory_entry_is_rejected() {
    let fixture = valid_package();
    fs::remove_file(fixture.directory.join(LICENSE_FILE_NAME)).unwrap();

    assert_invalid_package_contains(&fixture.directory, "directory entries");
}

#[test]
fn extra_directory_entry_is_rejected() {
    let fixture = valid_package();
    fs::write(fixture.directory.join("extra.txt"), b"unexpected").unwrap();

    assert_invalid_package_contains(&fixture.directory, "directory entries");
}

#[cfg(windows)]
#[test]
fn symlink_model_file_is_rejected_when_symlinks_are_available() {
    use std::os::windows::fs::symlink_file;

    let fixture = valid_package();
    let model = fixture.directory.join(MODEL_FILE_NAME);
    let target = fixture
        .directory
        .parent()
        .unwrap()
        .join("model-target.safetensors");
    fs::rename(&model, &target).unwrap();
    match symlink_file(&target, &model) {
        Ok(()) => {}
        Err(error)
            if matches!(
                error.kind(),
                std::io::ErrorKind::PermissionDenied | std::io::ErrorKind::Unsupported
            ) || error.raw_os_error() == Some(1314) =>
        {
            return;
        }
        Err(error) => panic!("failed to create test symlink: {error}"),
    }

    assert_invalid_package_contains(&fixture.directory, "symbolic link");
}

#[test]
fn model_over_sixteen_mib_is_rejected_before_store_load() {
    let fixture = valid_package();
    fs::write(
        fixture.directory.join(MODEL_FILE_NAME),
        vec![0_u8; 16 * 1024 * 1024 + 1],
    )
    .unwrap();

    assert_invalid_package_contains(&fixture.directory, "exceeds 16777216 bytes");
}

#[test]
fn missing_safetensors_tensor_is_rejected() {
    let mut fixture = valid_package();
    let model = fixture.original.clone();
    let missing = MissingVgg {
        conv1_1: model.conv1_1,
        conv1_2: model.conv1_2,
        conv2_1: model.conv2_1,
        conv2_2: model.conv2_2,
        conv3_1: model.conv3_1,
        conv3_2: model.conv3_2,
    };
    save_safetensors::<CpuBackend, _>(&missing, fixture.directory.join(MODEL_FILE_NAME)).unwrap();
    refresh_model_manifest(&fixture.directory, &mut fixture.manifest);

    assert_invalid_package_contains(&fixture.directory, "missing tensor conv3_3.bias");
}

#[test]
fn extra_safetensors_tensor_is_rejected() {
    let mut fixture = valid_package();
    let model = fixture.original.clone();
    let extra = ExtraVgg {
        conv1_1: model.conv1_1,
        conv1_2: model.conv1_2,
        conv2_1: model.conv2_1,
        conv2_2: model.conv2_2,
        conv3_1: model.conv3_1,
        conv3_2: model.conv3_2,
        conv3_3: model.conv3_3,
        extra: LinearConfig::new(1, 1).init(&Default::default()),
    };
    save_safetensors::<CpuBackend, _>(&extra, fixture.directory.join(MODEL_FILE_NAME)).unwrap();
    refresh_model_manifest(&fixture.directory, &mut fixture.manifest);

    assert_invalid_package_contains(&fixture.directory, "unexpected tensor extra.bias");
}

#[test]
fn empty_license_bundle_is_rejected() {
    let mut fixture = valid_package();
    write_license_bundle(
        &fixture.directory,
        &Vgg19LicenseBundle {
            schema_version: 1,
            entries: Vec::new(),
        },
    );
    refresh_license_manifest(&fixture.directory, &mut fixture.manifest);

    assert_invalid_package_contains(&fixture.directory, "at least one entry");
}

#[test]
fn blank_license_fields_are_rejected() {
    let mut fixture = valid_package();
    write_license_bundle(
        &fixture.directory,
        &Vgg19LicenseBundle {
            schema_version: 1,
            entries: vec![Vgg19LicenseEntry {
                component: " ".to_owned(),
                license_id: "LicenseRef-Test".to_owned(),
                source_url: "https://example.invalid/test".to_owned(),
                notice: "local test only".to_owned(),
            }],
        },
    );
    refresh_license_manifest(&fixture.directory, &mut fixture.manifest);

    assert_invalid_package_contains(&fixture.directory, "component must be non-empty");
}

#[test]
fn wrong_bgr_no_normalization_contract_is_rejected() {
    let mut fixture = valid_package();
    fixture.manifest.input.color_order = "rgb".to_owned();
    write_manifest(&fixture.directory, &fixture.manifest);

    assert_invalid_package_contains(&fixture.directory, "color_order");
}

struct PackageFixture {
    _temp: tempfile::TempDir,
    directory: PathBuf,
    original: Vgg19Conv3_3<CpuBackend>,
    manifest: Vgg19PackageManifest,
}

fn valid_package() -> PackageFixture {
    let temp = tempfile::tempdir().unwrap();
    let directory = temp.path().join("package");
    fs::create_dir(&directory).unwrap();
    let device = Default::default();
    let original = Vgg19Conv3_3::<CpuBackend>::new_for_import(&device);
    let model_path = directory.join(MODEL_FILE_NAME);
    save_safetensors::<CpuBackend, _>(&original, &model_path).unwrap();

    let licenses = Vgg19LicenseBundle {
        schema_version: 1,
        entries: vec![Vgg19LicenseEntry {
            component: "local VGG19 test fixture".to_owned(),
            license_id: "LicenseRef-Test-Only".to_owned(),
            source_url: "https://example.invalid/vgg19".to_owned(),
            notice: "Synthetic local package for loader tests only.".to_owned(),
        }],
    };
    write_license_bundle(&directory, &licenses);

    let manifest = Vgg19PackageManifest {
        schema_version: VGG19_PACKAGE_SCHEMA_VERSION,
        model_kind: VGG19_MODEL_KIND.to_owned(),
        architecture_version: VGG19_ARCHITECTURE_VERSION.to_owned(),
        source: Vgg19SourceManifest {
            framework: "torchvision".to_owned(),
            weight_id: "VGG19_Weights.IMAGENET1K_V1".to_owned(),
            url: VGG19_SOURCE_URL.to_owned(),
            sha256: "a".repeat(64),
        },
        input: Vgg19InputManifest {
            channels: 3,
            color_order: "bgr".to_owned(),
            value_range: "0..1".to_owned(),
            normalization: "none".to_owned(),
        },
        output_layer: "features.14".to_owned(),
        tensor_count: 14,
        total_elements: 1_735_488,
        model: file_manifest(&model_path),
        licenses: file_manifest(&directory.join(LICENSE_FILE_NAME)),
    };
    write_manifest(&directory, &manifest);

    PackageFixture {
        _temp: temp,
        directory,
        original,
        manifest,
    }
}

fn file_manifest(path: &std::path::Path) -> Vgg19FileManifest {
    Vgg19FileManifest {
        file_name: path.file_name().unwrap().to_str().unwrap().to_owned(),
        bytes: fs::metadata(path).unwrap().len(),
        sha256: sha256_file(path),
    }
}

fn refresh_model_manifest(directory: &std::path::Path, manifest: &mut Vgg19PackageManifest) {
    manifest.model = file_manifest(&directory.join(MODEL_FILE_NAME));
    write_manifest(directory, manifest);
}

fn refresh_license_manifest(directory: &std::path::Path, manifest: &mut Vgg19PackageManifest) {
    manifest.licenses = file_manifest(&directory.join(LICENSE_FILE_NAME));
    write_manifest(directory, manifest);
}

fn write_manifest(directory: &std::path::Path, manifest: &Vgg19PackageManifest) {
    let mut bytes = serde_json::to_vec_pretty(manifest).unwrap();
    bytes.push(b'\n');
    fs::write(directory.join(MANIFEST_FILE_NAME), bytes).unwrap();
}

fn write_license_bundle(directory: &std::path::Path, licenses: &Vgg19LicenseBundle) {
    let mut bytes = serde_json::to_vec_pretty(licenses).unwrap();
    bytes.push(b'\n');
    fs::write(directory.join(LICENSE_FILE_NAME), bytes).unwrap();
}

fn sha256_file(path: &std::path::Path) -> String {
    hex::encode(Sha256::digest(fs::read(path).unwrap()))
}

fn load_error(directory: &std::path::Path) -> TrainingError {
    load_vgg19_package::<CpuBackend>(directory, &Default::default()).unwrap_err()
}

fn assert_invalid_package_contains(directory: &std::path::Path, expected: &str) {
    let error = load_error(directory);
    assert!(
        matches!(&error, TrainingError::InvalidPackage(_)),
        "expected invalid package error, got {error:?}"
    );
    assert!(
        error.to_string().contains(expected),
        "expected {error:?} to contain {expected:?}"
    );
}

fn assert_module_snapshots_equal<B: Backend, M: ModuleSnapshot<B>>(first: &M, second: &M) {
    let first = first.collect(None, None, false);
    let second = second.collect(None, None, false);
    assert_eq!(first.len(), second.len());

    for (first, second) in first.iter().zip(second.iter()) {
        assert_eq!(first.full_path(), second.full_path());
        assert_eq!(first.shape, second.shape);
        assert_eq!(first.dtype, second.dtype);
        assert_eq!(first.to_data().unwrap(), second.to_data().unwrap());
    }
}

#[derive(Module, Debug)]
struct MissingVgg<B: Backend> {
    conv1_1: Conv2d<B>,
    conv1_2: Conv2d<B>,
    conv2_1: Conv2d<B>,
    conv2_2: Conv2d<B>,
    conv3_1: Conv2d<B>,
    conv3_2: Conv2d<B>,
}

#[derive(Module, Debug)]
struct ExtraVgg<B: Backend> {
    conv1_1: Conv2d<B>,
    conv1_2: Conv2d<B>,
    conv2_1: Conv2d<B>,
    conv2_2: Conv2d<B>,
    conv3_1: Conv2d<B>,
    conv3_2: Conv2d<B>,
    conv3_3: Conv2d<B>,
    extra: Linear<B>,
}
