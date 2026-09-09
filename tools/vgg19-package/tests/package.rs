use std::{fs, io, path::PathBuf};

use burn::{tensor::TensorData, tensor::backend::Backend};
use burn_store::ModuleSnapshot;
use feathertalk_training::{
    Vgg19Conv3_3, Vgg19LicenseBundle, Vgg19LicenseEntry, load_vgg19_package,
};
use feathertalk_vgg19_package::{PackageError, Vgg19PackageRequest, build_vgg19_package};
use feathertalk_weights::{LegacyImportRequest, LegacyModelKind, import_into};
use zip::ZipArchive;

type CpuBackend = burn::backend::NdArray<f32>;

#[test]
fn direct_fixture_builds_a_loadable_three_file_package() {
    let fixture = fixture();
    let destination = fixture.temp.path().join("published");

    let report = build_vgg19_package(&Vgg19PackageRequest {
        source: fixture.source.clone(),
        licenses: fixture.licenses.clone(),
        destination: destination.clone(),
    })
    .unwrap();

    assert_eq!(report.manifest.tensor_count, 14);
    assert_eq!(report.manifest.total_elements, 1_735_488);
    assert!(destination.join("manifest.json").is_file());
    assert!(destination.join("model.safetensors").is_file());
    assert!(destination.join("LICENSES.json").is_file());

    let loaded = load_vgg19_package::<CpuBackend>(&destination, &Default::default()).unwrap();
    let expected = import_fixture(&fixture.source);
    assert_module_data_equal(&expected, &loaded);
}

#[test]
fn existing_destination_is_rejected_without_overwrite() {
    let fixture = fixture();
    let destination = fixture.temp.path().join("published");
    fs::create_dir(&destination).unwrap();
    fs::write(destination.join("sentinel.txt"), b"keep").unwrap();

    let error = build_vgg19_package(&Vgg19PackageRequest {
        source: fixture.source,
        licenses: fixture.licenses,
        destination: destination.clone(),
    })
    .unwrap_err();

    assert!(matches!(error, PackageError::InvalidRequest(_)));
    assert_eq!(fs::read(destination.join("sentinel.txt")).unwrap(), b"keep");
}

#[test]
fn invalid_license_bundle_leaves_destination_absent() {
    let fixture = fixture();
    let destination = fixture.temp.path().join("published");
    fs::write(
        &fixture.licenses,
        serde_json::to_vec_pretty(&Vgg19LicenseBundle {
            schema_version: 1,
            entries: Vec::new(),
        })
        .unwrap(),
    )
    .unwrap();

    let error = build_vgg19_package(&Vgg19PackageRequest {
        source: fixture.source,
        licenses: fixture.licenses,
        destination: destination.clone(),
    })
    .unwrap_err();

    assert!(matches!(error, PackageError::Training(_)));
    assert!(!destination.exists());
}

#[test]
fn unexpected_source_tensor_leaves_destination_absent() {
    let fixture = fixture();
    let destination = fixture.temp.path().join("published");
    let unexpected = extract_fixture(&fixture.temp, "vgg19-unexpected.pth");

    let error = build_vgg19_package(&Vgg19PackageRequest {
        source: unexpected,
        licenses: fixture.licenses,
        destination: destination.clone(),
    })
    .unwrap_err();

    assert!(matches!(error, PackageError::WeightImport(_)));
    assert!(!destination.exists());
}

struct Fixture {
    temp: tempfile::TempDir,
    source: PathBuf,
    licenses: PathBuf,
}

fn fixture() -> Fixture {
    let temp = tempfile::tempdir().unwrap();
    let source = extract_fixture(&temp, "vgg19-direct.pth");
    let licenses = temp.path().join("LICENSES.input.json");
    let bundle = Vgg19LicenseBundle {
        schema_version: 1,
        entries: vec![Vgg19LicenseEntry {
            component: "synthetic VGG19 import fixture".to_owned(),
            license_id: "LicenseRef-Test-Only".to_owned(),
            source_url: "https://example.invalid/vgg19".to_owned(),
            notice: "Synthetic test fixture only.".to_owned(),
        }],
    };
    fs::write(&licenses, serde_json::to_vec_pretty(&bundle).unwrap()).unwrap();
    Fixture {
        temp,
        source,
        licenses,
    }
}

fn extract_fixture(temp: &tempfile::TempDir, member: &str) -> PathBuf {
    let destination = temp.path().join(member);
    let archive_path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/golden/vgg19-import-v1.zip");
    let archive = fs::File::open(archive_path).unwrap();
    let mut archive = ZipArchive::new(archive).unwrap();
    let mut source = archive.by_name(member).unwrap();
    let mut destination_file = fs::File::create(&destination).unwrap();
    io::copy(&mut source, &mut destination_file).unwrap();
    destination
}

fn import_fixture(path: &std::path::Path) -> Vgg19Conv3_3<CpuBackend> {
    let device = Default::default();
    let mut model = Vgg19Conv3_3::<CpuBackend>::new_for_import(&device);
    let report = import_into::<CpuBackend, _>(
        &mut model,
        &LegacyImportRequest {
            path: path.to_owned(),
            kind: LegacyModelKind::Vgg19Conv3_3,
            top_level_key: None,
            max_file_bytes: 16 * 1024 * 1024,
            max_tensor_count: 64,
            max_total_elements: 2_000_000,
        },
    )
    .unwrap();
    assert_eq!(report.applied.len(), 14);
    model
}

fn module_data<B: Backend, M: ModuleSnapshot<B>>(module: &M) -> Vec<(String, TensorData)> {
    module
        .collect(None, None, false)
        .into_iter()
        .map(|snapshot| (snapshot.full_path(), snapshot.to_data().unwrap()))
        .collect()
}

fn assert_module_data_equal<B: Backend, M: ModuleSnapshot<B>>(first: &M, second: &M) {
    assert_eq!(module_data(first), module_data(second));
}
