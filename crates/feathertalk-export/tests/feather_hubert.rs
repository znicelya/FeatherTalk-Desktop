use std::{
    fs::{self, File},
    io,
    path::PathBuf,
};

use burn::tensor::{Tensor, TensorData};
use feathertalk_export::{
    FeatherHubertPackageRequest, LicenseBundle, LicenseEntry, build_feather_hubert_package,
    load_model_package,
};
use feathertalk_models::{
    backend::CpuBackend,
    feather_hubert::{FeatherHubertConfig, FeatherHubertEncoder},
};
use zip::ZipArchive;

#[test]
fn micro_checkpoint_builds_strict_package_and_runs_after_reload() {
    let root = tempfile::tempdir().unwrap();
    let source = extract_fixture(root.path());
    let licenses = root.path().join("LICENSES.input.json");
    fs::write(
        &licenses,
        serde_json::to_vec_pretty(&LicenseBundle {
            schema_version: 1,
            entries: vec![LicenseEntry {
                component: "synthetic FeatherHuBERT fixture".to_owned(),
                license_id: "LicenseRef-Test-Only".to_owned(),
                source_url: "https://example.invalid/feather-hubert".to_owned(),
                notice: "Local conversion test only; not redistribution approval.".to_owned(),
            }],
        })
        .unwrap(),
    )
    .unwrap();
    let destination = root.path().join("feather-hubert-package");

    let report = build_feather_hubert_package(&FeatherHubertPackageRequest {
        source: source.clone(),
        licenses: licenses.clone(),
        destination: destination.clone(),
        created_at: "2026-08-27T00:00:00Z".to_owned(),
        minimum_app_version: "0.1.0".to_owned(),
    })
    .unwrap();

    assert_eq!(report.manifest.model_type, "feather_hubert");
    assert_eq!(
        report.manifest.architecture_version,
        "feather-hubert-burn-v1"
    );
    assert_eq!(report.manifest.source.sha256.len(), 64);
    assert_eq!(report.manifest.tensors.tensor_count, 35);
    assert_eq!(report.manifest.tensors.total_elements, 472_384);
    assert_eq!(report.manifest.outputs[0].shape, vec![1, -1, 64]);

    let device = Default::default();
    let (model, manifest) = load_model_package::<CpuBackend, FeatherHubertEncoder<CpuBackend>, _>(
        &destination,
        &report.manifest.description(),
        &device,
        |device| FeatherHubertConfig::parity_micro().init::<CpuBackend>(device),
    )
    .unwrap();
    assert_eq!(manifest, report.manifest);
    let waveform = Tensor::from_data(TensorData::new(vec![0.0_f32; 1360], [1, 1360]), &device);
    let output = model.forward(waveform);
    assert_eq!(output.dims(), [1, 4, 64]);
    assert!(
        output
            .into_data()
            .to_vec::<f32>()
            .unwrap()
            .iter()
            .all(|value| value.is_finite())
    );
}

fn extract_fixture(root: &std::path::Path) -> PathBuf {
    let archive_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/golden/burn-feasibility-v1.zip");
    let archive = File::open(archive_path).unwrap();
    let mut archive = ZipArchive::new(archive).unwrap();
    let mut source = archive.by_name("weights/feather_micro.pth").unwrap();
    let destination = root.join("feather_micro.pth");
    let mut destination_file = File::create(&destination).unwrap();
    io::copy(&mut source, &mut destination_file).unwrap();
    destination
}
