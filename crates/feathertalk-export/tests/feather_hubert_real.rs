use std::{
    fs::{self, File},
    io::{Read, Write},
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
use sha2::{Digest, Sha256};

const EXPECTED_BYTES: u64 = 40_436_613;
const EXPECTED_SHA256: &str = "58df96af118d75d7f69da441e1f3960096f28dda637a4e8f4265f108d27aeb52";

#[test]
fn configured_real_checkpoint_builds_and_reloads_without_source_mutation() {
    let Some(path) = std::env::var_os("FEATHERTALK_FEATHER_HUBERT_CHECKPOINT") else {
        eprintln!("FEATHERTALK_FEATHER_HUBERT_CHECKPOINT is not set; skipping local model");
        return;
    };
    let path = PathBuf::from(path);
    assert!(path.is_absolute());
    let before = audit_file(&path);
    assert_eq!(before.0, EXPECTED_BYTES);
    assert_eq!(before.1, EXPECTED_SHA256);

    let temporary = tempfile::tempdir().unwrap();
    let licenses_path = temporary.path().join("LICENSES.local.json");
    let licenses = LicenseBundle {
        schema_version: 1,
        entries: vec![LicenseEntry {
            component: "user-supplied FeatherHuBERT checkpoint".to_owned(),
            license_id: "LicenseRef-User-Supplied-Unreviewed".to_owned(),
            source_url: "https://example.invalid/local-conversion-record".to_owned(),
            notice: "Local conversion record only; not redistribution approval.".to_owned(),
        }],
    };
    let mut license_file = File::create(&licenses_path).unwrap();
    license_file
        .write_all(&serde_json::to_vec_pretty(&licenses).unwrap())
        .unwrap();
    license_file.sync_all().unwrap();

    let destination = temporary.path().join("feather-hubert-package");
    let report = build_feather_hubert_package(&FeatherHubertPackageRequest {
        source: path.clone(),
        licenses: licenses_path,
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
    assert_eq!(
        report.manifest.configuration,
        feathertalk_export::ModelConfiguration::FeatherHubert {
            channels: 256,
            expansion: 2,
            num_blocks: 8,
            output_dim: 1024,
            dropout: 0.0,
        }
    );
    assert_eq!(report.manifest.tensors.tensor_count, 65);
    assert_eq!(report.manifest.tensors.total_elements, 3_364_096);

    let mut entries = fs::read_dir(&destination)
        .unwrap()
        .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
        .collect::<Vec<_>>();
    entries.sort();
    assert_eq!(
        entries,
        ["LICENSES.json", "manifest.json", "model.safetensors"]
    );

    let device = Default::default();
    let (model, manifest) = load_model_package::<CpuBackend, FeatherHubertEncoder<CpuBackend>, _>(
        &destination,
        &report.manifest.description(),
        &device,
        |device| {
            FeatherHubertConfig {
                channels: 256,
                expansion: 2,
                num_blocks: 8,
                output_dim: 1024,
                dropout: 0.0,
            }
            .init::<CpuBackend>(device)
        },
    )
    .unwrap();
    assert_eq!(manifest, report.manifest);

    let samples = (0..1360)
        .map(|index| (index as f32 - 680.0) / 680.0)
        .collect::<Vec<_>>();
    let waveform = Tensor::from_data(TensorData::new(samples, [1, 1360]), &device);
    let output = model.forward(waveform);
    assert_eq!(output.dims(), [1, 4, 1024]);
    assert!(
        output
            .into_data()
            .to_vec::<f32>()
            .unwrap()
            .iter()
            .all(|value| value.is_finite())
    );

    let after = audit_file(&path);
    assert_eq!(after, before);
}

fn audit_file(path: &std::path::Path) -> (u64, String) {
    let metadata = fs::symlink_metadata(path).unwrap();
    assert!(metadata.is_file());
    assert!(!metadata.file_type().is_symlink());
    let mut file = File::open(path).unwrap();
    let mut digest = Sha256::new();
    let mut bytes = 0_u64;
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let count = file.read(&mut buffer).unwrap();
        if count == 0 {
            break;
        }
        bytes += count as u64;
        digest.update(&buffer[..count]);
    }
    (bytes, hex::encode(digest.finalize()))
}
