use feathertalk_models::{backend::CpuBackend, unet::OriginalUnetConfig};
use feathertalk_parity::archive::GoldenArchive;
use feathertalk_weights::{LegacyImportRequest, LegacyModelKind, import_into};

#[test]
fn unet_micro_checkpoint_applies_strictly_to_burn_model() {
    let root = env!("CARGO_MANIFEST_DIR");
    let archive = GoldenArchive::open(format!("{root}/../../tests/golden/burn-feasibility-v1.zip"))
        .expect("golden archive should open");
    let temp = tempfile::tempdir().expect("temporary fixture directory");
    let fixture = temp.path().join("fixture");
    archive
        .extract_to(&fixture)
        .expect("golden archive should extract");

    let device = Default::default();
    let mut model = OriginalUnetConfig::parity_micro().init::<CpuBackend>(&device);
    let report = import_into::<CpuBackend, _>(
        &mut model,
        &LegacyImportRequest {
            path: fixture.join("weights/unet_micro_train.pth"),
            kind: LegacyModelKind::OriginalUnet,
            ..Default::default()
        },
    )
    .expect("strict checkpoint import should have no missing or unexpected tensors");

    assert!(!report.applied.is_empty());
}
