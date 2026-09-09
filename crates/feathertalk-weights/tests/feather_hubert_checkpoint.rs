use std::{
    fs, io,
    path::{Path, PathBuf},
    sync::{Mutex, OnceLock},
};

use burn::tensor::{Tensor, TensorData};
use feathertalk_models::backend::CpuBackend;
use feathertalk_weights::{inspect_feather_hubert_checkpoint, load_feather_hubert_checkpoint};
use zip::ZipArchive;

#[test]
fn golden_micro_checkpoint_is_inferred_loaded_and_executed() {
    let path = extract_fixture("weights/feather_micro.pth");
    let inspection = inspect_feather_hubert_checkpoint(&path).unwrap();
    assert_eq!(inspection.config().channels, 32);
    assert_eq!(inspection.config().expansion, 2);
    assert_eq!(inspection.config().num_blocks, 2);
    assert_eq!(inspection.config().output_dim, 64);
    assert_eq!(inspection.config().dropout, 0.0);
    assert_eq!(inspection.tensor_count(), 35);
    assert_eq!(inspection.total_elements(), 472_384);
    assert_eq!(inspection.source_sha256().len(), 64);

    let device = Default::default();
    let (model, loaded) = load_feather_hubert_checkpoint::<CpuBackend>(&path, &device).unwrap();
    assert_eq!(loaded.source_sha256(), inspection.source_sha256());
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

fn extract_fixture(member: &str) -> PathBuf {
    static FIXTURE_DIR: OnceLock<PathBuf> = OnceLock::new();
    static EXTRACTION_LOCK: Mutex<()> = Mutex::new(());

    let directory = FIXTURE_DIR.get_or_init(|| {
        let directory = std::env::temp_dir().join(format!(
            "feathertalk-weights-feather-hubert-checkpoint-{}",
            std::process::id()
        ));
        fs::create_dir_all(&directory).unwrap();
        directory
    });
    let destination = directory.join(member.replace('/', "_"));
    let _guard = EXTRACTION_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);

    if !destination.exists() {
        let archive_path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/golden/burn-feasibility-v1.zip");
        let archive = fs::File::open(archive_path).unwrap();
        let mut archive = ZipArchive::new(archive).unwrap();
        let mut source = archive.by_name(member).unwrap();
        let mut destination_file = fs::File::create(&destination).unwrap();
        io::copy(&mut source, &mut destination_file).unwrap();
    }

    destination
}
