use std::path::PathBuf;

use burn::tensor::{Tensor, TensorData};
use feathertalk_models::backend::CpuBackend;
use feathertalk_weights::load_feather_hubert_checkpoint;

const EXPECTED_BYTES: u64 = 40_436_613;
const EXPECTED_SHA256: &str = "58df96af118d75d7f69da441e1f3960096f28dda637a4e8f4265f108d27aeb52";

#[test]
fn configured_user_checkpoint_loads_and_runs_on_cpu_without_writes() {
    let Some(path) = std::env::var_os("FEATHERTALK_FEATHER_HUBERT_CHECKPOINT") else {
        eprintln!("FEATHERTALK_FEATHER_HUBERT_CHECKPOINT is not set; skipping local model");
        return;
    };
    let path = PathBuf::from(path);
    assert!(path.is_absolute());
    let before = std::fs::metadata(&path).unwrap();
    assert_eq!(before.len(), EXPECTED_BYTES);

    let device = Default::default();
    let (model, checkpoint) = load_feather_hubert_checkpoint::<CpuBackend>(&path, &device).unwrap();
    assert_eq!(checkpoint.source_sha256(), EXPECTED_SHA256);
    assert_eq!(checkpoint.config().channels, 256);
    assert_eq!(checkpoint.config().expansion, 2);
    assert_eq!(checkpoint.config().num_blocks, 8);
    assert_eq!(checkpoint.config().output_dim, 1024);
    assert_eq!(checkpoint.config().dropout, 0.0);
    assert_eq!(checkpoint.tensor_count(), 65);
    assert_eq!(checkpoint.total_elements(), 3_364_096);

    let samples = (0..1360)
        .map(|index| (index as f32 - 680.0) / 680.0)
        .collect::<Vec<_>>();
    let waveform = Tensor::from_data(TensorData::new(samples, [1, 1360]), &device);
    let output = model.forward(waveform);
    assert_eq!(output.dims(), [1, 4, 1024]);
    let values = output.into_data().to_vec::<f32>().unwrap();
    assert!(values.iter().all(|value| value.is_finite()));

    let after = std::fs::metadata(&path).unwrap();
    assert_eq!(after.len(), before.len());
    assert_eq!(checkpoint.source_sha256(), EXPECTED_SHA256);
}
