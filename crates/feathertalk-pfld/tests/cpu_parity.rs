use std::{
    fs,
    path::{Path, PathBuf},
};

use burn::{
    backend::NdArray,
    tensor::{Tensor, TensorData},
};
use feathertalk_pfld::PfldRuntime;
use sha2::{Digest, Sha256};

type CpuBackend = NdArray<f32>;

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/pytorch_cpu_v1")
}

fn read_f32(path: &Path) -> Vec<f32> {
    fs::read(path)
        .unwrap()
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect()
}

fn compare(actual: &[f32], expected: &[f32]) -> (f32, f32) {
    assert_eq!(actual.len(), expected.len());
    let mut max_abs = 0.0_f32;
    let mut sum_abs = 0.0_f32;
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            actual.is_finite() && expected.is_finite(),
            "non-finite at {index}"
        );
        let difference = (actual - expected).abs();
        max_abs = max_abs.max(difference);
        sum_abs += difference;
    }
    (max_abs, sum_abs / actual.len() as f32)
}

/// Constructing the GhostOne graph moves a 125 768-byte runtime struct through
/// several frames, which overruns the default libtest thread stack on Windows
/// and aborts the whole test binary with `STATUS_STACK_OVERFLOW`.
/// `feathertalk-weights` already solves the same problem for its detached clone
/// with a dedicated 64 MiB stack; this mirrors that here.
const RUNTIME_LOAD_STACK_BYTES: usize = 64 * 1024 * 1024;

/// Runs `body` on a thread whose stack is large enough for the artifact load.
/// Panics travel back through `join`, so failed assertions still fail the test.
fn on_load_stack(name: &str, body: impl FnOnce() + Send + 'static) {
    std::thread::Builder::new()
        .name(name.to_owned())
        .stack_size(RUNTIME_LOAD_STACK_BYTES)
        .spawn(body)
        .expect("the loader thread starts")
        .join()
        .expect("the loader thread does not panic");
}

#[test]
fn committed_pfld_runtime_matches_python_on_all_220_cpu_outputs() {
    on_load_stack("pfld-cpu-parity", || {
        let root = fixture_dir();
        let input_values = read_f32(&root.join("input.f32"));
        let expected = read_f32(&root.join("output.f32"));
        let device = Default::default();
        let runtime = PfldRuntime::<CpuBackend>::load(
            &Path::new(env!("CARGO_MANIFEST_DIR")).join("artifacts/pfld_ghost_one"),
            &device,
        )
        .unwrap();
        let input = Tensor::<CpuBackend, 4>::from_data(
            TensorData::new(input_values, [1, 3, 192, 192]),
            &device,
        );
        let actual = runtime
            .forward(input)
            .unwrap()
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        let (max_abs, mean_abs) = compare(&actual, &expected);
        assert!(max_abs <= 1e-4, "max_abs={max_abs}, mean_abs={mean_abs}");
    });
}

#[test]
fn fixture_hash_change_is_detectable_before_forward() {
    let root = fixture_dir();
    let bytes = fs::read(root.join("output.f32")).unwrap();
    let hash = hex::encode(Sha256::digest(&bytes));
    let manifest: serde_json::Value =
        serde_json::from_slice(&fs::read(root.join("fixture.json")).unwrap()).unwrap();
    assert_eq!(
        hash,
        manifest["files"]["output.f32"]["sha256"].as_str().unwrap()
    );
}
