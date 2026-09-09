use std::{fs::OpenOptions, path::Path};

use burn::{
    backend::NdArray,
    tensor::{Tensor, TensorData},
};
use feathertalk_pfld::{PfldRuntime, PfldRuntimeError};

type CpuBackend = NdArray<f32>;

fn artifact_dir() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("artifacts/pfld_ghost_one")
}

fn copy_artifact() -> tempfile::TempDir {
    let temp = tempfile::tempdir().unwrap();
    for name in ["manifest.json", "model.safetensors"] {
        std::fs::copy(artifact_dir().join(name), temp.path().join(name)).unwrap();
    }
    temp
}

/// Constructing the GhostOne graph moves a 125 768-byte runtime struct through
/// several frames, which overruns the default libtest thread stack on Windows
/// and aborts the whole test binary with `STATUS_STACK_OVERFLOW`.
/// `feathertalk-weights` already solves the same problem for its detached clone
/// with a dedicated 64 MiB stack; this mirrors that here. The rejection tests
/// below stay on the libtest thread because they fail before construction.
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
fn committed_artifact_loads_and_runs_the_fixed_cpu_contract() {
    on_load_stack("pfld-cpu-contract", || {
        let device = Default::default();
        let runtime = PfldRuntime::<CpuBackend>::load(&artifact_dir(), &device).unwrap();
        assert_eq!(runtime.manifest().schema_version, 1);
        assert_eq!(runtime.tensor_count(), 1735);

        for shape in [
            [2, 3, 192, 192],
            [1, 1, 192, 192],
            [1, 3, 191, 192],
            [1, 3, 192, 191],
        ] {
            let input = Tensor::<CpuBackend, 4>::zeros(shape, &device);
            assert!(matches!(
                runtime.forward(input),
                Err(PfldRuntimeError::InvalidInputShape { actual }) if actual == shape
            ));
        }

        let mut values = vec![0.0_f32; 3 * 192 * 192];
        values[0] = f32::NAN;
        let input =
            Tensor::<CpuBackend, 4>::from_data(TensorData::new(values, [1, 3, 192, 192]), &device);
        assert!(matches!(
            runtime.forward(input),
            Err(PfldRuntimeError::NonFiniteInput)
        ));

        let output = runtime
            .forward(Tensor::<CpuBackend, 4>::zeros([1, 3, 192, 192], &device))
            .unwrap();
        assert_eq!(output.dims(), [1, 220]);
        assert!(
            output
                .into_data()
                .to_vec::<f32>()
                .unwrap()
                .iter()
                .all(|value| value.is_finite())
        );
    });
}

#[test]
fn artifact_loader_rejects_extra_entries_before_model_construction() {
    let temp = copy_artifact();
    std::fs::write(temp.path().join("extra.bin"), b"unexpected").unwrap();
    let device = Default::default();
    assert!(matches!(
        PfldRuntime::<CpuBackend>::load(temp.path(), &device),
        Err(PfldRuntimeError::UnexpectedArtifactEntry(_))
    ));
}

#[test]
fn artifact_loader_rejects_manifest_and_weight_tampering() {
    let device = Default::default();

    let manifest_temp = copy_artifact();
    let mut json: serde_json::Value =
        serde_json::from_slice(&std::fs::read(manifest_temp.path().join("manifest.json")).unwrap())
            .unwrap();
    json["model"]["sha256"] = serde_json::Value::String("0".repeat(64));
    std::fs::write(
        manifest_temp.path().join("manifest.json"),
        serde_json::to_vec(&json).unwrap(),
    )
    .unwrap();
    assert!(matches!(
        PfldRuntime::<CpuBackend>::load(manifest_temp.path(), &device),
        Err(PfldRuntimeError::InvalidManifest { field, .. }) if field == "model.sha256"
    ));

    let weight_temp = copy_artifact();
    let path = weight_temp.path().join("model.safetensors");
    let mut bytes = std::fs::read(&path).unwrap();
    let last = bytes.len() - 1;
    bytes[last] ^= 1;
    std::fs::write(&path, bytes).unwrap();
    assert!(matches!(
        PfldRuntime::<CpuBackend>::load(weight_temp.path(), &device),
        Err(PfldRuntimeError::HashMismatch {
            artifact: "weights",
            ..
        })
    ));
}

#[test]
fn artifact_loader_bounds_manifest_and_weight_reads() {
    let device = Default::default();
    let manifest_temp = copy_artifact();
    let file = OpenOptions::new()
        .write(true)
        .open(manifest_temp.path().join("manifest.json"))
        .unwrap();
    file.set_len(1024 * 1024 + 1).unwrap();
    assert!(matches!(
        PfldRuntime::<CpuBackend>::load(manifest_temp.path(), &device),
        Err(PfldRuntimeError::ManifestTooLarge { .. })
    ));

    let weight_temp = copy_artifact();
    let file = OpenOptions::new()
        .write(true)
        .open(weight_temp.path().join("model.safetensors"))
        .unwrap();
    file.set_len(32 * 1024 * 1024 + 1).unwrap();
    assert!(matches!(
        PfldRuntime::<CpuBackend>::load(weight_temp.path(), &device),
        Err(PfldRuntimeError::WeightsTooLarge { .. })
    ));
}
