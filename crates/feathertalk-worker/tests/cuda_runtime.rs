use std::process::{Command, Stdio};

use feathertalk_domain::{Backend, FrameReader, ServerFrame};

#[test]
fn missing_cuda_toolkit_keeps_the_worker_and_existing_backends_available() {
    let directory = tempfile::tempdir().unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_feathertalk-worker"))
        .env("CUDA_PATH", directory.path().join("missing-toolkit"))
        .env("FEATHERTALK_WORKER_BACKEND", "auto")
        .env_remove("FEATHERTALK_WORKER_ADAPTER")
        .stdin(Stdio::null())
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let mut reader = FrameReader::new(output.stdout.as_slice());
    let ServerFrame::Ready(ready) = reader.read_frame::<ServerFrame>().unwrap().unwrap() else {
        panic!("worker must still complete the handshake without CUDA");
    };
    assert!(ready.backends.contains(&Backend::Cpu));
    assert!(!ready.backends.contains(&Backend::Cuda));
    assert!(
        !ready
            .adapters
            .iter()
            .any(|adapter| adapter.backend == Backend::Cuda)
    );
}
