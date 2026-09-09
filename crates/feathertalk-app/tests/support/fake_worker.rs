use std::io::{BufReader, Write};

use feathertalk_domain::{
    AdapterInfo, AdapterKind, Backend, Capabilities, ClientFrame, FrameReader, PROTOCOL_VERSION,
    ReadyFrame, ServerFrame, TaskKind, encode_line,
};

fn main() {
    // The test copies this executable into its own temporary directory so every
    // probe has isolated evidence without mutating the test process environment.
    let executable = std::env::current_exe().unwrap();
    let directory = executable.parent().unwrap();
    std::fs::write(
        directory.join("probe-env.json"),
        serde_json::to_vec(&serde_json::json!({
            "backend": std::env::var("FEATHERTALK_WORKER_BACKEND").ok(),
            "adapter": std::env::var("FEATHERTALK_WORKER_ADAPTER").ok(),
        }))
        .unwrap(),
    )
    .unwrap();
    let frame = ServerFrame::Ready(ReadyFrame {
        protocol_version: PROTOCOL_VERSION,
        worker_version: "app-probe-test".into(),
        backends: vec![Backend::Cpu, Backend::Wgpu],
        adapters: vec![
            AdapterInfo {
                id: "cpu-0".into(),
                name: "Probe CPU".into(),
                backend: Backend::Cpu,
                kind: AdapterKind::Cpu,
                certified: true,
                vram_bytes: None,
            },
            AdapterInfo {
                id: "wgpu-fixture".into(),
                name: "Fixture GPU (Vulkan)".into(),
                backend: Backend::Wgpu,
                kind: AdapterKind::Discrete,
                certified: true,
                vram_bytes: Some(8 * 1024 * 1024 * 1024),
            },
            AdapterInfo {
                id: "wgpu-experimental".into(),
                name: "Fixture adapter (D3D12)".into(),
                backend: Backend::Wgpu,
                kind: AdapterKind::Integrated,
                certified: false,
                vram_bytes: None,
            },
        ],
        supported_commands: vec![
            TaskKind::ValidateProject,
            TaskKind::Train,
            TaskKind::Render,
            TaskKind::ExtractFrames,
            TaskKind::ExtractFeatures,
        ],
        capabilities: Capabilities {
            training: true,
            wgpu_training: true,
            onnx_validation: false,
            ffmpeg: false,
        },
    });
    println!("{}", encode_line(&frame).unwrap().trim_end());
    std::io::stdout().flush().unwrap();
    let mut reader = FrameReader::new(BufReader::new(std::io::stdin().lock()));
    match reader.read_frame::<ClientFrame>() {
        Some(Ok(ClientFrame::Shutdown(_))) => {
            std::fs::write(directory.join("shutdown.txt"), "shutdown").unwrap();
        }
        _ => std::process::exit(9),
    }
}
