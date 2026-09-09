use feathertalk_parity::{
    archive::GoldenArchive,
    fixture::{ForwardCase, run_wgpu_forward, run_wgpu_train_step},
    probe::GraphicsSelection,
};

fn golden_archive() -> GoldenArchive {
    let root = env!("CARGO_MANIFEST_DIR");
    GoldenArchive::open(format!("{root}/../../tests/golden/burn-feasibility-v1.zip")).unwrap()
}

fn certified_graphics() -> GraphicsSelection {
    #[cfg(target_os = "windows")]
    return GraphicsSelection::Vulkan;
    #[cfg(target_os = "macos")]
    return GraphicsSelection::Metal;
    #[cfg(target_os = "linux")]
    return GraphicsSelection::Vulkan;
    #[allow(unreachable_code)]
    GraphicsSelection::Auto
}

#[test]
#[ignore = "requires a certified WGPU adapter"]
fn feather_matches_python_on_wgpu() {
    let result = run_wgpu_forward(
        &golden_archive(),
        ForwardCase::FeatherMicro,
        certified_graphics(),
    )
    .unwrap();
    assert_eq!(result.execution.backend, "wgpu");
    assert!(!result.execution.used_cpu_fallback);
    assert!(result.metrics.max_abs <= 1e-3, "{result:?}");
}

#[test]
#[ignore = "requires a certified WGPU adapter"]
fn production_unet_matches_python_on_wgpu() {
    let result = run_wgpu_forward(
        &golden_archive(),
        ForwardCase::UnetProduction,
        certified_graphics(),
    )
    .unwrap();
    assert_eq!(result.execution.backend, "wgpu");
    assert!(!result.execution.used_cpu_fallback);
    assert!(result.metrics.max_abs <= 1e-3, "{result:?}");
}

#[test]
#[ignore = "requires a certified WGPU adapter with training capacity"]
fn production_unet_completes_one_adam_step_on_wgpu() {
    let result = run_wgpu_train_step(&golden_archive(), certified_graphics(), true).unwrap();
    assert_eq!(result.execution.backend, "wgpu");
    assert!(!result.execution.used_cpu_fallback);
    assert!(result.initial_loss.is_finite());
    assert!(result.gradient_norm.is_finite());
    assert!(result.gradient_norm > 0.0);
    assert!(result.output_weight_changed);
}
