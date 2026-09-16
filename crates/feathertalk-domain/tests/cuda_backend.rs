use feathertalk_domain::Backend;

#[test]
fn cuda_backend_survives_protocol_round_trip() {
    let backend = serde_json::from_str::<Backend>(r#""cuda""#);
    assert!(
        backend.is_ok(),
        "CUDA adapters must be understood: {backend:?}"
    );
    assert_eq!(
        serde_json::to_string(&backend.unwrap()).unwrap(),
        r#""cuda""#
    );
}

#[test]
fn rocm_backend_survives_protocol_round_trip() {
    let backend = serde_json::from_str::<Backend>(r#""rocm""#);
    assert!(
        backend.is_ok(),
        "ROCm adapters must be understood: {backend:?}"
    );
    assert_eq!(
        serde_json::to_string(&backend.unwrap()).unwrap(),
        r#""rocm""#
    );
}

#[test]
fn automatic_compute_prefers_cuda_then_rocm_then_wgpu() {
    assert!(
        Backend::Cuda.selection_priority() < Backend::Rocm.selection_priority(),
        "CUDA must remain the highest-priority native backend"
    );
    assert!(
        Backend::Rocm.selection_priority() < Backend::Wgpu.selection_priority(),
        "a certified ROCm device must outrank the generic wgpu path"
    );
}

#[test]
fn automatic_compute_choice_can_be_serialized() {
    let backend = serde_json::from_str::<Backend>(r#""auto""#);
    assert!(
        backend.is_ok(),
        "automatic selection must be accepted: {backend:?}"
    );
    assert_eq!(
        serde_json::to_string(&backend.unwrap()).unwrap(),
        r#""auto""#
    );
}
