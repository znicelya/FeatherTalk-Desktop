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
