use std::{
    fs, io,
    path::{Path, PathBuf},
    sync::{Mutex, OnceLock},
};

use burn::{
    backend::NdArray,
    nn::{Linear, LinearConfig},
    tensor::backend::Backend,
};
use burn_store::{ModuleSnapshot, SafetensorsStore};
use feathertalk_weights::{
    LegacyImportRequest, LegacyModelKind, WeightImportError, import_into, is_known_ignored_key,
    save_safetensors,
};
use sha2::{Digest, Sha256};
use zip::ZipArchive;

type CpuBackend = NdArray<f32>;

#[test]
fn nested_model_checkpoint_loads_all_expected_tensors() {
    let fixture = extract_fixture("weights/tiny_nested.pth");
    let device = Default::default();
    let mut model = LinearConfig::new(2, 2).init::<CpuBackend>(&device);
    let report =
        import_into::<CpuBackend, _>(&mut model, &request_for(fixture, Some("model"))).unwrap();
    assert_eq!(report.applied.len(), 2);
    assert!(report.ignored.is_empty());
    assert_eq!(report.tensor_count, 2);
    assert_eq!(report.total_elements, 6);
    assert_eq!(
        model.weight.val().to_data().to_vec::<f32>().unwrap(),
        vec![1.0, 3.0, 2.0, 4.0]
    );
    assert_eq!(
        model
            .bias
            .as_ref()
            .unwrap()
            .val()
            .to_data()
            .to_vec::<f32>()
            .unwrap(),
        vec![0.25, -0.5]
    );
}

#[test]
fn direct_state_dict_is_detected() {
    let fixture = extract_fixture("weights/tiny_direct.pth");
    let device = Default::default();
    let mut model = LinearConfig::new(2, 2).init::<CpuBackend>(&device);
    let report = import_into::<CpuBackend, _>(&mut model, &request_for(fixture, None)).unwrap();
    assert_eq!(report.applied.len(), 2);
}

#[test]
fn missing_requested_key_falls_back_to_model() {
    let fixture = extract_fixture("weights/tiny_nested.pth");
    let device = Default::default();
    let mut model = LinearConfig::new(2, 2).init::<CpuBackend>(&device);
    let report =
        import_into::<CpuBackend, _>(&mut model, &request_for(fixture, Some("not_present")))
            .unwrap();
    assert_eq!(report.applied.len(), 2);
}

#[test]
fn report_hash_matches_the_source_checkpoint() {
    let fixture = extract_fixture("weights/tiny_nested.pth");
    let expected_hash = hex::encode(Sha256::digest(fs::read(&fixture).unwrap()));
    let device = Default::default();
    let mut model = LinearConfig::new(2, 2).init::<CpuBackend>(&device);
    let report =
        import_into::<CpuBackend, _>(&mut model, &request_for(fixture, Some("model"))).unwrap();
    assert_eq!(report.source_sha256, expected_hash);
}

#[test]
fn missing_tensor_is_rejected() {
    let fixture = extract_fixture("weights/tiny_missing.pth");
    let device = Default::default();
    let mut model = LinearConfig::new(2, 2).init::<CpuBackend>(&device);
    let _ = model.weight.val();
    let _ = model.bias.as_ref().unwrap().val();
    let before = model.clone();
    let error =
        import_into::<CpuBackend, _>(&mut model, &request_for(fixture, Some("model"))).unwrap_err();
    assert!(matches!(error, WeightImportError::MissingTensor(_)));
    assert_module_snapshots_equal(&before, &model);
}

#[test]
fn unexpected_tensor_is_rejected() {
    let fixture = extract_fixture("weights/tiny_unexpected.pth");
    let device = Default::default();
    let mut model = LinearConfig::new(2, 2).init::<CpuBackend>(&device);
    let error =
        import_into::<CpuBackend, _>(&mut model, &request_for(fixture, Some("model"))).unwrap_err();
    assert!(matches!(error, WeightImportError::UnexpectedTensor(_)));
}

#[test]
fn shape_mismatch_is_rejected_without_mutating_the_module() {
    let fixture = extract_fixture("weights/tiny_nested.pth");
    let device = Default::default();
    let mut model = LinearConfig::new(3, 2).init::<CpuBackend>(&device);
    let _ = model.weight.val();
    let _ = model.bias.as_ref().unwrap().val();
    let before = model.clone();

    let error =
        import_into::<CpuBackend, _>(&mut model, &request_for(fixture, Some("model"))).unwrap_err();
    assert!(matches!(error, WeightImportError::ShapeMismatch(_)));
    assert_module_snapshots_equal(&before, &model);
}

#[test]
fn num_batches_tracked_is_ignored_as_a_known_buffer() {
    assert!(is_known_ignored_key("encoder.0.bn.num_batches_tracked"));
    assert!(!is_known_ignored_key("encoder.0.bn.running_mean"));
}

#[test]
fn imported_module_round_trips_through_safetensors() {
    let fixture = extract_fixture("weights/tiny_nested.pth");
    let device = Default::default();
    let mut first = LinearConfig::new(2, 2).init::<CpuBackend>(&device);
    import_into::<CpuBackend, _>(&mut first, &request_for(fixture, Some("model"))).unwrap();

    let temp = tempfile::tempdir().unwrap();
    let safe = temp.path().join("tiny.safetensors");
    save_safetensors::<CpuBackend, _>(&first, &safe).unwrap();
    let second = load_linear_safetensors::<CpuBackend>(&safe, &device).unwrap();
    assert_module_snapshots_equal(&first, &second);
}

#[test]
fn configured_tensor_limits_are_enforced_before_apply() {
    let fixture = extract_fixture("weights/tiny_nested.pth");
    let device = Default::default();
    let mut model = LinearConfig::new(2, 2).init::<CpuBackend>(&device);

    let mut count_request = request_for(fixture.clone(), Some("model"));
    count_request.max_tensor_count = 1;
    assert!(matches!(
        import_into::<CpuBackend, _>(&mut model, &count_request),
        Err(WeightImportError::UnsafeLimit(_))
    ));

    let mut element_request = request_for(fixture, Some("model"));
    element_request.max_total_elements = 5;
    assert!(matches!(
        import_into::<CpuBackend, _>(&mut model, &element_request),
        Err(WeightImportError::UnsafeLimit(_))
    ));
}

#[test]
fn source_file_limit_is_enforced_before_checkpoint_parse() {
    let fixture = extract_fixture("weights/tiny_nested.pth");
    let source_length = fs::metadata(&fixture).unwrap().len();
    let device = Default::default();
    let mut model = LinearConfig::new(2, 2).init::<CpuBackend>(&device);
    let mut request = request_for(fixture, Some("model"));
    request.max_file_bytes = source_length - 1;

    assert!(matches!(
        import_into::<CpuBackend, _>(&mut model, &request),
        Err(WeightImportError::UnsafeLimit(_))
    ));
}

#[test]
fn checkpoint_without_tensors_is_rejected_as_unsupported() {
    let temp = tempfile::tempdir().unwrap();
    let checkpoint = temp.path().join("empty-state-dict.pth");
    fs::write(&checkpoint, [0x80, 0x02, b'}', b'q', 0x00, b'.']).unwrap();
    let device = Default::default();
    let mut model = LinearConfig::new(2, 2).init::<CpuBackend>(&device);

    let error =
        import_into::<CpuBackend, _>(&mut model, &request_for(checkpoint, None)).unwrap_err();
    assert!(
        matches!(error, WeightImportError::UnsupportedStructure(_)),
        "{error:?}"
    );
}

fn extract_fixture(member: &str) -> PathBuf {
    static FIXTURE_DIR: OnceLock<PathBuf> = OnceLock::new();
    static EXTRACTION_LOCK: Mutex<()> = Mutex::new(());

    let directory = FIXTURE_DIR.get_or_init(|| {
        let directory = std::env::temp_dir().join(format!(
            "feathertalk-weights-legacy-import-{}",
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

fn request_for(path: PathBuf, top_level_key: Option<&str>) -> LegacyImportRequest {
    LegacyImportRequest {
        path,
        kind: LegacyModelKind::FeatherHubert,
        top_level_key: top_level_key.map(str::to_owned),
        max_file_bytes: 4 * 1024 * 1024 * 1024,
        max_tensor_count: 10_000,
        max_total_elements: 2_000_000_000,
    }
}

fn load_linear_safetensors<B: Backend>(
    path: &Path,
    device: &B::Device,
) -> Result<Linear<B>, burn_store::SafetensorsStoreError> {
    let mut model = LinearConfig::new(2, 2).init(device);
    let mut store = SafetensorsStore::from_file(path);
    model.load_from(&mut store)?;
    Ok(model)
}

fn assert_module_snapshots_equal<B: Backend, M: ModuleSnapshot<B>>(first: &M, second: &M) {
    let first = first.collect(None, None, false);
    let second = second.collect(None, None, false);
    assert_eq!(first.len(), second.len());

    for (first, second) in first.iter().zip(second.iter()) {
        assert_eq!(first.full_path(), second.full_path());
        assert_eq!(first.shape, second.shape);
        assert_eq!(first.dtype, second.dtype);
        assert_eq!(first.to_data().unwrap(), second.to_data().unwrap());
    }
}
