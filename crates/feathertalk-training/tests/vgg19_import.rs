use std::{fs, io, path::PathBuf};

use burn::{nn::conv::Conv2dConfig, tensor::backend::Backend};
use burn_store::ModuleSnapshot;
use feathertalk_training::Vgg19Conv3_3;
use feathertalk_weights::{LegacyImportRequest, LegacyModelKind, WeightImportError, import_into};
use zip::ZipArchive;

type CpuBackend = burn::backend::NdArray<f32>;

#[test]
fn vgg19_direct_state_imports_the_exact_truncated_tensor_set() {
    let (_temp, fixture) = extract_fixture("vgg19-direct.pth");
    let device = Default::default();
    let mut model = Vgg19Conv3_3::<CpuBackend>::new_for_import(&device);

    let report = import_into::<CpuBackend, _>(&mut model, &request_for(fixture)).unwrap();

    assert_eq!(report.applied.len(), 14);
    assert_eq!(report.ignored.len(), 24);
    assert_eq!(report.tensor_count, 14);
    assert_eq!(report.total_elements, 1_735_488);
    assert_eq!(
        model
            .conv1_1
            .weight
            .val()
            .to_data()
            .to_vec::<f32>()
            .unwrap()[0],
        0.001
    );
    assert_eq!(
        model
            .conv3_3
            .bias
            .as_ref()
            .unwrap()
            .val()
            .to_data()
            .to_vec::<f32>()
            .unwrap()[0],
        -0.07
    );
}

#[test]
fn vgg19_unexpected_tensor_is_rejected_without_mutating_the_module() {
    let (_temp, fixture) = extract_fixture("vgg19-unexpected.pth");
    let device = Default::default();
    let mut model = Vgg19Conv3_3::<CpuBackend>::new_for_import(&device);
    materialize_model(&model);
    let before = model.clone();

    let error = import_into::<CpuBackend, _>(&mut model, &request_for(fixture)).unwrap_err();

    assert!(
        matches!(error, WeightImportError::UnexpectedTensor(key) if key == "unexpected.weight")
    );
    assert_module_snapshots_equal(&before, &model);
}

#[test]
fn vgg19_shape_mismatch_is_rejected_without_mutating_the_module() {
    let (_temp, fixture) = extract_fixture("vgg19-direct.pth");
    let device = Default::default();
    let mut model = Vgg19Conv3_3::<CpuBackend>::new_for_import(&device);
    model.conv1_1 = Conv2dConfig::new([3, 63], [3, 3])
        .with_bias(true)
        .init(&device);
    materialize_model(&model);
    let before = model.clone();

    let error = import_into::<CpuBackend, _>(&mut model, &request_for(fixture)).unwrap_err();

    assert!(matches!(error, WeightImportError::ShapeMismatch(path) if path == "conv1_1.weight"));
    assert_module_snapshots_equal(&before, &model);
}

fn extract_fixture(member: &str) -> (tempfile::TempDir, PathBuf) {
    let temp = tempfile::tempdir().unwrap();
    let destination = temp.path().join(member);
    let archive_path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/golden/vgg19-import-v1.zip");
    let archive = fs::File::open(archive_path).unwrap();
    let mut archive = ZipArchive::new(archive).unwrap();
    let mut source = archive.by_name(member).unwrap();
    let mut destination_file = fs::File::create(&destination).unwrap();
    io::copy(&mut source, &mut destination_file).unwrap();
    (temp, destination)
}

fn request_for(path: PathBuf) -> LegacyImportRequest {
    LegacyImportRequest {
        path,
        kind: LegacyModelKind::Vgg19Conv3_3,
        top_level_key: None,
        max_file_bytes: 16 * 1024 * 1024,
        max_tensor_count: 64,
        max_total_elements: 2_000_000,
    }
}

fn materialize_model<B: Backend>(model: &Vgg19Conv3_3<B>) {
    for conv in [
        &model.conv1_1,
        &model.conv1_2,
        &model.conv2_1,
        &model.conv2_2,
        &model.conv3_1,
        &model.conv3_2,
        &model.conv3_3,
    ] {
        let _ = conv.weight.val();
        let _ = conv.bias.as_ref().unwrap().val();
    }
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
