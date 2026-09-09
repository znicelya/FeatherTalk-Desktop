use std::{
    rc::Rc,
    sync::atomic::{AtomicU64, Ordering},
};

use burn::{
    module::ParamId,
    tensor::{DType, Shape, TensorData},
};
use burn_store::TensorSnapshot;
use feathertalk_export::onnx::{
    InitializerSet, ONNX_FLOAT_DATA_TYPE, OnnxExportError, OnnxTensorProto,
    add_snapshot_initializers, initializer_from_snapshot,
};
use prost::Message;

fn snapshot_f32(name: &str, values: Vec<f32>, shape: Vec<usize>) -> TensorSnapshot {
    TensorSnapshot::from_data(
        TensorData::new(values, shape),
        name.split('.').map(str::to_owned).collect(),
        Vec::new(),
        ParamId::new(),
    )
}

fn snapshot_i32(name: &str, values: Vec<i32>, shape: Vec<usize>) -> TensorSnapshot {
    TensorSnapshot::from_data(
        TensorData::new(values, shape),
        name.split('.').map(str::to_owned).collect(),
        Vec::new(),
        ParamId::new(),
    )
}

#[test]
fn initializer_encodes_f32_values_as_little_endian_raw_data() {
    let snapshot = snapshot_f32("encoder.weight", vec![1.0, -2.5, 0.0], vec![1, 3]);

    let initializer = initializer_from_snapshot(&snapshot).unwrap();

    assert_eq!(initializer.name, "encoder.weight");
    assert_eq!(initializer.dims, vec![1, 3]);
    assert_eq!(initializer.data_type, ONNX_FLOAT_DATA_TYPE);
    assert_eq!(
        initializer.raw_data,
        [
            1.0_f32.to_le_bytes(),
            (-2.5_f32).to_le_bytes(),
            0.0_f32.to_le_bytes()
        ]
        .concat()
    );

    let decoded = OnnxTensorProto::decode(initializer.encode_to_vec().as_slice()).unwrap();
    assert_eq!(decoded, initializer);
}

#[test]
fn initializer_rejects_snapshot_with_non_f32_dtype() {
    let snapshot = snapshot_i32("encoder.indices", vec![1, 2], vec![2]);

    assert!(matches!(
        initializer_from_snapshot(&snapshot),
        Err(OnnxExportError::NonF32Initializer { .. })
    ));
}

#[test]
fn initializer_rejects_materialized_shape_mismatch() {
    let snapshot = TensorSnapshot::from_closure(
        Rc::new(|| Ok(TensorData::new(vec![1.0_f32], [1]))),
        DType::F32,
        Shape::new([2]),
        vec!["broken".to_owned()],
        Vec::new(),
        ParamId::new(),
    );

    assert!(matches!(
        initializer_from_snapshot(&snapshot),
        Err(OnnxExportError::SnapshotShapeMismatch { .. })
    ));
}

#[test]
fn initializer_set_is_sorted_and_rejects_duplicate_names() {
    let first = snapshot_f32("z.weight", vec![1.0], vec![1]);
    let second = snapshot_f32("a.weight", vec![2.0], vec![1]);
    let duplicate = snapshot_f32("z.weight", vec![3.0], vec![1]);
    let mut set = InitializerSet::new();

    add_snapshot_initializers(&mut set, [&first, &second]).unwrap();
    let names = set
        .iter()
        .map(|tensor| tensor.name.as_str())
        .collect::<Vec<_>>();
    assert_eq!(names, ["a.weight", "z.weight"]);
    assert_eq!(set.len(), 2);

    assert!(matches!(
        add_snapshot_initializers(&mut set, [&duplicate]),
        Err(OnnxExportError::DuplicateInitializer { name }) if name == "z.weight"
    ));
}

#[test]
fn initializer_rejects_element_count_mismatch_without_partial_output() {
    static CALLS: AtomicU64 = AtomicU64::new(0);
    let snapshot = TensorSnapshot::from_closure(
        Rc::new(|| {
            CALLS.fetch_add(1, Ordering::Relaxed);
            Ok(TensorData::from_bytes_vec(vec![0; 4], [2], DType::F32))
        }),
        DType::F32,
        Shape::new([1]),
        vec!["count".to_owned()],
        Vec::new(),
        ParamId::new(),
    );

    assert!(matches!(
        initializer_from_snapshot(&snapshot),
        Err(OnnxExportError::SnapshotShapeMismatch { .. })
    ));
    assert!(CALLS.load(Ordering::Relaxed) > 0);
}
