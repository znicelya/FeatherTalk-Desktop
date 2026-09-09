mod support;

use burn::{
    backend::NdArray,
    tensor::{Tensor, TensorData},
};
use feathertalk_scrfd::ScrfdModel;

type CpuBackend = NdArray<f32>;

#[test]
fn all_nine_outputs_match_opencv_cpu() {
    let fixture = support::load_and_verify_fixture().unwrap();
    let input = support::read_array(&fixture.root.join("input.npy")).unwrap();
    assert_eq!(input.shape(), &[1, 3, 640, 640]);

    let device = Default::default();
    let tensor = Tensor::<CpuBackend, 4>::from_data(
        TensorData::new(
            input.iter().copied().collect::<Vec<_>>(),
            input.shape().to_vec(),
        ),
        &device,
    );
    let model = ScrfdModel::<CpuBackend>::load(&support::artifact_paths(), &device).unwrap();
    let output = model.forward(tensor).unwrap();

    for (level_index, level) in output.levels.into_iter().enumerate() {
        let score_name = format!("out{level_index}.npy");
        let bbox_name = format!("out{}.npy", level_index + 3);
        let keypoint_name = format!("out{}.npy", level_index + 6);
        support::assert_cpu_tensor_matches_fixture(level.scores, &fixture.root.join(score_name));
        support::assert_cpu_tensor_matches_fixture(
            level.bbox_deltas,
            &fixture.root.join(bbox_name),
        );
        support::assert_cpu_tensor_matches_fixture(
            level.keypoint_deltas,
            &fixture.root.join(keypoint_name),
        );
    }
}

#[test]
fn parity_metric_rejects_non_finite_values() {
    assert!(support::compare_f32(&[f32::NAN], &[0.0]).is_err());
    assert!(support::compare_f32(&[0.0], &[f32::INFINITY]).is_err());
}
