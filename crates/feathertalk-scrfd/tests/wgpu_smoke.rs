mod support;

use burn::{
    backend::Wgpu,
    tensor::{Tensor, TensorData},
};
use feathertalk_scrfd::ScrfdModel;

type GpuBackend = Wgpu<f32, i32, u32>;

#[test]
#[ignore = "requires a compatible WGPU adapter"]
fn committed_scrfd_artifact_runs_on_wgpu() {
    let fixture = support::load_and_verify_fixture().unwrap();
    let input = support::read_array(&fixture.root.join("input.npy")).unwrap();
    let device = Default::default();
    let tensor = Tensor::<GpuBackend, 4>::from_data(
        TensorData::new(
            input.iter().copied().collect::<Vec<_>>(),
            input.shape().to_vec(),
        ),
        &device,
    );
    let model = ScrfdModel::<GpuBackend>::load(&support::artifact_paths(), &device).unwrap();
    let output = model.forward(tensor).unwrap();
    for (level, anchors) in output.levels.into_iter().zip([12_800, 3_200, 800]) {
        assert_eq!(level.scores.dims(), [1, anchors]);
        assert_eq!(level.bbox_deltas.dims(), [1, anchors, 4]);
        assert_eq!(level.keypoint_deltas.dims(), [1, anchors, 10]);
    }
}
