use burn::{
    backend::wgpu::{CubeBackend, WgpuRuntime},
    tensor::{Tensor, TensorData},
};
use cubek::{
    matmul::{
        definition::{MatmulElems, MatmulGlobalElems},
        launch::{Strategy, launch_ref},
        routines::{BlueprintStrategy, TileSizeSelection, double_unit::DoubleUnitSelectionArgs},
    },
    std::InputBinding,
};
use feathertalk_worker::ComputeRegistry;

/// SCRFD's 24-channel pointwise convolution exposed a double-buffered matmul
/// reading eight values from the next row when it rounded three K stages to four.
/// Force that kernel so this regression does not depend on autotune timings.
#[test]
#[ignore = "requires a certified native WGPU adapter"]
fn double_buffered_matmul_does_not_read_past_odd_k_stage_counts() {
    std::thread::Builder::new()
        .stack_size(64 * 1024 * 1024)
        .spawn(|| {
            type Gpu = CubeBackend<WgpuRuntime, f32, i32, u32>;
            let registry = ComputeRegistry::discover();
            let adapter = registry
                .adapters()
                .iter()
                .find(|adapter| adapter.certified && adapter.id.starts_with("wgpu-"))
                .expect("a certified native GPU is required");
            let context = registry.open_wgpu(&adapter.id).unwrap();
            let device = &context.device;
            let strategy =
                Strategy::DoubleUnit(BlueprintStrategy::Inferred(DoubleUnitSelectionArgs {
                    tile_size: TileSizeSelection::MaxTileSize,
                }));

            // Partial stages, odd whole-stage counts, and an even-count control.
            // Ones make the independently known dot product exactly K in FP32.
            for k in [24, 40, 17, 32] {
                let lhs = Tensor::<Gpu, 2>::from_data(
                    TensorData::new(vec![1.0_f32; 6_400 * k], [6_400, k]),
                    device,
                )
                .into_primitive()
                .tensor();
                let rhs = Tensor::<Gpu, 2>::from_data(
                    TensorData::new(vec![1.0_f32; 48 * k], [48, k]),
                    device,
                )
                .transpose()
                .into_primitive()
                .tensor();
                let output = Tensor::<Gpu, 2>::empty([6_400, 48], device);
                let out = output.clone().into_primitive().tensor();
                let mut dtypes = MatmulElems::from_globals(&MatmulGlobalElems {
                    lhs: lhs.dtype.into(),
                    rhs: rhs.dtype.into(),
                    out: out.dtype.into(),
                });
                let client = lhs.client.clone();
                launch_ref::<WgpuRuntime>(
                    &strategy,
                    &client,
                    InputBinding::new(lhs.binding(), dtypes.lhs_global),
                    InputBinding::new(rhs.binding(), dtypes.rhs_global),
                    out.binding(),
                    &mut dtypes,
                )
                .unwrap();

                for (index, value) in output
                    .into_data()
                    .into_vec::<f32>()
                    .unwrap()
                    .into_iter()
                    .enumerate()
                {
                    assert_eq!(value, k as f32, "K={k}, output element {index}");
                }
            }
            context.check().unwrap();
        })
        .unwrap()
        .join()
        .unwrap();
}

#[test]
#[ignore = "requires a certified native WGPU adapter"]
fn double_buffered_inner_product_does_not_read_the_next_batch() {
    std::thread::Builder::new()
        .stack_size(64 * 1024 * 1024)
        .spawn(|| {
            type Gpu = CubeBackend<WgpuRuntime, f32, i32, u32>;
            let registry = ComputeRegistry::discover();
            let adapter = registry
                .adapters()
                .iter()
                .find(|adapter| adapter.certified && adapter.id.starts_with("wgpu-"))
                .expect("a certified native GPU is required");
            let context = registry.open_wgpu(&adapter.id).unwrap();
            let device = &context.device;
            let strategy = Strategy::DoubleVecMat(BlueprintStrategy::Inferred(().into()));

            // Three 128-element stages on a 32-lane device. A second batch
            // makes an unchecked extra stage read real values from that batch.
            let lhs = Tensor::<Gpu, 3>::from_data(
                TensorData::new(vec![1.0_f32; 2 * 384], [2, 1, 384]),
                device,
            )
            .into_primitive()
            .tensor();
            let rhs = Tensor::<Gpu, 3>::from_data(
                TensorData::new(vec![1.0_f32; 2 * 384], [2, 1, 384]),
                device,
            )
            .transpose()
            .into_primitive()
            .tensor();
            let output = Tensor::<Gpu, 3>::empty([2, 1, 1], device);
            let out = output.clone().into_primitive().tensor();
            let mut dtypes = MatmulElems::from_globals(&MatmulGlobalElems {
                lhs: lhs.dtype.into(),
                rhs: rhs.dtype.into(),
                out: out.dtype.into(),
            });
            let client = lhs.client.clone();
            launch_ref::<WgpuRuntime>(
                &strategy,
                &client,
                InputBinding::new(lhs.binding(), dtypes.lhs_global),
                InputBinding::new(rhs.binding(), dtypes.rhs_global),
                out.binding(),
                &mut dtypes,
            )
            .unwrap();
            assert_eq!(
                output.into_data().into_vec::<f32>().unwrap(),
                [384.0, 384.0]
            );
            context.check().unwrap();
        })
        .unwrap()
        .join()
        .unwrap();
}
