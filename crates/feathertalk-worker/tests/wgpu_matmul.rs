use burn::backend::wgpu::WgpuRuntime;
use cubek::{
    cubecl::{
        Runtime,
        ir::{ElemType, FloatKind},
        prelude::TensorBinding,
        std::tensor::TensorHandle,
    },
    matmul::{
        definition::{MatmulElems, MatmulGlobalElems},
        launch::launch_ref,
        multi_level::{
            self,
            routines::{TileSizeSelection, batch::double_unit::DoubleUnitSelectionArgs},
        },
        routine::BlueprintStrategy,
        strategy::Strategy,
    },
    std::InputBinding,
};
use feathertalk_worker::ComputeRegistry;

const F32: ElemType = ElemType::Float(FloatKind::F32);
const F32_SIZE: usize = std::mem::size_of::<f32>();

/// Bytes of an all-ones `f32` buffer, laid out contiguously.
fn ones_bytes(len: usize) -> Vec<u8> {
    1.0_f32
        .to_le_bytes()
        .iter()
        .copied()
        .cycle()
        .take(len * F32_SIZE)
        .collect()
}

/// Decode a device readback into `f32`s.
fn to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(F32_SIZE)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect()
}

/// The transpose of a matrix's contiguous strides swaps its last two axes,
/// giving the column-major read the old `Tensor::transpose` produced.
fn transpose_last_two(strides: &[usize]) -> Vec<usize> {
    let mut strides = strides.to_vec();
    let rank = strides.len();
    strides.swap(rank - 2, rank - 1);
    strides
}

/// Borrow a handle as a launch binding without consuming it.
fn binding_of(handle: &TensorHandle<WgpuRuntime>) -> TensorBinding<WgpuRuntime> {
    handle.clone().binding()
}

/// SCRFD's 24-channel pointwise convolution exposed a double-buffered matmul
/// reading eight values from the next row when it rounded three K stages to four.
/// Force that kernel so this regression does not depend on autotune timings.
#[test]
#[ignore = "requires a certified native WGPU adapter"]
fn double_buffered_matmul_does_not_read_past_odd_k_stage_counts() {
    std::thread::Builder::new()
        .stack_size(64 * 1024 * 1024)
        .spawn(|| {
            let registry = ComputeRegistry::discover();
            let adapter = registry
                .adapters()
                .iter()
                .find(|adapter| adapter.certified && adapter.id.starts_with("wgpu-"))
                .expect("a certified native GPU is required");
            let context = registry.open_wgpu(&adapter.id).unwrap();
            let device = context
                .native_wgpu_device()
                .expect("wgpu context exposes a native device");
            let client = WgpuRuntime::client(&device);
            let strategy = Strategy::MultiLevel(multi_level::Strategy::DoubleUnit(
                BlueprintStrategy::Inferred(DoubleUnitSelectionArgs {
                    tile_size: TileSizeSelection::MaxTileSize,
                }),
            ));

            // Partial stages, odd whole-stage counts, and an even-count control.
            // Ones make the independently known dot product exactly K in FP32.
            for k in [24, 40, 17, 32] {
                let lhs_layout = client.create_tensor_from_slice(
                    &ones_bytes(6_400 * k),
                    [6_400, k].into(),
                    F32_SIZE,
                );
                let lhs = TensorHandle::<WgpuRuntime>::new(
                    lhs_layout.memory,
                    [6_400, k],
                    lhs_layout.strides,
                    F32,
                );
                // Contiguous `[48, k]` viewed transposed as `[k, 48]`.
                let rhs_layout =
                    client.create_tensor_from_slice(&ones_bytes(48 * k), [48, k].into(), F32_SIZE);
                let rhs_strides = transpose_last_two(&rhs_layout.strides);
                let rhs =
                    TensorHandle::<WgpuRuntime>::new(rhs_layout.memory, [k, 48], rhs_strides, F32);
                let output = TensorHandle::<WgpuRuntime>::empty(&client, [6_400, 48], F32);
                let mut dtypes = MatmulElems::from_globals(&MatmulGlobalElems {
                    lhs: F32,
                    rhs: F32,
                    out: F32,
                });
                launch_ref(
                    &strategy,
                    &client,
                    InputBinding::new(binding_of(&lhs), dtypes.lhs_global),
                    InputBinding::new(binding_of(&rhs), dtypes.rhs_global),
                    binding_of(&output),
                    &mut dtypes,
                )
                .unwrap();

                let values =
                    to_f32(&client.read_one_unchecked_tensor(output.into_copy_descriptor()));
                for (index, value) in values.into_iter().enumerate() {
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
            let registry = ComputeRegistry::discover();
            let adapter = registry
                .adapters()
                .iter()
                .find(|adapter| adapter.certified && adapter.id.starts_with("wgpu-"))
                .expect("a certified native GPU is required");
            let context = registry.open_wgpu(&adapter.id).unwrap();
            let device = context
                .native_wgpu_device()
                .expect("wgpu context exposes a native device");
            let client = WgpuRuntime::client(&device);
            let strategy = Strategy::MultiLevel(multi_level::Strategy::DoubleVecMat(
                BlueprintStrategy::Inferred(().into()),
            ));

            // Three 128-element stages on a 32-lane device. A second batch
            // makes an unchecked extra stage read real values from that batch.
            let lhs_layout =
                client.create_tensor_from_slice(&ones_bytes(2 * 384), [2, 1, 384].into(), F32_SIZE);
            let lhs = TensorHandle::<WgpuRuntime>::new(
                lhs_layout.memory,
                [2, 1, 384],
                lhs_layout.strides,
                F32,
            );
            // Contiguous `[2, 1, 384]` viewed transposed as `[2, 384, 1]`.
            let rhs_layout =
                client.create_tensor_from_slice(&ones_bytes(2 * 384), [2, 1, 384].into(), F32_SIZE);
            let rhs_strides = transpose_last_two(&rhs_layout.strides);
            let rhs =
                TensorHandle::<WgpuRuntime>::new(rhs_layout.memory, [2, 384, 1], rhs_strides, F32);
            let output = TensorHandle::<WgpuRuntime>::empty(&client, [2, 1, 1], F32);
            let mut dtypes = MatmulElems::from_globals(&MatmulGlobalElems {
                lhs: F32,
                rhs: F32,
                out: F32,
            });
            launch_ref(
                &strategy,
                &client,
                InputBinding::new(binding_of(&lhs), dtypes.lhs_global),
                InputBinding::new(binding_of(&rhs), dtypes.rhs_global),
                binding_of(&output),
                &mut dtypes,
            )
            .unwrap();
            assert_eq!(
                to_f32(&client.read_one_unchecked_tensor(output.into_copy_descriptor())),
                [384.0, 384.0]
            );
            context.check().unwrap();
        })
        .unwrap()
        .join()
        .unwrap();
}
