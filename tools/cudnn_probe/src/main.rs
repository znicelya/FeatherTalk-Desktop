// Standalone cuDNN vs (previously measured) cubecl convolution benchmark.
// Measures forward / dgrad (data grad) / wgrad (filter grad) for the hot
// FeatherTalk UNet conv shapes, in FP32, under two math modes:
//   - DEFAULT_MATH  : plain FP32 math
//   - TENSOR_OP_ALLOW_CONVERSION : lets cuDNN use TF32 tensor cores for FP32
// This does NOT touch burn/cubecl; it only tells us the cuDNN ceiling.

use std::time::Instant;

use cudarc::cudnn::sys::{
    cudnnConvolutionMode_t::CUDNN_CROSS_CORRELATION, cudnnMathType_t,
    cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
};
use cudarc::cudnn::{ConvBackwardData, ConvBackwardFilter, ConvForward, Cudnn};
use cudarc::driver::CudaContext;

#[derive(Clone, Copy)]
struct Case {
    name: &'static str,
    n: i32,
    cin: i32,
    cout: i32,
    hw: i32,
    groups: i32,
}

const K: i32 = 3;
const PAD: i32 = 1;

fn time_op<F: FnMut()>(stream: &cudarc::driver::CudaStream, warmup: usize, iters: usize, mut f: F) -> f64 {
    for _ in 0..warmup {
        f();
    }
    stream.synchronize().unwrap();
    let t0 = Instant::now();
    for _ in 0..iters {
        f();
    }
    stream.synchronize().unwrap();
    t0.elapsed().as_secs_f64() / iters as f64
}

fn run_case(cudnn: &std::sync::Arc<Cudnn>, stream: &std::sync::Arc<cudarc::driver::CudaStream>, c: Case, math: cudnnMathType_t, warmup: usize, iters: usize) {
    // pad=1, stride=1, dilation=1, k=3 -> output H/W == input H/W.
    let out_hw = c.hw;
    let cin_per_group = c.cin / c.groups;

    let mut conv = cudnn
        .create_conv2d::<f32>([PAD, PAD], [1, 1], [1, 1], CUDNN_CROSS_CORRELATION)
        .unwrap();
    conv.set_group_count(c.groups).unwrap();
    conv.set_math_type(math).unwrap();

    let x_desc = cudnn
        .create_4d_tensor::<f32>(CUDNN_TENSOR_NCHW, [c.n, c.cin, c.hw, c.hw])
        .unwrap();
    let w_desc = cudnn
        .create_4d_filter::<f32>(CUDNN_TENSOR_NCHW, [c.cout, cin_per_group, K, K])
        .unwrap();
    let y_desc = cudnn
        .create_4d_tensor::<f32>(CUDNN_TENSOR_NCHW, [c.n, c.cout, out_hw, out_hw])
        .unwrap();

    let x = stream.alloc_zeros::<f32>((c.n * c.cin * c.hw * c.hw) as usize).unwrap();
    let w = stream.alloc_zeros::<f32>((c.cout * cin_per_group * K * K) as usize).unwrap();
    let y = stream.alloc_zeros::<f32>((c.n * c.cout * out_hw * out_hw) as usize).unwrap();

    // ---- forward ----
    let fwd = {
        let op = ConvForward { conv: &conv, x: &x_desc, w: &w_desc, y: &y_desc };
        let algo = op.pick_algorithm().unwrap();
        let ws_bytes = op.get_workspace_size(algo).unwrap();
        let mut ws = stream.alloc_zeros::<u8>(ws_bytes.max(1)).unwrap();
        let mut y = y.clone();
        let x = x.clone();
        let w = w.clone();
        time_op(stream, warmup, iters, || unsafe {
            op.launch(algo, Some(&mut ws), (1.0f32, 0.0f32), &x, &w, &mut y).unwrap();
        })
    };

    // ---- dgrad (backward data) ----
    let dgrad = {
        let op = ConvBackwardData { conv: &conv, dx: &x_desc, w: &w_desc, dy: &y_desc };
        let algo = op.pick_algorithm().unwrap();
        let ws_bytes = op.get_workspace_size(algo).unwrap();
        let mut ws = stream.alloc_zeros::<u8>(ws_bytes.max(1)).unwrap();
        let mut dx = x.clone();
        let w = w.clone();
        let dy = y.clone();
        time_op(stream, warmup, iters, || unsafe {
            op.launch(algo, Some(&mut ws), (1.0f32, 0.0f32), &mut dx, &w, &dy).unwrap();
        })
    };

    // ---- wgrad (backward filter) ----
    let wgrad = {
        let op = ConvBackwardFilter { conv: &conv, x: &x_desc, dw: &w_desc, dy: &y_desc };
        let algo = op.pick_algorithm().unwrap();
        let ws_bytes = op.get_workspace_size(algo).unwrap();
        let mut ws = stream.alloc_zeros::<u8>(ws_bytes.max(1)).unwrap();
        let x = x.clone();
        let mut dw = w.clone();
        let dy = y.clone();
        time_op(stream, warmup, iters, || unsafe {
            op.launch(algo, Some(&mut ws), (1.0f32, 0.0f32), &x, &mut dw, &dy).unwrap();
        })
    };

    let bwd = dgrad + wgrad;
    println!(
        "{:32} fwd={:.4}s  dgrad={:.4}s wgrad={:.4}s  bwd={:.4}s  bwd/fwd={:.1}x",
        c.name, fwd, dgrad, wgrad, bwd, bwd / fwd.max(1e-9)
    );
}


fn main() {
    let batch: i32 = std::env::args()
        .skip_while(|a| a != "--batch")
        .nth(1)
        .and_then(|v| v.parse().ok())
        .unwrap_or(4);
    let iters = 50usize;
    let warmup = 15usize;

    let ctx = CudaContext::new(0).unwrap();
    let stream = ctx.default_stream();
    let cudnn = Cudnn::new(stream.clone()).unwrap();

    let cases = [
        Case { name: "depthwise c=32  hw=160", n: batch, cin: 32, cout: 32, hw: 160, groups: 32 },
        Case { name: "depthwise c=64  hw=80 ", n: batch, cin: 64, cout: 64, hw: 80, groups: 64 },
        Case { name: "depthwise c=128 hw=40 ", n: batch, cin: 128, cout: 128, hw: 40, groups: 128 },
        Case { name: "depthwise c=256 hw=20 ", n: batch, cin: 256, cout: 256, hw: 20, groups: 256 },
        Case { name: "depthwise c=512 hw=10 ", n: batch, cin: 512, cout: 512, hw: 10, groups: 512 },
        Case { name: "dense 32->32   hw=160", n: batch, cin: 32, cout: 32, hw: 160, groups: 1 },
        Case { name: "dense 32->64   hw=160", n: batch, cin: 32, cout: 64, hw: 160, groups: 1 },
        Case { name: "dense 64->64   hw=160", n: batch, cin: 64, cout: 64, hw: 160, groups: 1 },
        Case { name: "dense 64->128  hw=80 ", n: batch, cin: 64, cout: 128, hw: 80, groups: 1 },
        Case { name: "dense 128->256 hw=40 ", n: batch, cin: 128, cout: 256, hw: 40, groups: 1 },
        Case { name: "dense 256->256 hw=20 ", n: batch, cin: 256, cout: 256, hw: 20, groups: 1 },
    ];

    for (label, math) in [
        ("DEFAULT_MATH (plain fp32)", cudnnMathType_t::CUDNN_DEFAULT_MATH),
        ("TENSOR_OP_ALLOW_CONVERSION (tf32 tensor cores)", cudnnMathType_t::CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION),
    ] {
        println!("\n== cuDNN fp32 math={label} batch={batch} iters={iters} ==");
        for c in cases {
            run_case(&cudnn, &stream, c, math, warmup, iters);
        }
    }
}

