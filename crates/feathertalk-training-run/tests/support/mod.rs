#![allow(dead_code)]

use burn::{
    module::Module,
    tensor::{Tensor, backend::Backend},
};
use feathertalk_models::unet::{OriginalUnet, OriginalUnetConfig};
use feathertalk_training::{PerceptualFeatureExtractor, TrainingConfig, TrainingMode};

pub type CpuBackend = burn::backend::NdArray<f32>;
pub type CpuAutodiffBackend = burn::backend::Autodiff<CpuBackend>;
pub type CpuDevice = burn::backend::ndarray::NdArrayDevice;

/// A 160x160 forward plus backward through burn's autodiff graph overruns the
/// default 2 MiB libtest thread stack in a debug build and aborts the whole test
/// binary with `STATUS_STACK_OVERFLOW`. `feathertalk-pfld/tests/runtime.rs` and
/// `feathertalk-weights` already solve the same problem with a dedicated 64 MiB
/// stack; this mirrors that for training steps.
const STEP_STACK_BYTES: usize = 64 * 1024 * 1024;

/// Runs `body` on a thread whose stack is large enough for a training step.
/// Panics travel back through `join`, so failed assertions still fail the test.
pub fn on_step_stack(name: &str, body: impl FnOnce() + Send + 'static) {
    std::thread::Builder::new()
        .name(name.to_owned())
        .stack_size(STEP_STACK_BYTES)
        .spawn(body)
        .expect("the step thread starts")
        .join()
        .expect("the step thread does not panic");
}

#[derive(Debug, Clone, Copy)]
pub struct IdentityExtractor;

impl<B: Backend> PerceptualFeatureExtractor<B> for IdentityExtractor {
    fn forward(&self, image: Tensor<B, 4>) -> Tensor<B, 4> {
        image
    }
}

#[derive(Debug, Clone, Copy)]
pub struct NanExtractor;

impl<B: Backend> PerceptualFeatureExtractor<B> for NanExtractor {
    fn forward(&self, image: Tensor<B, 4>) -> Tensor<B, 4> {
        image.mul_scalar(f32::NAN)
    }
}

/// Builds the micro training model with every parameter already materialised.
///
/// Cloning a lazily initialised `Param` copies its *initializer*, not its value,
/// and "initializing one does not affect the other" (burn-core 0.21
/// `module/param/base.rs:439-463`), so two clones of a freshly `init`ed module
/// draw independent random weights. `fork` pushes every parameter through
/// `val()`, which makes later clones share one starting point - the tests that
/// compare two runs from the same weights depend on that.
pub fn model(device: &CpuDevice) -> OriginalUnet<CpuAutodiffBackend> {
    OriginalUnetConfig::parity_micro()
        .init::<CpuAutodiffBackend>(device)
        .fork(device)
}

pub fn training_config(
    mode: TrainingMode,
    batch_size: u64,
    total_epochs: u64,
    temporal_stride: u64,
) -> TrainingConfig {
    TrainingConfig {
        mode,
        batch_size,
        learning_rate: 1e-4,
        total_epochs,
        temporal_stride,
        mouth_weight: 4.0,
        temporal_weight: 0.5,
        temporal_mouth_weight: 4.0,
        perceptual_weight: 0.01,
    }
}

pub fn assert_close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() < 1e-5,
        "expected {expected}, got {actual}"
    );
}
