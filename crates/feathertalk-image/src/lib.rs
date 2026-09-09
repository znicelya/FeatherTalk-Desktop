//! Dependency-light pixel kernels that reproduce OpenCV's CPU results.
//!
//! Every kernel here is pinned by fixtures generated with OpenCV on the CPU;
//! see `tests/fixtures/opencv_cpu_v1/` and `rust/tools/image-parity/`.

mod error;
mod image;
mod jpeg;
mod laplacian;
mod resize;

pub use error::ImageError;
pub use image::{BgrImage, GrayImage, to_gray};
pub use jpeg::{decode_jpeg, jpeg_dimensions};
pub use laplacian::{laplacian_response, laplacian_variance};
pub use resize::{resize_area, resize_linear};
