//! FeatherHuBERT model components.
mod adapter;
mod config;
mod frontend;
mod model;
mod tcn;

pub use adapter::BurnFeatherHubertEncoder;
pub use config::{
    FeatherHubertConfig, HUBERT_KERNEL, HUBERT_STRIDE, SAMPLE_RATE, expected_hubert_frames,
    make_even_tokens, normalize_waveform,
};
pub use model::FeatherHubertEncoder;
