pub const PFLD_INPUT_CHANNELS: usize = 3;
pub const PFLD_OUTPUT_VALUES: usize = 220;

#[derive(Debug, Clone, PartialEq)]
pub struct PfldConfig {
    pub width_factor: f32,
    pub input_size: usize,
    pub landmark_count: usize,
    pub num_conv_branches: usize,
}

impl PfldConfig {
    pub const fn production() -> Self {
        Self {
            width_factor: 0.5,
            input_size: 192,
            landmark_count: 110,
            num_conv_branches: 6,
        }
    }

    pub const fn output_values(&self) -> usize {
        self.landmark_count * 2
    }

    pub const fn pooled_channels(&self) -> usize {
        32 + 40 + 48 + 72 + 64
    }
}
