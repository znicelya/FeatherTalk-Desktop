use burn::{tensor::Tensor, tensor::backend::Backend};

use crate::ScrfdError;

pub(crate) type GeneratedOutput<B> = (
    Tensor<B, 3>,
    Tensor<B, 3>,
    Tensor<B, 3>,
    Tensor<B, 3>,
    Tensor<B, 3>,
    Tensor<B, 3>,
    Tensor<B, 3>,
    Tensor<B, 3>,
    Tensor<B, 3>,
);

#[derive(Debug)]
pub struct ScrfdRawOutput<B: Backend> {
    pub levels: [ScrfdLevelOutput<B>; 3],
}

#[derive(Debug)]
pub struct ScrfdLevelOutput<B: Backend> {
    pub stride: u32,
    pub scores: Tensor<B, 2>,
    pub bbox_deltas: Tensor<B, 3>,
    pub keypoint_deltas: Tensor<B, 3>,
}

pub(crate) fn assemble<B: Backend>(
    outputs: GeneratedOutput<B>,
) -> Result<ScrfdRawOutput<B>, ScrfdError> {
    let (out0, out1, out2, out3, out4, out5, out6, out7, out8) = outputs;
    validate("out0", out0.dims().to_vec(), vec![1, 12_800, 1])?;
    validate("out1", out1.dims().to_vec(), vec![1, 3_200, 1])?;
    validate("out2", out2.dims().to_vec(), vec![1, 800, 1])?;
    validate("out3", out3.dims().to_vec(), vec![1, 12_800, 4])?;
    validate("out4", out4.dims().to_vec(), vec![1, 3_200, 4])?;
    validate("out5", out5.dims().to_vec(), vec![1, 800, 4])?;
    validate("out6", out6.dims().to_vec(), vec![1, 12_800, 10])?;
    validate("out7", out7.dims().to_vec(), vec![1, 3_200, 10])?;
    validate("out8", out8.dims().to_vec(), vec![1, 800, 10])?;

    Ok(ScrfdRawOutput {
        levels: [
            ScrfdLevelOutput {
                stride: 8,
                scores: out0.reshape([1, 12_800]),
                bbox_deltas: out3,
                keypoint_deltas: out6,
            },
            ScrfdLevelOutput {
                stride: 16,
                scores: out1.reshape([1, 3_200]),
                bbox_deltas: out4,
                keypoint_deltas: out7,
            },
            ScrfdLevelOutput {
                stride: 32,
                scores: out2.reshape([1, 800]),
                bbox_deltas: out5,
                keypoint_deltas: out8,
            },
        ],
    })
}

fn validate(
    name: &'static str,
    actual: Vec<usize>,
    expected: Vec<usize>,
) -> Result<(), ScrfdError> {
    if actual != expected {
        return Err(ScrfdError::InvalidOutputShape {
            name,
            expected,
            actual,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::{NdArray, ndarray::NdArrayDevice};

    type Cpu = NdArray<f32>;

    fn valid(device: &NdArrayDevice) -> GeneratedOutput<Cpu> {
        (
            Tensor::zeros([1, 12_800, 1], device),
            Tensor::zeros([1, 3_200, 1], device),
            Tensor::zeros([1, 800, 1], device),
            Tensor::zeros([1, 12_800, 4], device),
            Tensor::zeros([1, 3_200, 4], device),
            Tensor::zeros([1, 800, 4], device),
            Tensor::zeros([1, 12_800, 10], device),
            Tensor::zeros([1, 3_200, 10], device),
            Tensor::zeros([1, 800, 10], device),
        )
    }

    #[test]
    fn valid_generated_outputs_assemble_with_only_score_rank_removed() {
        let device = Default::default();
        let output = assemble(valid(&device)).unwrap();
        assert_eq!(output.levels[0].scores.dims(), [1, 12_800]);
        assert_eq!(output.levels[1].bbox_deltas.dims(), [1, 3_200, 4]);
        assert_eq!(output.levels[2].keypoint_deltas.dims(), [1, 800, 10]);
    }

    #[test]
    fn every_score_bbox_and_keypoint_shape_is_checked() {
        for index in 0..9 {
            let device = Default::default();
            let mut outputs = valid(&device);
            match index {
                0 => outputs.0 = Tensor::zeros([1, 12_800, 2], &device),
                1 => outputs.1 = Tensor::zeros([1, 3_200, 2], &device),
                2 => outputs.2 = Tensor::zeros([1, 800, 2], &device),
                3 => outputs.3 = Tensor::zeros([1, 12_800, 5], &device),
                4 => outputs.4 = Tensor::zeros([1, 3_200, 5], &device),
                5 => outputs.5 = Tensor::zeros([1, 800, 5], &device),
                6 => outputs.6 = Tensor::zeros([1, 12_800, 11], &device),
                7 => outputs.7 = Tensor::zeros([1, 3_200, 11], &device),
                8 => outputs.8 = Tensor::zeros([1, 800, 11], &device),
                _ => unreachable!(),
            }
            assert!(
                assemble(outputs).is_err(),
                "output index {index} was accepted"
            );
        }
    }
}
