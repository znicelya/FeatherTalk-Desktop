use crate::TrainingError;

const SPLITMIX_GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;
const MIX_MULTIPLIER_ONE: u64 = 0xBF58_476D_1CE4_E5B9;
const MIX_MULTIPLIER_TWO: u64 = 0x94D0_49BB_1331_11EB;
const POSITION_MULTIPLIER: u64 = 0xD1B5_4A32_D192_ED03;
const SHUFFLE_DOMAIN: u64 = 0x5348_5546_464C_455F;
const REFERENCE_DOMAIN: u64 = 0x5245_4645_5245_4E43;

pub(super) fn epoch_permutation(
    sample_count: u64,
    seed: u64,
    epoch: u64,
) -> Result<Vec<u64>, TrainingError> {
    let length = usize::try_from(sample_count).map_err(|_| TrainingError::DataLoaderOverflow {
        operation: "converting sample count",
    })?;
    let mut permutation = Vec::new();
    permutation.try_reserve_exact(length).map_err(|source| {
        TrainingError::PermutationAllocation {
            samples: sample_count,
            source,
        }
    })?;
    permutation.extend(0..sample_count);

    let mut random = SplitMix64::new(derive_seed(seed, epoch, 0, SHUFFLE_DOMAIN));
    for index in (1..length).rev() {
        let upper = u64::try_from(index)
            .map_err(|_| TrainingError::DataLoaderOverflow {
                operation: "converting shuffle index",
            })?
            .checked_add(1)
            .ok_or(TrainingError::DataLoaderOverflow {
                operation: "computing shuffle bound",
            })?;
        let swap_index = usize::try_from(random.bounded(upper)?).map_err(|_| {
            TrainingError::DataLoaderOverflow {
                operation: "converting bounded shuffle index",
            }
        })?;
        permutation.swap(index, swap_index);
    }
    Ok(permutation)
}

pub(super) fn reference_index(
    frame_count: u64,
    seed: u64,
    epoch: u64,
    position: u64,
) -> Result<u64, TrainingError> {
    let mut random = SplitMix64::new(derive_seed(seed, epoch, position, REFERENCE_DOMAIN));
    random.bounded(frame_count)
}

struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(state: u64) -> Self {
        Self { state }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(SPLITMIX_GAMMA);
        mix64(self.state)
    }

    fn bounded(&mut self, upper: u64) -> Result<u64, TrainingError> {
        if upper == 0 {
            return Err(TrainingError::InvalidDataLoaderConfig(
                "random upper bound must be non-zero".into(),
            ));
        }
        let threshold = upper.wrapping_neg() % upper;
        loop {
            let value = self.next_u64();
            if value >= threshold {
                return Ok(value % upper);
            }
        }
    }
}

fn derive_seed(base: u64, epoch: u64, position: u64, domain: u64) -> u64 {
    mix64(
        base ^ domain
            ^ epoch.wrapping_mul(SPLITMIX_GAMMA)
            ^ position.wrapping_mul(POSITION_MULTIPLIER),
    )
}

fn mix64(mut value: u64) -> u64 {
    value = (value ^ (value >> 30)).wrapping_mul(MIX_MULTIPLIER_ONE);
    value = (value ^ (value >> 27)).wrapping_mul(MIX_MULTIPLIER_TWO);
    value ^ (value >> 31)
}

#[cfg(test)]
mod tests {
    use super::{epoch_permutation, reference_index};
    use crate::TrainingError;

    #[test]
    fn version_one_shuffle_and_reference_literals_are_stable() {
        assert_eq!(epoch_permutation(5, 7, 0).unwrap(), vec![0, 4, 2, 1, 3]);
        assert_eq!(epoch_permutation(5, 7, 1).unwrap(), vec![0, 3, 1, 4, 2]);
        assert_eq!(epoch_permutation(6, 42, 0).unwrap(), vec![2, 0, 4, 1, 3, 5]);
        assert_eq!(
            (0..5)
                .map(|position| reference_index(5, 7, 0, position).unwrap())
                .collect::<Vec<_>>(),
            vec![1, 2, 4, 2, 2],
        );
        assert_eq!(
            (0..5)
                .map(|position| reference_index(5, 7, 1, position).unwrap())
                .collect::<Vec<_>>(),
            vec![0, 4, 1, 0, 0],
        );
        assert_eq!(
            (0..6)
                .map(|position| reference_index(8, 42, 0, position).unwrap())
                .collect::<Vec<_>>(),
            vec![4, 1, 6, 6, 4, 5],
        );
    }

    #[test]
    fn reference_selection_preserves_python_self_reference_behavior() {
        assert_eq!(epoch_permutation(5, 0, 0).unwrap(), vec![2, 0, 3, 1, 4]);
        assert_eq!(reference_index(5, 0, 0, 1).unwrap(), 0);
    }

    #[test]
    fn impossible_permutation_lengths_fail_without_allocating() {
        assert!(matches!(
            epoch_permutation(u64::MAX, 7, 0),
            Err(TrainingError::DataLoaderOverflow { .. })
                | Err(TrainingError::PermutationAllocation { .. })
        ));
    }
}
