use crate::PreprocessError;

pub fn audio_window_indices(
    frame_index: usize,
    frame_count: usize,
) -> Result<[Option<usize>; 8], PreprocessError> {
    if frame_count == 0 || frame_index >= frame_count {
        return Err(PreprocessError::FrameIndexOutOfRange {
            frame_index,
            frame_count,
        });
    }
    Ok(std::array::from_fn(|slot| {
        let index = if slot < 4 {
            frame_index.checked_sub(4 - slot)
        } else {
            frame_index.checked_add(slot - 4)
        };
        index.filter(|value| *value < frame_count)
    }))
}
