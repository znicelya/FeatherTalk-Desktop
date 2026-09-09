use std::ops::Range;

use crate::AudioError;

pub const HUBERT_KERNEL: usize = 400;
pub const HUBERT_STRIDE: usize = 320;
pub const DEFAULT_CHUNK_SAMPLES: usize = HUBERT_STRIDE * 1000;
pub const MAX_CHUNKS: usize = 1_000_000;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChunkRange {
    index: usize,
    start: usize,
    end: usize,
}

impl ChunkRange {
    pub fn index(&self) -> usize {
        self.index
    }

    pub fn start(&self) -> usize {
        self.start
    }

    pub fn end(&self) -> usize {
        self.end
    }

    pub fn range(&self) -> Range<usize> {
        self.start..self.end
    }
}

impl From<Range<usize>> for ChunkRange {
    fn from(range: Range<usize>) -> Self {
        Self {
            index: 0,
            start: range.start,
            end: range.end,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChunkPlan {
    total_samples: usize,
    target_tokens: usize,
    ranges: Vec<ChunkRange>,
}

impl ChunkPlan {
    pub fn total_samples(&self) -> usize {
        self.total_samples
    }

    pub fn target_tokens(&self) -> usize {
        self.target_tokens
    }

    pub fn ranges(&self) -> &[ChunkRange] {
        &self.ranges
    }
}

pub fn expected_hubert_frames(samples: usize) -> usize {
    if samples < HUBERT_KERNEL {
        0
    } else {
        (samples - (HUBERT_KERNEL - HUBERT_STRIDE)) / HUBERT_STRIDE
    }
}

pub fn plan_chunks(samples: usize, chunk_samples: usize) -> Result<ChunkPlan, AudioError> {
    if chunk_samples == 0 {
        return Err(AudioError::InvalidChunkSize);
    }
    let overlap = HUBERT_KERNEL
        .checked_sub(HUBERT_STRIDE)
        .ok_or(AudioError::ArithmeticOverflow)?;
    let _bounded_total = samples
        .checked_add(overlap)
        .ok_or(AudioError::ArithmeticOverflow)?;
    let complete_chunks = samples / chunk_samples;
    let remainder = samples % chunk_samples;
    let tail_needed = if complete_chunks == 0 {
        samples >= HUBERT_KERNEL
    } else {
        remainder >= HUBERT_KERNEL
    };
    let chunk_count = complete_chunks
        .checked_add(usize::from(tail_needed))
        .ok_or(AudioError::ArithmeticOverflow)?;
    if chunk_count > MAX_CHUNKS {
        return Err(AudioError::TooManyChunks {
            actual: chunk_count,
            limit: MAX_CHUNKS,
        });
    }

    if samples < HUBERT_KERNEL {
        return Ok(ChunkPlan {
            total_samples: samples,
            target_tokens: 0,
            ranges: Vec::new(),
        });
    }

    let mut ranges = Vec::with_capacity(chunk_count);
    for index in 0..complete_chunks {
        let start = index
            .checked_mul(chunk_samples)
            .ok_or(AudioError::ArithmeticOverflow)?;
        let requested_end = start
            .checked_add(chunk_samples)
            .and_then(|value| value.checked_add(overlap))
            .ok_or(AudioError::ArithmeticOverflow)?;
        ranges.push(ChunkRange {
            index,
            start,
            end: requested_end.min(samples),
        });
    }
    if tail_needed {
        let start = complete_chunks
            .checked_mul(chunk_samples)
            .ok_or(AudioError::ArithmeticOverflow)?;
        ranges.push(ChunkRange {
            index: complete_chunks,
            start,
            end: samples,
        });
    }
    Ok(ChunkPlan {
        total_samples: samples,
        target_tokens: expected_hubert_frames(samples),
        ranges,
    })
}
