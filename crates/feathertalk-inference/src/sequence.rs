use crate::InferenceError;

#[derive(Debug, Clone)]
pub struct PingPongFrames {
    frame_count: usize,
    period: usize,
    phase: usize,
}

impl PingPongFrames {
    pub fn new(frame_count: usize) -> Result<Self, InferenceError> {
        if frame_count < 2 {
            return Err(InferenceError::FrameCountTooSmall {
                actual: frame_count,
                minimum: 2,
            });
        }
        let period = (frame_count - 1)
            .checked_mul(2)
            .ok_or(InferenceError::ArithmeticOverflow)?;
        Ok(Self {
            frame_count,
            period,
            phase: 0,
        })
    }

    pub fn frame_count(&self) -> usize {
        self.frame_count
    }

    pub fn position(&self) -> usize {
        reflected_index(self.phase, self.period, self.frame_count)
    }

    // The public contract returns a bare source index; the Iterator implementation below
    // provides the conventional Option-returning adapter for generic iterator consumers.
    #[allow(clippy::should_implement_trait)]
    pub fn next(&mut self) -> usize {
        self.advance()
    }

    fn advance(&mut self) -> usize {
        let index = reflected_index(self.phase, self.period, self.frame_count);
        self.phase = (self.phase + 1) % self.period;
        index
    }
}

impl Iterator for PingPongFrames {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        Some(self.advance())
    }
}

pub(crate) fn period_for(frame_count: usize) -> Result<usize, InferenceError> {
    if frame_count < 2 {
        return Err(InferenceError::FrameCountTooSmall {
            actual: frame_count,
            minimum: 2,
        });
    }
    (frame_count - 1)
        .checked_mul(2)
        .ok_or(InferenceError::ArithmeticOverflow)
}

pub(crate) fn index_at(output_index: usize, frame_count: usize) -> Result<usize, InferenceError> {
    let period = period_for(frame_count)?;
    let phase = output_index % period;
    Ok(reflected_index(phase, period, frame_count))
}

fn reflected_index(phase: usize, period: usize, frame_count: usize) -> usize {
    if phase < frame_count {
        phase
    } else {
        period - phase
    }
}
