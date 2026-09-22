use std::thread;

use crate::TrainingError;
use crate::data::{TrainingDataset, TrainingSample};

pub(crate) fn load_samples<D>(
    dataset: &D,
    samples: &[TrainingSample],
) -> Result<Vec<D::Item>, TrainingError>
where
    D: TrainingDataset + Sync,
    D::Item: Send,
{
    if samples.len() < 2 {
        return samples
            .iter()
            .map(|sample| dataset.load_sample(sample))
            .collect();
    }
    let worker_count = samples.len().min(4);
    let chunk_size = samples.len().div_ceil(worker_count);
    thread::scope(|scope| {
        let handles: Vec<_> = samples
            .chunks(chunk_size)
            .map(|chunk| scope.spawn(move || load_chunk(dataset, chunk)))
            .collect();
        join_handles(handles, samples.len())
    })
}

fn load_chunk<D>(dataset: &D, samples: &[TrainingSample]) -> Result<Vec<D::Item>, TrainingError>
where
    D: TrainingDataset,
{
    samples
        .iter()
        .map(|sample| dataset.load_sample(sample))
        .collect()
}

fn join_handles<T>(
    handles: Vec<thread::ScopedJoinHandle<'_, Result<Vec<T>, TrainingError>>>,
    capacity: usize,
) -> Result<Vec<T>, TrainingError> {
    let mut items = Vec::with_capacity(capacity);
    for handle in handles {
        items.extend(join_one(handle)?);
    }
    Ok(items)
}

fn join_one<T>(
    handle: thread::ScopedJoinHandle<'_, Result<Vec<T>, TrainingError>>,
) -> Result<Vec<T>, TrainingError> {
    handle.join().unwrap_or_else(|_| Err(worker_panic()))
}

fn worker_panic() -> TrainingError {
    TrainingError::InvalidInput("data worker panic".into())
}
