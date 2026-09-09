use std::path::PathBuf;

use feathertalk_audio::{
    AudioError, ChunkEncoder, ChunkPlan, DEFAULT_CHUNK_SAMPLES, MAX_FEATURE_FILE_BYTES,
    drop_odd_token, expected_hubert_frames, extract_long_audio, normalize_waveform, plan_chunks,
    read_wav_16k_mono, write_feature_file_no_clobber,
};
use feathertalk_domain::{ExtractFeaturesParams, Progress, TaskStage};
use feathertalk_media::CancellationToken;

use crate::{
    CommandOutcome, TaskReporter,
    admission::{check_project_dir, invalid_request},
    audio_task_error, feature_to_json, is_audio_cancellation,
};

/// The asset directory `normalize_media` writes into.
const ASSETS_DIR: &str = "assets";

/// The feature subdirectory. `feathertalk-audio` owns the file name but nothing
/// owns the directory, so the worker decides the layout.
const FEATURES_DIR: &str = "features";

/// The published feature file. `feathertalk-audio` keeps its own copy of this
/// name private (`src/commit.rs:15`), so the literal is duplicated here.
const FEATURE_FILE_NAME: &str = "feather_hubert.f32";

/// The fixed feature header. `feathertalk-audio` computes the same number as a
/// `pub(crate) usize` (`src/format.rs`), so admission keeps a `u64` copy to
/// project a file size without a cast.
const FEATURE_HEADER_BYTES: u64 = 44;

/// Extract the FeatherHuBERT features of a normalised wav into a project's
/// asset directory.
///
/// The encoder arrives by mutable reference so a caller can drive the command
/// without loading weights; `FeatureModel` in this crate supplies the real one.
/// It is a type parameter rather than a trait object because
/// `feathertalk_audio::extract_long_audio` needs a sized encoder.
pub fn execute_extract_features<E: ChunkEncoder>(
    params: &ExtractFeaturesParams,
    token: &CancellationToken,
    reporter: &dyn TaskReporter,
    encoder: &mut E,
    model_sha256: &str,
) -> CommandOutcome {
    // One stage before the first chunk: the caller has already spent seconds
    // loading the model and admission reads the whole wav, and the CLI would
    // otherwise print nothing until the first chunk lands.
    reporter.report(TaskStage::Preparing, None);
    let admitted = match admit(params, encoder.output_dim()) {
        Ok(admitted) => admitted,
        Err(outcome) => return outcome,
    };
    // The runtime checks the token before dispatch; this second check covers
    // the seconds admission spent reading a large wav file.
    if token.is_cancelled() {
        return CommandOutcome::Cancelled;
    }
    let normalized = match normalize_waveform(&admitted.samples) {
        Ok(normalized) => normalized,
        Err(error) => return audio_failure(&error),
    };
    let total = admitted.plan.ranges().len() as u64;
    let mut progress = ChunkProgress {
        inner: encoder,
        reporter,
        token,
        total,
        completed: 0,
    };
    let matrix = match extract_long_audio(&normalized, &mut progress, DEFAULT_CHUNK_SAMPLES) {
        Ok(matrix) => matrix,
        Err(error) => return audio_failure(&error),
    };
    // Two tokens per video frame, so an odd one has no frame to belong to.
    let matrix = drop_odd_token(matrix);
    if let Err(error) = reporter.before_publish() {
        return CommandOutcome::Failed(error);
    }
    match write_feature_file_no_clobber(&admitted.destination, &matrix) {
        Ok(artifact) => {
            let payload = feature_to_json(&admitted.output_dir, &artifact, model_sha256);
            CommandOutcome::Completed(Some(payload))
        }
        Err(error) => audio_failure(&error),
    }
}

/// Everything admission established, so the command body never re-reads the
/// request.
struct Admitted {
    samples: Vec<f32>,
    plan: ChunkPlan,
    output_dir: PathBuf,
    destination: PathBuf,
}

/// Everything that has to hold before the encoder runs, ordered so that the
/// cheapest refusal happens first.
///
/// The plan is computed here and again inside `extract_long_audio`, which costs
/// nothing -- it walks no samples -- and it is what lets admission refuse an
/// oversized feature file before a forward pass rather than after all of them.
fn admit(params: &ExtractFeaturesParams, dims: usize) -> Result<Admitted, CommandOutcome> {
    check_project_dir(&params.project_dir).map_err(CommandOutcome::Failed)?;
    if !params.audio.is_absolute() {
        return Err(CommandOutcome::Failed(invalid_request(
            "音频文件必须是绝对路径",
            format!("audio {} is not absolute", params.audio.display()),
        )));
    }
    let output_dir = params.project_dir.join(ASSETS_DIR).join(FEATURES_DIR);
    let destination = output_dir.join(FEATURE_FILE_NAME);
    // `write_feature_file_no_clobber` refuses the collision anyway, but only
    // after the encoder has run. This slice has no `force` flag, so the
    // cheapest correct answer is to refuse now.
    if destination.exists() {
        return Err(CommandOutcome::Failed(invalid_request(
            "特征文件已存在",
            format!("{} already exists", destination.display()),
        )));
    }
    let samples = read_wav_16k_mono(&params.audio)
        .map_err(|error| CommandOutcome::Failed(audio_task_error(&error)))?;
    let frames = expected_hubert_frames(samples.len());
    if frames < 2 {
        return Err(CommandOutcome::Failed(invalid_request(
            "音频太短，无法提取特征",
            format!(
                "{} samples yield {frames} FeatherHuBERT frame(s), at least 2 are required",
                samples.len()
            ),
        )));
    }
    let plan = plan_chunks(samples.len(), DEFAULT_CHUNK_SAMPLES)
        .map_err(|error| CommandOutcome::Failed(audio_task_error(&error)))?;
    // Overflow takes the rejection path too: a token count that cannot be
    // turned into a byte count is over the limit by definition.
    let projected = (plan.target_tokens() as u64)
        .checked_mul(dims as u64)
        .and_then(|values| values.checked_mul(4))
        .and_then(|bytes| bytes.checked_add(FEATURE_HEADER_BYTES))
        .unwrap_or(u64::MAX);
    if projected > MAX_FEATURE_FILE_BYTES {
        return Err(CommandOutcome::Failed(invalid_request(
            "音频过长，特征文件会超出上限",
            format!(
                "{} tokens at {dims} dims need {projected} bytes, over {MAX_FEATURE_FILE_BYTES}",
                plan.target_tokens()
            ),
        )));
    }
    Ok(Admitted {
        samples,
        plan,
        output_dir,
        destination,
    })
}

/// Bridges the encoder onto the worker's reporter and token.
///
/// One chunk is the granularity, which `DEFAULT_CHUNK_SAMPLES` fixes at just
/// over twenty seconds of audio: fine enough for a progress bar, coarse enough
/// that reporting is never the bottleneck.
struct ChunkProgress<'a, E: ChunkEncoder> {
    inner: &'a mut E,
    reporter: &'a dyn TaskReporter,
    token: &'a CancellationToken,
    total: u64,
    completed: u64,
}

impl<E: ChunkEncoder> ChunkEncoder for ChunkProgress<'_, E> {
    fn output_dim(&self) -> usize {
        self.inner.output_dim()
    }

    fn encode(&mut self, chunk_index: usize, samples: &[f32]) -> Result<Vec<f32>, AudioError> {
        // Between chunks is the only place this command can be interrupted: a
        // chunk is a single forward pass with no seam inside it.
        if self.token.is_cancelled() {
            return Err(AudioError::Cancelled {
                operation: "extract_features",
            });
        }
        let output = self.inner.encode(chunk_index, samples)?;
        self.completed += 1;
        self.reporter.report(
            TaskStage::ExtractingFeatures,
            Some(Progress {
                completed: self.completed,
                total: Some(self.total),
            }),
        );
        Ok(output)
    }
}

/// Cancellation is not a failure: the encoder reports it as an error and the
/// runtime needs it back as `Cancelled`.
fn audio_failure(error: &AudioError) -> CommandOutcome {
    if is_audio_cancellation(error) {
        CommandOutcome::Cancelled
    } else {
        CommandOutcome::Failed(audio_task_error(error))
    }
}
