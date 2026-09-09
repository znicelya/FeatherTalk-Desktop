//! Verify a prepared asset package, then make it immutable.

use std::path::PathBuf;

use feathertalk_audio::{
    FeatureCommitSpec, FeatureMatrix, commit_feature_artifact, fit_feature_tokens,
    read_feature_file,
};
use feathertalk_domain::{ProjectDirParams, TaskStage};
use feathertalk_frame_pipeline::{QualityReport, read_quality_report};
use feathertalk_media::CancellationToken;
use feathertalk_pfld::PFLD_MODEL_SHA256;
use feathertalk_project::{AssetPackageState, read_asset_manifest};

use crate::{
    CommandOutcome, TaskReporter,
    admission::{check_project_dir, invalid_request},
    asset_scan::{count_asset_files, verify_frames},
    audio_task_error, lock_to_json, pipeline_task_error, project_task_error,
};

/// The asset directory the earlier commands write into.
const ASSETS_DIR: &str = "assets";

/// The package manifest. `feathertalk-project` keeps its own private copy of
/// this name (`src/package.rs:56`), so the literal is duplicated the way
/// `admission.rs` duplicates `project.json`.
const MANIFEST_FILE: &str = "assets.json";

/// The report `extract_frames` publishes, relative to the asset directory.
const QUALITY_FILE: &str = "quality.json";

/// The feature file `extract_features` publishes, relative to the project
/// root.
const FEATURE_FILE: &str = "assets/features/feather_hubert.f32";

/// What a lockable package must already contain, relative to the project root.
/// Frames and landmarks are absent on purpose: they are checked against the
/// quality report, one by one, rather than by name.
const REQUIRED_FILES: [&str; 3] = [
    "assets/video_25fps.mp4",
    "assets/audio_16k_mono.wav",
    FEATURE_FILE,
];

/// How far the feature stream may sit from the frame count and still be
/// fitted.
///
/// 50 tokens is one second of audio at 25 fps. The real drift between a wav
/// and the frames cut from the same clip is under four tokens; anything past a
/// second means the two inputs do not belong together, and the lock says so
/// instead of quietly padding a mismatch into an immutable package.
const MAX_TOKEN_FIT_DELTA: i64 = 50;

/// Verify a prepared asset package, then commit it as locked.
///
/// `feature_model_sha256` is the digest of the FeatherHuBERT package installed
/// in this worker, which the caller reads out of the package manifest. It
/// records which encoder was present when the package was locked; the feature
/// file carries no provenance of its own, so this is a claim about the worker,
/// not a proof about the file.
pub fn execute_lock_asset_package(
    params: &ProjectDirParams,
    token: &CancellationToken,
    reporter: &dyn TaskReporter,
    feature_model_sha256: &str,
) -> CommandOutcome {
    // Admission reads a quality report and a feature file that can both be
    // large, so the stage lands before the first byte.
    reporter.report(TaskStage::Preparing, None);
    let admitted = match admit(params) {
        Ok(admitted) => admitted,
        Err(outcome) => return outcome,
    };
    // The runtime checks the token before dispatch; this covers the seconds
    // admission just spent reading.
    if token.is_cancelled() {
        return CommandOutcome::Cancelled;
    }
    let (frame_width, frame_height) =
        match verify_frames(&admitted.assets, &admitted.report, token, reporter) {
            Ok(geometry) => geometry,
            Err(outcome) => return outcome,
        };
    if let Err(outcome) = count_asset_files(&admitted.assets, admitted.report.frame_count()) {
        return outcome;
    }
    let spec = FeatureCommitSpec {
        project_root: params.project_dir.clone(),
        frame_count: admitted.report.frame_count(),
        frame_width,
        frame_height,
        landmark_model_sha256: PFLD_MODEL_SHA256.to_owned(),
        feature_model_sha256: feature_model_sha256.to_owned(),
    };
    // The commit sits past the last cancellation point on purpose: it stages,
    // backs up, renames and rolls back on failure, and interrupting it halfway
    // is exactly what would leave a package neither prepared nor locked.
    match commit_feature_artifact(&spec, &admitted.matrix) {
        Ok(artifact) => {
            let payload = lock_to_json(
                &params.project_dir,
                &spec,
                &artifact,
                admitted.token_adjustment,
            );
            CommandOutcome::Completed(Some(payload))
        }
        Err(error) => CommandOutcome::Failed(audio_task_error(&error)),
    }
}

/// Everything admission established, so the command body never re-reads the
/// package.
struct Admitted {
    assets: PathBuf,
    report: QualityReport,
    matrix: FeatureMatrix,
    token_adjustment: i64,
}

/// Everything that has to hold before the package is walked, ordered so that
/// the cheapest refusal happens first.
fn admit(params: &ProjectDirParams) -> Result<Admitted, CommandOutcome> {
    check_project_dir(&params.project_dir).map_err(CommandOutcome::Failed)?;
    let assets = params.project_dir.join(ASSETS_DIR);
    let manifest_path = assets.join(MANIFEST_FILE);
    // `commit_feature_artifact` refuses to mutate a locked package as well,
    // but only after the whole package has been walked, and it reports the
    // refusal as a mutation rather than as the state the operator asked about.
    if manifest_path.exists() {
        match read_asset_manifest(&manifest_path) {
            Ok(manifest) if manifest.state == AssetPackageState::Locked => {
                return Err(CommandOutcome::Failed(invalid_request(
                    "素材包已加锁",
                    format!("{} is already locked", manifest_path.display()),
                )));
            }
            Ok(_) => {}
            Err(error) => return Err(CommandOutcome::Failed(project_task_error(&error))),
        }
    }
    let report = read_quality_report(&assets.join(QUALITY_FILE))
        .map_err(|error| CommandOutcome::Failed(pipeline_task_error(&error)))?;
    if !report.anomalies().is_empty() {
        return Err(CommandOutcome::Failed(invalid_request(
            "素材包仍有异常帧",
            format!("the report carries {} anomalies", report.anomalies().len()),
        )));
    }
    if report.accepted_count() != report.frame_count() {
        return Err(CommandOutcome::Failed(invalid_request(
            "仍有帧未被接受",
            format!(
                "the report accepted {} of {} frames",
                report.accepted_count(),
                report.frame_count()
            ),
        )));
    }
    for relative in REQUIRED_FILES {
        let path = params.project_dir.join(relative);
        if !path.is_file() {
            return Err(CommandOutcome::Failed(invalid_request(
                "素材包缺少必需文件",
                format!("{} is missing or not a regular file", path.display()),
            )));
        }
    }
    let matrix = read_feature_file(&params.project_dir.join(FEATURE_FILE))
        .map_err(|error| CommandOutcome::Failed(audio_task_error(&error)))?;
    // Two tokens per frame is the shape `commit_feature_artifact` enforces.
    let target_tokens = report
        .frame_count()
        .checked_mul(2)
        .and_then(|tokens| usize::try_from(tokens).ok())
        .ok_or_else(|| {
            CommandOutcome::Failed(invalid_request(
                "素材帧数过多",
                format!(
                    "{} frames need more tokens than this platform can address",
                    report.frame_count()
                ),
            ))
        })?;
    let token_adjustment = target_tokens as i64 - matrix.tokens() as i64;
    if token_adjustment.abs() > MAX_TOKEN_FIT_DELTA {
        return Err(CommandOutcome::Failed(invalid_request(
            "特征令牌数与帧数不匹配",
            format!(
                "the feature file holds {} tokens, {} frames need {target_tokens}",
                matrix.tokens(),
                report.frame_count()
            ),
        )));
    }
    let matrix = fit_feature_tokens(matrix, target_tokens)
        .map_err(|error| CommandOutcome::Failed(audio_task_error(&error)))?;
    Ok(Admitted {
        assets,
        report,
        matrix,
        token_adjustment,
    })
}
