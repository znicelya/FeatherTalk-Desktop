//! The Chinese copy the shell paints.
//!
//! User-visible strings live in the bundled Chinese catalog and page fragments.
//! The fragments merge over the base catalog before the shell reads any key.

use thiserror::Error;
use yororen_ui::i18n::{LoadError, TranslationMap, parse_translation_value};

/// The default locale used by the shell at startup.
pub const LOCALE_TAG: &str = "zh-CN";

/// Locale tags whose catalogs are bundled with the application.
pub const LOCALE_TAGS: &[&str] = &["zh-CN", "en"];

/// The catalog source, embedded so the shell has copy before it reads any file.
pub const RAW: &str = include_str!("../locales/zh-CN/base.json");
pub const ADDITIONAL: &[&str] = &[
    include_str!("../locales/zh-CN/ui.json"),
    include_str!("../locales/zh-CN/workflow.json"),
    include_str!("../locales/zh-CN/models.json"),
];

/// The keys the shell chrome reads. Page keys come from `navigation::Page`.
pub const SHELL_KEYS: &[&str] = &[
    "shell.title",
    "shell.subtitle",
    "shell.navigation",
    "shell.worker.ready",
    "shell.worker.missing",
    "shell.worker.path",
    "shell.worker.probed",
    "shell.worker.unset",
    "shell.worker.source.cli",
    "shell.worker.source.env",
    "shell.worker.source.sibling",
    "compute.label",
    "compute.refresh",
    "compute.automatic",
    "compute.discovering",
    "compute.unavailable",
    "compute.experimental",
    "compute.vram",
    "compute.vram_unavailable",
    "compute.invalid",
    "compute.blocked",
];

/// The keys the task page reads outside the protocol's own enums.
///
/// The enum-derived keys are not listed here: `tests/catalog.rs` walks
/// `TaskStatus::ALL`, `TaskStage::ALL_UNIT_SAMPLES`, `TaskKind::ALL` and the
/// recovery list instead, so a new variant upstream fails the test without
/// anybody remembering to extend a second list.
pub const TASKS_PAGE_KEYS: &[&str] = &[
    "tasks.submit",
    "tasks.blocked.no_project",
    "tasks.blocked.no_worker",
    "tasks.blocked.busy",
    "tasks.incomplete.title",
    "tasks.incomplete.description",
    "tasks.incomplete.resume",
    "tasks.incomplete.discard",
    "tasks.incomplete.updated",
    "tasks.incomplete.unknown_kind",
    "tasks.list.title",
    "tasks.empty.title",
    "tasks.empty.description",
    "tasks.cancel",
    "tasks.cancel_hint",
    "tasks.attempts",
    "tasks.detail",
    "tasks.error_code",
    "tasks.recovery",
    "tasks.crash_log",
    "tasks.failure.crashed",
    "tasks.failure.unavailable",
    "tasks.failure.rejected",
    "tasks.failure.unsupported",
    "tasks.notes.title",
    "tasks.note.scan_failed",
    "tasks.note.log_failed",
    "tasks.note.journal_failed",
    "tasks.note.manifest_failed",
    "tasks.note.task_id_failed",
    "tasks.note.submit_failed",
    "tasks.note.resolve_failed",
];

/// The keys the asset page reads outside its own step enum.
///
/// `Step::label_key` and `Step::hint_key` are not listed here: `tests/catalog.rs`
/// walks `Step::ALL` instead, so a fifth step fails the test without anybody
/// remembering to extend a second list.
pub const ASSETS_PAGE_KEYS: &[&str] = &[
    "assets.project.label",
    "assets.project.unset",
    "assets.project.pick",
    "assets.refresh",
    "assets.video.label",
    "assets.video.unset",
    "assets.video.pick",
    "assets.state.ready",
    "assets.state.done",
    "assets.state.locked",
    "assets.blocked.no_source",
    "assets.blocked.no_video",
    "assets.blocked.no_audio",
    "assets.blocked.no_features",
    "assets.blocked.locked",
    "assets.package.title",
    "assets.package.state",
    "assets.package.fps",
    "assets.package.sample_rate",
    "assets.package.channels",
    "assets.package.frames",
    "assets.package.resolution",
    "assets.package.duration",
    "assets.package.tokens",
    "assets.package.landmark_model",
    "assets.package.feature_model",
    "assets.package.preparing",
    "assets.package.locked",
    "assets.failure.hint",
    "assets.notes.title",
    "assets.note.pick_failed",
    "assets.note.absolute_failed",
    "assets.note.manifest_failed",
];

/// The keys the training page reads outside the protocol's own enums.
///
/// The three preset names and hints and the two variant names are not listed
/// here: `tests/catalog.rs` walks `training::ALL_MODES` and
/// `training::ALL_VARIANTS` through their key functions instead, so a fourth mode
/// upstream fails the test without anybody remembering to extend a second list.
/// `tasks.note.manifest_failed` is not here either -- writing the manifest happens
/// in `submit`, on behalf of every page, so the note belongs to the task page's
/// namespace.
pub const TRAINING_PAGE_KEYS: &[&str] = &[
    "training.form.title",
    "training.form.preset",
    "training.form.variant",
    "training.form.epochs",
    "training.form.epochs_hint",
    "training.form.batch_size",
    "training.form.batch_size_hint",
    "training.form.batch_size_resume",
    "training.form.resume",
    "training.form.resume_hint",
    "training.device.label",
    "training.device.cpu",
    "training.output.label",
    "training.submit",
    "training.blocked.not_locked",
    "training.blocked.no_checkpoint",
    "training.blocked.epoch_reached",
    "training.blocked.batch_size_mismatch",
    "training.warn.fresh_over_checkpoint",
    "training.live.title",
    "training.live.position",
    "training.live.steps",
    "training.metrics.title",
    "training.metrics.empty",
    "training.metrics.source",
    "training.metrics.mode",
    "training.metrics.total_loss",
    "training.metrics.full_loss",
    "training.metrics.perceptual_loss",
    "training.metrics.mouth_loss",
    "training.metrics.temporal_loss",
    "training.metrics.temporal_mouth_loss",
    "training.metrics.samples_seen",
    "training.metrics.samples_per_second",
    "training.metrics.remaining",
    "training.metrics.gpu_memory",
    "training.metrics.gpu_memory_none",
    "training.metrics.mode_baseline",
    "training.metrics.mode_mouth",
    "training.metrics.mode_temporal",
    "training.checkpoint.title",
    "training.checkpoint.empty",
    "training.checkpoint.latest",
    "training.checkpoint.epoch",
    "training.checkpoint.count",
    "training.checkpoint.preview_count",
    "training.config.title",
    "training.config.empty",
    "training.config.batch_size",
    "training.config.learning_rate",
    "training.config.total_epochs",
    "training.config.temporal_stride",
    "training.config.mouth_weight",
    "training.config.temporal_weight",
    "training.config.temporal_mouth_weight",
    "training.config.perceptual_weight",
    "training.config.random_seed",
    "training.failure.hint",
    "training.notes.title",
    "training.note.metrics_failed",
    "training.note.state_failed",
];

/// The keys the generate page reads outside the protocol's own enums.
///
/// The three training mode names are not listed here: `training::mode_value`
/// maps a checkpoint's mode spelling onto the same keys the training page uses,
/// so a mode is named one way on both pages rather than twice in this catalog.
pub const GENERATE_PAGE_KEYS: &[&str] = &[
    "generate.form.title",
    "generate.checkpoint.label",
    "generate.checkpoint.latest",
    "generate.checkpoint.picked",
    "generate.checkpoint.step",
    "generate.checkpoint.epoch",
    "generate.checkpoint.mode",
    "generate.checkpoint.pick",
    "generate.checkpoint.use_latest",
    "generate.checkpoint.unreadable",
    "generate.checkpoint.none",
    "generate.audio.label",
    "generate.audio.project",
    "generate.audio.picked",
    "generate.audio.pick",
    "generate.audio.note",
    "generate.audio.missing",
    "generate.preview.label",
    "generate.preview.hint",
    "generate.preview.frames",
    "generate.preview.seconds",
    "generate.preview.full",
    "generate.output.label",
    "generate.output.pick",
    "generate.submit",
    "generate.blocked.not_locked",
    "generate.blocked.no_checkpoint",
    "generate.blocked.no_audio",
    "generate.format.title",
    "generate.format.resolution",
    "generate.format.fps",
    "generate.format.video_codec",
    "generate.format.audio_codec",
    "generate.format.quality",
    "generate.format.quality_default",
    "generate.live.title",
    "generate.live.position",
    "generate.renders.title",
    "generate.renders.empty",
    "generate.renders.play",
    "generate.renders.reveal",
    "generate.renders.count",
    "generate.failure.hint",
    "generate.notes.title",
    "generate.note.checkpoint_failed",
    "generate.note.output_dir_failed",
    "generate.note.pick_failed",
];

/// Why the bundled catalog could not be turned into a translation map.
#[derive(Debug, Error)]
pub enum CatalogError {
    #[error("the bundled catalog is not valid JSON: {0}")]
    Json(#[from] serde_json::Error),
    #[error("the bundled catalog is not a translation object: {0}")]
    Load(#[from] LoadError),
    #[error("unsupported locale: {0}")]
    UnsupportedLocale(String),
}

/// Parse the bundled catalog into a translation map.
///
/// `yororen_ui::locale::parse_bundled_translations` does the same thing but
/// panics on malformed input. The shell returns the error instead and falls back
/// to the framework's own copy, because a broken catalog is not worth a crash on
/// the user's machine.
pub fn translations(locale_tag: &str) -> Result<TranslationMap, CatalogError> {
    if locale_tag != LOCALE_TAG {
        return Err(CatalogError::UnsupportedLocale(locale_tag.to_owned()));
    }
    let value: serde_json::Value = serde_json::from_str(RAW)?;
    let mut translations = parse_translation_value(value)?;
    for raw in ADDITIONAL {
        translations.merge(parse_translation_value(serde_json::from_str(raw)?)?);
    }
    Ok(translations)
}
