//! Every string the user sees.
//!
//! Separated from `run.rs` so the whole output surface can be asserted without
//! spawning a process.

use feathertalk_client::{ClientError, ENV_WORKER_BIN, EventSink, WorkerPathSource};
use feathertalk_domain::{
    Event, ReadyFrame, Recovery, RejectedFrame, TaskError, TaskKind, TaskStage,
};

/// The worker's own variables for locating its media tools. Written as
/// literals because the CLI must not link the worker crate;
/// `feathertalk-worker`'s `ENV_FFPROBE` and `ENV_FFMPEG` are the source of
/// truth for the names.
const ENV_WORKER_FFPROBE: &str = "FEATHERTALK_WORKER_FFPROBE";
const ENV_WORKER_FFMPEG: &str = "FEATHERTALK_WORKER_FFMPEG";

/// The worker's variables for the two model directories, literals for the same
/// reason: `feathertalk-worker`'s `ENV_SCRFD_DIR` and `ENV_PFLD_DIR` are the
/// source of truth for these names.
const ENV_WORKER_SCRFD_DIR: &str = "FEATHERTALK_WORKER_SCRFD_DIR";
const ENV_WORKER_PFLD_DIR: &str = "FEATHERTALK_WORKER_PFLD_DIR";

/// The worker's variable for the FeatherHuBERT package directory, a literal for
/// the same reason: `feathertalk-worker`'s `ENV_HUBERT_DIR` is the source of
/// truth for this name.
const ENV_WORKER_HUBERT_DIR: &str = "FEATHERTALK_WORKER_HUBERT_DIR";

/// The worker's variable for the VGG19 package directory, a literal for the same
/// reason as the others: `feathertalk-worker`'s `ENV_VGG19_DIR` is the source of
/// truth for this name.
const ENV_WORKER_VGG19_DIR: &str = "FEATHERTALK_WORKER_VGG19_DIR";

/// The Chinese name of every stage.
///
/// No `_` arm on purpose: adding a stage to the protocol must break this match.
/// A stage the CLI cannot name is a stage the user cannot understand.
pub fn stage_label(stage: &TaskStage) -> String {
    match stage {
        TaskStage::Queued => "排队中".to_string(),
        TaskStage::Preparing => "准备中".to_string(),
        TaskStage::ExtractingAudio => "正在提取音频".to_string(),
        TaskStage::ExtractingFrames => "正在提取视频帧".to_string(),
        TaskStage::DetectingFaces => "正在检测人脸".to_string(),
        TaskStage::ExtractingFeatures => "正在提取特征".to_string(),
        TaskStage::Training { epoch, step, loss } => {
            format!("正在训练 轮次 {epoch} 步 {step} 损失 {loss:.4}")
        }
        TaskStage::Importing => "正在导入".to_string(),
        TaskStage::Exporting => "正在导出".to_string(),
        TaskStage::Rendering { frame, total } => format!("正在渲染 第 {frame}/{total} 帧"),
        TaskStage::Completed => "已完成".to_string(),
        TaskStage::Failed { code, message } => format!("已失败 {} {message}", code.as_wire()),
        TaskStage::Cancelled => "已取消".to_string(),
    }
}

/// One line per event. The slug is kept alongside the Chinese label so a user
/// can search the logs or a spec for the same token the protocol uses.
pub fn event_line(event: &Event) -> String {
    let mut line = format!("[{}] {}", event.stage.as_slug(), stage_label(&event.stage));
    if let Some(text) = progress_text(event) {
        line.push(' ');
        line.push_str(&text);
    }
    if let Some(text) = metrics_text(event) {
        line.push(' ');
        line.push_str(&text);
    }
    line
}

fn progress_text(event: &Event) -> Option<String> {
    let progress = event.progress.as_ref()?;
    match progress.total {
        // A percentage needs a denominator, and zero is not one.
        Some(total) if total > 0 => Some(format!(
            "进度 {}/{} ({:.1}%)",
            progress.completed,
            total,
            progress.completed as f64 * 100.0 / total as f64
        )),
        Some(total) => Some(format!("进度 {}/{}", progress.completed, total)),
        None => Some(format!("已处理 {}", progress.completed)),
    }
}

fn metrics_text(event: &Event) -> Option<String> {
    let mut parts = Vec::new();
    if let Some(rate) = event.metrics.samples_per_second {
        parts.push(format!("速率 {rate:.2}/秒"));
    }
    if let Some(eta) = event.metrics.eta_seconds {
        parts.push(format!("预计剩余 {eta} 秒"));
    }
    if let Some(vram) = event.metrics.vram_bytes {
        parts.push(format!("显存 {}", mebibytes(vram)));
    }
    if parts.is_empty() {
        None
    } else {
        Some(parts.join(" "))
    }
}

fn mebibytes(bytes: u64) -> String {
    format!("{:.1} MiB", bytes as f64 / (1024.0 * 1024.0))
}

/// What the user can do about a failure. Also exhaustive by design.
pub fn recovery_label(recovery: &Recovery) -> &'static str {
    match recovery {
        Recovery::Retry => "可以直接重试该任务",
        Recovery::ResumeFromCheckpoint => "可以从最近的检查点继续",
        Recovery::FreeDiskSpace => "请清理磁盘空间后重试",
        Recovery::SelectDifferentAdapter => "请改用其他计算设备后重试",
        Recovery::ExcludeBadFrames => "请排除有问题的视频帧后重试",
        Recovery::ReimportModel => "请重新导入模型文件",
        Recovery::NotRecoverable => "该错误无法自动恢复，请检查输入与环境",
    }
}

/// The failure report.
///
/// `summary` is the worker's Chinese sentence and `detail` is its English
/// diagnostic. Both are printed verbatim: translating either would put a second
/// author between the operator and what actually happened.
pub fn failure_block(error: &TaskError) -> String {
    [
        format!("任务失败：{}", error.summary),
        format!("错误码: {}", error.code.as_wire()),
        format!("阶段: {}", error.stage.as_slug()),
        format!("建议: {}", recovery_label(&error.recovery)),
        format!("详情: {}", error.detail),
    ]
    .join("\n")
}

/// The handshake, in Chinese. Built from `ready` alone — the CLI probes nothing
/// itself, so what it prints is exactly what the worker claims.
pub fn capabilities_report(ready: &ReadyFrame) -> String {
    let mut lines = vec![
        format!("工作进程版本: {}", ready.worker_version),
        format!("协议版本: {}", ready.protocol_version),
        format!(
            "后端: {}",
            ready
                .backends
                .iter()
                .map(slug)
                .collect::<Vec<_>>()
                .join(", ")
        ),
        "计算设备:".to_string(),
    ];
    for adapter in &ready.adapters {
        let vram = match adapter.vram_bytes {
            Some(bytes) => format!(" 显存 {}", mebibytes(bytes)),
            None => String::new(),
        };
        lines.push(format!(
            "  {} {} 类型 {} 后端 {} 认证 {}{vram}",
            adapter.id,
            adapter.name,
            slug(&adapter.kind),
            slug(&adapter.backend),
            yes_no(adapter.certified)
        ));
    }
    lines.push(format!(
        "支持的命令: {}",
        ready
            .supported_commands
            .iter()
            .copied()
            .map(TaskKind::as_slug)
            .collect::<Vec<_>>()
            .join(", ")
    ));
    lines.push(format!(
        "能力: 训练 {} wgpu 训练 {} onnx 校验 {} ffmpeg {}",
        yes_no(ready.capabilities.training),
        yes_no(ready.capabilities.wgpu_training),
        yes_no(ready.capabilities.onnx_validation),
        yes_no(ready.capabilities.ffmpeg)
    ));
    lines.join("\n")
}

fn yes_no(value: bool) -> &'static str {
    if value { "是" } else { "否" }
}

/// The wire spelling of a type that has no `as_slug` — `Backend`, `AdapterKind`,
/// `ErrorCode`. Taken from serde so the CLI can never drift from the protocol.
pub fn slug<T: serde::Serialize>(value: &T) -> String {
    serde_json::to_string(value)
        .unwrap_or_default()
        .trim_matches('"')
        .to_string()
}

/// Human output. Events go to stderr so stdout stays a clean result channel and
/// `feathertalk probe-media a.mp4 > info.json` produces a usable file.
pub struct HumanSink {
    quiet: bool,
}

impl HumanSink {
    pub fn new(quiet: bool) -> Self {
        Self { quiet }
    }
}

impl EventSink for HumanSink {
    fn on_event(&mut self, event: &Event, raw: &str) {
        let _ = raw;
        // `--quiet` drops progress, never the result or the failure report.
        if self.quiet {
            return;
        }
        eprintln!("{}", event_line(event));
    }

    fn on_rejected(&mut self, rejected: &RejectedFrame, raw: &str) {
        let _ = raw;
        eprintln!("工作进程拒绝了请求：{}", rejected.reason);
    }
}

/// Machine output: every frame exactly as the worker wrote it, one per line, on
/// stdout. Never a re-serialisation — this workspace builds `serde_json` without
/// `preserve_order`, so a round trip would silently reorder object keys.
pub struct JsonSink;

impl EventSink for JsonSink {
    fn on_event(&mut self, event: &Event, raw: &str) {
        let _ = event;
        println!("{raw}");
    }

    fn on_rejected(&mut self, rejected: &RejectedFrame, raw: &str) {
        let _ = rejected;
        println!("{raw}");
    }
}

/// Session-level errors, in Chinese, each one ending in something to try.
///
/// `ClientError`'s own `Display` is English and stays that way: it is a
/// developer-facing diagnostic, and this is the user-facing translation of it.
pub fn render_client_error(error: &ClientError) -> String {
    let mut text = match error {
        ClientError::WorkerNotFound { probed } => {
            let mut lines = vec!["找不到工作进程可执行文件。已按以下顺序查找：".to_string()];
            for candidate in probed {
                let shown = match candidate.path.as_ref() {
                    Some(path) => path.display().to_string(),
                    None => "未设置".to_string(),
                };
                lines.push(format!("  {}: {shown}", source_label(candidate.source)));
            }
            lines.push(format!(
                "请用 --worker 指定路径，或设置环境变量 {ENV_WORKER_BIN}。"
            ));
            lines.join("\n")
        }
        ClientError::Spawn { path, source } => format!(
            "无法启动工作进程 {}：{source}\n请确认该文件存在并且可以执行。",
            path.display()
        ),
        ClientError::Handshake { reason, .. } => {
            format!("工作进程握手失败：{reason}\n请确认 --worker 指向的是 feathertalk-worker。")
        }
        ClientError::ProtocolVersion { expected, actual } => format!(
            "协议版本不匹配：本客户端支持 {expected}，工作进程使用 {actual}。\n请让两者来自同一次构建。"
        ),
        ClientError::Rejected { reason } => format!("工作进程拒绝了本次请求：{reason}"),
        ClientError::UnsupportedCommand {
            requested,
            supported,
        } => {
            let mut text = format!(
                "工作进程不支持命令 {requested}。它声明支持：{}。",
                supported.join(", ")
            );
            if matches!(*requested, "probe_media" | "normalize_media") {
                text.push_str(&format!(
                    "\n{requested} 需要可用的 ffprobe 与 ffmpeg。请安装 ffmpeg，或用环境变量 \
                     {ENV_WORKER_FFPROBE} 与 {ENV_WORKER_FFMPEG} 指定它们的完整路径。"
                ));
            } else if *requested == "extract_frames" {
                text.push_str(&format!(
                    "\n{requested} 需要媒体工具与人脸模型。请用环境变量 {ENV_WORKER_FFPROBE}、\
                     {ENV_WORKER_FFMPEG}、{ENV_WORKER_SCRFD_DIR}、{ENV_WORKER_PFLD_DIR} \
                     指定它们的完整路径。"
                ));
            } else if *requested == "extract_features" {
                text.push_str(&format!(
                    "\n{requested} 需要 FeatherHuBERT 特征模型。请用环境变量 \
                     {ENV_WORKER_HUBERT_DIR} 指定模型包目录的完整路径。"
                ));
            } else if *requested == "lock_asset_package" {
                text.push_str(&format!(
                    "\n{requested} 需要 FeatherHuBERT 模型包来记录编码器摘要。\
                     请用环境变量 {ENV_WORKER_HUBERT_DIR} 指定模型包目录的完整路径。"
                ));
            } else if *requested == "train" {
                text.push_str(&format!(
                    "\n{requested} 需要 VGG19 感知损失模型包。请用环境变量 \
                     {ENV_WORKER_VGG19_DIR} 指定模型包目录的完整路径。"
                ));
            } else if *requested == "render" {
                text.push_str(&format!(
                    "\n{requested} 需要可用的 ffprobe 与 ffmpeg 来写出视频。请安装 ffmpeg，\
                     或用环境变量 {ENV_WORKER_FFPROBE} 与 {ENV_WORKER_FFMPEG} \
                     指定它们的完整路径。"
                ));
            }
            text
        }
        ClientError::Protocol(source) => {
            format!("协议错误：{source}\n工作进程与客户端的版本可能不一致。")
        }
        ClientError::Io(source) => format!("读写工作进程时出错：{source}"),
        ClientError::WorkerGone { status, .. } => match status {
            Some(code) => format!("工作进程已退出（退出码 {code}），任务没有完成。"),
            None => "工作进程已不可用，任务没有完成。".to_string(),
        },
    };
    let tail = error.stderr_tail();
    if !tail.is_empty() {
        text.push_str("\n工作进程最后的输出：");
        for line in tail {
            text.push_str(&format!("\n  {line}"));
        }
    }
    text
}

fn source_label(source: WorkerPathSource) -> &'static str {
    match source {
        WorkerPathSource::CliOption => "--worker 选项",
        WorkerPathSource::EnvVar => "环境变量 FEATHERTALK_WORKER_BIN",
        WorkerPathSource::SiblingOfCurrentExe => "与本程序同目录",
    }
}

#[cfg(test)]
mod tests {
    use feathertalk_domain::ErrorCode;

    use super::*;

    #[test]
    fn every_stage_has_a_chinese_label() {
        for stage in TaskStage::ALL_UNIT_SAMPLES {
            let label = stage_label(&stage);
            assert!(
                !label.is_ascii(),
                "{stage:?} must have a Chinese label, got {label:?}"
            );
        }
    }

    #[test]
    fn every_recovery_has_advice() {
        for recovery in [
            Recovery::Retry,
            Recovery::ResumeFromCheckpoint,
            Recovery::FreeDiskSpace,
            Recovery::SelectDifferentAdapter,
            Recovery::ExcludeBadFrames,
            Recovery::ReimportModel,
            Recovery::NotRecoverable,
        ] {
            assert!(!recovery_label(&recovery).is_empty(), "{recovery:?}");
        }
    }

    #[test]
    fn error_codes_are_shown_in_their_wire_form() {
        // `as_wire` and serde must agree; this is the guard against drift.
        for code in ErrorCode::ALL {
            assert_eq!(slug(&code), code.as_wire(), "{code:?}");
        }
    }
}
