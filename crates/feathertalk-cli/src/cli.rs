use std::path::PathBuf;

use clap::{Parser, Subcommand, ValueEnum};

/// The command line. Help text is Chinese, because the user is.
#[derive(Debug, Parser)]
#[command(
    name = "feathertalk",
    version,
    about = "FeatherTalk 命令行客户端",
    long_about = "通过标准输入输出驱动 feathertalk-worker 执行单个任务。\n\n\
                  标准输出只有结果，进度输出在标准错误，因此可以安全重定向。\n\
                  退出码：0 完成，1 任务失败，2 已取消，3 会话错误。"
)]
pub struct Cli {
    /// 工作进程可执行文件路径，默认依次查找环境变量与本程序同目录
    #[arg(long, global = true, value_name = "PATH")]
    pub worker: Option<PathBuf>,

    /// 计算后端，默认自动选择 CUDA、wgpu 或 CPU
    #[arg(long, global = true, value_enum)]
    pub backend: Option<BackendArg>,

    /// capabilities 中的设备 ID，根据 cuda- 前缀或 cpu-0 推断后端，其余为 wgpu
    #[arg(long, global = true, value_name = "ID")]
    pub adapter: Option<String>,

    /// 按行输出原始协议帧，供程序解析
    #[arg(long, global = true)]
    pub json: bool,

    /// 不输出进度，只保留结果与错误
    #[arg(long, global = true, conflicts_with = "json")]
    pub quiet: bool,

    /// 指定任务 ID：13 位毫秒时间戳、连字符、8 位小写十六进制
    #[arg(long, global = true, value_name = "ID")]
    pub task_id: Option<String>,

    #[command(subcommand)]
    pub command: Command,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum BackendArg {
    Auto,
    Cpu,
    Wgpu,
    Cuda,
}

impl From<BackendArg> for feathertalk_domain::Backend {
    fn from(value: BackendArg) -> Self {
        match value {
            BackendArg::Auto => Self::Auto,
            BackendArg::Cpu => Self::Cpu,
            BackendArg::Wgpu => Self::Wgpu,
            BackendArg::Cuda => Self::Cuda,
        }
    }
}

/// The task commands, kebab-cased by clap: `validate-project`, `probe-media`,
/// `normalize-media`, `extract-frames`, `extract-features`,
/// `lock-asset-package`, `train`, `render`, `inspect-model`,
/// `import-legacy-model`, `migrate-legacy-features`, `export-model-package`,
/// `export-onnx`, `capabilities`.
#[derive(Debug, Subcommand)]
pub enum Command {
    /// 校验工程目录
    ValidateProject {
        /// 工程目录
        project_dir: PathBuf,
    },
    /// 探测媒体文件信息
    ProbeMedia {
        /// 输入的音视频文件
        input: PathBuf,
    },
    /// 归一化媒体文件：输出 25fps 视频与 16kHz 单声道音频
    NormalizeMedia {
        /// 输入的音视频文件
        input: PathBuf,
        /// 输出目录，归一化后的视频与音频写入其中
        output_dir: PathBuf,
    },
    /// 抽取视频帧并检测人脸关键点
    ExtractFrames {
        /// 工程目录
        project_dir: PathBuf,
        /// 已归一化的 25fps 视频，位于工程目录的 assets 下
        video: PathBuf,
    },
    /// 提取音频的 FeatherHuBERT 特征
    ExtractFeatures {
        /// 工程目录
        project_dir: PathBuf,
        /// 已归一化的 16kHz 单声道音频，位于工程目录的 assets 下
        audio: PathBuf,
    },
    /// 写入素材清单并加锁素材包
    LockAssetPackage {
        /// 工程目录
        project_dir: PathBuf,
    },
    /// 训练 U-Net：读取已加锁的工程，按轮数训练并写出检查点与诊断产物
    Train {
        /// 工程目录
        project_dir: PathBuf,
        /// 训练模式
        #[arg(long, value_enum, default_value_t = TrainMode::Baseline)]
        mode: TrainMode,
        /// 模型变体
        #[arg(long, value_enum, default_value_t = TrainVariant::OriginalUnet)]
        variant: TrainVariant,
        /// 训练轮数
        #[arg(long)]
        epochs: u32,
        /// 每次更新使用的样本数，续训时须与检查点一致
        #[arg(long, default_value_t = feathertalk_domain::DEFAULT_BATCH_SIZE)]
        batch_size: u32,
        /// 从最新检查点继续训练，没有检查点时报错
        #[arg(long)]
        resume: bool,
    },
    /// 渲染视频：用检查点权重逐帧推理，并混入指定音轨
    Render {
        /// 工程目录
        project_dir: PathBuf,
        /// 检查点目录，例如 models/unet/checkpoint-00000004
        checkpoint: PathBuf,
        /// 混入输出视频的音频文件
        audio: PathBuf,
        /// 输出的 mp4 文件，不能已存在
        output: PathBuf,
        /// 最多渲染多少帧，默认渲染整个工程
        #[arg(long, value_name = "N")]
        max_output_frames: Option<u64>,
    },
    /// 检视模型：读取模型包或训练检查点的清单，报告类型、参数量、哈希与兼容状态
    InspectModel {
        /// 模型包目录或训练检查点目录
        source: PathBuf,
    },
    /// 将旧版模型权重导入标准模型包
    ImportLegacyModel {
        /// 旧版 .pth 或 .pth.tar 文件
        source: PathBuf,
        /// 旧模型类型
        #[arg(value_enum)]
        kind: LegacyModelKindArg,
        /// 新模型包目录，必须不存在
        destination: PathBuf,
    },
    /// 将旧版 NumPy 音频特征迁移为标准特征文件
    MigrateLegacyFeatures {
        /// 旧版 .npy 特征文件
        source: PathBuf,
        /// 目标特征文件，必须不存在
        destination: PathBuf,
    },
    /// 将训练检查点导出为标准模型包
    ExportModelPackage {
        /// 训练检查点目录，例如 models/unet/checkpoint-00000004
        source: PathBuf,
        /// 新模型包目录，必须不存在
        destination: PathBuf,
    },
    /// 将模型包导出为 ONNX 模型（opset 17）
    ExportOnnx {
        /// 模型包目录
        source: PathBuf,
        /// 导出的模型类型
        #[arg(value_enum)]
        kind: OnnxExportKindArg,
        /// 目标 .onnx 文件，必须不存在
        destination: PathBuf,
    },
    /// 打印工作进程的握手信息：后端、设备、支持的命令
    Capabilities,
}

/// The training modes, mirrored from `feathertalk-domain` because `ValueEnum`
/// has to be derived on a local type. `run.rs` maps them onto the domain enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum TrainMode {
    /// 基线：整幅 L1 加感知损失
    Baseline,
    /// 基线之上加嘴部 ROI 权重
    MouthRoi,
    /// 嘴部 ROI 之上加相邻帧的时序一致性
    Temporal,
}

/// The U-Net variants, mirrored for the same reason.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum TrainVariant {
    /// 原版 U-Net
    OriginalUnet,
    /// MobileOne U-Net
    ///
    /// Spelled the way the model is spelled everywhere else -- the checkpoint
    /// manifest and the ONNX export both say `mobileone_unet` -- rather than the
    /// `mobile-one-unet` clap would derive from the variant name.
    #[value(name = "mobileone-unet")]
    MobileOneUnet,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum LegacyModelKindArg {
    #[value(name = "feather-hubert")]
    FeatherHubert,
    Pfld,
    #[value(name = "original-unet")]
    OriginalUnet,
    #[value(name = "mobileone-unet")]
    MobileOneUnet,
}

/// The models with an ONNX graph. Narrower than `LegacyModelKindArg`: the face
/// landmark model has no export.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum OnnxExportKindArg {
    #[value(name = "feather-hubert")]
    FeatherHubert,
    #[value(name = "original-unet")]
    OriginalUnet,
    #[value(name = "mobileone-unet")]
    MobileOneUnet,
}
