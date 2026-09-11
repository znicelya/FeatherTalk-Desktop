# FeatherTalk Desktop

FeatherTalk 是一个离线优先的数字人/说话头像工作台。它把视频中的人脸关键点、音频语音特征与 U-Net 训练/推理串成一条可复现流水线，并提供桌面 GUI、脚本化 CLI 和独立 worker 三种入口。项目以 Rust 2024 workspace 组织，模型权重和媒体处理工具均通过可校验的 manifest、SHA-256 与许可证元数据管理。

> 当前仓库版本：`0.1.0`（协议版本 `3`，客户端和 worker 需要配套更新）。桌面应用与安装器主要面向 Windows 10/11 x64；核心 crate 也可在支持的 Linux/macOS Rust 环境中构建。GPU 加速使用 NVIDIA CUDA 或 wgpu 的 Vulkan/Metal 后端，取决于运行环境和适配器能力。

## 能做什么

- 媒体探测与标准化：调用 FFprobe/FFmpeg，将输入转换为 25 fps 视频和 16 kHz、单声道 WAV。
- 视频帧处理：抽帧、SCRFD 人脸检测、PFLD 关键点提取，并把结果写入项目资产包。
- 音频特征：使用 FeatherHuBERT 生成 `[帧数, 2, 1024]` 特征矩阵。
- 训练与渲染：训练 Original U-Net 或 MobileOne U-Net，支持 `baseline`、`mouth-roi`、`temporal` 模式、断点续训、指标和预览；使用检查点逐帧渲染并合成带音频 MP4。
- 模型工具链：导入旧版 `.pth`/`.pth.tar`/`.npy`，检查模型包，导出标准模型包和 ONNX（opset 17），并进行结构或运行时契约验证。
- 可审计资产：`project.json`、`assets/assets.json`、模型 `manifest.json` 和权重哈希在每个阶段校验；拒绝路径穿越、符号链接和不完整资产。

## 架构概览

```text
┌────────────────────┐       JSON 行协议       ┌─────────────────────────┐
│ feathertalk-app    │ ─────────────────────▶ │ feathertalk-worker      │
│ gpui + yororen_ui  │                         │ 媒体 / 人脸 / 训练 / 推理 │
└─────────┬──────────┘                         └──────────┬──────────────┘
          │                                               │
          │                           ┌───────────────────┴──────────────────┐
          ▼                           ▼                                      ▼
   feathertalk-client          feathertalk-domain                  Burn + CUDA / wgpu
          ▲                           ▲
          └──────────── feathertalk-cli（脚本与 CI）
```

根 workspace 默认包含 `crates/feathertalk-*` 核心库、`feathertalk-cli`、`feathertalk-worker` 及 `tools/*`。`crates/feathertalk-app` 是独立 workspace（gpui 依赖较重），需要使用它自己的 manifest 构建。`vendor/cubek-matmul` 是带 FeatherTalk 补丁的本地依赖。

主要 crate：

| crate | 职责 |
| --- | --- |
| `feathertalk-domain` | 请求/事件/错误/协议帧与任务模型 |
| `feathertalk-client` | 启动 worker、握手、取消与事件流 |
| `feathertalk-worker` | JSON 行协议服务端和全部任务执行器 |
| `feathertalk-project` | 项目与资产 manifest、原子写入和安全校验 |
| `feathertalk-media` / `feathertalk-audio` | FFmpeg 媒体、音频特征文件处理 |
| `feathertalk-scrfd` / `feathertalk-pfld` / `feathertalk-frame-*` | 人脸检测、关键点和帧流水线 |
| `feathertalk-models` / `feathertalk-training*` / `feathertalk-inference` | Burn 模型、训练、数据集和推理 |
| `feathertalk-weights` / `feathertalk-export` | 权重导入、模型包和 ONNX 导出 |
| `feathertalk-app` | 中文桌面工作台（gpui + yororen_ui） |

## 环境要求

源码构建需要：

- Rust `1.94`（仓库通过 `rust-toolchain.toml` 固定）和 Cargo；C++ MSVC Build Tools（Windows）。
- Windows 桌面运行建议 Windows 10/11 x64；Linux/macOS 需自行准备对应图形驱动和 FFmpeg。
- FFmpeg 与 FFprobe。worker 不修改 PATH，可通过环境变量指定绝对路径。
- 模型目录：SCRFD、PFLD、FeatherHuBERT、VGG19，目录内必须有 `manifest.json` 与 `model.safetensors`（VGG19 另含 `LICENSES.json`）。
- GPU（可选）：Windows/Linux 上的 NVIDIA 显卡优先使用 `burn-cuda`，需要兼容驱动及 CUDA Toolkit 12.x 或更新版本（含 NVRTC 和开发头文件）。CUDA 不可用时自动选择现有 wgpu 后端（Windows/Linux 的 Vulkan、macOS 的 Metal），再回退 CPU。构建和启动不强制要求安装 CUDA；先运行 `capabilities` 检查实际能力。

## 从源码构建与测试

在仓库根目录执行：

```powershell
# 构建 workspace 中的库、CLI 和 worker
cargo build --locked --workspace

# 运行核心测试（app 不在根 workspace）
cargo test --locked --workspace --all-targets

# 构建并运行桌面应用
cargo run --manifest-path crates/feathertalk-app/Cargo.toml -- --project .\path\to\project

# 仅运行桌面应用测试
cargo test --manifest-path crates/feathertalk-app/Cargo.toml
```

发布构建使用 `cargo build --release --locked`。开发 profile 已将依赖优化到 `opt-level=2`，以避免 Burn 内核在测试中的极慢编译/执行；不要在 app workspace 中执行 `cargo update`，其锁文件依赖已下架的 `gpui-ce 0.3.3`。

## 首次运行配置

CLI 默认按以下顺序寻找 worker：`--worker <path>`、`FEATHERTALK_WORKER_BIN`、与 CLI 同目录的 `feathertalk-worker(.exe)`。显式设置计算后端时使用 `--backend auto|cpu|wgpu|cuda` 和可选的 `--adapter <ID>`；不指定则沿用 worker 环境，环境未配置时使用 `auto`（CUDA → wgpu → CPU）。显式指定后端或设备时，设备不可用会报错，不会替换为另一设备。

```powershell
$env:FEATHERTALK_WORKER_FFMPEG  = 'C:\tools\ffmpeg\bin\ffmpeg.exe'
$env:FEATHERTALK_WORKER_FFPROBE = 'C:\tools\ffmpeg\bin\ffprobe.exe'
$env:FEATHERTALK_WORKER_SCRFD_DIR = 'C:\FeatherTalk\models\scrfd_2_5g'
$env:FEATHERTALK_WORKER_PFLD_DIR  = 'C:\FeatherTalk\models\pfld_ghost_one'
$env:FEATHERTALK_WORKER_HUBERT_DIR = 'C:\FeatherTalk\models\feather_hubert'
$env:FEATHERTALK_WORKER_VGG19_DIR  = 'C:\FeatherTalk\models\vgg19'

cargo run --locked -p feathertalk-cli -- capabilities
```

也可以设置 `FEATHERTALK_WORKER_MEDIA_TIMEOUT_MS`（默认 `300000`）、`FEATHERTALK_WORKER_BACKEND=auto|cpu|wgpu|cuda` 和 `FEATHERTALK_WORKER_ADAPTER=<ID>`。worker 启动时会返回握手帧，列出协议版本、适配器、可用命令及模型/FFmpeg 能力。

### NVIDIA CUDA 加速

安装 NVIDIA 驱动和 CUDA Toolkit 后，让 `CUDA_PATH` 指向 Toolkit 根目录，并将它的 `bin` 目录加入进程的 `PATH`（Linux 还需让动态链接器能找到 CUDA 库）。例如在当前 PowerShell 会话中：

```powershell
$env:CUDA_PATH = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6'
$env:PATH = "$env:CUDA_PATH\bin;$env:PATH"
cargo run --locked -p feathertalk-cli -- capabilities
```

worker 会检测驱动、NVRTC、头文件，并实际编译执行一个 Burn 内核。只有通过检测的 NVIDIA 设备才会出现在 `cuda` 后端中；仅有 `nvidia-smi` 或 `nvcc` 不代表运行环境可用。CUDA 11.x 不满足当前 Burn 版本要求，缺失或不兼容时会在 stderr 记录原因，自动模式继续使用 wgpu/CPU。

CUDA 覆盖两个 U-Net 的训练和渲染、VGG19 感知损失、FeatherHuBERT 音频特征，以及 SCRFD/PFLD 人脸和关键点提取。抽帧和视频标准化由 FFmpeg 使用所选 CUDA 设备硬件解码，保留现有输出格式和编码参数；解码失败会清理本次临时输出并使用软件解码重试。任务取消或超时不会重试，已经开始的模型任务失败也不会自动换后端重跑。

可用 `--backend cuda` 强制 CUDA、`--backend cpu` 强制 CPU，或通过 `--adapter cuda-uuid-<UUID>` 指定 `capabilities` 中的稳定设备 ID。真实 GPU 测试需要手动运行：

```powershell
cargo test --locked -p feathertalk-worker --test cuda_execution -- --ignored --test-threads=1
cargo test --locked -p feathertalk-worker --test wgpu_execution -- --ignored --test-threads=1
```

设置前述 `FEATHERTALK_WORKER_FFMPEG` / `FEATHERTALK_WORKER_FFPROBE` 后，可运行 `cargo test --locked -p feathertalk-worker --test cuda_media -- --ignored --test-threads=1`，验证实际硬件解码和无效 CUDA 设备触发的软件重试。该测试需要 FFmpeg 支持 `libx264`；多显卡时可用 `FEATHERTALK_TEST_CUDA_DEVICE` 指定测试设备序号（默认 `0`）。

## CLI 快速开始

下面示例假定 `$project` 是一个有效项目目录，`$video` 是原始音视频文件：

```powershell
$project = 'D:\FeatherTalk\projects\demo'
$video   = 'D:\media\input.mp4'

# 诊断环境和媒体
cargo run --locked -p feathertalk-cli -- capabilities
cargo run --locked -p feathertalk-cli -- probe-media $video > probe.json
cargo run --locked -p feathertalk-cli -- normalize-media $video "$project\assets"

# 生成训练资产并锁定清单
cargo run --locked -p feathertalk-cli -- extract-frames $project "$project\assets\video_25fps.mp4"
cargo run --locked -p feathertalk-cli -- extract-features $project "$project\assets\audio_16k_mono.wav"
cargo run --locked -p feathertalk-cli -- lock-asset-package $project
cargo run --locked -p feathertalk-cli -- validate-project $project

# 训练、检查点渲染
cargo run --locked -p feathertalk-cli -- train $project --mode baseline --variant original-unet --epochs 10 --batch-size 1
cargo run --locked -p feathertalk-cli -- render $project "$project\models\unet\checkpoint-00000010" "$project\assets\audio_16k_mono.wav" "$project\outputs\demo.mp4"
```

所有命令都支持全局 `--json`（逐行 JSON，适合脚本）和 `--quiet`（隐藏进度，保留结果/错误，不能与 `--json` 同时使用）。退出码固定为：`0` 完成、`1` 任务失败、`2` 取消、`3` 会话/参数错误。使用 `--help` 查看当前版本完整参数。

命令一览：

| 命令 | 作用 |
| --- | --- |
| `validate-project <dir>` | 校验项目、锁定资产和必需文件 |
| `probe-media <input>` | 输出媒体元数据 |
| `normalize-media <input> <output-dir>` | 生成 25 fps 视频与 16 kHz 单声道 WAV |
| `extract-frames <project> <video>` | 抽帧、人脸检测和关键点 |
| `extract-features <project> <audio>` | 生成 FeatherHuBERT 特征 |
| `lock-asset-package <project>` | 写入最终资产清单与模型哈希 |
| `train <project>` | 训练 U-Net（`--mode`、`--variant`、`--epochs`、`--resume`） |
| `render <project> <checkpoint> <audio> <output>` | 渲染 MP4，可用 `--max-output-frames N` 预览 |
| `inspect-model <source>` | 检查模型包或训练检查点 |
| `import-legacy-model <source> <kind> <destination>` | 导入旧版权重（`feather-hubert`、`pfld`、`original-unet`、`mobileone-unet`） |
| `migrate-legacy-features <source.npy> <destination>` | 迁移旧 NumPy 特征 |
| `export-model-package <checkpoint> <destination>` | 导出标准模型包 |
| `export-onnx <source> <kind> <destination.onnx>` | 导出 FeatherHuBERT/Original U-Net/MobileOne U-Net ONNX |
| `capabilities` | 查看 worker 握手、适配器和支持命令 |

## 项目目录与 manifest

项目根目录至少包含 `project.json` 和以下资产树：

```text
demo/
├─ project.json
├─ assets/
│  ├─ assets.json
│  ├─ video_25fps.mp4
│  ├─ audio_16k_mono.wav
│  ├─ features/feather_hubert.f32
│  ├─ frames/
│  └─ landmarks/
├─ models/                         # 训练检查点
│  └─ unet/checkpoint-00000010/
└─ outputs/                        # 指标、预览和渲染结果
```

最小 `project.json` 示例（schema 版本为 `1`）：

```json
{
  "schema_version": 1,
  "project_id": "demo",
  "display_name": "Demo",
  "asset_package": "assets/assets.json",
  "default_model": "original_unet",
  "task_history": []
}
```

锁定后的 `assets/assets.json` 会记录 `video_fps: 25`、`audio_sample_rate: 16000`、`audio_channels: 1`、帧尺寸/数量、`feature_type: "feather_hubert"`、特征形状 `[frame_count, 2, 1024]` 以及人脸/特征模型 SHA-256。资产锁定后不应手工修改；重新生成资产时请从准备状态重新执行流水线。

## 模型包与 ONNX 工具

`tools/model-package` 提供模型导入、迁移、ONNX 导出和验证：

```powershell
cargo run --locked -p feathertalk-model-package -- --help
cargo run --locked -p feathertalk-model-package -- onnx validate --source model.onnx --kind original-unet
cargo run --locked -p feathertalk-onnx-validate -- --model model.onnx --kind original-unet --structural-only
```

ONNX 运行时对比需要带 `ort-runtime` feature 的构建，并同时提供 `--input` 与 `--expected-output`；无法使用 ONNX Runtime 时使用 `--structural-only`。SCRFD 导入器的完整流程和固定源哈希见 [`tools/scrfd-import/README.md`](tools/scrfd-import/README.md)。

## Windows 安装器

Windows 安装器脚本位于 [`installer/windows/`](installer/windows/)，会打包 `feathertalk-app.exe`、`feathertalk-worker.exe`、`feathertalk.exe`、FFmpeg、运行库和转换后的模型。普通用户直接运行发布的 `FeatherTalk-<version>-x64.msi`；安装后可执行：

```powershell
& 'C:\Program Files\FeatherTalk\feathertalk.exe' capabilities
```

从源码构建 MSI 需要 Windows x64、Rust 1.94、MSVC、.NET 8 运行时和 FFmpeg Windows 发布目录（含 `LICENSE`、`README.txt`）：

```powershell
powershell -NoProfile -File .\installer\windows\build.ps1 -FfmpegDirectory 'D:\environment\ffmpeg' -SourceRepositoryDirectory '..\FeatherTalk'
powershell -NoProfile -File .\installer\windows\verify.ps1 -MsiPath .\dist\FeatherTalk-0.1.0-x64.msi
```

脚本会校验模型源哈希、MSI 文件清单、开始菜单入口和升级/降级规则；详细参数、模型转换和输出位置请阅读 [`installer/windows/README.md`](installer/windows/README.md)。

## 故障排查

- `capabilities` 中没有媒体命令：检查 FFmpeg/FFprobe 路径和可执行权限。
- 没有 `extract-frames`：同时检查 SCRFD、PFLD 两个模型目录；目录必须是转换后的包而不是原始 `.onnx`/`.pth` 文件。
- 没有 `extract-features` 或 `train`：分别检查 FeatherHuBERT、VGG19 包及其 `manifest.json`、权重和许可证清单。
- GPU 初始化失败：先运行 `--backend cpu`；使用 `capabilities` 查看适配器 ID，再通过 `--adapter` 选择，并更新显卡驱动/Vulkan 或 Metal 支持。
- 项目校验失败：确认路径使用正斜杠相对 manifest 路径、必需文件非空，且资产目录中没有符号链接。
- 任务失败后：根据错误中的恢复建议重试、从最近检查点续训、释放磁盘空间、排除坏帧或重新导入模型。worker 的协议输出在 stdout，诊断信息在 stderr，便于重定向和 CI 集成。

## 项目来源与致谢

本项目的功能和实现思路源自 [FeatherTalk](https://github.com/anliyuan/FeatherTalk)。我们在此基础上将原项目的 Python 代码迁移并重写为 Rust，实现了对应的处理流水线、命令行工具和 worker，同时开发了桌面 GUI 客户端。

感谢 [@anliyuan](https://github.com/anliyuan) 及 FeatherTalk 社区对数字人/说话头像技术的开源贡献。本项目仅作为 Rust 实现与桌面客户端的延伸，相关上游代码、模型和第三方依赖仍分别遵循其原有许可证；使用或再发布时请同时阅读上游项目及依赖项的许可说明。

上游项目：<https://github.com/anliyuan/FeatherTalk>

## 许可证与第三方声明

本项目代码采用 Apache-2.0（发布包会附带许可证文本）。FFmpeg、Burn/CubeCL、gpui、yororen_ui 及模型权重分别遵循各自许可证；发布包会附带 `FFmpeg-LICENSE.txt`、`FFmpeg-README.txt`、`THIRD-PARTY-NOTICES.txt` 和模型目录中的 `LICENSES.json`。模型许可证不会因 FeatherTalk 的 Apache-2.0 许可而改变。对 `vendor/cubek-matmul` 的修改说明位于 [`vendor/cubek-matmul/FEATHERTALK-PATCH.md`](vendor/cubek-matmul/FEATHERTALK-PATCH.md)。

## 贡献与开发约定

提交变更前建议运行 `cargo fmt --all`、`cargo test --locked --workspace --all-targets` 和针对受影响工具的测试；协议字段、manifest schema、模型哈希或安装器文件清单发生变化时，请同步更新测试与文档。涉及 GPU 的测试应明确标注所需后端和适配器，避免把硬件特定结果当作 CPU 通过条件。
