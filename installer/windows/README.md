# FeatherTalk Windows 安装包

安装包适用于 Windows 10/11 x64，包含 FeatherTalk 工作台、后台 worker、命令行工具、
FFmpeg/FFprobe、四个模型包和 Microsoft Visual C++ 运行库。使用程序无需安装 Rust、Python、
.NET 或 WiX。Windows 上检测到 NVIDIA 显卡及可用的 CUDA Toolkit 12.x 或更新版本时，
模型计算优先使用 `burn-cuda`，抽帧和视频标准化使用 FFmpeg CUDA 硬件解码。
CUDA 是可选组件，安装包不包含 Toolkit；未安装或检测失败时自动使用现有 Vulkan / SPIR-V 后端，
没有可用 GPU 时使用 CPU。显式选择设备会保留该选择；可在“计算设备”中刷新设备列表。

## 安装和使用

双击 `FeatherTalk-0.1.0-x64.msi`，按中文向导选择安装位置。安装需要管理员权限，
默认安装到 `C:\Program Files\FeatherTalk`，完成后从开始菜单启动 FeatherTalk。
可通过 Windows“已安装的应用”卸载，更高版本以及同版本重新构建的 MSI 会替换现有安装，
不会并存两个 FeatherTalk 产品。当前版本号使用三段数字；使用当前配置生成的同版本 MSI 可相互替换，
更高版本已安装时仍拒绝降级。

允许同版本替换会触发 WiX 的 ICE61 提示；验证脚本会检查升级版本上限恰为当前版本，
并检查更高版本的降级保护，防止扩大覆盖范围。

在 PowerShell 中检查已安装程序：

```powershell
& 'C:\Program Files\FeatherTalk\feathertalk.exe' capabilities
& 'C:\Program Files\FeatherTalk\feathertalk.exe' --help
```

worker 优先使用 `FEATHERTALK_WORKER_FFMPEG` 和 `FEATHERTALK_WORKER_FFPROBE` 的显式
配置；没有配置时自动使用安装目录内的工具。安装程序不修改 PATH 或全局环境变量。

启用 CUDA 需自行安装 NVIDIA 驱动及 CUDA Toolkit，设置 `CUDA_PATH` 为 Toolkit 根目录，
并确保其 `bin` 目录在启动程序的 `PATH` 中。检测会检查 NVRTC、头文件和实际内核运行；
`capabilities` 中出现 `cuda` 设备即表示可用。CUDA 11.x 不满足当前 Burn 要求。
CLI 支持 `--backend auto|cpu|wgpu|cuda`（默认 `auto`）和 `--adapter <ID>`。
详情及当前会话的配置示例见项目 [README](../../README.md#nvidia-cuda-加速)。

四个模型已转换为 Rust 可加载的 `manifest.json` 与 `model.safetensors`，
随程序安装到以下位置；FeatherHuBERT 和 VGG19 另带 `LICENSES.json`：

| 原始模型 | 安装目录内的位置 | 功能 |
| --- | --- | --- |
| `scrfd_2.5g_kps.onnx` | `models/scrfd_2_5g` | 人脸检测 |
| `checkpoint_epoch_335.pth.tar` | `models/pfld_ghost_one` | 人脸关键点 |
| `feather_hubert_188_latest_99.pth` | `models/feather_hubert` | 音频特征 |
| `vgg19-dcbb9e9d.pth` | `models/vgg19` | VGG19 conv3_3 感知损失 |

程序自动从安装目录查找这四个模型，无需设置环境变量。仍可分别用
`FEATHERTALK_WORKER_SCRFD_DIR`、`FEATHERTALK_WORKER_PFLD_DIR`、
`FEATHERTALK_WORKER_HUBERT_DIR`、`FEATHERTALK_WORKER_VGG19_DIR`
指向其他已转换模型包的绝对路径，显式配置优先。
原始 `.pth`/`.onnx` 文件不能直接作为模型包目录。

VGG19 使用当前训练所需的 `conv3_3` 特征提取器，转换后的权重约 6.6 MiB。
项目数据请选择保存在安装目录外的可写目录。随包模型会随程序卸载，用户项目和
外部模型不会随卸载删除。

## 构建 MSI

构建机器需要 Windows x64、Rust 1.94/MSVC C++ Build Tools、.NET 8 运行时，以及带
`LICENSE` 和 `README.txt` 的 FFmpeg Windows 发行目录。脚本从 NuGet 下载固定的
WiX Toolset 5.0.2 与 UI 扩展，并校验 SHA-256；缓存保存在 `target/wix-tools`。
不要求安装 .NET SDK 或全局 WiX。

在仓库根目录运行：

```powershell
powershell -NoProfile -File .\installer\windows\build.ps1 -FfmpegDirectory 'D:\environment\ffmpeg' -SourceRepositoryDirectory '..\FeatherTalk'
```

不传 `-FfmpegDirectory` 时会定位 PATH 中的 FFmpeg。`-VCRuntimeDirectory` 可指定
`x64\Microsoft.VC143.CRT` 目录，默认由 `vswhere` 定位。`-OutputDirectory` 可指定输出
位置。正常构建从 Cargo 的编译输出读取实际程序路径，支持显式配置的 x64 target。
仅调整 WiX 文件时可用 `-SkipBuild` 复用默认 `target/release` 目录中的原生构建，
使用自定义编译 target 时请执行完整构建。

独立桌面仓库通过 `-SourceRepositoryDirectory` 指定原始 FeatherTalk 仓库的位置，
用于读取项目 `LICENSE` 和默认的 FeatherHuBERT 检查点；上面的示例假定两个仓库为同级目录。
不传该参数时沿用旧布局，查找当前 Rust workspace 的上级目录。

模型打包默认复用 `crates/feathertalk-scrfd/artifacts/scrfd_2_5g` 和
`crates/feathertalk-pfld/artifacts/pfld_ghost_one` 的转换结果，并从
源仓库内的 `demo/kanghui_training_video_featherhubert_188_latest/feather_hubert_188_latest_99.pth`
通过刚构建的 Rust worker 转换 FeatherHuBERT。转换输入保存在独立构建目录，
不写入原始模型目录。VGG19 默认读取 `target/installer/models/vgg19` 的转换包。
`-ScrfdModelDirectory`、`-PfldModelDirectory`、`-HubertModelDirectory`、
`-Vgg19ModelDirectory` 可指定已有转换包；每个包均核对原始模型身份与权重哈希。

如缺少 SCRFD/PFLD 转换结果，可先在仓库根目录运行：

```powershell
cargo run --locked --manifest-path tools/scrfd-import/Cargo.toml --bin generate -- --repo-root .. --destination target/installer/scrfd-conversion
cargo run --locked --release -p feathertalk-pfld-artifact -- target/installer/pfld-conversion
```

随后构建时将 `-ScrfdModelDirectory` 指向生成的
`scrfd-conversion/artifacts/scrfd_2_5g`，`-PfldModelDirectory` 指向
`pfld-conversion`，使用相对于当前目录的路径或绝对路径。

首次准备 VGG19 时，使用[官方 torchvision 权重](https://download.pytorch.org/models/vgg19-dcbb9e9d.pth)，
在仓库根目录运行以下命令，将 `$vggCheckpoint` 改为本地权重的实际路径：

```powershell
$vggCheckpoint = 'C:\models\vgg19-dcbb9e9d.pth'
New-Item -ItemType Directory -Path target/installer/models -Force | Out-Null
cargo run --release --locked -p feathertalk-vgg19-package -- --source $vggCheckpoint --licenses installer/windows/vgg19-licenses.json --destination target/installer/models/vgg19
```

转换器只导出感知损失所需的 14 个张量，并重新加载验证。安装包构建时检查原始
官方权重的 SHA-256，因此测试用或其他来源的 VGG19 包不能误入安装包。

脚本只打包 `feathertalk-app.exe`、`feathertalk-worker.exe`、`feathertalk.exe` 三个
产品程序，以及运行依赖、转换后的基础模型和说明；原始模型、测试用 worker、
Cargo 缓存、PDB 和源码不进入 MSI。
版本号从两个 Cargo workspace 读取并检查一致性。输出位于 `dist`：

- `FeatherTalk-<version>-x64.msi`
- 同名 `.msi.sha256` 校验文件
- `FeatherTalk-<version>-x64.payload.json` 相对路径文件清单、哈希及模型来源

每次构建使用独立暂存目录，位于 `target/installer`。默认生成未签名的 MSI；
正式发布时可使用项目自己的代码签名证书签名。

## 验证 MSI

```powershell
powershell -NoProfile -File .\installer\windows\verify.ps1 -MsiPath .\dist\FeatherTalk-0.1.0-x64.msi
```

验证器检查 MSI 属性、同版本替换与防降级规则、开始菜单入口和文件清单，然后执行管理员映像提取，
逐个核对文件哈希与模型来源，在清除开发机 FeatherTalk 配置的子进程内检查 CLI、
worker 与媒体处理，并使用包内模型在 CPU 上执行一帧人脸/关键点检测及一秒音频
特征提取，再使用随包 VGG19 完成一个训练步骤并保存检查点。
人脸测试使用仓库内 `demo_frame_v1/frame.jpg` 测试素材。
提取过程不注册产品安装，验证输出保存在 `target/installer`。

FeatherTalk 使用 Apache-2.0。随包 FFmpeg 的许可证和源码版本见
`FFmpeg-LICENSE.txt`、`FFmpeg-README.txt`；其他运行依赖说明见 `THIRD-PARTY-NOTICES.txt`。
本地修正的 Cubek 矩阵乘法内核，其上游许可证、来源和修改说明位于 `licenses/cubek-matmul`。
模型保留各自清单中的来源和许可证元数据，FeatherHuBERT、VGG19 的记录在各自的 `LICENSES.json`
中；应用的 Apache-2.0 许可证不会替代模型权重的许可证。
