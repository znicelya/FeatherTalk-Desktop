# FeatherTalk Linux 打包

Linux x86_64 打包脚本把 FeatherTalk 工作台、后台 worker、命令行工具、
可选的 FFmpeg/FFprobe、四个模型包和许可/说明文件打成一个可重定位的
`.tar.gz`，附带 `install.sh`、桌面入口、SHA-256 校验文件和 `payload.json`
文件清单。使用程序无需安装 Rust、Python 或 .NET。

Linux 上检测到 NVIDIA 显卡及可用的 CUDA Toolkit 12.x 或更新版本时，模型计算
优先使用 `burn-cuda`，并用 FFmpeg CUDA 硬件解码抽帧与标准化。AMD 计算卡可使用
`burn-rocm`（需要兼容的 ROCm，不依赖 Vulkan）。CUDA/ROCm 不可用时自动回退到
wgpu 的 Vulkan 后端，没有可用 GPU 时使用 CPU。CUDA/ROCm 不随包分发，需自行安装
驱动与运行库。

脚本只打包 `feathertalk-app`、`feathertalk-worker`、`feathertalk` 三个产品程序，
以及运行依赖、转换后的基础模型和说明；原始模型、测试用 worker、Cargo 缓存和源码
不进入归档。版本号从两个 Cargo workspace 读取并检查一致。

## 构建

构建机需要 Linux x86_64、Rust 1.95（见 `rust-toolchain.toml`）、Python 3，以及
可选的 FFmpeg。若打包 FFmpeg，需提供含 `ffmpeg`、`ffprobe` 可执行文件的目录。

在仓库根目录运行：

```bash
installer/linux/build.sh --ffmpeg-dir /usr/bin
```

常用选项：

- `--ffmpeg-dir DIR`：FFmpeg/FFprobe 所在目录（可直接是 `bin/` 或其父目录）；
  不传时从 `PATH` 定位。
- `--no-ffmpeg`：不打包 FFmpeg，运行时使用系统 `ffmpeg` 或
  `FEATHERTALK_WORKER_FFMPEG`/`FEATHERTALK_WORKER_FFPROBE` 指定的绝对路径。
- `--source-repo DIR`：读取项目 `LICENSE` 的目录。默认直接用仓库根的
  `LICENSE`（Apache-2.0），通常无需指定；仅当仓库根缺少 `LICENSE` 的旧布局才需要
  指向包含 `LICENSE` 的仓库。
- `--output-dir DIR`：产物输出目录，默认 `<repo>/dist`。
- `--skip-build`：复用 `target/release` 中已有的二进制，不重新编译。
- `--scrfd-dir` / `--pfld-dir` / `--hubert-dir` / `--vgg19-dir`：覆盖为其他已转换
  的模型包目录；每个包都会核对原始模型身份与权重哈希。

模型打包默认直接使用仓库 `models/` 下的四个预构建包：`models/scrfd_2_5g`、
`models/pfld_ghost_one`、`models/feather_hubert`、`models/vgg19`。每个包已自带
`manifest.json`、`model.safetensors` 与许可元数据（FeatherHuBERT、VGG19 另带
`LICENSES.json`），构建时不再从原始 `.pth`/`.onnx` 检查点转换。模型身份校验逻辑
与 Windows 共享同一套哈希，集中在 `installer/linux/models.py`。

输出位于 `dist`（或 `--output-dir`）：

- `FeatherTalk-<version>-linux-x86_64.tar.gz`
- 同名 `.tar.gz.sha256` 校验文件
- `FeatherTalk-<version>-linux-x86_64.payload.json` 文件清单、哈希及模型来源

每次构建使用 `target/installer` 下的独立暂存目录，结束后清理。

## 安装与使用

解压后运行包内 `install.sh`：

```bash
tar -xzf FeatherTalk-0.1.0-linux-x86_64.tar.gz
cd FeatherTalk-0.1.0-linux-x86_64
sudo ./install.sh --prefix /opt/feathertalk --bindir /usr/local/bin
```

安装器把 `bin/`、`models/` 和说明文件复制到前缀目录，并在 `bindir`
创建 `feathertalk-app`、`feathertalk` 的软链接，再写入桌面入口。卸载：

```bash
sudo /opt/feathertalk/install.sh --uninstall
```

归档本身是可重定位的，也可以不安装，直接从解压目录运行
`bin/feathertalk-app` 或 `bin/feathertalk`。

程序自动从二进制同级的 `models/` 目录查找四个模型，并优先使用同目录内的
`ffmpeg`/`ffprobe`；因此 `bin/` 内的三个程序、FFmpeg 工具与 `models/` 需保持在同一
安装目录内。仍可用 `FEATHERTALK_WORKER_FFMPEG`、`FEATHERTALK_WORKER_FFPROBE`、
`FEATHERTALK_WORKER_SCRFD_DIR`、`FEATHERTALK_WORKER_PFLD_DIR`、
`FEATHERTALK_WORKER_HUBERT_DIR`、`FEATHERTALK_WORKER_VGG19_DIR` 指向其他绝对路径，
显式配置优先。CLI 支持 `--backend auto|cpu|wgpu|cuda|rocm`（默认 `auto`）和
`--adapter <ID>`，详情见项目 [README](../../README.md#nvidia-cuda-加速)。

```bash
feathertalk --help
feathertalk --backend cpu capabilities
```

## 验证

```bash
installer/linux/verify.sh --archive dist/FeatherTalk-0.1.0-linux-x86_64.tar.gz
```

验证器校验归档 SHA-256，解压到临时目录后逐个核对文件大小与哈希（并拒绝清单外的
多余文件），核对四个模型包的来源与权重，最后在 CPU 上运行随包 CLI 的
`capabilities` 确认二进制可加载。验证不改动系统，临时目录在退出时清理。

## 许可

FeatherTalk 使用 Apache-2.0（归档内 `LICENSE.txt`）。打包 FFmpeg 时附带
`FFmpeg-LICENSE.txt`、`FFmpeg-README.txt`（存在时）；其余第三方说明见
`THIRD-PARTY-NOTICES.txt`。Cubek 矩阵乘法内核来自 crates.io 的 `cubek-matmul 0.3.0`
依赖（MIT 或 Apache-2.0）。模型保留各自清单中的来源与许可元数据，
FeatherHuBERT、VGG19 的记录在各自的 `LICENSES.json`；应用的 Apache-2.0 许可不会
替代模型权重的许可证。
