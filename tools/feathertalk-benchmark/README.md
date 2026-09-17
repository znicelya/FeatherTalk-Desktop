# FeatherTalk worker 命令基准测试

`feathertalk-benchmark` 直接链接 `feathertalk-worker`，在基准进程内执行并统计 worker 命令的耗时。它不会构建或启动独立的 worker 可执行文件，因此计时只覆盖命令本身（模型加载、媒体处理、计算），不包含进程启动与握手开销。

每条命令先执行 1 次（first run，记为 repeat 0），再重复执行 `--repeats` 次（warm runs，repeat 1…N）。最终报告给出首次耗时与预热样本的中位数/最小值/最大值。

## 使用流程

### 前置条件

- **Rust 工具链**：仓库根目录的 `rust-toolchain.toml` 锁定 `1.94.0`。安装了 rustup 时进入仓库会自动切换到该版本，无需手动指定。
- **FFmpeg / FFprobe**：`normalize_media`、`extract_frames` 等媒体命令通过它们解码与标准化视频/音频。
- **模型产物**：SCRFD、PFLD、FeatherHuBERT、VGG19 四个目录。克隆仓库后 `models/` 下已自带 `scrfd_2_5g`、`pfld_ghost_one`、`feather_hubert`、`vgg19`，可直接使用；也可以指向自行下载的目录。
- **GPU 加速（可选）**：NVIDIA CUDA 或 Linux ROCm。未配置时 worker 使用 `auto`（CUDA → ROCm → wgpu → CPU）自动回退。

### 1. 克隆仓库

```powershell
git clone https://github.com/znicelya/FeatherTalk-Desktop.git
cd FeatherTalk-Desktop
```

基准工具位于 `tools/feathertalk-benchmark`，属于根 workspace，不需要单独进入该目录，在仓库根目录执行后续所有 cargo 命令即可。

### 2. 配置环境变量

worker 的工具链与模型路径全部从环境变量读取（与独立 worker 使用的变量完全相同）。在**仓库根目录**打开终端，按当前平台设置：

```powershell
# PowerShell（当前会话）
$env:FEATHERTALK_WORKER_FFMPEG     = 'C:\tools\ffmpeg\bin\ffmpeg.exe'
$env:FEATHERTALK_WORKER_FFPROBE    = 'C:\tools\ffmpeg\bin\ffprobe.exe'
$env:FEATHERTALK_WORKER_SCRFD_DIR  = '.\models\scrfd_2_5g'
$env:FEATHERTALK_WORKER_PFLD_DIR   = '.\models\pfld_ghost_one'
$env:FEATHERTALK_WORKER_HUBERT_DIR = '.\models\feather_hubert'
$env:FEATHERTALK_WORKER_VGG19_DIR  = '.\models\vgg19'
```

```bash
# bash / zsh（当前会话）
export FEATHERTALK_WORKER_FFMPEG=/usr/bin/ffmpeg
export FEATHERTALK_WORKER_FFPROBE=/usr/bin/ffprobe
export FEATHERTALK_WORKER_SCRFD_DIR=./models/scrfd_2_5g
export FEATHERTALK_WORKER_PFLD_DIR=./models/pfld_ghost_one
export FEATHERTALK_WORKER_HUBERT_DIR=./models/feather_hubert
export FEATHERTALK_WORKER_VGG19_DIR=./models/vgg19
```

完整变量说明：

| 变量 | 说明 | 默认值 |
| --- | --- | --- |
| `FEATHERTALK_WORKER_FFMPEG` | ffmpeg 可执行文件路径 | PATH 中的 `ffmpeg` |
| `FEATHERTALK_WORKER_FFPROBE` | ffprobe 可执行文件路径 | PATH 中的 `ffprobe` |
| `FEATHERTALK_WORKER_SCRFD_DIR` | SCRFD 人脸检测模型目录 | 无 |
| `FEATHERTALK_WORKER_PFLD_DIR` | PFLD 关键点模型目录 | 无 |
| `FEATHERTALK_WORKER_HUBERT_DIR` | FeatherHuBERT 音频模型包目录 | 无 |
| `FEATHERTALK_WORKER_VGG19_DIR` | VGG19 感知损失模型目录 | 无 |
| `FEATHERTALK_WORKER_MEDIA_TIMEOUT_MS` | 媒体工具超时（毫秒） | `300000` |
| `FEATHERTALK_WORKER_BACKEND` | 计算后端：`auto`/`cpu`/`wgpu`/`cuda`/`rocm` | 未设置时为 `auto` |
| `FEATHERTALK_WORKER_ADAPTER` | 指定适配器 ID（见 `capabilities` 输出） | 无 |

- `probe_media`、`normalize_media`、`render`：只需 `FEATHERTALK_WORKER_FFMPEG` 与 `FEATHERTALK_WORKER_FFPROBE`。
- `extract_frames`：媒体工具链 + `FEATHERTALK_WORKER_SCRFD_DIR` 与 `FEATHERTALK_WORKER_PFLD_DIR`。
- `extract_features`、`lock_asset_package`：只需 `FEATHERTALK_WORKER_HUBERT_DIR`。
- `train`：只需 `FEATHERTALK_WORKER_VGG19_DIR`（训练不读媒体文件）。

用不到的变量可以不设置；缺少工具链的命令会被 worker 拒绝，在基准报告中以任务失败的形式报错。本仓库 `models/` 目录已自带全部四个模型包，克隆后即可使用。

### 3. 准备请求文件

请求文件是一个 JSON 数组，元素为 worker 的 `Request` 对象（`command` + `params`，与 JSON 线协议一致）：

```json
[
  {
    "command": "probe_media",
    "params": {
      "input": "tools/feathertalk-benchmark/fixtures/kanghui_5s.mp4"
    }
  },
  {
    "command": "normalize_media",
    "params": {
      "input": "tools/feathertalk-benchmark/fixtures/kanghui_5s.mp4",
      "output_dir": "target/benchmark/normalize-{{repeat}}"
    }
  }
]
```

**`{{repeat}}` 占位符**：任何字符串字段都可以包含 `{{repeat}}`。首次运行时它被替换为 `0`，之后的每次预热运行被替换为 `1`、`2`、…… 这样让写输出的命令在每次重复时写到不同目录，避免重复执行覆盖前一次结果。不需要隔离输出的命令（如只读的 `probe_media`）可以省略该占位符。

每个重复开始前请求文件会被重新读取，因此**数组长度与命令顺序必须在所有重复中保持一致**，否则会报 `the request file changed command order between repeats`。数组不能为空。

可用命令（snake_case，与线协议一致）：

| 命令 | 必需参数 |
| --- | --- |
| `probe_media` | `input` |
| `normalize_media` | `input`, `output_dir` |
| `validate_project` | `project_dir` |
| `lock_asset_package` | `project_dir` |
| `extract_frames` | `project_dir`, `video` |
| `extract_features` | `project_dir`, `audio` |
| `train` | `project_dir`, `mode`（`baseline`/`mouth_roi`/`temporal`）, `variant`（`original_unet`/`mobileone_unet`）, `epochs`, `resume`；`batch_size` 可选，默认 `1` |
| `render` | `project_dir`, `checkpoint`, `audio`, `output`, `max_output_frames`（`null` 表示渲染全长） |
| `inspect_model` | `source` |
| `import_legacy_model` | `source`, `kind`（`feather_hubert`/`pfld`/`original_unet`/`mobileone_unet`）, `destination` |
| `export_model_package` | `source`, `destination` |
| `export_onnx` | `source`, `kind`（`feather_hubert`/`original_unet`/`mobileone_unet`）, `destination` |
| `migrate_legacy_features` | `source`, `destination` |

示例使用仓库自带的片段 `tools/feathertalk-benchmark/fixtures/kanghui_5s.mp4`（1280×720 HEVC + AAC，约 5 秒，2.5 MB）。这两条命令只需要 ffmpeg/ffprobe，不需要任何模型目录。首次运行（`{{repeat}}` → `0`）就会真正执行命令并写出产物。

### 4. 首次构建

```powershell
cargo build --locked -p feathertalk-benchmark
```

首次编译需要构建整个 worker 依赖链（含 Burn），耗时较长；后续增量编译很快。

### 5. 运行基准测试

直接用第 3 步的 JSON 生成请求文件，然后运行（两条命令都只依赖 ffmpeg/ffprobe）：

```powershell
New-Item -ItemType Directory -Force target/benchmark | Out-Null

$json = @'
[
  {
    "command": "probe_media",
    "params": {
      "input": "tools/feathertalk-benchmark/fixtures/kanghui_5s.mp4"
    }
  },
  {
    "command": "normalize_media",
    "params": {
      "input": "tools/feathertalk-benchmark/fixtures/kanghui_5s.mp4",
      "output_dir": "target/benchmark/normalize-{{repeat}}"
    }
  }
]
'@
[System.IO.File]::WriteAllText("$PWD\target\benchmark\requests.json", $json, [System.Text.UTF8Encoding]::new($false))

cargo run --locked -p feathertalk-benchmark -- `
    --request-file target/benchmark/requests.json `
    --repeats 3 `
    --label main
```

命令行参数：

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `--request-file <PATH>` | 请求文件路径（JSON 数组） | 必填 |
| `--repeats <N>` | 首次执行之后的预热重复次数，`0` 表示只跑一次 | `3` |
| `--timeout-secs <F>` | 单次命令超时秒数，超时后取消该命令并报错 | `900` |
| `--label <STRING>` | 报告标签，用于区分不同批次 | `main` |
| `--backend <auto\|cpu\|wgpu\|cuda\|rocm>` | 覆盖计算后端（优先级高于环境变量） | 无（沿用 worker 环境变量） |
| `--adapter <ID>` | 指定计算适配器 ID | 无 |
| `--json` | 输出机器可读的 JSON 报告 | 关闭（输出表格） |

> 说明：`--backend` 提供 `auto`/`cpu`/`wgpu`/`cuda`/`rocm`。显式指定的后端在设备不可用时会直接报错，不会自动换用其他设备。

### 6. 阅读报告

默认输出表格（以下为示意输出，数值随实际测试变化）：

```text
FeatherTalk worker command benchmark
label: main
requests: target/benchmark/requests.json
repeats: 3

command                 first      warm median      warm min      warm max      warm n
probe_media              0.310s         0.298s       0.291s       0.305s          3
normalize_media          4.290s         1.882s       1.864s       1.898s          3
```

- `first`：全新执行 1 次的耗时（repeat 0），包含首次模型加载、编译缓存写充等一次性开销，通常明显高于预热样本。
- `warm median`/`warm min`/`warm max`：预热样本的中位数、最小值、最大值，用于比较稳态性能。
- `warm n`：预热样本数，等于 `--repeats`。

需要后续分析时加 `--json`，输出为 `BenchmarkReport` 结构（含每条命令的完整样本）：

```powershell
cargo run --locked -p feathertalk-benchmark -- `
    --request-file target/benchmark/requests.json --json `
    | Out-File target/benchmark/report-main.json
```

### 7. 对比不同计算后端

用 `--backend` 在同一批请求上分别测量，配合不同 `--label` 区分报告：

```powershell
cargo run --locked --release -p feathertalk-benchmark -- --request-file requests.json --backend cpu  --label cpu  --json | Out-File report-cpu.json
cargo run --locked --release -p feathertalk-benchmark -- --request-file requests.json --backend wgpu --label wgpu --json | Out-File report-wgpu.json
cargo run --locked --release -p feathertalk-benchmark -- --request-file requests.json --backend cuda --label cuda --json | Out-File report-cuda.json
cargo run --locked --release -p feathertalk-benchmark -- --request-file requests.json --backend rocm --label rocm --json | Out-File report-rocm.json
```

指定 GPU 设备时用 `--adapter <ID>`，ID 来自 `cargo run --locked -p feathertalk-cli -- capabilities` 的适配器列表。显式指定的后端或设备不可用时**会直接报错**，不会自动换用其他设备。

## 常见问题

| 报错 | 原因与处理 |
| --- | --- |
| `invalid request file ...` | JSON 不符合 `Request` 结构（字段拼写错误、缺少必需参数、混入未知字段——协议使用 `deny_unknown_fields`）。按第 3 步的参数表核对。 |
| `the request file must contain at least one request` | 请求文件是空数组。 |
| `the request file changed command order between repeats` | 重复之间命令顺序或数量发生了变化。`{{repeat}}` 只替换字符串值，不要用它增删数组元素。 |
| `... did not finish within ...s` | 命令超过 `--timeout-secs`，已被取消。调大超时或缩短任务（例如渲染时设置 `max_output_frames`）。 |
| `... failed (CODE): summary` | worker 命令本身失败，错误码与摘要来自 worker 的任务错误。常见于模型路径未配置或媒体文件不存在。 |
| `... was cancelled` | 命令被取消令牌终止（例如超时触发）。 |
| `benchmark failed: I/O error: ...` | 请求文件读取失败（路径错误、无权限）。 |

所有命令**串行执行**，互不并行。不要在测试期间同时跑其他训练或 GPU 压测任务，否则样本不可比。

## 测试

```powershell
cargo test --locked -p feathertalk-benchmark
```

测试使用注入的执行器（`run_benchmark_with_executor`），不依赖真实的模型产物或媒体文件，可在任何环境运行。