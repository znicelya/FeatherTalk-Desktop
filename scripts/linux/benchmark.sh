#!/usr/bin/env bash
# FeatherTalk Linux 基准测试脚本
#   - FFmpeg: BtbN/FFmpeg-Builds gpl 变体
#   - Rustup: 华为云镜像 + 官方源回退
#   - 命令: probe_media, normalize_media, extract_frames,
#           extract_features, train, render
# 用法: ./run_benchmark.sh [OPTIONS]

set -euo pipefail

# ---------- 默认配置 ----------
REPO_URL="https://github.com/znicelya/FeatherTalk-Desktop.git"
REPEATS=3
BACKENDS=("cpu" "wgpu" "cuda" "rocm")
PROJECT_ROOT="target/benchmark/full-pipeline"
FIXTURE="tools/feathertalk-benchmark/fixtures/kanghui_5s.mp4"
REPORT_DIR="target/benchmark/reports"
TIMEOUT_SECS=10800
EPOCHS=3

# FFmpeg 静态包配置（BtbN/FFmpeg-Builds，gpl 变体）
FFMPEG_INSTALL_DIR="/opt/ffmpeg"
FFMPEG_VARIANT="gpl"
FFMPEG_ARCHIVE="ffmpeg-master-latest-linux64-${FFMPEG_VARIANT}.tar.xz"
FFMPEG_URL="https://gh-proxy.org/https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/${FFMPEG_ARCHIVE}"
FFMPEG_DIR_NAME="ffmpeg-master-latest-linux64-${FFMPEG_VARIANT}"
FFMPEG_MIN_MAJOR=5

# ---------- Rustup 镜像（华为云） ----------
export RUSTUP_DIST_SERVER="${RUSTUP_DIST_SERVER:-https://repo.huaweicloud.com/rustup}"
export RUSTUP_UPDATE_ROOT="${RUSTUP_UPDATE_ROOT:-https://repo.huaweicloud.com/rustup/rustup}"

# ---------- 帮助信息 ----------
usage() {
  cat <<EOF
FeatherTalk Linux 基准测试脚本

用法:
  $0 [OPTIONS]

选项:
  --repo-url <URL>        指定仓库地址（默认: $REPO_URL）
  --repeats <N>           预热重复次数，0 表示只跑一次（默认: $REPEATS）
  --backends <LIST>       逗号分隔的后端列表
                          可选: cpu,wgpu,cuda,rocm（默认: cpu,wgpu,cuda,rocm）
  --project-root <PATH>    full pipeline 临时项目根目录（默认: $PROJECT_ROOT）
  --skip-ffmpeg           跳过 FFmpeg 静态包安装
  --skip-build            跳过 cargo build
  --timeout-secs <N>     单次命令超时秒数（默认: $TIMEOUT_SECS）
  --epochs <N>           训练 epoch 数（默认: $EPOCHS）
  -h, --help              显示帮助

示例:
  $0
  $0 --backends cpu --repeats 5
  $0 --project-root /data/my_project
  $0 --skip-ffmpeg --backends cpu,cuda

环境变量:
  RUSTUP_DIST_SERVER               rustup 工具链下载源（默认华为云）
  RUSTUP_UPDATE_ROOT               rustup 自更新源（默认华为云）
  FEATHERTALK_WORKER_FFMPEG        ffmpeg 可执行文件路径
  FEATHERTALK_WORKER_FFPROBE       ffprobe 可执行文件路径
  FEATHERTALK_WORKER_SCRFD_DIR     SCRFD 人脸检测模型目录
  FEATHERTALK_WORKER_PFLD_DIR      PFLD 关键点模型目录
  FEATHERTALK_WORKER_HUBERT_DIR    FeatherHuBERT 音频模型目录
  FEATHERTALK_WORKER_VGG19_DIR     VGG19 感知损失模型目录

输出:
  target/benchmark/reports/report-<backend>.json   各后端 JSON 报告
EOF
}

# ---------- 参数解析 ----------
SKIP_FFMPEG=false
SKIP_BUILD=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo-url)     REPO_URL="$2"; shift 2 ;;
    --repeats)      REPEATS="$2"; shift 2 ;;
    --backends)
      IFS=',' read -r -a BACKENDS <<< "$2"
      shift 2
      ;;
    --project-root) PROJECT_ROOT="$2"; shift 2 ;;
    --timeout-secs) TIMEOUT_SECS="$2"; shift 2 ;;
    --epochs)       EPOCHS="$2"; shift 2 ;;
    --skip-ffmpeg)  SKIP_FFMPEG=true; shift ;;
    --skip-build)   SKIP_BUILD=true; shift ;;
    -h|--help)      usage; exit 0 ;;
    *) echo "未知参数: $1"; echo; usage; exit 1 ;;
  esac
done

log()  { printf '\033[1;34m[bench]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[warn]\033[0m %s\n' "$*"; }
err()  { printf '\033[1;31m[err]\033[0m %s\n' "$*" >&2; }

VALID_BACKENDS=("cpu" "wgpu" "cuda" "rocm")
for b in "${BACKENDS[@]}"; do
  if [[ ! " ${VALID_BACKENDS[*]} " =~ " $b " ]]; then
    err "无效后端: $b（可选: ${VALID_BACKENDS[*]}）"
    exit 1
  fi
done

# ---------- FFmpeg 版本解析 ----------
parse_ffmpeg_major() {
  local ver_line="$1"
  if [[ "$ver_line" =~ ffmpeg\ version\ ([0-9]+)\.[0-9]+ ]]; then
    echo "${BASH_REMATCH[1]}"
  elif [[ "$ver_line" =~ ffmpeg\ version\ N- ]]; then
    echo "99"
  else
    echo "0"
  fi
}

# ================================================================
# 1. 环境检测
# ================================================================
log "检测系统环境..."

if [[ -f /etc/os-release ]]; then
  . /etc/os-release
  DISTRO="${ID:-unknown}"
  log "发行版: ${PRETTY_NAME:-$DISTRO}"
else
  DISTRO="unknown"
fi

if command -v apt-get &>/dev/null; then
  PKG_MGR="apt-get"; PKG_INSTALL="sudo apt-get install -y"
elif command -v dnf &>/dev/null; then
  PKG_MGR="dnf";     PKG_INSTALL="sudo dnf install -y"
elif command -v yum &>/dev/null; then
  PKG_MGR="yum";     PKG_INSTALL="sudo yum install -y"
elif command -v pacman &>/dev/null; then
  PKG_MGR="pacman";  PKG_INSTALL="sudo pacman -S --noconfirm"
else
  err "未找到受支持的包管理器 (apt/dnf/yum/pacman)"
  exit 1
fi
log "包管理器: $PKG_MGR"

HAS_NVIDIA=false
HAS_AMD=false

if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
  HAS_NVIDIA=true
  log "检测到 NVIDIA GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
fi

if command -v rocminfo &>/dev/null && rocminfo &>/dev/null 2>&1; then
  HAS_AMD=true
  log "检测到 AMD ROCm 环境"
elif [[ -d /dev/kfd ]] && [[ -d /dev/dri ]]; then
  HAS_AMD=true
  log "检测到 AMD GPU 设备节点 (/dev/kfd, /dev/dri)"
fi

HAS_VULKAN=false
if command -v vulkaninfo >/dev/null 2>&1 && { vulkaninfo --summary >/dev/null 2>&1 || vulkaninfo >/dev/null 2>&1; }; then
  HAS_VULKAN=true
  log "detected Vulkan"
else
  log "no Vulkan runtime/adapters detected"
fi


# ================================================================
# 2. 安装基础系统依赖
# ================================================================
log "安装基础系统依赖..."

case "$PKG_MGR" in
  apt-get)
    sudo apt-get update
    $PKG_INSTALL curl git wget xz-utils build-essential pkg-config libssl-dev
    ;;
  dnf|yum)
    $PKG_INSTALL curl git wget xz gcc gcc-c++ make pkgconfig openssl-devel
    ;;
  pacman)
    $PKG_INSTALL curl git wget xz base-devel openssl
    ;;
esac

# ================================================================
# 3. 检查/安装 FFmpeg
# ================================================================
if $SKIP_FFMPEG; then
  warn "--skip-ffmpeg 已指定，跳过 FFmpeg 安装"
fi

if ! $SKIP_FFMPEG; then
  log "检查 FFmpeg..."

  INSTALL_FFMPEG=true

  if command -v ffmpeg &>/dev/null && command -v ffprobe &>/dev/null; then
    FFMPEG_VER_LINE=$(ffmpeg -version 2>/dev/null | head -1)
    CURRENT_MAJOR=$(parse_ffmpeg_major "$FFMPEG_VER_LINE")
    if [[ -n "$CURRENT_MAJOR" && "$CURRENT_MAJOR" -ge "$FFMPEG_MIN_MAJOR" ]]; then
      log "已检测到满足要求的 FFmpeg: $FFMPEG_VER_LINE"
      INSTALL_FFMPEG=false
    else
      warn "现有 FFmpeg 版本不满足要求（major=$CURRENT_MAJOR），将安装静态包覆盖"
    fi
  else
    log "未检测到 ffmpeg，将安装静态包"
  fi

  if $INSTALL_FFMPEG; then
    log "安装 FFmpeg 静态构建（BtbN latest, linux64-gpl）..."

    sudo mkdir -p "$FFMPEG_INSTALL_DIR"
    cd "$FFMPEG_INSTALL_DIR"

    if [[ ! -f "$FFMPEG_ARCHIVE" ]] || ! xz -t "$FFMPEG_ARCHIVE" 2>/dev/null; then
      log "下载 $FFMPEG_URL ..."
      sudo rm -f "$FFMPEG_ARCHIVE"
      sudo curl -fL --retry 3 --retry-delay 5 --connect-timeout 30 \
        -o "$FFMPEG_ARCHIVE" "$FFMPEG_URL"
      if ! xz -t "$FFMPEG_ARCHIVE" 2>/dev/null; then
        err "下载的 FFmpeg 压缩包损坏: $FFMPEG_ARCHIVE"
        sudo rm -f "$FFMPEG_ARCHIVE"
        exit 1
      fi
    fi

    sudo tar xf "$FFMPEG_ARCHIVE"
    FFMPEG_DIR="$FFMPEG_INSTALL_DIR/$FFMPEG_DIR_NAME"
    if [[ ! -d "$FFMPEG_DIR" ]]; then
      err "未找到解压后的目录: $FFMPEG_DIR"
      exit 1
    fi

    sudo cp "$FFMPEG_DIR/bin/ffmpeg"  /usr/local/bin/
    sudo cp "$FFMPEG_DIR/bin/ffprobe" /usr/local/bin/
    sudo chmod +x /usr/local/bin/ffmpeg /usr/local/bin/ffprobe

    log "FFmpeg 静态包安装完成（gpl 变体）"
  fi
fi

export PATH="/usr/local/bin:$PATH"
hash -r 2>/dev/null || true

if ! command -v ffmpeg &>/dev/null || ! command -v ffprobe &>/dev/null; then
  err "ffmpeg/ffprobe 不可用"
  exit 1
fi

FFMPEG_VER_LINE=$(ffmpeg -version 2>/dev/null | head -1)
FFMPEG_MAJOR=$(parse_ffmpeg_major "$FFMPEG_VER_LINE")
if [[ -z "$FFMPEG_MAJOR" || "$FFMPEG_MAJOR" -lt "$FFMPEG_MIN_MAJOR" ]]; then
  err "FFmpeg 版本不满足要求（需要 >= ${FFMPEG_MIN_MAJOR}.0）"
  err "当前: $FFMPEG_VER_LINE"
  exit 1
fi
log "ffmpeg: $FFMPEG_VER_LINE"
log "ffprobe: $(ffprobe -version 2>&1 | head -1)"

# ================================================================
# 4. 检查/安装 Rust 工具链（华为云镜像 + 官方源回退）
# ================================================================
log "检查 Rust 工具链..."

if [ -f "$HOME/.cargo/env" ]; then
    source "$HOME/.cargo/env"
fi

# 已安装则完全跳过安装流程
if command -v rustup &>/dev/null && command -v cargo &>/dev/null; then
  log "已检测到 Rust 工具链，跳过安装"
  log "  rustup: $(rustup --version 2>&1 | head -1)"
  log "  cargo:  $(cargo --version 2>&1 | head -1)"
  log "  rustc:  $(rustc --version 2>&1 | head -1)"

  # 确保 PATH 包含 cargo bin（有些环境 rustup 装了但 PATH 未刷新）
  export PATH="$HOME/.cargo/bin:$PATH"

  # 仍导出镜像变量，供后续 rustup toolchain install 使用
  export RUSTUP_DIST_SERVER
  export RUSTUP_UPDATE_ROOT

  SKIP_RUST_INSTALL=true
else
  SKIP_RUST_INSTALL=false
  log "  RUSTUP_DIST_SERVER = $RUSTUP_DIST_SERVER"
  log "  RUSTUP_UPDATE_ROOT = $RUSTUP_UPDATE_ROOT"
  log "未检测到完整 Rust 工具链，将安装..."
fi

if ! $SKIP_RUST_INSTALL; then
  # 缺 rustup 或 cargo，先装 rustup
  if ! command -v rustup &>/dev/null; then
    log "安装 rustup（使用华为云镜像）..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | \
      sh -s -- -y --default-toolchain none --no-modify-path
    # shellcheck source=/dev/null
    source "$HOME/.cargo/env"
  else
    log "已存在 rustup，跳过 rustup 安装"
  fi

  export PATH="$HOME/.cargo/bin:$PATH"
  export RUSTUP_DIST_SERVER
  export RUSTUP_UPDATE_ROOT

  if ! command -v rustup &>/dev/null; then
    err "rustup 未安装，请先安装 rustup"
    exit 1
  fi

  log "rustup: $(rustup --version 2>&1)"
fi

# ---------- 安装 rust-toolchain.toml 锁定的工具链 ----------
# 无论 rustup 是否新装，都要确保锁定版本存在
TOOLCHAIN_CHANNEL="1.94.0"
if [[ -f "rust-toolchain.toml" ]]; then
  PARSED_CHANNEL=$(grep -E '^\s*channel\s*=' rust-toolchain.toml 2>/dev/null | head -1 | sed -E 's/.*"([^"]+)".*/\1/')
  if [[ -n "$PARSED_CHANNEL" ]]; then
    TOOLCHAIN_CHANNEL="$PARSED_CHANNEL"
  fi
fi

# 若该工具链已安装，直接跳过
if rustup toolchain list 2>/dev/null | grep -qE "^${TOOLCHAIN_CHANNEL}(-|\s|$)"; then
  log "工具链 $TOOLCHAIN_CHANNEL 已安装，跳过下载"
else
  log "安装工具链 $TOOLCHAIN_CHANNEL（先试华为云镜像）..."
  if rustup toolchain install "$TOOLCHAIN_CHANNEL" 2>/dev/null; then
    log "工具链 $TOOLCHAIN_CHANNEL 安装成功（华为云镜像）"
  else
    warn "华为云镜像上没有工具链 $TOOLCHAIN_CHANNEL，回退到官方源..."
    rm -rf "$HOME/.rustup/tmp/"* 2>/dev/null || true
    if RUSTUP_DIST_SERVER= RUSTUP_UPDATE_ROOT= rustup toolchain install "$TOOLCHAIN_CHANNEL"; then
      log "工具链 $TOOLCHAIN_CHANNEL 安装成功（官方源）"
    else
      err "工具链 $TOOLCHAIN_CHANNEL 安装失败"
      exit 1
    fi
  fi
fi

if [[ "$TOOLCHAIN_CHANNEL" != "stable" && "$TOOLCHAIN_CHANNEL" != "beta" && "$TOOLCHAIN_CHANNEL" != "nightly" ]]; then
  rustup default "$TOOLCHAIN_CHANNEL" >/dev/null 2>&1 || true
fi

log "cargo: $(cargo --version 2>&1)"
log "rustc: $(rustc --version 2>&1)"

# ================================================================
# 5. 克隆仓库
# ================================================================
if [[ ! -f "Cargo.toml" ]] || [[ ! -f "rust-toolchain.toml" ]]; then
  log "未检测到仓库，克隆 $REPO_URL ..."
  CLONE_DIR="${REPO_URL##*/}"
  CLONE_DIR="${CLONE_DIR%.git}"
  if [[ -d "$CLONE_DIR" ]]; then
    warn "目录 $CLONE_DIR 已存在，尝试直接进入"
    cd "$CLONE_DIR"
  else
    git clone "$REPO_URL"
    cd "$CLONE_DIR"
  fi
else
  log "已在仓库根目录: $(pwd)"
fi

log "解析项目工具链 (rust-toolchain.toml)..."
rustup show active-toolchain 2>/dev/null || rustup show

# ================================================================
# 6. 配置 worker 环境变量
# ================================================================
log "配置 worker 环境变量..."

export FEATHERTALK_WORKER_FFMPEG="${FEATHERTALK_WORKER_FFMPEG:-$(command -v ffmpeg)}"
export FEATHERTALK_WORKER_FFPROBE="${FEATHERTALK_WORKER_FFPROBE:-$(command -v ffprobe)}"
export FEATHERTALK_WORKER_SCRFD_DIR="${FEATHERTALK_WORKER_SCRFD_DIR:-$PWD/models/scrfd_2_5g}"
export FEATHERTALK_WORKER_PFLD_DIR="${FEATHERTALK_WORKER_PFLD_DIR:-$PWD/models/pfld_ghost_one}"
export FEATHERTALK_WORKER_HUBERT_DIR="${FEATHERTALK_WORKER_HUBERT_DIR:-$PWD/models/feather_hubert}"
export FEATHERTALK_WORKER_VGG19_DIR="${FEATHERTALK_WORKER_VGG19_DIR:-$PWD/models/vgg19}"

log "  FFMPEG     = $FEATHERTALK_WORKER_FFMPEG"
log "  FFPROBE    = $FEATHERTALK_WORKER_FFPROBE"
log "  SCRFD_DIR  = $FEATHERTALK_WORKER_SCRFD_DIR"

for dir in "$FEATHERTALK_WORKER_SCRFD_DIR" "$FEATHERTALK_WORKER_PFLD_DIR" \
           "$FEATHERTALK_WORKER_HUBERT_DIR" "$FEATHERTALK_WORKER_VGG19_DIR"; do
  if [[ -d "$dir" ]]; then
    log "  模型目录存在: $dir"
  else
    warn "模型目录不存在: $dir（依赖该模型的命令会失败）"
  fi
done

if [[ ! -f "$FIXTURE" ]]; then
  err "测试 fixture 缺失: $FIXTURE"
  exit 1
fi

# ================================================================
# 7. 准备完整流水线
# ================================================================
log "准备完整流水线..."
log "每个 repeat 的项目由 feathertalk-benchmark 自动初始化: $PROJECT_ROOT"

mkdir -p target/benchmark
mkdir -p "$REPORT_DIR"

# ================================================================
# 9. 首次构建
# ================================================================
if $SKIP_BUILD; then
  warn "--skip-build 已指定，跳过 cargo build"
else
  log "构建 feathertalk-benchmark（首次编译可能较久）..."
  export CARGO_SOURCE_CRATES_IO_REPLACE_WITH=ustc
  export CARGO_SOURCE_USTC_REGISTRY='sparse+https://mirrors.ustc.edu.cn/crates.io-index/'
  cargo build --locked --release -p feathertalk-benchmark
fi

# ================================================================
# 10. 按后端运行基准测试（单个后端失败不中断）
# ================================================================
log "开始基准测试（repeats=$REPEATS，backends=${BACKENDS[*]}）..."

for backend in "${BACKENDS[@]}"; do
  case "$backend" in
    wgpu)
      if ! $HAS_VULKAN; then
        warn "skip wgpu: no Vulkan runtime/adapters detected"
        continue
      fi
      ;;
    cuda)
      if ! $HAS_NVIDIA; then
        warn "跳过 cuda: 未检测到 NVIDIA GPU"
        continue
      fi
      ;;
    rocm)
      if ! $HAS_AMD; then
        warn "跳过 rocm: 未检测到 AMD GPU 或 ROCm 环境"
        continue
      fi
      ;;
  esac

  log "===== 后端: $backend ====="

  REPORT_JSON="$REPORT_DIR/report-${backend}.json"

  if FEATHERTALK_WORKER_BACKEND="$backend" \
     cargo run --locked --release -p feathertalk-benchmark -- \
       --input "$FIXTURE" \
       --project-root "$PROJECT_ROOT" \
       --repeats "$REPEATS" \
       --epochs "$EPOCHS" \
       --timeout-secs "$TIMEOUT_SECS" \
       --label "$backend" \
       --backend "$backend" \
       --json \
       > "$REPORT_JSON"; then
    log "报告已保存: $REPORT_JSON"
  else
    rm -f "$REPORT_JSON"

    warn "后端 $backend 失败，继续下一个"
  fi
  echo
done

# ================================================================
# 11. 汇总
# ================================================================
log "基准测试完成。报告目录: $REPORT_DIR"
ls -lh "$REPORT_DIR"/*.json 2>/dev/null || warn "未生成任何 JSON 报告"
