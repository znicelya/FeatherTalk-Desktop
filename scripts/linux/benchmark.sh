#!/usr/bin/env bash
# FeatherTalk Linux 基准测试脚本（含 FFmpeg 静态包安装）
# 用法: ./run_benchmark.sh [OPTIONS]

set -euo pipefail

# ---------- 默认配置 ----------
REPO_URL="https://github.com/znicelya/FeatherTalk-Desktop.git"
REPEATS=3
BACKENDS=("cpu" "wgpu" "cuda" "rocm")
REQUEST_FILE="target/benchmark/requests.json"
FIXTURE="tools/feathertalk-benchmark/fixtures/kanghui_5s.mp4"
REPORT_DIR="target/benchmark/reports"

# FFmpeg 静态包配置
FFMPEG_INSTALL_DIR="/opt/ffmpeg"
FFMPEG_VERSION="7.0.2"
FFMPEG_ARCHIVE="ffmpeg-release-amd64-static.tar.xz"
FFMPEG_URL="https://johnvansickle.com/ffmpeg/releases/${FFMPEG_ARCHIVE}"
FFMPEG_MIN_MAJOR=5

# ---------- 帮助信息 ----------
usage() {
  cat <<EOF
FeatherTalk Linux 基准测试脚本

用法:
  $0 [OPTIONS]

选项:
  --repo-url <URL>     指定仓库地址（默认: $REPO_URL）
  --repeats <N>        预热重复次数，0 表示只跑一次（默认: $REPEATS）
  --backends <LIST>    指定要测试的后端，逗号分隔
                       可选: cpu,wgpu,cuda,rocm（默认: cpu,wgpu,cuda,rocm）
  --skip-ffmpeg        跳过 FFmpeg 静态包安装（使用系统已有版本）
  --skip-build         跳过 cargo build 步骤（仅 cargo run）
  -h, --help           显示本帮助信息并退出

示例:
  # 默认：安装 FFmpeg 静态包，四后端各跑 3 次预热
  $0

  # 只测 CPU 和 WGPU，预热 5 次
  $0 --backends cpu,wgpu --repeats 5

  # 使用已有 FFmpeg，指定 fork 仓库
  $0 --skip-ffmpeg --repo-url https://github.com/yourfork/FeatherTalk-Desktop.git

环境变量（可在运行前 export 覆盖）:
  FEATHERTALK_WORKER_FFMPEG        ffmpeg 可执行文件路径
  FEATHERTALK_WORKER_FFPROBE       ffprobe 可执行文件路径
  FEATHERTALK_WORKER_SCRFD_DIR     SCRFD 人脸检测模型目录
  FEATHERTALK_WORKER_PFLD_DIR      PFLD 关键点模型目录
  FEATHERTALK_WORKER_HUBERT_DIR    FeatherHuBERT 音频模型目录
  FEATHERTALK_WORKER_VGG19_DIR     VGG19 感知损失模型目录

输出:
  target/benchmark/reports/report-<backend>.json   各后端 JSON 报告

退出码:
  0   全部成功
  1   参数错误或环境不满足
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
    --skip-ffmpeg)  SKIP_FFMPEG=true; shift ;;
    --skip-build)   SKIP_BUILD=true; shift ;;
    -h|--help)      usage; exit 0 ;;
    *) echo "未知参数: $1"; echo; usage; exit 1 ;;
  esac
done

# 校验后端取值
VALID_BACKENDS=("cpu" "wgpu" "cuda" "rocm")
for b in "${BACKENDS[@]}"; do
  if [[ ! " ${VALID_BACKENDS[*]} " =~ " $b " ]]; then
    err "无效后端: $b（可选: ${VALID_BACKENDS[*]}）"
    exit 1
  fi
done

log()  { printf '\033[1;34m[bench]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[warn]\033[0m %s\n' "$*"; }
err()  { printf '\033[1;31m[err]\033[0m %s\n' "$*" >&2; }

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
  PKG_MGR="apt-get"
  PKG_INSTALL="sudo apt-get install -y"
elif command -v dnf &>/dev/null; then
  PKG_MGR="dnf"
  PKG_INSTALL="sudo dnf install -y"
elif command -v yum &>/dev/null; then
  PKG_MGR="yum"
  PKG_INSTALL="sudo yum install -y"
elif command -v pacman &>/dev/null; then
  PKG_MGR="pacman"
  PKG_INSTALL="sudo pacman -S --noconfirm"
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

# ================================================================
# 2. 安装基础系统依赖（不含 FFmpeg）
# ================================================================
log "安装基础系统依赖..."

case "$PKG_MGR" in
  apt-get)  $PKG_INSTALL curl git wget xz-utils build-essential pkg-config libssl-dev ;;
  dnf|yum)  $PKG_INSTALL curl git wget xz gcc gcc-c++ make pkgconfig openssl-devel ;;
  pacman)   $PKG_INSTALL curl git wget xz base-devel openssl ;;
esac

# ================================================================
# 3. 安装 FFmpeg 静态包（>= 5.x）
# ================================================================
if $SKIP_FFMPEG; then
  warn "--skip-ffmpeg 已指定，跳过 FFmpeg 安装"
fi

if ! $SKIP_FFMPEG; then
  log "安装 FFmpeg 静态构建（${FFMPEG_VERSION} release）..."

  if command -v ffmpeg &>/dev/null && command -v ffprobe &>/dev/null; then
    CURRENT_MAJOR=$(ffmpeg -version 2>/dev/null | head -1 | sed -n 's/.*ffmpeg version \([0-9]*\).*/\1/p')
    if [[ -n "$CURRENT_MAJOR" && "$CURRENT_MAJOR" -ge "$FFMPEG_MIN_MAJOR" ]]; then
      log "已检测到满足要求的 FFmpeg: $(ffmpeg -version 2>&1 | head -1)"
    else
      warn "现有 FFmpeg 版本过低（major=${CURRENT_MAJOR:-unknown}），将安装静态包覆盖"
    fi
  fi

  INSTALL_FFMPEG=true
  if [[ -x /usr/local/bin/ffmpeg ]]; then
    LOCAL_MAJOR=$(/usr/local/bin/ffmpeg -version 2>/dev/null | head -1 | sed -n 's/.*ffmpeg version \([0-9]*\).*/\1/p')
    if [[ -n "$LOCAL_MAJOR" && "$LOCAL_MAJOR" -ge "$FFMPEG_MIN_MAJOR" ]]; then
      INSTALL_FFMPEG=false
      log "已存在 /usr/local/bin/ffmpeg (major=$LOCAL_MAJOR)，跳过安装"
    fi
  fi

  if $INSTALL_FFMPEG; then
    sudo mkdir -p "$FFMPEG_INSTALL_DIR"
    cd "$FFMPEG_INSTALL_DIR"

    if [[ ! -f "$FFMPEG_ARCHIVE" ]]; then
      log "下载 $FFMPEG_URL ..."
      sudo wget -q "$FFMPEG_URL"
      sudo wget -q "${FFMPEG_URL}.md5"
    fi

    if sudo md5sum -c "${FFMPEG_ARCHIVE}.md5" 2>/dev/null | grep -q "OK"; then
      log "MD5 校验通过"
    else
      err "MD5 校验失败，请检查下载文件"
      exit 1
    fi

    sudo tar xf "$FFMPEG_ARCHIVE"
    FFMPEG_DIR=$(find . -maxdepth 1 -type d -name "ffmpeg-*-static" | head -1)
    if [[ -z "$FFMPEG_DIR" ]]; then
      err "未找到解压后的 ffmpeg-*-static 目录"
      exit 1
    fi
    cd "$FFMPEG_DIR"

    sudo cp ffmpeg ffprobe /usr/local/bin/
    sudo chmod +x /usr/local/bin/ffmpeg /usr/local/bin/ffprobe

    log "FFmpeg 静态包安装完成"
  fi
fi

export PATH="/usr/local/bin:$PATH"
hash -r 2>/dev/null || true

if ! command -v ffmpeg &>/dev/null || ! command -v ffprobe &>/dev/null; then
  err "ffmpeg/ffprobe 不可用"
  exit 1
fi

FFMPEG_MAJOR=$(ffmpeg -version 2>/dev/null | head -1 | sed -n 's/.*ffmpeg version \([0-9]*\).*/\1/p')
if [[ -z "$FFMPEG_MAJOR" || "$FFMPEG_MAJOR" -lt "$FFMPEG_MIN_MAJOR" ]]; then
  err "FFmpeg 版本不满足要求（需要 >= ${FFMPEG_MIN_MAJOR}.0，当前: $(ffmpeg -version 2>&1 | head -1)）"
  exit 1
fi
log "ffmpeg: $(ffmpeg -version 2>&1 | head -1)"
log "ffprobe: $(ffprobe -version 2>&1 | head -1)"

# ================================================================
# 4. 安装 Rust 工具链
# ================================================================
log "检查 Rust 工具链..."

if ! command -v cargo &>/dev/null; then
  log "安装 rustup..."
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain none
  # shellcheck source=/dev/null
  source "$HOME/.cargo/env"
fi

export PATH="$HOME/.cargo/bin:$PATH"

if ! command -v rustup &>/dev/null; then
  err "rustup 未安装，请先安装 rustup"
  exit 1
fi

log "rustup: $(rustup --version 2>&1)"
log "cargo: $(cargo --version 2>&1)"

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
# 6. 配置环境变量
# ================================================================
log "配置 worker 环境变量..."

export FEATHERTALK_WORKER_FFMPEG="${FEATHERTALK_WORKER_FFMPEG:-$(command -v ffmpeg)}"
export FEATHERTALK_WORKER_FFPROBE="${FEATHERTALK_WORKER_FFPROBE:-$(command -v ffprobe)}"
export FEATHERTALK_WORKER_SCRFD_DIR="${FEATHERTALK_WORKER_SCRFD_DIR:-./models/scrfd_2_5g}"
export FEATHERTALK_WORKER_PFLD_DIR="${FEATHERTALK_WORKER_PFLD_DIR:-./models/pfld_ghost_one}"
export FEATHERTALK_WORKER_HUBERT_DIR="${FEATHERTALK_WORKER_HUBERT_DIR:-./models/feather_hubert}"
export FEATHERTALK_WORKER_VGG19_DIR="${FEATHERTALK_WORKER_VGG19_DIR:-./models/vgg19}"

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
  err "请确认仓库完整克隆（git clone 后文件应存在）"
  exit 1
fi

# ================================================================
# 7. 生成请求文件
# ================================================================
log "生成请求文件..."

mkdir -p target/benchmark
mkdir -p "$REPORT_DIR"

cat > "$REQUEST_FILE" <<'JSON'
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
JSON

log "请求文件: $REQUEST_FILE"

# ================================================================
# 8. 首次构建
# ================================================================
if $SKIP_BUILD; then
  warn "--skip-build 已指定，跳过 cargo build"
else
  log "构建 feathertalk-benchmark（首次编译可能较久）..."
  cargo build --locked -p feathertalk-benchmark
fi

# ================================================================
# 9. 按后端运行基准测试
# ================================================================
log "开始基准测试（repeats=$REPEATS，backends=${BACKENDS[*]}）..."

for backend in "${BACKENDS[@]}"; do
  case "$backend" in
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

  RUN_BACKEND="$backend"
  if [[ "$backend" == "rocm" ]]; then
    RUN_BACKEND="auto"
  fi

  REPORT_JSON="$REPORT_DIR/report-${backend}.json"

  FEATHERTALK_WORKER_BACKEND="$backend" \
  cargo run --locked -p feathertalk-benchmark -- \
    --request-file "$REQUEST_FILE" \
    --repeats "$REPEATS" \
    --label "$backend" \
    --backend "$RUN_BACKEND" \
    --json \
    | tee "$REPORT_JSON"

  log "报告已保存: $REPORT_JSON"
  echo
done

# ================================================================
# 10. 汇总
# ================================================================
log "基准测试完成。报告目录: $REPORT_DIR"
ls -lh "$REPORT_DIR"/*.json 2>/dev/null || warn "未生成任何 JSON 报告"