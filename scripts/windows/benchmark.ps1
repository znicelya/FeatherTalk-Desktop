#Requires -Version 5.1
<#
.SYNOPSIS
    FeatherTalk Windows 基准测试脚本（PowerShell）

.DESCRIPTION
    环境检测、FFmpeg 静态包安装、Rust 工具链、克隆仓库、生成请求文件、
    按后端（cpu/wgpu/cuda/rocm）运行 feathertalk-benchmark 并输出 JSON 报告。

    文件必须保存为 UTF-8 with BOM，否则 PowerShell 5.1 会按 GBK 解析导致中文乱码。

.PARAMETER RepoUrl
    仓库地址，默认 https://github.com/znicelya/FeatherTalk-Desktop.git

.PARAMETER Repeats
    预热重复次数，默认 3

.PARAMETER Backends
    逗号分隔的后端列表，默认 cpu,wgpu,cuda,rocm

.PARAMETER SkipFfmpeg
    跳过 FFmpeg 安装，使用系统已有版本

.PARAMETER SkipBuild
    跳过 cargo build，直接 cargo run

.EXAMPLE
    .\benchmark.ps1

.EXAMPLE
    .\benchmark.ps1 -Backends cpu,wgpu -Repeats 5

.EXAMPLE
    .\benchmark.ps1 -SkipFfmpeg -RepoUrl https://github.com/yourfork/FeatherTalk-Desktop.git
#>

[CmdletBinding()]
param(
    [string]$RepoUrl = "https://github.com/znicelya/FeatherTalk-Desktop.git",
    [int]$Repeats = 3,
    [string]$Backends = "cpu,wgpu,cuda,rocm",
    [switch]$SkipFfmpeg,
    [switch]$SkipBuild
)

$ErrorActionPreference = "Stop"

# ---------- 配置 ----------
$RequestFile = "target\benchmark\requests.json"
$Fixture     = "tools\feathertalk-benchmark\fixtures\kanghui_5s.mp4"
$ReportDir   = "target\benchmark\reports"

# FFmpeg（BtbN Builds，Windows 静态包）
$FfmpegDir      = "C:\ffmpeg"
$FfmpegUrl      = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip"
$FfmpegMinMajor = 5

# ---------- 日志函数 ----------
function Write-Log  { param($msg) Write-Host "[bench] $msg" -ForegroundColor Cyan }
function Write-Warn { param($msg) Write-Host "[warn]  $msg" -ForegroundColor Yellow }
function Write-Err  { param($msg) Write-Host "[err]   $msg" -ForegroundColor Red }

# ---------- 外部命令 helper ----------
# PowerShell 5.1 会把原生命令写到 stderr 的内容当成 NativeCommandError，
# 配合 $ErrorActionPreference = "Stop" 会中断脚本。此 helper 临时关闭该行为。
function Invoke-Native {
    param(
        [Parameter(Mandatory = $true)][string]$Command,
        [Parameter(ValueFromRemainingArguments = $true)][string[]]$Args
    )
    $prev = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $output = & $Command @Args 2>&1
        return $output
    } finally {
        $ErrorActionPreference = $prev
    }
}

# ---------- FFmpeg 版本解析 ----------
# 兼容:
#   "ffmpeg version 7.0.2 ..."                -> 7
#   "ffmpeg version 2026-06-15-git-... ..."   -> 视为开发版，返回 99
function Get-FfmpegMajor {
    param([string]$VersionLine)
    if ($VersionLine -match "ffmpeg version (\d+)\.(\d+)") {
        return [int]$Matches[1]
    }
    if ($VersionLine -match "ffmpeg version (\d{4})-") {
        return 99
    }
    return 0
}

# ---------- 校验后端 ----------
$ValidBackends = @("cpu", "wgpu", "cuda", "rocm")
$BackendList = $Backends.Split(",") | ForEach-Object { $_.Trim() } | Where-Object { $_ -ne "" }
foreach ($b in $BackendList) {
    if ($ValidBackends -notcontains $b) {
        Write-Err "无效后端: $b（可选: $($ValidBackends -join ',')）"
        exit 1
    }
}

Write-Log "FeatherTalk Windows 基准测试"
Write-Log "后端: $($BackendList -join ',') | 预热次数: $Repeats"

# ================================================================
# 1. 环境检测
# ================================================================
Write-Log "检测系统环境..."

if ($PSVersionTable.PSEdition -eq "Core" -and -not $IsWindows) {
    Write-Err "此脚本仅支持 Windows。Linux 请使用 run_benchmark.sh"
    exit 1
}

Write-Log "PowerShell: $($PSVersionTable.PSVersion) ($($PSVersionTable.PSEdition))"
Write-Log "OS: $([System.Environment]::OSVersion.VersionString)"

foreach ($cmd in @("git", "curl")) {
    if (-not (Get-Command $cmd -ErrorAction SilentlyContinue)) {
        Write-Err "未找到 $cmd，请先安装（Git for Windows / Windows 10 1803+ 自带 curl）"
        exit 1
    }
}

# GPU 检测
$HasNvidia = $false
$HasAmd    = $false

if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    $gpuName = Invoke-Native nvidia-smi --query-gpu=name --format=csv,noheader |
               Select-Object -First 1
    if ($gpuName -and $gpuName -notmatch "failed|error") {
        $HasNvidia = $true
        Write-Log "检测到 NVIDIA GPU: $gpuName"
    }
}
if (-not $HasNvidia) { Write-Log "未检测到 NVIDIA GPU（nvidia-smi 不可用）" }

try {
    $vc = Get-CimInstance Win32_VideoController -ErrorAction Stop |
          Where-Object { $_.Name -match "AMD|Radeon" }
    if ($vc) {
        $HasAmd = $true
        Write-Log "检测到 AMD GPU: $($vc[0].Name)"
    }
} catch { }
if (-not $HasAmd) { Write-Log "未检测到 AMD GPU" }

# ================================================================
# 2. 安装 FFmpeg 静态包
# ================================================================
if ($SkipFfmpeg) {
    Write-Warn "-SkipFfmpeg 已指定，跳过 FFmpeg 安装"
} else {
    Write-Log "检查 FFmpeg..."

    $ffmpegOk = $false
    $existing = Get-Command ffmpeg -ErrorAction SilentlyContinue
    if ($existing) {
        $verLine = Invoke-Native ffmpeg -version | Select-Object -First 1
        if ($verLine) {
            $curMajor = Get-FfmpegMajor $verLine
            if ($curMajor -ge $FfmpegMinMajor) {
                Write-Log "已检测到满足要求的 FFmpeg: major=$curMajor (>= $FfmpegMinMajor)"
                $ffmpegOk = $true
            } else {
                Write-Warn "现有 FFmpeg 版本过低（major=$curMajor），将安装静态包覆盖"
            }
        }
    }

    if (-not $ffmpegOk) {
        if (-not (Test-Path $FfmpegDir)) {
            New-Item -ItemType Directory -Path $FfmpegDir -Force | Out-Null
        }
        Push-Location $FfmpegDir
        try {
            $zipPath = Join-Path $FfmpegDir "ffmpeg.zip"
            if (-not (Test-Path $zipPath)) {
                Write-Log "下载 FFmpeg 静态包（BtbN Builds）..."
                Invoke-Native curl.exe -L -o $zipPath $FfmpegUrl | Out-Null
                if ($LASTEXITCODE -ne 0) { throw "FFmpeg 下载失败" }
            }

            Write-Log "解压 FFmpeg..."
            $extractDir = Join-Path $FfmpegDir "ffmpeg-master-latest-win64-gpl"
            if (Test-Path $extractDir) { Remove-Item $extractDir -Recurse -Force }
            Expand-Archive -Path $zipPath -DestinationPath $FfmpegDir -Force

            $binDir = Join-Path $FfmpegDir "bin"
            if (-not (Test-Path $binDir)) { New-Item -ItemType Directory -Path $binDir -Force | Out-Null }

            Copy-Item (Join-Path $extractDir "bin\ffmpeg.exe")  $binDir -Force
            Copy-Item (Join-Path $extractDir "bin\ffprobe.exe") $binDir -Force

            Write-Log "FFmpeg 安装完成: $binDir"
        } finally {
            Pop-Location
        }
    }
}

# 将 C:\ffmpeg\bin 加入当前会话 PATH
$FfmpegBinDir = Join-Path $FfmpegDir "bin"
if (Test-Path $FfmpegBinDir) {
    if ($env:PATH -notlike "*$FfmpegBinDir*") {
        $env:PATH = "$FfmpegBinDir;$env:PATH"
    }
}

if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
    Write-Err "ffmpeg 不可用，请检查安装"
    exit 1
}
if (-not (Get-Command ffprobe -ErrorAction SilentlyContinue)) {
    Write-Err "ffprobe 不可用"
    exit 1
}

$ffVer = Invoke-Native ffmpeg -version | Select-Object -First 1
Write-Log $ffVer

# ================================================================
# 3. 安装 Rust 工具链
# ================================================================
Write-Log "检查 Rust 工具链..."

if (-not (Get-Command cargo -ErrorAction SilentlyContinue)) {
    Write-Log "安装 rustup..."
    $rustupInit = Join-Path $env:TEMP "rustup-init.exe"
    Invoke-Native curl.exe -L -o $rustupInit "https://static.rust-lang.org/rustup/dist/x86_64-pc-windows-msvc/rustup-init.exe" | Out-Null
    if ($LASTEXITCODE -ne 0) { Write-Err "rustup-init.exe 下载失败"; exit 1 }

    & $rustupInit -y --default-toolchain none
    if ($LASTEXITCODE -ne 0) { Write-Err "rustup 安装失败"; exit 1 }

    $cargoBin = Join-Path $env:USERPROFILE ".cargo\bin"
    if ($env:PATH -notlike "*$cargoBin*") {
        $env:PATH = "$cargoBin;$env:PATH"
    }
}

if (-not (Get-Command rustup -ErrorAction SilentlyContinue)) {
    Write-Err "rustup 未安装"
    exit 1
}

Write-Log ((Invoke-Native rustup --version) -join "`n")
Write-Log ((Invoke-Native cargo  --version) -join "`n")

# ================================================================
# 4. 克隆仓库
# ================================================================
$needClone = -not (Test-Path "Cargo.toml") -or -not (Test-Path "rust-toolchain.toml")
if ($needClone) {
    Write-Log "未检测到仓库，克隆 $RepoUrl ..."
    $cloneDir = [System.IO.Path]::GetFileNameWithoutExtension($RepoUrl.TrimEnd("/"))
    if (Test-Path $cloneDir) {
        Write-Warn "目录 $cloneDir 已存在，尝试直接进入"
        Set-Location $cloneDir
    } else {
        Invoke-Native git clone $RepoUrl | ForEach-Object { Write-Host $_ }
        if ($LASTEXITCODE -ne 0) { Write-Err "git clone 失败"; exit 1 }
        Set-Location $cloneDir
    }
} else {
    Write-Log "已在仓库根目录: $(Get-Location)"
}

Write-Log "解析项目工具链 (rust-toolchain.toml)..."
$tcOut = Invoke-Native rustup show active-toolchain
if ($LASTEXITCODE -ne 0 -or -not $tcOut) {
    $tcOut = Invoke-Native rustup show
}
$tcOut | ForEach-Object { Write-Log $_ }

# ================================================================
# 5. 配置环境变量
# ================================================================
Write-Log "配置 worker 环境变量..."

if (-not $env:FEATHERTALK_WORKER_FFMPEG) {
    $env:FEATHERTALK_WORKER_FFMPEG = (Get-Command ffmpeg).Source
}
if (-not $env:FEATHERTALK_WORKER_FFPROBE) {
    $env:FEATHERTALK_WORKER_FFPROBE = (Get-Command ffprobe).Source
}
if (-not $env:FEATHERTALK_WORKER_SCRFD_DIR)  { $env:FEATHERTALK_WORKER_SCRFD_DIR  = ".\models\scrfd_2_5g" }
if (-not $env:FEATHERTALK_WORKER_PFLD_DIR)   { $env:FEATHERTALK_WORKER_PFLD_DIR   = ".\models\pfld_ghost_one" }
if (-not $env:FEATHERTALK_WORKER_HUBERT_DIR) { $env:FEATHERTALK_WORKER_HUBERT_DIR = ".\models\feather_hubert" }
if (-not $env:FEATHERTALK_WORKER_VGG19_DIR)  { $env:FEATHERTALK_WORKER_VGG19_DIR  = ".\models\vgg19" }

Write-Log "  FFMPEG     = $env:FEATHERTALK_WORKER_FFMPEG"
Write-Log "  FFPROBE    = $env:FEATHERTALK_WORKER_FFPROBE"
Write-Log "  SCRFD_DIR  = $env:FEATHERTALK_WORKER_SCRFD_DIR"

foreach ($d in @(
    $env:FEATHERTALK_WORKER_SCRFD_DIR,
    $env:FEATHERTALK_WORKER_PFLD_DIR,
    $env:FEATHERTALK_WORKER_HUBERT_DIR,
    $env:FEATHERTALK_WORKER_VGG19_DIR
)) {
    if (Test-Path $d) {
        Write-Log "  模型目录存在: $d"
    } else {
        Write-Warn "模型目录不存在: $d（依赖该模型的命令会失败）"
    }
}

if (-not (Test-Path $Fixture)) {
    Write-Err "测试 fixture 缺失: $Fixture"
    Write-Err "请确认仓库完整克隆"
    exit 1
}

# ================================================================
# 6. 生成请求文件
# ================================================================
Write-Log "生成请求文件..."

New-Item -ItemType Directory -Path "target\benchmark" -Force | Out-Null
New-Item -ItemType Directory -Path $ReportDir -Force | Out-Null

$requests = @(
    [ordered]@{
        command = "probe_media"
        params  = [ordered]@{
            input = "tools/feathertalk-benchmark/fixtures/kanghui_5s.mp4"
        }
    },
    [ordered]@{
        command = "normalize_media"
        params  = [ordered]@{
            input      = "tools/feathertalk-benchmark/fixtures/kanghui_5s.mp4"
            output_dir = "target/benchmark/normalize-{{repeat}}"
        }
    }
)

$json = $requests | ConvertTo-Json -Depth 10
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
[System.IO.File]::WriteAllText(
    (Join-Path (Get-Location) $RequestFile),
    $json,
    $utf8NoBom
)

Write-Log "请求文件: $RequestFile"

# ================================================================
# 7. 首次构建
# ================================================================
if ($SkipBuild) {
    Write-Warn "-SkipBuild 已指定，跳过 cargo build"
} else {
    Write-Log "构建 feathertalk-benchmark（首次编译可能较久）..."
    Invoke-Native cargo build --locked -p feathertalk-benchmark | ForEach-Object { Write-Host $_ }
    if ($LASTEXITCODE -ne 0) {
        Write-Err "构建失败"
        exit 1
    }
}

# ================================================================
# 8. 按后端运行基准测试
# ================================================================
Write-Log "开始基准测试（repeats=$Repeats，backends=$($BackendList -join ',')）..."

foreach ($backend in $BackendList) {
    if ($backend -eq "cuda" -and -not $HasNvidia) {
        Write-Warn "跳过 cuda: 未检测到 NVIDIA GPU"
        continue
    }
    if ($backend -eq "rocm" -and -not $HasAmd) {
        Write-Warn "跳过 rocm: 未检测到 AMD GPU 或 ROCm 环境"
        continue
    }

    Write-Log "===== 后端: $backend ====="

    # ROCm 通过环境变量控制（--backend 只支持 auto/cpu/wgpu/cuda）
    $runBackend = $backend
    if ($backend -eq "rocm") { $runBackend = "auto" }

    $reportJson = Join-Path $ReportDir "report-$backend.json"

    $env:FEATHERTALK_WORKER_BACKEND = $backend

    $args = @(
        "run", "--locked", "-p", "feathertalk-benchmark", "--",
        "--request-file", $RequestFile,
        "--repeats", $Repeats,
        "--label", $backend,
        "--backend", $runBackend,
        "--json"
    )

    $output = Invoke-Native cargo @args
    $output | Out-File -FilePath $reportJson -Encoding utf8

    if ($LASTEXITCODE -ne 0) {
        Write-Warn "后端 $backend 测试失败，检查 $reportJson"
    } else {
        Write-Log "报告已保存: $reportJson"
    }
    Write-Host ""
}

# ================================================================
# 9. 汇总
# ================================================================
Write-Log "基准测试完成。报告目录: $ReportDir"
$reports = Get-ChildItem -Path $ReportDir -Filter "*.json" -ErrorAction SilentlyContinue
if ($reports) {
    $reports | Format-Table Name, Length, LastWriteTime -AutoSize
} else {
    Write-Warn "未生成任何 JSON 报告"
}