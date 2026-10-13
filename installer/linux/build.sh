#!/usr/bin/env bash
# FeatherTalk Linux x86_64 packager.
#
# Builds the three production binaries (feathertalk-app, feathertalk-worker,
# feathertalk), stages them next to FFmpeg/FFprobe, the four prebuilt model
# packages and the license/notice files, then produces a relocatable
# `.tar.gz` with an `install.sh`, a desktop entry, a SHA-256 checksum and a
# `payload.json` manifest. This mirrors installer/windows/build.ps1 for Linux.
#
# The runtime resolves bundled models and media tools relative to the worker
# executable (see feathertalk-worker/src/config.rs), so every binary, the
# `models/` directory and the FFmpeg tools stay in one install directory.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
rust_root="$(cd -- "$script_dir/../.." >/dev/null 2>&1 && pwd)"

# -- defaults -----------------------------------------------------------------
ffmpeg_dir=""
bundle_ffmpeg=1
source_repo=""
output_dir=""
skip_build=0
scrfd_dir=""
pfld_dir=""
hubert_dir=""
vgg19_dir=""

usage() {
    cat <<'USAGE'
用法: installer/linux/build.sh [选项]

选项:
  --ffmpeg-dir DIR        FFmpeg/FFprobe 所在目录（含 ffmpeg、ffprobe，可含 bin/）。
                          不指定时从 PATH 定位。
  --no-ffmpeg             不打包 FFmpeg；运行时使用系统 ffmpeg 或
                          FEATHERTALK_WORKER_FFMPEG/FFPROBE 配置。
  --source-repo DIR       读取项目 LICENSE 的目录。默认优先用仓库根的 LICENSE，
                          找不到时再取当前 workspace 上级目录。
  --output-dir DIR        产物输出目录，默认 <repo>/dist。
  --skip-build            复用 target/release 现有二进制，不重新编译。
  --scrfd-dir DIR         覆盖 SCRFD 模型包目录（默认 models/scrfd_2_5g）。
  --pfld-dir DIR          覆盖 PFLD 模型包目录（默认 models/pfld_ghost_one）。
  --hubert-dir DIR        覆盖 FeatherHuBERT 模型包目录（默认 models/feather_hubert）。
  --vgg19-dir DIR         覆盖 VGG19 模型包目录（默认 models/vgg19）。
  -h, --help              显示本帮助。
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --ffmpeg-dir) ffmpeg_dir="${2:?--ffmpeg-dir 需要参数}"; shift 2;;
        --no-ffmpeg) bundle_ffmpeg=0; shift;;
        --source-repo) source_repo="${2:?--source-repo 需要参数}"; shift 2;;
        --output-dir) output_dir="${2:?--output-dir 需要参数}"; shift 2;;
        --skip-build) skip_build=1; shift;;
        --scrfd-dir) scrfd_dir="${2:?}"; shift 2;;
        --pfld-dir) pfld_dir="${2:?}"; shift 2;;
        --hubert-dir) hubert_dir="${2:?}"; shift 2;;
        --vgg19-dir) vgg19_dir="${2:?}"; shift 2;;
        -h|--help) usage; exit 0;;
        *) echo "未知选项: $1" >&2; usage >&2; exit 2;;
    esac
done

# -- environment checks -------------------------------------------------------
if [[ "$(uname -s)" != "Linux" ]]; then
    echo "此脚本仅支持在 Linux 上构建。" >&2
    exit 1
fi
if [[ "$(uname -m)" != "x86_64" ]]; then
    echo "当前仅支持 x86_64；检测到 $(uname -m)。" >&2
    exit 1
fi
for tool in cargo python3 tar sha256sum install; do
    command -v "$tool" >/dev/null 2>&1 || { echo "缺少必需工具: $tool" >&2; exit 1; }
done

models_py="$script_dir/models.py"

# Reject anything that is not a 64-bit x86-64 ELF so a cross-built or wrong-arch
# artifact never enters the tarball.
assert_elf_x86_64() {
    python3 - "$1" <<'PY'
import sys
path = sys.argv[1]
with open(path, "rb") as handle:
    header = handle.read(20)
if header[:4] != b"\x7fELF":
    sys.exit(f"不是 ELF 文件: {path}")
if header[4] != 2:
    sys.exit(f"不是 64 位 ELF: {path}")
machine = int.from_bytes(header[18:20], "little")
if machine != 0x3E:
    sys.exit(f"不是 x86-64 ELF (e_machine=0x{machine:x}): {path}")
PY
}

# Resolve the project LICENSE. An explicit --source-repo wins; otherwise the
# repository ships its own Apache-2.0 LICENSE at the root, and the external
# source-repo layout remains a fallback for older checkouts.
if [[ -n "$source_repo" ]]; then
    repository_root="$(cd -- "$source_repo" >/dev/null 2>&1 && pwd)"
    license_path="$repository_root/LICENSE"
elif [[ -f "$rust_root/LICENSE" ]]; then
    license_path="$rust_root/LICENSE"
else
    repository_root="$(cd -- "$rust_root/.." >/dev/null 2>&1 && pwd)"
    license_path="$repository_root/LICENSE"
fi
if [[ ! -f "$license_path" ]]; then
    echo "找不到项目 LICENSE，请用 --source-repo 指定包含 LICENSE 的仓库。" >&2
    exit 1
fi

workspace_manifest="$rust_root/Cargo.toml"
app_manifest="$rust_root/crates/feathertalk-app/Cargo.toml"

cargo_version() {
    # Reads a package version from `cargo metadata` without extra tooling.
    local manifest="$1" package="$2"
    ( cd "$rust_root" && cargo metadata --manifest-path "$manifest" --locked --no-deps --format-version 1 ) \
        | python3 -c "import json,sys; d=json.load(sys.stdin); print(next(p['version'] for p in d['packages'] if p['name']=='$package'))"
}

worker_version="$(cargo_version "$workspace_manifest" feathertalk-worker)"
app_version="$(cargo_version "$app_manifest" feathertalk-app)"
if [[ "$worker_version" != "$app_version" ]]; then
    echo "app 与 worker 版本不一致: $app_version != $worker_version" >&2
    exit 1
fi
if [[ ! "$worker_version" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    echo "版本号必须是三段数字: $worker_version" >&2
    exit 1
fi
version="$worker_version"

# -- locate FFmpeg ------------------------------------------------------------
ffmpeg_bin=""
if [[ "$bundle_ffmpeg" -eq 1 ]]; then
    if [[ -z "$ffmpeg_dir" ]]; then
        if command -v ffmpeg >/dev/null 2>&1; then
            ffmpeg_dir="$(dirname -- "$(command -v ffmpeg)")"
        else
            echo "未找到 ffmpeg；用 --ffmpeg-dir 指定，或用 --no-ffmpeg 跳过打包。" >&2
            exit 1
        fi
    fi
    ffmpeg_dir="$(cd -- "$ffmpeg_dir" >/dev/null 2>&1 && pwd)"
    ffmpeg_bin="$ffmpeg_dir"
    if [[ -x "$ffmpeg_dir/bin/ffmpeg" ]]; then
        ffmpeg_bin="$ffmpeg_dir/bin"
    fi
    for tool in ffmpeg ffprobe; do
        if [[ ! -x "$ffmpeg_bin/$tool" ]]; then
            echo "FFmpeg 目录缺少可执行的 $tool: $ffmpeg_bin/$tool" >&2
            exit 1
        fi
        assert_elf_x86_64 "$ffmpeg_bin/$tool"
    done
fi

# -- build release binaries ---------------------------------------------------
if [[ "$skip_build" -eq 0 ]]; then
    echo "编译 release worker 与 CLI ..."
    ( cd "$rust_root" && cargo build --manifest-path "$workspace_manifest" --release --locked \
        -p feathertalk-worker -p feathertalk-cli --bin feathertalk-worker --bin feathertalk )
    echo "编译 release 桌面程序 ..."
    ( cd "$rust_root" && cargo build --manifest-path "$app_manifest" --release --locked --bin feathertalk-app )
fi

workspace_target="$( ( cd "$rust_root" && cargo metadata --manifest-path "$workspace_manifest" --locked --no-deps --format-version 1 ) | python3 -c "import json,sys;print(json.load(sys.stdin)['target_directory'])")"
app_target="$( ( cd "$rust_root" && cargo metadata --manifest-path "$app_manifest" --locked --no-deps --format-version 1 ) | python3 -c "import json,sys;print(json.load(sys.stdin)['target_directory'])")"

app_executable="$app_target/release/feathertalk-app"
worker_executable="$workspace_target/release/feathertalk-worker"
cli_executable="$workspace_target/release/feathertalk"
for binary in "$app_executable" "$worker_executable" "$cli_executable"; do
    if [[ ! -f "$binary" ]]; then
        echo "找不到构建产物: $binary（可能需要去掉 --skip-build）" >&2
        exit 1
    fi
    assert_elf_x86_64 "$binary"
done

# -- stage payload ------------------------------------------------------------
package_name="FeatherTalk-$version-linux-x86_64"
# build_dir="$rust_root/target/installer/build-$(date +%s)-$$"
build_dir="$HOME/.feathertalk-installer/build-$(date +%s)-$$"
stage="$build_dir/$package_name"
rm -rf "$build_dir"
mkdir -p "$stage/bin" "$stage/models" "$stage/licenses"

install -m 0755 "$app_executable" "$stage/bin/feathertalk-app"
install -m 0755 "$worker_executable" "$stage/bin/feathertalk-worker"
install -m 0755 "$cli_executable" "$stage/bin/feathertalk"
if [[ "$bundle_ffmpeg" -eq 1 ]]; then
    install -m 0755 "$ffmpeg_bin/ffmpeg" "$stage/bin/ffmpeg"
    install -m 0755 "$ffmpeg_bin/ffprobe" "$stage/bin/ffprobe"
fi

# -- bundle and validate the four model packages -----------------------------
declare -A model_sources=(
    [scrfd_2_5g]="${scrfd_dir:-$rust_root/models/scrfd_2_5g}"
    [pfld_ghost_one]="${pfld_dir:-$rust_root/models/pfld_ghost_one}"
    [feather_hubert]="${hubert_dir:-$rust_root/models/feather_hubert}"
    [vgg19]="${vgg19_dir:-$rust_root/models/vgg19}"
)
model_records="[]"
while IFS= read -r directory; do
    source_dir="$(cd -- "${model_sources[$directory]}" >/dev/null 2>&1 && pwd)"
    destination="$stage/models/$directory"
    mkdir -p "$destination"
    while IFS= read -r name; do
        install -m 0644 "$source_dir/$name" "$destination/$name"
    done < <(python3 "$models_py" files "$directory")
    record="$(python3 "$models_py" validate "$directory" "$destination")"
    model_records="$(python3 -c "import json,sys; recs=json.loads(sys.argv[1]); recs.append(json.loads(sys.argv[2])); print(json.dumps(recs))" "$model_records" "$record")"
done < <(python3 "$models_py" directories)
echo "SCRFD、PFLD、FeatherHuBERT、VGG19 模型包通过来源与权重校验。"

# -- licenses and notices -----------------------------------------------------
install -m 0644 "$license_path" "$stage/LICENSE.txt"
if [[ "$bundle_ffmpeg" -eq 1 ]]; then
    ffmpeg_root="$ffmpeg_bin"
    [[ -f "$ffmpeg_root/LICENSE" ]] || ffmpeg_root="$(dirname -- "$ffmpeg_bin")"
    for name in LICENSE README.txt; do
        if [[ -f "$ffmpeg_root/$name" ]]; then
            install -m 0644 "$ffmpeg_root/$name" "$stage/FFmpeg-${name%.txt}.txt" 2>/dev/null || \
                install -m 0644 "$ffmpeg_root/$name" "$stage/FFmpeg-$name"
        fi
    done
    # Normalize to the two names referenced by the notices, when present.
    [[ -f "$stage/FFmpeg-LICENSE" ]] && mv "$stage/FFmpeg-LICENSE" "$stage/FFmpeg-LICENSE.txt"
    [[ -f "$stage/FFmpeg-README" ]] && mv "$stage/FFmpeg-README" "$stage/FFmpeg-README.txt"
fi

mkdir -p "$stage/licenses/cubek-matmul"
for name in LICENSE-MIT LICENSE-APACHE FEATHERTALK-PATCH.md; do
    install -m 0644 "$rust_root/vendor/cubek-matmul/$name" "$stage/licenses/cubek-matmul/$name"
done

cat > "$stage/THIRD-PARTY-NOTICES.txt" <<'NOTICES'
FeatherTalk is licensed under Apache-2.0; see LICENSE.txt.

A local fix to the Cubek matrix-multiplication kernels is used. The upstream
license, source reference and modification notice are included under
licenses/cubek-matmul.
Cubek project: https://github.com/tracel-ai/cubek

FFmpeg and FFprobe, when bundled, are separate executables distributed without
modification. Their license and build configuration are in FFmpeg-LICENSE.txt
and FFmpeg-README.txt when present. These licenses also apply to the components
incorporated by that FFmpeg distribution.
FFmpeg project: https://ffmpeg.org/

Bundled models are under models/scrfd_2_5g, models/pfld_ghost_one,
models/feather_hubert and models/vgg19. Their manifests retain the original
checkpoint identities and license metadata. FeatherHuBERT and VGG19 also carry
LICENSES.json provenance records. VGG19 contains the conv3_3 feature extractor
used by perceptual loss. The application's Apache-2.0 license does not relicense
model weights.
NOTICES

install -m 0644 "$script_dir/README.md" "$stage/README.txt"

# -- desktop entry and installer ----------------------------------------------
cat > "$stage/FeatherTalk.desktop" <<'DESKTOP'
[Desktop Entry]
Type=Application
Name=FeatherTalk
Comment=FeatherTalk 工作台
Exec=feathertalk-app
Terminal=false
Categories=AudioVideo;Video;
DESKTOP

# install.sh lives inside the tarball and copies the extracted tree to a prefix,
# then drops a launcher and desktop entry. It never touches models or FFmpeg
# resolution, which stay relative to the installed binaries.
cat > "$stage/install.sh" <<'INSTALL'
#!/usr/bin/env bash
# Installs FeatherTalk from this extracted package into a prefix.
set -euo pipefail
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

prefix="${PREFIX:-/opt/feathertalk}"
bindir="${BINDIR:-/usr/local/bin}"
uninstall=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --prefix) prefix="${2:?}"; shift 2;;
        --bindir) bindir="${2:?}"; shift 2;;
        --uninstall) uninstall=1; shift;;
        -h|--help)
            echo "用法: ./install.sh [--prefix /opt/feathertalk] [--bindir /usr/local/bin] [--uninstall]"
            exit 0;;
        *) echo "未知选项: $1" >&2; exit 2;;
    esac
done

launcher="$bindir/feathertalk-app"
cli="$bindir/feathertalk"
desktop_dir="${XDG_DATA_HOME:-$prefix/share}/applications"
desktop_file="$desktop_dir/FeatherTalk.desktop"

if [[ "$uninstall" -eq 1 ]]; then
    rm -f "$launcher" "$cli" "$desktop_file"
    rm -rf "$prefix"
    echo "已卸载 FeatherTalk（删除 $prefix 及启动器）。"
    exit 0
fi

echo "安装到 $prefix ..."
mkdir -p "$prefix"
cp -a "$here/bin" "$here/models" "$here/licenses" "$prefix/"
for f in "$here"/*.txt "$here"/*.desktop; do
    [[ -e "$f" ]] && cp -a "$f" "$prefix/"
done
# Copy this installer into the prefix so `--uninstall` can run from there later.
cp -a "$here/install.sh" "$prefix/install.sh"
chmod 0755 "$prefix/install.sh"

mkdir -p "$bindir"
ln -sf "$prefix/bin/feathertalk-app" "$launcher"
ln -sf "$prefix/bin/feathertalk" "$cli"

mkdir -p "$desktop_dir"
sed "s|^Exec=feathertalk-app$|Exec=$prefix/bin/feathertalk-app|" \
    "$here/FeatherTalk.desktop" > "$desktop_file"

echo "完成。启动命令: feathertalk-app（或 $prefix/bin/feathertalk-app）"
echo "命令行工具: feathertalk --help"
INSTALL
chmod 0755 "$stage/install.sh"

# -- archive, checksum and manifest -------------------------------------------
if [[ -z "$output_dir" ]]; then
    output_dir="$rust_root/dist"
fi
mkdir -p "$output_dir"
output_dir="$(cd -- "$output_dir" >/dev/null 2>&1 && pwd)"
tarball="$output_dir/$package_name.tar.gz"

echo "打包 $tarball ..."
# Deterministic-ish archive: sorted names, fixed owner.
tar --sort=name --numeric-owner --owner=0 --group=0 \
    -C "$build_dir" -czf "$tarball" "$package_name"

# payload.json: relative path, size and sha256 of every staged file, plus the
# model provenance records, matching the Windows payload manifest intent.
python3 - "$stage" "$package_name" "$version" "$model_records" "$output_dir/$package_name.payload.json" <<'PY'
import hashlib, json, os, sys
stage, package_name, version, model_records, out = sys.argv[1:6]
files = []
for root, _dirs, names in os.walk(stage):
    for name in sorted(names):
        full = os.path.join(root, name)
        rel = os.path.relpath(full, stage).replace(os.sep, "/")
        digest = hashlib.sha256()
        with open(full, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        files.append({"name": rel, "bytes": os.path.getsize(full), "sha256": digest.hexdigest()})
files.sort(key=lambda entry: entry["name"])
manifest = {
    "product": "FeatherTalk",
    "version": version,
    "platform": "linux",
    "architecture": "x86_64",
    "archive": f"{package_name}.tar.gz",
    "models": json.loads(model_records),
    "files": files,
}
with open(out, "w", encoding="utf-8") as handle:
    json.dump(manifest, handle, indent=2, ensure_ascii=False)
    handle.write("\n")
PY

checksum="$(cd "$output_dir" && sha256sum "$package_name.tar.gz" | cut -d' ' -f1)"
printf '%s  %s\n' "$checksum" "$package_name.tar.gz" > "$tarball.sha256"

size_mib="$(python3 -c "import os,sys;print(round(os.path.getsize(sys.argv[1])/1048576,1))" "$tarball")"
echo "已生成 $tarball ($size_mib MiB)"
echo "SHA256 $checksum"
echo "清理暂存目录 $build_dir"
rm -rf "$build_dir"
