#!/usr/bin/env bash
# Verifies a FeatherTalk Linux package built by installer/linux/build.sh.
#
# Checks the archive checksum, extracts to a temporary directory, re-validates
# every file against payload.json and the four model packages against their
# source identities, then runs the bundled CLI `capabilities` on CPU to confirm
# the binaries load. This is the Linux counterpart of installer/windows/verify.ps1.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
rust_root="$(cd -- "$script_dir/../.." >/dev/null 2>&1 && pwd)"
models_py="$script_dir/models.py"

archive=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --archive) archive="${2:?--archive 需要参数}"; shift 2;;
        -h|--help)
            echo "用法: installer/linux/verify.sh --archive dist/FeatherTalk-<version>-linux-x86_64.tar.gz"
            exit 0;;
        *) echo "未知选项: $1" >&2; exit 2;;
    esac
done
if [[ -z "$archive" ]]; then
    echo "必须用 --archive 指定要校验的 .tar.gz。" >&2
    exit 2
fi
archive="$(cd -- "$(dirname -- "$archive")" >/dev/null 2>&1 && pwd)/$(basename -- "$archive")"
if [[ ! -f "$archive" ]]; then
    echo "找不到归档: $archive" >&2
    exit 1
fi

for tool in python3 tar sha256sum; do
    command -v "$tool" >/dev/null 2>&1 || { echo "缺少必需工具: $tool" >&2; exit 1; }
done

# -- checksum -----------------------------------------------------------------
if [[ -f "$archive.sha256" ]]; then
    echo "校验归档 SHA-256 ..."
    expected="$(cut -d' ' -f1 < "$archive.sha256")"
    actual="$(sha256sum "$archive" | cut -d' ' -f1)"
    if [[ "$expected" != "$actual" ]]; then
        echo "归档校验和不匹配: 期望 $expected 实际 $actual" >&2
        exit 1
    fi
fi

payload="${archive%.tar.gz}.payload.json"
if [[ ! -f "$payload" ]]; then
    echo "找不到 payload 清单: $payload" >&2
    exit 1
fi

# -- extract ------------------------------------------------------------------
work="$(mktemp -d "${TMPDIR:-/tmp}/feathertalk-verify.XXXXXX")"
trap 'rm -rf "$work"' EXIT
tar -C "$work" -xzf "$archive"
root="$(find "$work" -maxdepth 1 -mindepth 1 -type d | head -n1)"
if [[ -z "$root" ]]; then
    echo "归档内没有顶层目录。" >&2
    exit 1
fi

# -- payload file hashes ------------------------------------------------------
echo "逐个核对 payload 文件哈希 ..."
python3 - "$root" "$payload" <<'PY'
import hashlib, json, os, sys
root, payload = sys.argv[1], sys.argv[2]
manifest = json.load(open(payload, encoding="utf-8"))
errors = 0
seen = set()
for entry in manifest["files"]:
    full = os.path.join(root, entry["name"])
    seen.add(entry["name"])
    if not os.path.isfile(full):
        print(f"缺少文件: {entry['name']}", file=sys.stderr); errors += 1; continue
    size = os.path.getsize(full)
    if size != entry["bytes"]:
        print(f"大小不符 {entry['name']}: {size} != {entry['bytes']}", file=sys.stderr); errors += 1
    digest = hashlib.sha256()
    with open(full, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != entry["sha256"]:
        print(f"哈希不符: {entry['name']}", file=sys.stderr); errors += 1
# Any staged file not listed in the manifest is unexpected.
for walk_root, _dirs, names in os.walk(root):
    for name in names:
        rel = os.path.relpath(os.path.join(walk_root, name), root).replace(os.sep, "/")
        if rel not in seen:
            print(f"清单外的多余文件: {rel}", file=sys.stderr); errors += 1
sys.exit(1 if errors else 0)
PY

# -- model provenance ---------------------------------------------------------
echo "核对模型来源与权重 ..."
while IFS= read -r directory; do
    python3 "$models_py" validate "$directory" "$root/models/$directory" >/dev/null
done < <(python3 "$models_py" directories)

# -- run the CLI --------------------------------------------------------------
echo "运行随包 CLI capabilities（CPU）..."
cli="$root/bin/feathertalk"
worker="$root/bin/feathertalk-worker"
if [[ ! -x "$cli" || ! -x "$worker" ]]; then
    echo "随包 CLI 或 worker 缺失或不可执行。" >&2
    exit 1
fi
# Isolate from any developer config and resolve the sibling worker.
if ! FEATHERTALK_WORKER_BIN="$worker" "$cli" --backend cpu capabilities; then
    echo "CLI capabilities 运行失败。" >&2
    exit 1
fi

echo "校验通过。"
