from __future__ import annotations

import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


ARCHIVE_NAME = "vgg19-conv3-3-v1.zip"
FIXTURE_NAME = "vgg19-conv3-3-v1"
FIXED_ZIP_TIME = (1980, 1, 1, 0, 0, 0)
REQUIRED_KEYS = [
    "features.0.weight",
    "features.0.bias",
    "features.2.weight",
    "features.2.bias",
    "features.5.weight",
    "features.5.bias",
    "features.7.weight",
    "features.7.bias",
    "features.10.weight",
    "features.10.bias",
    "features.12.weight",
    "features.12.bias",
    "features.14.weight",
    "features.14.bias",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the committed VGG19 conv3_3 Python parity fixture."
    )
    parser.add_argument("--source", required=True, type=Path)
    return parser.parse_args()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(64 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def npy_bytes(tensor: torch.Tensor) -> bytes:
    output = io.BytesIO()
    array = tensor.detach().cpu().numpy().astype(np.float32, copy=False)
    np.save(output, array, allow_pickle=False)
    return output.getvalue()


def array_manifest(path: str, data: bytes, shape: tuple[int, ...]) -> dict[str, object]:
    return {
        "path": path,
        "shape": list(shape),
        "dtype": "float32",
        "sha256": sha256_bytes(data),
    }


def load_state(source: Path) -> dict[str, torch.Tensor]:
    if not source.is_file():
        raise FileNotFoundError(f"VGG19 source is not a file: {source}")
    value = torch.load(source, map_location="cpu", weights_only=True)
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise TypeError("VGG19 source must be a direct string-keyed state dict")
    missing = [key for key in REQUIRED_KEYS if key not in value]
    if missing:
        raise KeyError(f"VGG19 source is missing required keys: {missing}")
    state: dict[str, torch.Tensor] = {}
    for key in REQUIRED_KEYS:
        tensor = value[key]
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"VGG19 state entry is not a tensor: {key}")
        if tensor.dtype != torch.float32:
            raise TypeError(f"VGG19 state entry must be float32: {key} ({tensor.dtype})")
        state[key] = tensor.detach().cpu()
    return state


def forward_conv3_3(state: dict[str, torch.Tensor], input_tensor: torch.Tensor) -> torch.Tensor:
    x = F.relu(F.conv2d(input_tensor, state["features.0.weight"], state["features.0.bias"], padding=1))
    x = F.relu(F.conv2d(x, state["features.2.weight"], state["features.2.bias"], padding=1))
    x = F.max_pool2d(x, 2, 2)
    x = F.relu(F.conv2d(x, state["features.5.weight"], state["features.5.bias"], padding=1))
    x = F.relu(F.conv2d(x, state["features.7.weight"], state["features.7.bias"], padding=1))
    x = F.max_pool2d(x, 2, 2)
    x = F.relu(F.conv2d(x, state["features.10.weight"], state["features.10.bias"], padding=1))
    x = F.relu(F.conv2d(x, state["features.12.weight"], state["features.12.bias"], padding=1))
    return F.conv2d(x, state["features.14.weight"], state["features.14.bias"], padding=1)


def write_archive(source: Path) -> tuple[Path, str]:
    repository_root = Path(__file__).resolve().parents[3]
    golden = repository_root / "rust" / "tests" / "golden"
    golden.mkdir(parents=True, exist_ok=True)
    archive_path = golden / ARCHIVE_NAME

    state = load_state(source)
    input_tensor = torch.linspace(
        0.0, 1.0, steps=1 * 3 * 16 * 16, dtype=torch.float32
    ).reshape(1, 3, 16, 16)
    with torch.no_grad():
        expected = forward_conv3_3(state, input_tensor)

    input_data = npy_bytes(input_tensor)
    expected_data = npy_bytes(expected)
    manifest = {
        "schema_version": 1,
        "fixture": FIXTURE_NAME,
        "source_sha256": sha256_file(source),
        "output_layer": "features.14",
        "input_contract": {
            "channels": 3,
            "color_order": "bgr",
            "value_range": "0..1",
            "normalization": "none",
        },
        "input": array_manifest("input.npy", input_data, tuple(input_tensor.shape)),
        "expected": array_manifest("expected.npy", expected_data, tuple(expected.shape)),
    }
    manifest_data = (
        json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")

    with zipfile.ZipFile(
        archive_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for name, data in [
            ("manifest.json", manifest_data),
            ("input.npy", input_data),
            ("expected.npy", expected_data),
        ]:
            info = zipfile.ZipInfo(name, date_time=FIXED_ZIP_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, data, compresslevel=9)

    archive_hash = sha256_file(archive_path)
    archive_path.with_suffix(".sha256").write_text(
        f"{archive_hash}  {archive_path.name}\n", encoding="ascii", newline="\n"
    )
    return archive_path, archive_hash


def main() -> None:
    args = parse_args()
    archive_path, archive_hash = write_archive(args.source.resolve())
    print(f"archive={archive_path}")
    print(f"sha256={archive_hash}")


if __name__ == "__main__":
    main()
