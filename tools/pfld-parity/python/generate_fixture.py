#!/usr/bin/env python3
"""Generate a deterministic CPU PFLD parity fixture."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import torch


INPUT_SHAPE = (1, 3, 192, 192)
OUTPUT_SHAPE = (1, 220)
SOURCE_SHA256 = "bada866661ad5fa1080a085f51fe9c016c69958c406951afa4afc7840f856de0"
MODEL_TYPE = "pfld_ghost_one"
ARTIFACT_MODEL_NAME = "model.safetensors"
ARTIFACT_MODEL_SHA256 = "e131dd764236fde54a27b2f7084906119f06c28b140bf127b459ec967e92915b"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def deterministic_input() -> np.ndarray:
    height, width = INPUT_SHAPE[2:]
    yy, xx = np.indices((height, width), dtype=np.uint32)
    channels = np.stack(
        [
            (3 * xx + 5 * yy + 17) % 256,
            (7 * xx + 11 * yy + 29) % 256,
            (13 * xx + 17 * yy + 43) % 256,
        ],
        axis=-1,
    ).astype(np.uint8, copy=False)
    nchw = channels.astype(np.float32) / np.float32(255.0)
    nchw = np.transpose(nchw, (2, 0, 1))[None, ...]
    assert nchw.shape == INPUT_SHAPE
    assert nchw.dtype == np.float32
    assert np.isfinite(nchw).all()
    return np.ascontiguousarray(nchw)


def descriptor(path: Path, shape: tuple[int, ...]) -> dict[str, object]:
    data = path.read_bytes()
    return {
        "file_name": path.name,
        "dtype": "f32-le",
        "shape": list(shape),
        "bytes": len(data),
        "sha256": sha256_bytes(data),
    }


def generate(checkpoint: Path, artifact_dir: Path, output_dir: Path) -> None:
    checkpoint = checkpoint.resolve()
    artifact_dir = artifact_dir.resolve()
    output_dir = output_dir.resolve()
    if sha256_file(checkpoint) != SOURCE_SHA256:
        raise RuntimeError("source checkpoint hash changed")
    artifact_manifest = json.loads((artifact_dir / "manifest.json").read_text(encoding="utf-8"))
    if artifact_manifest["source"]["sha256"] != SOURCE_SHA256:
        raise RuntimeError("artifact source hash does not match checkpoint")
    artifact_model = artifact_dir / ARTIFACT_MODEL_NAME
    artifact_model_sha256 = sha256_file(artifact_model)
    if (
        artifact_model_sha256 != artifact_manifest["model"]["sha256"]
        or artifact_model_sha256 != ARTIFACT_MODEL_SHA256
    ):
        raise RuntimeError("committed artifact model hash is invalid")

    # Keep all CPU kernels deterministic and avoid environment-dependent thread reductions.
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    repo_root = checkpoint.parent.parent
    sys.path.insert(0, str(repo_root / "data_utils"))
    from pfld_mobileone import PFLD_GhostOne  # pylint: disable=import-outside-toplevel

    checkpoint_object = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = PFLD_GhostOne(width_factor=0.5, input_size=192, landmark_number=110, inference_mode=False)
    model.load_state_dict(checkpoint_object["pfld_backbone"], strict=True)
    model.eval()

    input_array = deterministic_input()
    input_tensor = torch.from_numpy(input_array)
    with torch.inference_mode():
        output_array = model(input_tensor).detach().cpu().numpy().astype(np.float32, copy=False)
    output_array = np.ascontiguousarray(output_array)
    if output_array.shape != OUTPUT_SHAPE or output_array.dtype != np.float32:
        raise RuntimeError(f"unexpected Python output contract: {output_array.shape} {output_array.dtype}")
    if not np.isfinite(output_array).all():
        raise RuntimeError("Python output contains a non-finite value")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = output_dir.with_name(output_dir.name + ".staging")
    if staging.exists():
        raise RuntimeError(f"staging directory already exists: {staging}")
    staging.mkdir()
    try:
        input_path = staging / "input.f32"
        output_path = staging / "output.f32"
        input_array.astype("<f4", copy=False).tofile(input_path)
        output_array.astype("<f4", copy=False).tofile(output_path)
        manifest = {
            "schema_version": 1,
            "case": "pfld_cpu_v1",
            "model_type": MODEL_TYPE,
            "source": {
                "file_name": checkpoint.name,
                "sha256": SOURCE_SHA256,
            },
            "artifact": {
                "file_name": ARTIFACT_MODEL_NAME,
                "sha256": artifact_model_sha256,
            },
            "generator": {
                "python_version": platform.python_version(),
                "torch_version": torch.__version__,
                "numpy_version": np.__version__,
                "platform": platform.platform(),
                "threads": 1,
                "input_formula": "bgr_u8_channel_affine_v1",
            },
            "files": {
                "input.f32": descriptor(input_path, INPUT_SHAPE),
                "output.f32": descriptor(output_path, OUTPUT_SHAPE),
            },
        }
        manifest_path = staging / "fixture.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")
        if output_dir.exists():
            existing = {name: (output_dir / name).read_bytes() for name in ["fixture.json", "input.f32", "output.f32"]}
            current = {name: (staging / name).read_bytes() for name in ["fixture.json", "input.f32", "output.f32"]}
            if existing == current:
                return
            raise RuntimeError("fixture destination exists with different bytes")
        staging.replace(output_dir)
    finally:
        if staging.exists():
            for child in staging.iterdir():
                child.unlink()
            staging.rmdir()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    generate(args.checkpoint, args.artifact_dir, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
