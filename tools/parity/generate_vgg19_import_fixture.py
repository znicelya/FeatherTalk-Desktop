from __future__ import annotations

import hashlib
import json
import tempfile
import zipfile
from pathlib import Path

import torch


RELEVANT = {
    0: (64, 3),
    2: (64, 64),
    5: (128, 64),
    7: (128, 128),
    10: (256, 128),
    12: (256, 256),
    14: (256, 256),
}
IGNORED_FEATURES = [16, 19, 21, 23, 25, 28, 30, 32, 34]
IGNORED_CLASSIFIER = [0, 3, 6]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(64 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def direct_state() -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    for ordinal, (index, (out_channels, in_channels)) in enumerate(
        RELEVANT.items(), 1
    ):
        state[f"features.{index}.weight"] = torch.full(
            (out_channels, in_channels, 3, 3),
            ordinal / 1000.0,
            dtype=torch.float32,
        )
        state[f"features.{index}.bias"] = torch.full(
            (out_channels,), -ordinal / 100.0, dtype=torch.float32
        )

    for index in IGNORED_FEATURES:
        state[f"features.{index}.weight"] = torch.tensor([1.0])
        state[f"features.{index}.bias"] = torch.tensor([1.0])
    for index in IGNORED_CLASSIFIER:
        state[f"classifier.{index}.weight"] = torch.tensor([1.0])
        state[f"classifier.{index}.bias"] = torch.tensor([1.0])
    return state


def write_fixture(repository_root: Path) -> None:
    golden = repository_root / "rust" / "tests" / "golden"
    golden.mkdir(parents=True, exist_ok=True)
    archive_path = golden / "vgg19-import-v1.zip"
    sidecar_path = golden / "vgg19-import-v1.sha256"

    with tempfile.TemporaryDirectory(prefix="feathertalk-vgg19-import-") as temp:
        temp_path = Path(temp)
        direct_path = temp_path / "vgg19-direct.pth"
        unexpected_path = temp_path / "vgg19-unexpected.pth"
        manifest_path = temp_path / "manifest.json"

        state = direct_state()
        torch.save(state, direct_path)

        unexpected = dict(state)
        unexpected["unexpected.weight"] = torch.tensor([1.0])
        torch.save(unexpected, unexpected_path)

        manifest = {
            "schema_version": 1,
            "tensor_key_count": len(state),
            "members": {
                direct_path.name: sha256(direct_path),
                unexpected_path.name: sha256(unexpected_path),
            },
        }
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

        with zipfile.ZipFile(
            archive_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
        ) as archive:
            for path in [direct_path, unexpected_path, manifest_path]:
                info = zipfile.ZipInfo(path.name, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.create_system = 3
                info.external_attr = 0o100644 << 16
                archive.writestr(info, path.read_bytes(), compresslevel=9)

    sidecar_path.write_text(
        f"{sha256(archive_path)}  {archive_path.name}\n", encoding="ascii"
    )


if __name__ == "__main__":
    write_fixture(Path(__file__).resolve().parents[3])
