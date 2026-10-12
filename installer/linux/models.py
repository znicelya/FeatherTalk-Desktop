#!/usr/bin/env python3
"""Shared bundled-model identities and package checks for the Linux packager.

This is the Linux counterpart of ``installer/windows/models.ps1``. It keeps the
four prebuilt model packages' source checkpoint identities and validates a
staged package directory the same way the Windows build does: manifest kind,
original source file name and SHA-256, bundled weight hash, and (where present)
the license provenance file hash.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

# The same identities enforced by installer/windows/models.ps1. Each entry is a
# prebuilt package under models/<directory>.
DEFINITIONS = [
    {
        "directory": "scrfd_2_5g",
        "kind": "scrfd_2.5g_kps",
        "source_file": "scrfd_2.5g_kps.onnx",
        "source_sha256": "32d20c77b9e2dc1d07e94c2ab9d25bdd5cd05eddbe0b46e7b38e7a1eca22e99a",
        "files": ["manifest.json", "model.safetensors"],
    },
    {
        "directory": "pfld_ghost_one",
        "kind": "pfld_ghost_one",
        "source_file": "checkpoint_epoch_335.pth.tar",
        "source_sha256": "bada866661ad5fa1080a085f51fe9c016c69958c406951afa4afc7840f856de0",
        "files": ["manifest.json", "model.safetensors"],
    },
    {
        "directory": "feather_hubert",
        "kind": "feather_hubert",
        "source_file": "feather_hubert_188_latest_99.pth",
        "source_sha256": "58df96af118d75d7f69da441e1f3960096f28dda637a4e8f4265f108d27aeb52",
        "files": ["manifest.json", "model.safetensors", "LICENSES.json"],
    },
    {
        "directory": "vgg19",
        "kind": "vgg19-conv3-3",
        "source_file": "vgg19-dcbb9e9d.pth",
        "source_sha256": "dcbb9e9dad569fff7a846263a77324fc34978fea2bfb039c012d710e1776ae44",
        "source_url": "https://download.pytorch.org/models/vgg19-dcbb9e9d.pth",
        "files": ["manifest.json", "model.safetensors", "LICENSES.json"],
    },
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate(directory: Path, definition: dict) -> dict:
    """Validate one staged package directory and return its provenance record.

    Raises ValueError with a human message on any mismatch, mirroring the
    throws in Assert-BundledModelPackage.
    """
    for name in definition["files"]:
        if not (directory / name).is_file():
            raise ValueError(
                f"Incomplete {definition['kind']} model package: missing {name} in {directory}"
            )

    package = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))

    if definition["directory"] == "scrfd_2_5g":
        kind = package.get("model_kind")
        weights = package.get("weights", {})
        source_matches = package.get("source", {}).get("file_name") == definition["source_file"]
    elif definition["directory"] == "vgg19":
        kind = package.get("model_kind")
        weights = package.get("model", {})
        source = package.get("source", {})
        # The VGG19 runtime format identifies its official source by URL and weight id.
        source_matches = (
            source.get("framework") == "torchvision"
            and source.get("weight_id") == "VGG19_Weights.IMAGENET1K_V1"
            and source.get("url") == definition["source_url"]
        )
        if (
            package.get("architecture_version") != "torchvision-vgg19-conv3-3-v1"
            or package.get("output_layer") != "features.14"
            or package.get("tensor_count") != 14
            or package.get("total_elements") != 1735488
        ):
            raise ValueError(f"Unexpected VGG19 perceptual feature layout in {directory}")
    else:
        kind = package.get("model_type")
        weights = package.get("model", {})
        source_matches = package.get("source", {}).get("file_name") == definition["source_file"]

    if (
        kind != definition["kind"]
        or not source_matches
        or package.get("source", {}).get("sha256") != definition["source_sha256"]
    ):
        raise ValueError(f"Unexpected model kind or source checkpoint in {directory}")

    weights_hash = _sha256(directory / "model.safetensors")
    if weights.get("file_name") != "model.safetensors" or (
        str(weights.get("sha256", "")).lower() != weights_hash
    ):
        raise ValueError(f"Model weight checksum mismatch in {directory}")

    if "LICENSES.json" in definition["files"]:
        license_hash = _sha256(directory / "LICENSES.json")
        licenses = package.get("licenses", {})
        if licenses.get("file_name") != "LICENSES.json" or (
            str(licenses.get("sha256", "")).lower() != license_hash
        ):
            raise ValueError(f"Model license checksum mismatch in {directory}")

    return {
        "directory": f"models/{definition['directory']}",
        "model_kind": kind,
        "source_file": definition["source_file"],
        "source_sha256": definition["source_sha256"],
        "model_sha256": weights_hash,
    }


def _find(directory_name: str) -> dict:
    for definition in DEFINITIONS:
        if definition["directory"] == directory_name:
            return definition
    raise SystemExit(f"unknown model directory: {directory_name}")


def main(argv) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("directories", help="print the four package directory names, one per line")

    files_parser = sub.add_parser("files", help="print the files of one package, one per line")
    files_parser.add_argument("directory")

    validate_parser = sub.add_parser(
        "validate", help="validate a staged package directory and print its JSON record"
    )
    validate_parser.add_argument("directory")
    validate_parser.add_argument("staged_path")

    args = parser.parse_args(argv)

    if args.command == "directories":
        for definition in DEFINITIONS:
            print(definition["directory"])
        return 0

    if args.command == "files":
        for name in _find(args.directory)["files"]:
            print(name)
        return 0

    if args.command == "validate":
        definition = _find(args.directory)
        try:
            record = validate(Path(args.staged_path), definition)
        except ValueError as error:
            print(str(error), file=sys.stderr)
            return 1
        print(json.dumps(record))
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
