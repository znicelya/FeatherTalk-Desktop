import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

if sys.version_info[:2] != (3, 11):
    raise SystemExit(
        f"Python 3.11 is required, got {sys.version_info.major}.{sys.version_info.minor}"
    )

import cv2
import numpy as np

SOURCE_BYTES = 3_291_017
SOURCE_SHA256 = "32d20c77b9e2dc1d07e94c2ab9d25bdd5cd05eddbe0b46e7b38e7a1eca22e99a"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(64 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--destination", type=Path)
    mode.add_argument("--verify-against", type=Path)
    args = parser.parse_args()

    if np.__version__ != "2.2.6":
        raise SystemExit(f"NumPy 2.2.6 is required, got {np.__version__}")
    if cv2.__version__ != "4.12.0":
        raise SystemExit(f"OpenCV 4.12.0 is required, got {cv2.__version__}")

    repo_root = args.repo_root.resolve()
    onnx_path = repo_root / "data_utils" / "scrfd_2.5g_kps.onnx"
    onnx_bytes = onnx_path.read_bytes()
    if len(onnx_bytes) != SOURCE_BYTES:
        raise SystemExit(
            f"source ONNX size mismatch: expected {SOURCE_BYTES}, got {len(onnx_bytes)}"
        )
    onnx_hash = hashlib.sha256(onnx_bytes).hexdigest()
    if onnx_hash != SOURCE_SHA256:
        raise SystemExit(
            f"source ONNX SHA-256 mismatch: expected {SOURCE_SHA256}, got {onnx_hash}"
        )

    if args.destination is not None:
        destination = args.destination.resolve()
        if destination.exists() or destination.is_symlink():
            raise SystemExit(f"destination already exists: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging_parent = destination.parent
    else:
        destination = None
        staging_parent = Path(tempfile.gettempdir()).resolve()

    with tempfile.TemporaryDirectory(
        prefix="scrfd-fixture-", dir=staging_parent
    ) as temporary:
        staging = Path(temporary)

        x = np.arange(640, dtype=np.uint16)[None, :]
        y = np.arange(640, dtype=np.uint16)[:, None]
        image = np.empty((640, 640, 3), dtype=np.uint8)
        image[..., 0] = ((3 * x + 5 * y + 17) % 256).astype(np.uint8)
        image[..., 1] = ((7 * x + 11 * y + 29) % 256).astype(np.uint8)
        image[..., 2] = ((13 * x + 17 * y + 43) % 256).astype(np.uint8)

        cv2.setNumThreads(1)
        cv2.ocl.setUseOpenCL(False)
        if cv2.getNumThreads() != 1:
            raise SystemExit(f"OpenCV thread count is {cv2.getNumThreads()}, expected 1")
        if cv2.ocl.useOpenCL():
            raise SystemExit("OpenCL must be disabled")

        net = cv2.dnn.readNetFromONNX(str(onnx_path))
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        names = tuple(net.getUnconnectedOutLayersNames())
        expected_names = tuple(f"out{index}" for index in range(9))
        if names != expected_names:
            raise SystemExit(f"unexpected output names: expected {expected_names}, got {names}")

        blob = cv2.dnn.blobFromImage(
            image,
            scalefactor=1.0 / 128.0,
            size=(640, 640),
            mean=(127.5, 127.5, 127.5),
            swapRB=True,
            crop=False,
        )
        net.setInput(blob)
        outputs = net.forward(names)

        arrays = {"input.npy": blob}
        arrays.update(
            {f"out{index}.npy": output for index, output in enumerate(outputs)}
        )
        expected_shapes = {
            "input.npy": [1, 3, 640, 640],
            "out0.npy": [1, 12800, 1],
            "out1.npy": [1, 3200, 1],
            "out2.npy": [1, 800, 1],
            "out3.npy": [1, 12800, 4],
            "out4.npy": [1, 3200, 4],
            "out5.npy": [1, 800, 4],
            "out6.npy": [1, 12800, 10],
            "out7.npy": [1, 3200, 10],
            "out8.npy": [1, 800, 10],
        }
        files = {}
        for name, array in arrays.items():
            if array.dtype != np.float32:
                raise SystemExit(f"{name}: expected float32, got {array.dtype}")
            if list(array.shape) != expected_shapes[name]:
                raise SystemExit(
                    f"{name}: expected shape {expected_shapes[name]}, got {list(array.shape)}"
                )
            if not np.isfinite(array).all():
                raise SystemExit(f"{name}: contains non-finite values")
            path = staging / name
            np.save(path, array, allow_pickle=False)
            files[name] = {
                "dtype": "float32",
                "shape": expected_shapes[name],
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }

        manifest = {
            "schema_version": 1,
            "case": "opencv_cpu_v1",
            "source": {
                "file_name": "scrfd_2.5g_kps.onnx",
                "file_bytes": SOURCE_BYTES,
                "sha256": SOURCE_SHA256,
            },
            "generator": {
                "python_version": "3.11",
                "numpy_version": np.__version__,
                "opencv_version": cv2.__version__,
                "backend": "opencv",
                "target": "cpu",
                "threads": cv2.getNumThreads(),
                "opencl": cv2.ocl.useOpenCL(),
            },
            "files": files,
        }
        (staging / "fixture.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )

        expected_file_names = {"fixture.json", *arrays}
        actual_file_names = {path.name for path in staging.iterdir()}
        if actual_file_names != expected_file_names:
            raise SystemExit(
                f"generated file set mismatch: expected {sorted(expected_file_names)}, "
                f"got {sorted(actual_file_names)}"
            )
        if destination is not None:
            os.rename(staging, destination)
        else:
            committed = args.verify_against.resolve()
            committed_names = {path.name for path in committed.iterdir()}
            if committed_names != expected_file_names:
                raise SystemExit(
                    f"committed file set mismatch: expected {sorted(expected_file_names)}, "
                    f"got {sorted(committed_names)}"
                )
            for name in sorted(expected_file_names):
                actual = staging / name
                expected = committed / name
                if actual.read_bytes() != expected.read_bytes():
                    raise SystemExit(f"fixture differs: {expected}")


if __name__ == "__main__":
    main()
