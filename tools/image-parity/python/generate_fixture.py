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

CASE = "opencv_cpu_v1"
PATTERN = "bgr_u8_channel_affine_v1"
PATTERN_EDGE = 640
NUMPY_VERSION = "2.4.6"
OPENCV_VERSION = "5.0.0"

# (name, source (width, height), destination (width, height), interpolation)
RESIZE_CASES = (
    ("area_int_2x2", (8, 8), (4, 4), cv2.INTER_AREA),
    ("area_int_4x4", (8, 8), (2, 2), cv2.INTER_AREA),
    ("area_shrink", (13, 9), (7, 5), cv2.INTER_AREA),
    ("area_upscale", (5, 5), (8, 8), cv2.INTER_AREA),
    ("linear_shrink", (200, 200), (192, 192), cv2.INTER_LINEAR),
    ("linear_upscale", (61, 47), (192, 192), cv2.INTER_LINEAR),
)
GRAY_SIZE = (64, 64)  # (width, height)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(64 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def pattern(width: int, height: int) -> np.ndarray:
    """Top-left `height x width` crop of the fixed 640-edge BGR pattern."""
    if width > PATTERN_EDGE or height > PATTERN_EDGE:
        raise SystemExit(f"crop {width}x{height} exceeds the {PATTERN_EDGE} pattern edge")
    x = np.arange(width, dtype=np.uint16)[None, :]
    y = np.arange(height, dtype=np.uint16)[:, None]
    image = np.empty((height, width, 3), dtype=np.uint8)
    image[..., 0] = ((3 * x + 5 * y + 17) % 256).astype(np.uint8)
    image[..., 1] = ((7 * x + 11 * y + 29) % 256).astype(np.uint8)
    image[..., 2] = ((13 * x + 17 * y + 43) % 256).astype(np.uint8)
    return image


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate or verify the feathertalk-image OpenCV CPU parity fixture."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--destination", type=Path)
    mode.add_argument("--verify-against", type=Path)
    args = parser.parse_args()

    if np.__version__ != NUMPY_VERSION:
        raise SystemExit(f"NumPy {NUMPY_VERSION} is required, got {np.__version__}")
    if cv2.__version__ != OPENCV_VERSION:
        raise SystemExit(f"OpenCV {OPENCV_VERSION} is required, got {cv2.__version__}")

    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
    if cv2.getNumThreads() != 1:
        raise SystemExit(f"OpenCV thread count is {cv2.getNumThreads()}, expected 1")
    if cv2.ocl.useOpenCL():
        raise SystemExit("OpenCL must be disabled")

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
        prefix="image-parity-", dir=staging_parent
    ) as temporary:
        staging = Path(temporary) / CASE
        staging.mkdir()

        arrays = {}
        expectations = {}
        for name, (src_w, src_h), (dst_w, dst_h), interpolation in RESIZE_CASES:
            source = pattern(src_w, src_h)
            resized = cv2.resize(source, (dst_w, dst_h), interpolation=interpolation)
            arrays[f"{name}_src.npy"] = source
            arrays[f"{name}_dst.npy"] = resized
            expectations[f"{name}_src.npy"] = ("uint8", [src_h, src_w, 3])
            expectations[f"{name}_dst.npy"] = ("uint8", [dst_h, dst_w, 3])

        gray_width, gray_height = GRAY_SIZE
        gray_source = pattern(gray_width, gray_height)
        gray = cv2.cvtColor(gray_source, cv2.COLOR_BGR2GRAY)
        response = cv2.Laplacian(gray, cv2.CV_64F)
        arrays["gray_src.npy"] = gray_source
        arrays["gray_dst.npy"] = gray
        arrays["laplacian_response.npy"] = response
        expectations["gray_src.npy"] = ("uint8", [gray_height, gray_width, 3])
        expectations["gray_dst.npy"] = ("uint8", [gray_height, gray_width])
        expectations["laplacian_response.npy"] = ("float64", [gray_height, gray_width])

        files = {}
        for name in sorted(arrays):
            array = arrays[name]
            dtype, shape = expectations[name]
            if array.dtype != np.dtype(dtype):
                raise SystemExit(f"{name}: expected {dtype}, got {array.dtype}")
            if list(array.shape) != shape:
                raise SystemExit(f"{name}: expected shape {shape}, got {list(array.shape)}")
            if not np.isfinite(array).all():
                raise SystemExit(f"{name}: contains non-finite values")
            path = staging / name
            np.save(path, array, allow_pickle=False)
            files[name] = {
                "dtype": dtype,
                "shape": shape,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }

        manifest = {
            "schema_version": 1,
            "case": CASE,
            "source": {
                "kind": "synthetic",
                "pattern": PATTERN,
                "pattern_edge": PATTERN_EDGE,
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
            "scalars": {"laplacian_variance": float(response.var())},
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
                if (staging / name).read_bytes() != (committed / name).read_bytes():
                    raise SystemExit(f"fixture differs: {committed / name}")


if __name__ == "__main__":
    main()
