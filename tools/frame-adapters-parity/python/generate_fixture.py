"""Generate or verify the frame-adapters OpenCV CPU parity fixture."""

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

if sys.version_info[:2] != (3, 11):
    raise SystemExit(
        "Python 3.11 is required, got "
        f"{sys.version_info.major}.{sys.version_info.minor}"
    )

import cv2
import numpy as np
import torch

CASE = "opencv_cpu_v1"
PATTERN = "bgr_u8_channel_affine_v1"
FRAME_EDGE = 640
CROP_EDGE = 192
LETTERBOX_WIDTH = 1280
LETTERBOX_HEIGHT = 720
NUMPY_VERSION = "2.4.6"
OPENCV_VERSION = "5.0.0"
TORCH_VERSION = "2.13.0"

JPEG_PARAMS = [
    cv2.IMWRITE_JPEG_QUALITY,
    90,
    cv2.IMWRITE_JPEG_OPTIMIZE,
    0,
    cv2.IMWRITE_JPEG_PROGRESSIVE,
    0,
    cv2.IMWRITE_JPEG_SAMPLING_FACTOR,
    cv2.IMWRITE_JPEG_SAMPLING_FACTOR_420,
]

REFERENCE_RELATIVE_PATH = "../feathertalk-scrfd/tests/fixtures/opencv_cpu_v1"
REFERENCE_INPUT_SHA256 = (
    "3d1bcdaf3874b28af337d5b596902143b59655a1bf8411c034b1aab1162f04db"
)
REFERENCE_FILES = (
    ("input.npy", [1, 3, 640, 640]),
    ("out0.npy", [1, 12800, 1]),
    ("out1.npy", [1, 3200, 1]),
    ("out2.npy", [1, 800, 1]),
    ("out3.npy", [1, 12800, 4]),
    ("out4.npy", [1, 3200, 4]),
    ("out5.npy", [1, 800, 4]),
    ("out6.npy", [1, 12800, 10]),
    ("out7.npy", [1, 3200, 10]),
    ("out8.npy", [1, 800, 10]),
)

STRIDES = (8, 16, 32)
ANCHORS_PER_LOCATION = 2
CONFIDENCE_THRESHOLD = 0.02
NMS_IOU_THRESHOLD = 0.4

CROP_CASES = (
    ("in_bounds", "crop_blob.npy", [220.0, 180.0, 160.0, 200.0]),
    ("padded", "crop_blob_padded.npy", [-100.0, -80.0, 900.0, 860.0]),
)

LETTERBOX_KEY = "letterbox_1280x720"
# (channel, row, column): the top padding band, the last padded row, the first
# resized row at both edges, two interior samples, the last resized row, then
# the bottom padding band.
LETTERBOX_SAMPLES = (
    (0, 0, 0),
    (1, 138, 320),
    (2, 139, 0),
    (0, 139, 639),
    (1, 300, 320),
    (2, 499, 639),
    (0, 500, 0),
    (1, 639, 639),
)

PFLD_CHECKPOINT = "checkpoint_epoch_335.pth.tar"
PFLD_MEAN_FACE = "mean_face.txt"
PFLD_LANDMARK_COUNT = 110


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(64 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def pattern(width: int, height: int) -> np.ndarray:
    """The `bgr_u8_channel_affine_v1` pattern at any size.

    Unlike Task 3's helper this one is unbounded, because the letterbox case
    needs a 1280x720 source while the committed arrays are 640x640.
    """
    x = np.arange(width, dtype=np.int64)[None, :]
    y = np.arange(height, dtype=np.int64)[:, None]
    image = np.empty((height, width, 3), dtype=np.uint8)
    image[..., 0] = ((3 * x + 5 * y + 17) % 256).astype(np.uint8)
    image[..., 1] = ((7 * x + 11 * y + 29) % 256).astype(np.uint8)
    image[..., 2] = ((13 * x + 17 * y + 43) % 256).astype(np.uint8)
    return image


def blob_from(square: np.ndarray) -> np.ndarray:
    """The SCRFD input blob, exactly as `data_utils/detect_face.py` builds it."""
    if square.shape != (FRAME_EDGE, FRAME_EDGE, 3):
        raise SystemExit(f"blob source must be {FRAME_EDGE} square, got {square.shape}")
    blob = cv2.dnn.blobFromImage(
        square,
        1.0 / 128,
        (FRAME_EDGE, FRAME_EDGE),
        (127.5, 127.5, 127.5),
        swapRB=True,
    )
    return np.ascontiguousarray(blob, dtype=np.float32)


def letterbox(image: np.ndarray) -> dict:
    """Port of `resize_image` for a landscape frame."""
    height, width = image.shape[:2]
    if width <= height:
        raise SystemExit(f"letterbox expects a landscape frame, got {width}x{height}")
    new_width = FRAME_EDGE
    new_height = int(np.floor(FRAME_EDGE * (height / width))) + 1
    pad_x = (FRAME_EDGE - new_width) // 2
    pad_y = (FRAME_EDGE - new_height) // 2
    resized = cv2.resize(
        image, (new_width, new_height), interpolation=cv2.INTER_AREA
    )
    padded = cv2.copyMakeBorder(
        resized,
        pad_y,
        FRAME_EDGE - new_height - pad_y,
        pad_x,
        FRAME_EDGE - new_width - pad_x,
        cv2.BORDER_CONSTANT,
        value=0,
    )
    return {
        "blob": blob_from(padded),
        "new_width": new_width,
        "new_height": new_height,
        "pad_x": pad_x,
        "pad_y": pad_y,
    }


def crop_geometry(bbox: list, width: int, height: int) -> dict:
    """Port of `compute_face_crop_geometry`, including its f32 then f64 order."""
    left = np.float32(bbox[0])
    top = np.float32(bbox[1])
    x1 = int(np.trunc(np.float64(left)))
    y1 = int(np.trunc(np.float64(top)))
    x2 = int(np.trunc(np.float64(left + np.float32(bbox[2]))))
    y2 = int(np.trunc(np.float64(top + np.float32(bbox[3]))))
    if x2 <= x1 or y2 <= y1:
        raise SystemExit(f"integer edges must define a positive rectangle: {bbox}")
    size = int(np.trunc(max(x2 - x1, y2 - y1) * 1.05))
    origin_x = (x1 + x2) // 2 - size // 2
    origin_y = (y1 + y2) // 2 - size // 2
    source_left = min(max(origin_x, 0), width)
    source_top = min(max(origin_y, 0), height)
    source_right = min(max(origin_x + size, 0), width)
    source_bottom = min(max(origin_y + size, 0), height)
    if source_right <= source_left or source_bottom <= source_top:
        raise SystemExit(f"requested crop does not intersect image: {bbox}")
    return {
        "size": size,
        "origin_x": origin_x,
        "origin_y": origin_y,
        "padding": [
            max(0, -origin_x),
            max(0, -origin_y),
            max(0, origin_x + size - width),
            max(0, origin_y + size - height),
        ],
        "source": [
            source_left,
            source_top,
            source_right - source_left,
            source_bottom - source_top,
        ],
    }


def crop_blob(image: np.ndarray, geometry: dict) -> np.ndarray:
    """Port of `pfld_input`: pad into a square canvas, resize, scale to 0..1."""
    size = geometry["size"]
    pad_left, pad_top = geometry["padding"][0], geometry["padding"][1]
    source_x, source_y, source_width, source_height = geometry["source"]
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    canvas[
        pad_top : pad_top + source_height, pad_left : pad_left + source_width
    ] = image[
        source_y : source_y + source_height, source_x : source_x + source_width
    ]
    resized = cv2.resize(
        canvas, (CROP_EDGE, CROP_EDGE), interpolation=cv2.INTER_LINEAR
    )
    blob = (resized.astype(np.float32) / np.float32(255.0)).transpose(2, 0, 1)[None]
    return np.ascontiguousarray(blob, dtype=np.float32)


def anchor_centers(stride: int) -> np.ndarray:
    """Port of `generate_anchor_centers`: row major, two anchors per location."""
    rows = FRAME_EDGE // stride
    steps = np.arange(rows, dtype=np.float32) * np.float32(stride)
    grid = np.empty((rows, rows, ANCHORS_PER_LOCATION, 2), dtype=np.float32)
    grid[..., 0] = steps[None, :, None]
    grid[..., 1] = steps[:, None, None]
    return grid.reshape(-1, 2)


def clamp(value: np.float32, upper: np.float32) -> np.float32:
    return min(max(value, np.float32(0.0)), upper)


def load_reference(root: Path, name: str) -> np.ndarray:
    array = np.load(root / name, allow_pickle=False)
    if array.dtype != np.float32:
        raise SystemExit(f"{name}: expected float32, got {array.dtype}")
    if not np.isfinite(array).all():
        raise SystemExit(f"{name}: contains non-finite values")
    return array


def decode_levels(root: Path) -> tuple:
    """Port of `decode_level` over the three committed SCRFD output levels."""
    edge = np.float32(FRAME_EDGE)
    threshold = np.float32(CONFIDENCE_THRESHOLD)
    candidates = []
    level_max_scores = []
    degenerate = 0
    for level, stride in enumerate(STRIDES):
        centers = anchor_centers(stride)
        anchors = centers.shape[0]
        scores = load_reference(root, f"out{level}.npy").reshape(-1)
        distances = load_reference(root, f"out{level + 3}.npy").reshape(-1, 4)
        keypoints = load_reference(root, f"out{level + 6}.npy").reshape(-1, 10)
        if scores.shape[0] != anchors:
            raise SystemExit(f"level {level}: {scores.shape[0]} scores, want {anchors}")
        if distances.shape[0] != anchors:
            raise SystemExit(f"level {level}: {distances.shape[0]} boxes, want {anchors}")
        if keypoints.shape[0] != anchors:
            raise SystemExit(
                f"level {level}: {keypoints.shape[0]} keypoint rows, want {anchors}"
            )
        level_max_scores.append(float(scores.max()))
        step = np.float32(stride)
        for index in range(anchors):
            score = scores[index]
            if score < threshold:
                continue
            center_x, center_y = centers[index]
            left, top, right, bottom = distances[index]
            x1 = clamp(center_x - left * step, edge)
            y1 = clamp(center_y - top * step, edge)
            x2 = clamp(center_x + right * step, edge)
            y2 = clamp(center_y + bottom * step, edge)
            if not (x2 > x1 and y2 > y1):
                degenerate += 1
                continue
            points = []
            for point in range(5):
                point_x = clamp(center_x + keypoints[index][point * 2] * step, edge)
                point_y = clamp(center_y + keypoints[index][point * 2 + 1] * step, edge)
                points.append([float(point_x), float(point_y)])
            candidates.append(
                {
                    "score": float(score),
                    "bbox": [
                        float(x1),
                        float(y1),
                        float(x2 - x1),
                        float(y2 - y1),
                    ],
                    "keypoints": points,
                }
            )
    return candidates, level_max_scores, degenerate


def iou(left: dict, right: dict) -> np.float32:
    a = np.asarray(left["bbox"], dtype=np.float32)
    b = np.asarray(right["bbox"], dtype=np.float32)
    width = max(min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]), np.float32(0.0))
    height = max(min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]), np.float32(0.0))
    intersection = width * height
    union = a[2] * a[3] + b[2] * b[3] - intersection
    if not union > np.float32(0.0):
        return np.float32(0.0)
    return intersection / union


def non_max_suppression(candidates: list) -> list:
    """Port of `non_max_suppression`: score descending, index breaks ties."""
    threshold = np.float32(NMS_IOU_THRESHOLD)
    order = sorted(
        range(len(candidates)),
        key=lambda index: (-candidates[index]["score"], index),
    )
    kept = []
    for index in order:
        if all(
            iou(candidates[index], candidates[other]) <= threshold for other in kept
        ):
            kept.append(index)
    return kept


def landmark_points(data_utils: Path, blob: np.ndarray, crop: dict) -> list:
    """Run PFLD on one crop blob and decode it the way `decode_landmarks` does."""
    from pfld_mobileone import PFLD_GhostOne

    mean_face = np.asarray(
        (data_utils / PFLD_MEAN_FACE).read_text(encoding="utf-8").split(" "),
        dtype=np.float32,
    )
    if mean_face.shape != (PFLD_LANDMARK_COUNT * 2,):
        raise SystemExit(f"mean face must hold 220 floats, got {mean_face.shape}")
    checkpoint = torch.load(
        data_utils / PFLD_CHECKPOINT, map_location="cpu", weights_only=True
    )
    model = PFLD_GhostOne()
    model.load_state_dict(checkpoint["pfld_backbone"])
    model.eval()
    output = model(torch.from_numpy(blob)).numpy().reshape(-1)
    if output.shape != (PFLD_LANDMARK_COUNT * 2,):
        raise SystemExit(f"PFLD must emit 220 values, got {output.shape}")
    normalized = (output + mean_face).reshape(-1, 2).astype(np.float64)
    scaled = np.trunc(normalized * np.float64(crop["size"])).astype(np.int64)
    scaled[:, 0] += crop["origin_x"]
    scaled[:, 1] += crop["origin_y"]
    return [[int(point[0]), int(point[1])] for point in scaled]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--destination", type=Path)
    mode.add_argument("--verify-against", type=Path)
    parser.add_argument(
        "--dump-letterbox",
        type=Path,
        help="write the hash pinned letterbox blob here instead of committing it",
    )
    args = parser.parse_args()

    if np.__version__ != NUMPY_VERSION:
        raise SystemExit(f"NumPy {NUMPY_VERSION} is required, got {np.__version__}")
    if cv2.__version__ != OPENCV_VERSION:
        raise SystemExit(f"OpenCV {OPENCV_VERSION} is required, got {cv2.__version__}")
    torch_version = torch.__version__.split("+")[0]
    if torch_version != TORCH_VERSION:
        raise SystemExit(f"torch {TORCH_VERSION} is required, got {torch.__version__}")

    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
    if cv2.getNumThreads() != 1:
        raise SystemExit(f"OpenCV thread count is {cv2.getNumThreads()}, expected 1")
    if cv2.ocl.useOpenCL():
        raise SystemExit("OpenCL must be disabled")
    torch.set_num_threads(1)
    torch.set_grad_enabled(False)

    repo_root = args.repo_root.resolve()
    data_utils = repo_root / "data_utils"
    reference_root = (
        repo_root / "rust" / "crates" / "feathertalk-scrfd" / "tests" / "fixtures" / CASE
    )
    if not data_utils.is_dir():
        raise SystemExit(f"missing torch assets: {data_utils}")
    if not reference_root.is_dir():
        raise SystemExit(f"missing reference fixture: {reference_root}")
    sys.path.insert(0, str(data_utils))

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
        prefix="frame-adapters-parity-", dir=staging_parent
    ) as temporary:
        staging = Path(temporary) / CASE
        staging.mkdir()

        frame = pattern(FRAME_EDGE, FRAME_EDGE)
        committed_input = reference_root / "input.npy"
        if not np.array_equal(
            blob_from(frame), np.load(committed_input, allow_pickle=False)
        ):
            raise SystemExit(f"pattern blob no longer matches {committed_input}")
        if sha256_file(committed_input) != REFERENCE_INPUT_SHA256:
            raise SystemExit(f"unexpected digest for {committed_input}")

        encoded, buffer = cv2.imencode(".jpg", frame, JPEG_PARAMS)
        if not encoded:
            raise SystemExit("cv2.imencode failed on the pattern frame")
        (staging / "frame.jpg").write_bytes(buffer.tobytes())
        frame_decode = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        if frame_decode is None:
            raise SystemExit("cv2.imdecode rejected the encoded frame")
        if frame_decode.shape != (FRAME_EDGE, FRAME_EDGE, 3):
            raise SystemExit(f"decoded shape is {frame_decode.shape}")

        pin = letterbox(pattern(LETTERBOX_WIDTH, LETTERBOX_HEIGHT))
        if args.dump_letterbox is not None:
            np.save(args.dump_letterbox, pin["blob"], allow_pickle=False)

        arrays = {"frame_decode.npy": frame_decode}
        expectations = {"frame_decode.npy": ("uint8", [FRAME_EDGE, FRAME_EDGE, 3])}
        crops = {}
        landmark_input = None
        for name, array_name, bbox in CROP_CASES:
            geometry = crop_geometry(bbox, FRAME_EDGE, FRAME_EDGE)
            blob = crop_blob(frame_decode, geometry)
            arrays[array_name] = blob
            expectations[array_name] = ("float32", [1, 3, CROP_EDGE, CROP_EDGE])
            crops[name] = {
                "array": array_name,
                "bbox": [float(value) for value in bbox],
                "origin_x": geometry["origin_x"],
                "origin_y": geometry["origin_y"],
                "padding": geometry["padding"],
                "size": geometry["size"],
                "source": geometry["source"],
            }
            if name == "in_bounds":
                landmark_input = (geometry, blob)

        candidates, level_max_scores, degenerate = decode_levels(reference_root)
        kept = non_max_suppression(candidates)
        (staging / "detections_thr002.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "confidence_threshold": CONFIDENCE_THRESHOLD,
                    "nms_iou_threshold": NMS_IOU_THRESHOLD,
                    "level_max_scores": level_max_scores,
                    "candidate_count": len(candidates),
                    "degenerate_count": degenerate,
                    "detections": [candidates[index] for index in kept],
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
            newline="\n",
        )

        geometry, blob = landmark_input
        (staging / "landmarks.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "crop": {
                        "bbox": crops["in_bounds"]["bbox"],
                        "origin_x": geometry["origin_x"],
                        "origin_y": geometry["origin_y"],
                        "size": geometry["size"],
                    },
                    "points": landmark_points(data_utils, blob, geometry),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
            newline="\n",
        )

        array_manifest = {}
        for name in sorted(arrays):
            array = arrays[name]
            dtype, shape = expectations[name]
            if array.dtype != np.dtype(dtype):
                raise SystemExit(f"{name}: expected {dtype}, got {array.dtype}")
            if list(array.shape) != shape:
                raise SystemExit(
                    f"{name}: expected shape {shape}, got {list(array.shape)}"
                )
            if not np.isfinite(array).all():
                raise SystemExit(f"{name}: contains non-finite values")
            path = staging / name
            np.save(path, array, allow_pickle=False)
            array_manifest[name] = {
                "dtype": dtype,
                "shape": shape,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }

        blob_manifest = {}
        for name in ("detections_thr002.json", "frame.jpg", "landmarks.json"):
            path = staging / name
            blob_manifest[name] = {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }

        reference_manifest = {}
        for name, shape in REFERENCE_FILES:
            path = reference_root / name
            array = np.load(path, allow_pickle=False)
            if array.dtype != np.float32:
                raise SystemExit(f"{path}: expected float32, got {array.dtype}")
            if list(array.shape) != shape:
                raise SystemExit(
                    f"{path}: expected shape {shape}, got {list(array.shape)}"
                )
            reference_manifest[name] = {
                "dtype": "float32",
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
                "width": FRAME_EDGE,
                "height": FRAME_EDGE,
            },
            "generator": {
                "python_version": "3.11",
                "numpy_version": np.__version__,
                "opencv_version": cv2.__version__,
                "torch_version": torch_version,
                "backend": "opencv",
                "target": "cpu",
                "threads": cv2.getNumThreads(),
                "opencl": cv2.ocl.useOpenCL(),
            },
            "jpeg": {
                "quality": 90,
                "optimize": 0,
                "progressive": 0,
                "sampling": "420",
            },
            "scalars": {"level_max_scores": level_max_scores},
            "arrays": array_manifest,
            "blobs": blob_manifest,
            "crops": crops,
            "hashed_arrays": {
                LETTERBOX_KEY: {
                    "dtype": "float32",
                    "shape": list(pin["blob"].shape),
                    "sha256": hashlib.sha256(pin["blob"].tobytes()).hexdigest(),
                    "samples": [
                        [
                            channel,
                            row,
                            column,
                            float(pin["blob"][0, channel, row, column]),
                        ]
                        for channel, row, column in LETTERBOX_SAMPLES
                    ],
                    "source_width": LETTERBOX_WIDTH,
                    "source_height": LETTERBOX_HEIGHT,
                    "new_width": pin["new_width"],
                    "new_height": pin["new_height"],
                    "pad_x": pin["pad_x"],
                    "pad_y": pin["pad_y"],
                }
            },
            "reference_fixture": {
                "case": CASE,
                "relative_path": REFERENCE_RELATIVE_PATH,
                "files": reference_manifest,
            },
        }
        (staging / "fixture.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )

        expected_file_names = {"fixture.json", *array_manifest, *blob_manifest}
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
