"""Generate or verify the frame-adapters demo video frame fixture."""

import argparse
import hashlib
import json
import os
import subprocess
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

from generate_fixture import (
    JPEG_PARAMS,
    NMS_IOU_THRESHOLD,
    STRIDES,
    anchor_centers,
    clamp,
    crop_blob,
    crop_geometry,
    landmark_points,
    letterbox,
    non_max_suppression,
    sha256_file,
)

CASE = "demo_frame_v1"
VIDEO_RELATIVE_PATH = "demo/feathertalk_demo_latest_188.mp4"
VIDEO_SHA256 = "9353ad796089aa104765d651ca99f158349cfd203644923b2fa72f68b44e9ac1"
FRAME_INDEX = 750
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
FRAME_COUNT = 1511
FRAME_FPS = 25.0
EXTRACTED_FRAME = "frame_750.png"

BLUR_KERNEL = 19
BLUR_SIGMA = 3.0
CONFIDENCE_THRESHOLD = 0.5

NUMPY_VERSION = "2.4.6"
OPENCV_VERSION = "5.0.0"
TORCH_VERSION = "2.13.0"

SCRFD_ONNX = "scrfd_2.5g_kps.onnx"
SCRFD_OUTPUT_NAMES = tuple(f"out{index}" for index in range(9))


def probe_video(path: Path) -> None:
    """Refuse to run against a clip whose container metadata has moved."""
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise SystemExit(f"OpenCV cannot open {path}")
    try:
        actual = {
            "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "frame_count": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
            "fps": float(capture.get(cv2.CAP_PROP_FPS)),
        }
    finally:
        capture.release()
    expected = {
        "width": FRAME_WIDTH,
        "height": FRAME_HEIGHT,
        "frame_count": FRAME_COUNT,
        "fps": FRAME_FPS,
    }
    if actual != expected:
        raise SystemExit(f"video probe mismatch: expected {expected}, got {actual}")


def extract_frame(ffmpeg: str, repo_root: Path, png_path: Path) -> list:
    """Decode exactly one frame into a lossless PNG next to the staging tree."""
    argv = [
        ffmpeg,
        "-v",
        "error",
        "-y",
        "-i",
        VIDEO_RELATIVE_PATH,
        "-vf",
        f"select=eq(n\\,{FRAME_INDEX})",
        "-fps_mode",
        "passthrough",
        "-frames:v",
        "1",
        str(png_path),
    ]
    completed = subprocess.run(argv, cwd=repo_root, capture_output=True)
    if completed.returncode != 0:
        raise SystemExit(
            f"ffmpeg exited {completed.returncode}: "
            f"{completed.stderr.decode('utf-8', 'replace').strip()}"
        )
    if not png_path.is_file():
        raise SystemExit(f"ffmpeg reported success but {png_path} is missing")
    # The manifest records a machine independent command: argv[0] collapses to
    # the bare executable name and the temporary output path to a relative file
    # name, so a reader can re-run it verbatim from the repository root.
    return ["ffmpeg"] + argv[1:-1] + [EXTRACTED_FRAME]


def decode_detections(outputs: tuple, transform: dict) -> tuple:
    """Port of `decode_level` for a letterboxed frame, at threshold 0.50."""
    width = np.float32(FRAME_WIDTH)
    height = np.float32(FRAME_HEIGHT)
    pad_x = np.float32(transform["pad_x"])
    pad_y = np.float32(transform["pad_y"])
    scale_x = np.float32(FRAME_WIDTH) / np.float32(transform["new_width"])
    scale_y = np.float32(FRAME_HEIGHT) / np.float32(transform["new_height"])
    threshold = np.float32(CONFIDENCE_THRESHOLD)
    candidates = []
    level_max_scores = []
    for level, stride in enumerate(STRIDES):
        centers = anchor_centers(stride)
        anchors = centers.shape[0]
        scores = np.ascontiguousarray(outputs[level], dtype=np.float32).reshape(-1)
        distances = np.ascontiguousarray(
            outputs[level + 3], dtype=np.float32
        ).reshape(-1, 4)
        keypoints = np.ascontiguousarray(
            outputs[level + 6], dtype=np.float32
        ).reshape(-1, 10)
        if scores.shape[0] != anchors:
            raise SystemExit(f"level {level}: {scores.shape[0]} scores, want {anchors}")
        if distances.shape[0] != anchors:
            raise SystemExit(
                f"level {level}: {distances.shape[0]} boxes, want {anchors}"
            )
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
            x1 = clamp((center_x - left * step - pad_x) * scale_x, width)
            y1 = clamp((center_y - top * step - pad_y) * scale_y, height)
            x2 = clamp((center_x + right * step - pad_x) * scale_x, width)
            y2 = clamp((center_y + bottom * step - pad_y) * scale_y, height)
            if not (x2 > x1 and y2 > y1):
                continue
            points = []
            for point in range(5):
                offset_x = keypoints[index][point * 2]
                offset_y = keypoints[index][point * 2 + 1]
                point_x = clamp((center_x + offset_x * step - pad_x) * scale_x, width)
                point_y = clamp((center_y + offset_y * step - pad_y) * scale_y, height)
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
    return candidates, level_max_scores


def scrfd_network(data_utils: Path) -> cv2.dnn.Net:
    """Load SCRFD on the OpenCV CPU backend, as `data_utils/detect_face.py` does."""
    net = cv2.dnn.readNetFromONNX(str(data_utils / SCRFD_ONNX))
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    names = tuple(net.getUnconnectedOutLayersNames())
    if names != SCRFD_OUTPUT_NAMES:
        raise SystemExit(f"unexpected SCRFD output names: {names}")
    return net


def evaluate_frame(data_utils: Path, net: cv2.dnn.Net, image: np.ndarray) -> dict:
    """Everything the Rust pipeline derives from one decoded frame."""
    if image.shape != (FRAME_HEIGHT, FRAME_WIDTH, 3):
        raise SystemExit(f"frame shape is {image.shape}")
    transform = letterbox(image)
    net.setInput(transform["blob"])
    outputs = net.forward(list(SCRFD_OUTPUT_NAMES))
    candidates, level_max_scores = decode_detections(outputs, transform)
    kept = non_max_suppression(candidates)
    # `choose_primary` returns `MultipleFaces` for two survivors and `NoFace`
    # for none, so anything but one face breaks Task 15's accepted path.
    if len(kept) != 1:
        raise SystemExit(
            f"expected exactly one face at threshold {CONFIDENCE_THRESHOLD}, "
            f"got {len(kept)} from {len(candidates)} candidates"
        )
    detection = candidates[kept[0]]
    geometry = crop_geometry(detection["bbox"], FRAME_WIDTH, FRAME_HEIGHT)
    blob = crop_blob(image, geometry)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    response = cv2.Laplacian(gray, cv2.CV_64F)
    return {
        "crop": {
            "origin_x": geometry["origin_x"],
            "origin_y": geometry["origin_y"],
            "padding": geometry["padding"],
            "size": geometry["size"],
            "source": geometry["source"],
        },
        "detection": detection,
        "landmarks": landmark_points(data_utils, blob, geometry),
        "laplacian_variance": float(response.var()),
        "level_max_scores": level_max_scores,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument(
        "--ffmpeg",
        default="ffmpeg",
        help="ffmpeg executable, if it is not on PATH",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--destination", type=Path)
    mode.add_argument("--verify-against", type=Path)
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
    video = repo_root / VIDEO_RELATIVE_PATH
    if not data_utils.is_dir():
        raise SystemExit(f"missing torch assets: {data_utils}")
    if not video.is_file():
        raise SystemExit(f"missing demo video: {video}")
    if sha256_file(video) != VIDEO_SHA256:
        raise SystemExit(f"unexpected digest for {video}")
    sys.path.insert(0, str(data_utils))
    probe_video(video)

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
        prefix="demo-frame-parity-", dir=staging_parent
    ) as temporary:
        scratch = Path(temporary)
        staging = scratch / CASE
        staging.mkdir()

        # The PNG stays in the scratch directory, one level above the fixture,
        # so `os.rename` moves only the files that belong in the repository.
        png_path = scratch / EXTRACTED_FRAME
        extraction = extract_frame(args.ffmpeg, repo_root, png_path)
        raw = cv2.imread(str(png_path), cv2.IMREAD_COLOR)
        if raw is None:
            raise SystemExit(f"OpenCV cannot read {png_path}")
        if raw.shape != (FRAME_HEIGHT, FRAME_WIDTH, 3):
            raise SystemExit(f"extracted frame shape is {raw.shape}")
        raw_bgr_sha256 = hashlib.sha256(
            np.ascontiguousarray(raw).tobytes()
        ).hexdigest()

        net = scrfd_network(data_utils)
        frames = {}
        for name, source in (
            ("sharp", raw),
            (
                "blurred",
                cv2.GaussianBlur(raw, (BLUR_KERNEL, BLUR_KERNEL), BLUR_SIGMA),
            ),
        ):
            blob_name = "frame.jpg" if name == "sharp" else "frame_blurred.jpg"
            encoded, buffer = cv2.imencode(".jpg", source, JPEG_PARAMS)
            if not encoded:
                raise SystemExit(f"cv2.imencode failed on the {name} frame")
            (staging / blob_name).write_bytes(buffer.tobytes())
            # The committed JPEG, decoded again, is the truth the Rust adapters
            # are measured against - not the PNG that produced it.
            decoded = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
            if decoded is None:
                raise SystemExit(f"cv2.imdecode rejected the {name} frame")
            frames[name] = {
                "blob": blob_name,
                **evaluate_frame(data_utils, net, decoded),
            }

        blob_manifest = {}
        for name in ("frame.jpg", "frame_blurred.jpg"):
            path = staging / name
            blob_manifest[name] = {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }

        manifest = {
            "schema_version": 1,
            "case": CASE,
            "source": {
                "kind": "video_frame",
                "video": VIDEO_RELATIVE_PATH,
                "sha256": VIDEO_SHA256,
                "raw_bgr_sha256": raw_bgr_sha256,
                "width": FRAME_WIDTH,
                "height": FRAME_HEIGHT,
                "fps": FRAME_FPS,
                "frame_count": FRAME_COUNT,
                "frame_index": FRAME_INDEX,
                "extraction": {"tool": "ffmpeg", "arguments": extraction},
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
            "blur": {"kernel": BLUR_KERNEL, "sigma": BLUR_SIGMA},
            "detection_config": {
                "confidence_threshold": CONFIDENCE_THRESHOLD,
                "nms_iou_threshold": NMS_IOU_THRESHOLD,
            },
            "blobs": blob_manifest,
            "frames": frames,
        }
        (staging / "fixture.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )

        expected_file_names = {"fixture.json", *blob_manifest}
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
            recorded = json.loads(
                (committed / "fixture.json").read_text(encoding="utf-8")
            )
            if recorded["source"]["raw_bgr_sha256"] != raw_bgr_sha256:
                raise SystemExit(
                    "the extracted frame changed: your ffmpeg produced "
                    f"{raw_bgr_sha256}, the fixture records "
                    f"{recorded['source']['raw_bgr_sha256']}"
                )
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
