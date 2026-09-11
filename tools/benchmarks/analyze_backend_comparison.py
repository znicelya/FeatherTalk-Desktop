"""Export measured timings and numerical checks; run after timing is finished.

Requires numpy for numerical comparisons. Does not run a model or alter outputs.
"""

import argparse
import csv
import json
from pathlib import Path
import statistics
import struct
import subprocess

import numpy as np


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def read_lines(path):
    path = Path(path)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line] if path.exists() else []


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def difference(reference, actual):
    assert reference.shape == actual.shape, (reference.shape, actual.shape)
    assert np.isfinite(reference).all() and np.isfinite(actual).all()
    reference, actual = reference.astype(np.float64), actual.astype(np.float64)
    delta = actual - reference
    rms = float(np.sqrt(np.mean(delta * delta)))
    scale = float(np.sqrt(np.mean(reference * reference)))
    return {"elements": reference.size, "all_finite": True,
            "max_absolute_error": float(np.max(np.abs(delta))),
            "mean_absolute_error": float(np.mean(np.abs(delta))),
            "root_mean_square_error": rms, "relative_rms_error": rms / scale if scale else None}


def export_cudnn_comparison(output, summary, cli, install, verify_media):
    """Compare the additional environment run without claiming a new backend."""
    cases = [r for r in cli if r["run_label"] == "cudnn-available"
             and r["status"] == "completed" and r["validation"]["ok"]]
    if not cases:
        return
    comparisons, checks = [], []
    for workload in ("features", "frames", "normalize", "render", "train"):
        additional = next((s for s in summary if s["run_label"] == "cudnn-available"
                           and s["workload"] == workload and s["backend"] == "cuda"), None)
        if additional is None:
            continue
        baseline = {s["backend"]: s for s in summary if s["run_label"] == "cli" and s["workload"] == workload}
        comparisons.append({"workload": workload, "baseline": baseline,
                            "cuda_with_cudnn_available": additional,
                            "elapsed_change_vs_baseline_cuda_percent":
                                (additional["median_seconds"] / baseline["cuda"]["median_seconds"] - 1) * 100})
    for record in cases:
        workload = record["workload"]
        reference = next(r for r in cli if r["run_label"] == "cli" and r["workload"] == workload
                         and r["backend_requested"] == "cuda" and r["repeat"] == record["repeat"]
                         and r["status"] == "completed" and r["validation"]["ok"])
        result = record["result"]
        check = {"workload": workload, "repeat": record["repeat"], "task_dir": record["task_dir"],
                 "baseline_task_dir": reference["task_dir"], "validation_ok": True}
        if workload != "normalize":
            assert result["backend"] == "cuda" and result["graphics_api"].lower() == "cuda"
            assert result["used_cpu_fallback"] is False
            check.update(reported_backend=result["backend"], used_cpu_fallback=result["used_cpu_fallback"])
        if workload in ("features", "train"):
            relative, offset = (("assets/features/feather_hubert.f32", 44) if workload == "features"
                                else ("outputs/preview/step-00000002/prediction.f32", 32))
            arrays = [np.frombuffer((Path(r["request"]["params"]["project_dir"]) / relative).read_bytes()[offset:], dtype="<f4")
                      for r in (reference, record)]
            check["output_vs_baseline_cuda"] = difference(*arrays)
            if workload == "train":
                check["loss"] = result["total_loss"]
                check["loss_absolute_error_vs_baseline_cuda"] = abs(result["total_loss"] - reference["result"]["total_loss"])
        elif workload == "frames":
            arrays = []
            for sample in (reference, record):
                project = Path(sample["request"]["params"]["project_dir"])
                count = read_json(project / "assets/quality.json")["frame_count"]
                arrays.append(np.concatenate([np.loadtxt(project / f"assets/landmarks/{i:06}.lms") for i in range(count)]))
            check["landmarks_vs_baseline_cuda"] = difference(*arrays)
        elif workload == "render" and verify_media:
            arrays = []
            for sample in (reference, record):
                proc = subprocess.run([str(install / "ffmpeg.exe"), "-hide_banner", "-loglevel", "error",
                    "-i", sample["request"]["params"]["output"], "-map", "0:v:0", "-f", "rawvideo",
                    "-pix_fmt", "rgb24", "pipe:1"], capture_output=True, check=True, timeout=60)
                arrays.append(np.frombuffer(proc.stdout, dtype=np.uint8))
            check["render_rgb8_vs_baseline_cuda"] = difference(*arrays)
        checks.append(check)
    write_json(output / "cudnn-comparison.json", {
        "interpretation": "cuDNN is available in the child environment; this is the same installed Burn-CUDA backend, not a cuDNN operator backend.",
        "cudnn_directories": sorted({r["cudnn_directory"] for r in cases}),
        "valid_cli_measurements": len(cases), "comparisons": comparisons,
        "correctness_reference": "Original CUDA CLI run with the same repeat index and fixture",
        "correctness": checks})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--verify-media", action="store_true", help="Decode/probe outputs after timing is finished")
    args = parser.parse_args()
    root, output = args.root.resolve(), args.out.resolve()
    output.mkdir(parents=True, exist_ok=True)
    workers = read_lines(root / "results.ndjson")
    cli = read_lines(root / "cli-results.ndjson")
    all_records = workers + cli
    rows, failures, sessions = [], [], {}
    for record in all_records:
        if record["status"] != "completed" or not record["validation"]["ok"]:
            failures.append(record)
            continue
        source = "cli" if "end_to_end_seconds" in record else "worker"
        phase = "fresh_cli" if source == "cli" else ("first_in_worker" if record["repeat"] == 0 else "repeat_in_worker")
        result = record["result"]
        session = {}
        if source == "worker":
            name = record["session"]
            if name not in sessions:
                sessions[name] = read_json(root / "sessions" / name / "session.json")
            session = sessions[name]
        rows.append({"run_label": record["run_label"], "workload": record["workload"],
                     "backend": record["backend_requested"], "phase": phase, "repeat": record["repeat"],
                     "seconds": record["end_to_end_seconds"] if source == "cli" else record["task_seconds"],
                     "startup_seconds": record.get("startup_seconds"), "started_utc": record["started_utc"],
                     "reported_backend": result.get("backend"), "graphics_api": result.get("graphics_api"),
                     "used_cpu_fallback": result.get("used_cpu_fallback"), "loss": result.get("total_loss"),
                     "worker_cwd": record.get("worker_cwd", session.get("worker_cwd")),
                     "cudnn_directory": record.get("cudnn_directory", session.get("cudnn_directory")),
                     "measurement_file": str(Path(record["task_dir"]) / "measurement.json")})
    with open(output / "measurements.csv", "w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    groups = {}
    for row in rows:
        groups.setdefault((row["run_label"], row["phase"], row["workload"], row["backend"]), []).append(row["seconds"])
    summary = []
    for (label, phase, workload, backend), values in groups.items():
        summary.append({"run_label": label, "phase": phase, "workload": workload, "backend": backend,
                        "n": len(values), "median_seconds": statistics.median(values),
                        "min_seconds": min(values), "max_seconds": max(values),
                        "mean_seconds": statistics.mean(values),
                        "sample_stddev_seconds": statistics.stdev(values) if len(values) > 1 else None,
                        "values_seconds": values})
    write_json(output / "summary.json", summary)
    write_json(output / "excluded.json", failures)
    inventory = read_json(root / "inventory.json")
    metadata = {"inventory": inventory, "windows_hardware": read_json(root / "hardware-windows.json"),
                "fixtures": read_json(root / "fixtures.json"), "training_seed": read_json(root / "training-seed.json")}
    write_json(output / "environment.json", metadata)

    def measured(workload, backend):
        return next(r for r in workers if r["workload"] == workload and r["backend_requested"] == backend
                    and r["repeat"] == 3 and r["status"] == "completed" and r["validation"]["ok"])

    correctness = {"features": {}, "landmarks": {}, "training_loss": {}, "training_prediction": {},
                   "valid_measurement_count": len(rows), "excluded_count": len(failures),
                   "baseline_valid_count": sum(r["run_label"] in ("main", "main-training", "cli") for r in rows),
                   "diagnostic_valid_count": sum(r["run_label"].startswith("cache-") for r in rows),
                   "cudnn_valid_count": sum(r["run_label"].startswith("cudnn-") for r in rows)}
    cache_diagnostics = []
    for backend in ("cuda", "wgpu"):
        for workload in ("features", "render"):
            samples = [r for r in cli if r["run_label"] == f"cache-{backend}" and r["workload"] == workload
                       and r["status"] == "completed" and r["validation"]["ok"]]
            repeated = [r["end_to_end_seconds"] for r in samples if r["repeat"] > 0]
            if not repeated:
                continue
            baseline = next(s["median_seconds"] for s in summary if s["run_label"] == "cli"
                            and s["workload"] == workload and s["backend"] == backend)
            median = statistics.median(repeated)
            cache_diagnostics.append({"backend": backend, "workload": workload,
                "baseline_median_seconds": baseline, "cached_median_seconds": median,
                "cached_n": len(repeated), "cached_values_seconds": repeated,
                "cache_population_seconds": next(r["end_to_end_seconds"] for r in samples if r["repeat"] == 0),
                "speedup": baseline / median,
                "config": (root / "diagnostics" / backend / "CubeCL.toml").read_text(encoding="utf-8")})
    write_json(output / "cache-diagnostics.json", cache_diagnostics)
    for workload, section, relative, offset in (
        ("features", "features", "assets/features/feather_hubert.f32", 44),
        ("train", "training_prediction", "outputs/preview/step-00000002/prediction.f32", 32),
    ):
        cpu = measured(workload, "cpu")
        cpu_path = Path(cpu["request"]["params"]["project_dir"]) / relative
        reference = np.frombuffer(cpu_path.read_bytes()[offset:], dtype="<f4")
        if workload == "features":
            magic, version, tokens, pair, dims, size = struct.unpack("<8sIQQQQ", cpu_path.read_bytes()[:44])
            assert magic == b"FTF32\0\0\0" and version == 1 and pair == 2 and dims == 1024
            correctness["feature_shape"] = [tokens, dims]
        for backend in ("wgpu", "cuda"):
            record = measured(workload, backend)
            path = Path(record["request"]["params"]["project_dir"]) / relative
            values = np.frombuffer(path.read_bytes()[offset:], dtype="<f4")
            correctness[section][backend] = difference(reference, values)
    reference_frames = Path(measured("frames", "cpu")["request"]["params"]["project_dir"])
    quality_ref = read_json(reference_frames / "assets/quality.json")
    count = quality_ref["frame_count"]
    landmarks_cpu = np.concatenate([np.loadtxt(reference_frames / f"assets/landmarks/{i:06}.lms") for i in range(count)])
    for backend in ("wgpu", "cuda"):
        project = Path(measured("frames", backend)["request"]["params"]["project_dir"])
        landmarks = np.concatenate([np.loadtxt(project / f"assets/landmarks/{i:06}.lms") for i in range(count)])
        correctness["landmarks"][backend] = difference(landmarks_cpu, landmarks)
        quality = read_json(project / "assets/quality.json")
        assert quality["accepted_count"] == count and not quality["anomalies"]
        correctness["landmarks"][backend]["accepted_frames"] = count
        correctness["landmarks"][backend]["changed_coordinates"] = int(np.count_nonzero(landmarks_cpu != landmarks))
    loss_cpu = measured("train", "cpu")["result"]["total_loss"]
    for backend in ("cpu", "wgpu", "cuda"):
        loss = measured("train", backend)["result"]["total_loss"]
        correctness["training_loss"][backend] = {"value": loss, "absolute_error_vs_cpu": abs(loss - loss_cpu)}
    write_json(output / "correctness.json", correctness)

    export_cudnn_comparison(output, summary, cli, Path(inventory["install"]), args.verify_media)

    wgpu_prediction = Path(measured("train", "wgpu")["request"]["params"]["project_dir"]) / "outputs/preview/step-00000002/prediction.f32"
    cuda_prediction = Path(measured("train", "cuda")["request"]["params"]["project_dir"]) / "outputs/preview/step-00000002/prediction.f32"
    correctness["training_prediction_cuda_vs_wgpu"] = difference(
        np.frombuffer(wgpu_prediction.read_bytes()[32:], dtype="<f4"),
        np.frombuffer(cuda_prediction.read_bytes()[32:], dtype="<f4"))
    if args.verify_media:
        install = Path(inventory["install"])
        verified_media = []
        for record in all_records:
            if record["status"] != "completed" or not record["validation"]["ok"]:
                continue
            params = record["request"]["params"]
            if record["workload"] == "render":
                paths = [Path(params["output"])]
            elif record["workload"] == "normalize":
                paths = sorted(Path(params["output_dir"]).iterdir())
            else:
                continue
            for path in paths:
                proc = subprocess.run([str(install / "ffprobe.exe"), "-v", "error", "-count_frames",
                    "-show_entries", "stream=codec_name,codec_type,width,height,nb_read_frames,r_frame_rate,sample_rate,channels,duration",
                    "-of", "json", str(path)], capture_output=True, check=True, timeout=60)
                streams = json.loads(proc.stdout)["streams"]
                for stream in streams:
                    if stream["codec_type"] == "video":
                        assert stream["width"] == 1280 and stream["height"] == 720
                        assert stream["r_frame_rate"] == "25/1"
                        expected = metadata["fixtures"]["render_frames"] if record["workload"] == "render" else round(metadata["fixtures"]["normalize_seconds"] * 25)
                        assert int(stream["nb_read_frames"]) == expected
                    elif stream["codec_type"] == "audio" and record["workload"] == "normalize":
                        assert stream["sample_rate"] == "16000" and stream["channels"] == 1
                verified_media.append({"file": str(path), "streams": streams})
        write_json(output / "media-verification.json", verified_media)
        decoded = {}
        for backend in ("cpu", "wgpu", "cuda"):
            video = measured("render", backend)["request"]["params"]["output"]
            proc = subprocess.run([str(install / "ffmpeg.exe"), "-hide_banner", "-loglevel", "error", "-i", video,
                "-map", "0:v:0", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"], capture_output=True, check=True, timeout=60)
            decoded[backend] = np.frombuffer(proc.stdout, dtype=np.uint8)
            assert decoded[backend].size == metadata["fixtures"]["render_frames"] * 1280 * 720 * 3
        correctness["render_rgb8_vs_cpu"] = {backend: difference(decoded["cpu"], decoded[backend]) for backend in ("wgpu", "cuda")}
        correctness["verified_media_file_count"] = len(verified_media)
    write_json(output / "correctness.json", correctness)

    lines = []
    for phase, label in (("fresh_cli", "每次启动 CLI（总耗时）"), ("first_in_worker", "新 worker 首项任务（不含 Ready）"),
                         ("repeat_in_worker", "同 worker 重复任务（不含 Ready）")):
        lines += [label, "", "| 工作负载 | CPU 秒 | Vulkan 秒 | CUDA 秒 | CUDA / Vulkan |", "| --- | ---: | ---: | ---: | ---: |"]
        for workload in ("features", "frames", "normalize", "render", "train"):
            selected = {row["backend"]: row for row in summary if row["phase"] == phase and row["workload"] == workload
                        and row["run_label"] in ("main", "main-training", "cli")}
            if len(selected) != 3:
                continue
            values = [selected[backend]["median_seconds"] for backend in ("cpu", "wgpu", "cuda")]
            lines.append(f"| {workload} | {values[0]:.3f} | {values[1]:.3f} | {values[2]:.3f} | {values[2]/values[1]:.2f}× |")
        lines.append("")
    (output / "tables.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(json.dumps(correctness, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
