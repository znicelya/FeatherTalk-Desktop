"""Measure installed FeatherTalk workers without rebuilding or editing the install.

Each workload gets a fresh worker, then repeats in that same worker. Timings use
perf_counter and end at the terminal protocol event (including publication).
Inputs, raw protocol, stderr, timing events and outputs stay under --root.
No driver caches, system environment variables or power settings are changed.
"""

from __future__ import annotations

import argparse
import array
import copy
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import secrets
import shutil
import statistics
import struct
import subprocess
import sys
import threading
import time
import wave
from datetime import datetime, timezone

try:
    import psutil
except ImportError:
    psutil = None


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_info(path):
    path = Path(path)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def child_env(args, backend):
    env = dict(os.environ)
    # Select tools and models from the installed payload, regardless of dev env.
    for key in list(env):
        if key.startswith("FEATHERTALK_WORKER_"):
            del env[key]
    env.update({
        "CUDA_PATH": str(args.cuda),
        "CUDA_ROOT": str(args.cuda),
        "CUDA_HOME": str(args.cuda),
        "FEATHERTALK_WORKER_BACKEND": backend,
        "FEATHERTALK_WORKER_FFMPEG": str(args.install / "ffmpeg.exe"),
        "FEATHERTALK_WORKER_FFPROBE": str(args.install / "ffprobe.exe"),
        "FEATHERTALK_WORKER_SCRFD_DIR": str(args.install / "models/scrfd_2_5g"),
        "FEATHERTALK_WORKER_PFLD_DIR": str(args.install / "models/pfld_ghost_one"),
        "FEATHERTALK_WORKER_HUBERT_DIR": str(args.install / "models/feather_hubert"),
        "FEATHERTALK_WORKER_VGG19_DIR": str(args.install / "models/vgg19"),
    })
    search_paths = [str(args.cuda / "bin"), str(args.install)]
    if args.cudnn:
        cudnn_bin = args.cudnn / "bin/12.6"
        if not (cudnn_bin / "cudnn64_9.dll").is_file():
            raise ValueError(f"cuDNN 9 CUDA 12.6 DLL directory missing: {cudnn_bin}")
        env["CUDNN_PATH"] = str(args.cudnn)
        env["CUDNN_HOME"] = str(args.cudnn)
        search_paths.insert(1, str(cudnn_bin))
    env["PATH"] = os.pathsep.join(search_paths + [env.get("PATH", "")])
    return env


def external(argv, *, env=None, cwd=None, timeout=180):
    started = time.perf_counter()
    proc = subprocess.run([str(x) for x in argv], capture_output=True, text=True,
                          encoding="utf-8", errors="replace", env=env, cwd=cwd,
                          timeout=timeout, creationflags=subprocess.CREATE_NO_WINDOW)
    return {"argv": [str(x) for x in argv], "returncode": proc.returncode,
            "seconds": time.perf_counter() - started, "stdout": proc.stdout, "stderr": proc.stderr}


class Worker:
    def __init__(self, args, backend, name):
        self.args, self.backend, self.name = args, backend, name
        self.directory = args.root / "sessions" / name
        self.directory.mkdir(parents=True, exist_ok=False)
        self.queue = queue.Queue()
        self.stderr = open(self.directory / "stderr.log", "w", encoding="utf-8")
        self.stdout = open(self.directory / "stdout.ndjson", "w", encoding="utf-8")
        self.events = open(self.directory / "received.ndjson", "w", encoding="utf-8")
        self.started = time.perf_counter()
        self.proc = subprocess.Popen([str(args.install / "feathertalk-worker.exe")],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.stderr,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
            env=child_env(args, backend), cwd=args.worker_cwd,
            creationflags=subprocess.CREATE_NO_WINDOW)
        self.thread = threading.Thread(target=self.read, daemon=True)
        self.thread.start()
        self.telemetry_stop = threading.Event()
        self.telemetry_thread = threading.Thread(target=self.sample_process, daemon=True)
        self.telemetry_thread.start()
        try:
            received, ready = self.next_frame(180)
            if ready.get("frame") != "ready":
                raise RuntimeError(f"Expected Ready: {ready}")
            self.ready = ready["data"]
            self.startup_seconds = received - self.started
            self.adapter = next(a for a in self.ready["adapters"]
                                if a["backend"] == backend and a["certified"])
            self.info = {"name": name, "backend": backend, "pid": self.proc.pid,
                         "worker_cwd": str(args.worker_cwd),
                         "cudnn_directory": str(args.cudnn) if args.cudnn else None,
                         "startup_seconds": self.startup_seconds, "ready": self.ready,
                         "selected_adapter": self.adapter, "loaded_gpu_modules": self.modules()}
            write_json(self.directory / "session.json", self.info)
            print(f"READY {name}: {self.startup_seconds:.3f}s {self.adapter['name']}", flush=True)
        except BaseException:
            self.close()
            raise

    def modules(self):
        if psutil is None:
            return []
        try:
            return sorted({m.path for m in psutil.Process(self.proc.pid).memory_maps()
                           if any(k in m.path.lower() for k in ("cuda", "nvrtc", "vulkan", "nvcuda", "cudnn", "cublas"))})
        except (psutil.Error, OSError):
            return []

    def sample_process(self):
        if psutil is None:
            return
        with open(self.directory / "process.csv", "w", newline="", encoding="utf-8") as output:
            writer = csv.writer(output)
            writer.writerow(["session_seconds", "pid", "name", "cpu_user_seconds", "cpu_system_seconds", "rss_bytes", "threads"])
            while not self.telemetry_stop.is_set():
                try:
                    parent = psutil.Process(self.proc.pid)
                    for proc in [parent] + parent.children(recursive=True):
                        try:
                            cpu = proc.cpu_times()
                            writer.writerow([time.perf_counter() - self.started, proc.pid, proc.name(),
                                             cpu.user, cpu.system, proc.memory_info().rss, proc.num_threads()])
                        except psutil.Error:
                            pass
                    output.flush()
                except psutil.Error:
                    break
                self.telemetry_stop.wait(1)

    def read(self):
        for line in self.proc.stdout:
            received = time.perf_counter()
            self.stdout.write(line)
            self.stdout.flush()
            try:
                frame = json.loads(line)
            except json.JSONDecodeError:
                self.queue.put((received, {"invalid_protocol_line": line}))
                continue
            self.events.write(json.dumps({"session_seconds": received - self.started,
                                          "frame": frame}, ensure_ascii=False) + "\n")
            self.events.flush()
            self.queue.put((received, frame))
        self.queue.put((time.perf_counter(), {"eof": True}))

    def next_frame(self, timeout):
        try:
            received, frame = self.queue.get(timeout=timeout)
        except queue.Empty:
            raise TimeoutError(f"No protocol response within {timeout}s") from None
        if "eof" in frame or "invalid_protocol_line" in frame:
            raise RuntimeError(f"Worker protocol ended: {frame}; see {self.directory / 'stderr.log'}")
        return received, frame

    def send(self, value):
        self.proc.stdin.write(json.dumps(value, ensure_ascii=False) + "\n")
        self.proc.stdin.flush()

    def task(self, command, params, label, cancel_after_step=None):
        task_id = f"{int(time.time() * 1000):013d}-{secrets.token_hex(4)}"
        request = {"command": command, "params": {k: str(v) if isinstance(v, Path) else v
                                                  for k, v in params.items()}}
        write_json(self.directory / f"{label}.request.json", request)
        started = time.perf_counter()
        self.send({"frame": "start", "data": {"protocol_version": 3,
                   "task_id": task_id, "request": request}})
        record = {"session": self.name, "label": label, "backend_requested": self.backend,
                  "command": command, "request": request, "started_utc": datetime.now(timezone.utc).isoformat(),
                  "startup_seconds": self.startup_seconds, "events": []}
        last_stage, last_notice = "waiting", started
        print(f"START {self.name}/{label} {command}", flush=True)
        try:
            while True:
                now = time.perf_counter()
                if now - started > self.args.timeout:
                    self.send({"frame": "cancel", "data": {"protocol_version": 3, "task_id": task_id}})
                    raise TimeoutError(f"Task exceeded {self.args.timeout}s at {last_stage}")
                if now - last_notice >= 20:
                    print(f"WAIT {self.name}/{label} {now-started:.1f}s stage={last_stage}", flush=True)
                    last_notice = now
                try:
                    received, frame = self.next_frame(min(5, self.args.timeout - (now - started)))
                except TimeoutError:
                    continue
                if frame.get("frame") != "event":
                    raise RuntimeError(f"Unexpected frame: {frame}")
                event = frame["data"]
                if event["task_id"] != task_id:
                    raise RuntimeError(f"Unexpected task id: {event['task_id']}")
                last_stage = event["stage"]["stage"]
                record["events"].append({"seconds": received - started, "event": event})
                if (cancel_after_step is not None and last_stage == "training"
                        and event["stage"]["data"]["step"] >= cancel_after_step):
                    self.send({"frame": "cancel", "data": {"protocol_version": 3, "task_id": task_id}})
                    record["cancel_requested_after_step"] = cancel_after_step
                    cancel_after_step = None
                if last_stage in ("completed", "failed", "cancelled"):
                    record.update(status=last_stage, task_seconds=received - started,
                                  result=event.get("result"), error=event.get("error"))
                    break
        except Exception as error:
            record.update(status="harness_error", task_seconds=time.perf_counter() - started,
                          error=f"{type(error).__name__}: {error}")
        write_json(self.directory / f"{label}.result.json", record)
        print(f"END {self.name}/{label}: {record['status']} {record['task_seconds']:.3f}s", flush=True)
        return record

    def close(self):
        if hasattr(self, "telemetry_stop"):
            self.telemetry_stop.set()
            self.telemetry_thread.join(timeout=3)
        if hasattr(self, "proc") and self.proc.poll() is None:
            try:
                self.send({"frame": "shutdown", "data": {"protocol_version": 3}})
                self.proc.wait(timeout=10)
            except (OSError, subprocess.TimeoutExpired):
                if psutil:
                    try:
                        for child in psutil.Process(self.proc.pid).children(recursive=True):
                            child.kill()
                    except psutil.Error:
                        pass
                self.proc.kill()
                self.proc.wait(timeout=10)
        if hasattr(self, "thread"):
            self.thread.join(timeout=5)
        for stream in (self.stderr, self.stdout, self.events):
            stream.close()


def inventory(args):
    env = child_env(args, "cpu")
    files = [file_info(path) for path in sorted(args.install.rglob("*")) if path.is_file()]
    payload_path = args.repo / "dist/FeatherTalk-0.1.0-x64.payload.json"
    payload = json.loads(payload_path.read_text(encoding="utf-8-sig"))
    by_name = {str(Path(x["path"]).relative_to(args.install)).replace("\\", "/"): x for x in files}
    mismatches = [x["name"] for x in payload["files"]
                  if x["name"].replace("\\", "/") not in by_name or
                  by_name[x["name"].replace("\\", "/")]["sha256"] != x["sha256"]]
    info = {"measured_utc": datetime.now(timezone.utc).isoformat(), "install": str(args.install),
            "toolkit": str(args.cuda), "python": sys.version, "files": files,
            "payload_manifest": file_info(payload_path), "payload_mismatches": mismatches,
            "controlled_environment": {k: v for k, v in env.items()
                if k.startswith(("CUDA", "FEATHERTALK", "CUBECL", "BURN", "WGPU", "RAYON"))},
            "inherited_gpu_environment": {k: v for k, v in os.environ.items()
                if k.startswith(("CUDA", "FEATHERTALK", "CUBECL", "BURN", "WGPU", "RAYON"))},
            "inherited_cuda_path_entries": [x for x in os.environ.get("PATH", "").split(os.pathsep)
                                             if "cuda" in x.lower()]}
    info["gpu"] = external(["nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total,pstate,temperature.gpu,power.limit,utilization.gpu,clocks.sm,clocks.mem", "--format=csv"])
    info["nvcc"] = external([args.cuda / "bin/nvcc.exe", "--version"], env=env)
    info["ffmpeg"] = external([args.install / "ffmpeg.exe", "-version"], env=env)
    info["default_capabilities"] = external([args.install / "feathertalk.exe", "--json", "capabilities"], cwd=args.worker_cwd)
    if psutil:
        info["memory"] = psutil.virtual_memory()._asdict()
        info["cpu_logical_count"] = psutil.cpu_count()
        info["cpu_physical_count"] = psutil.cpu_count(logical=False)
    write_json(args.root / "inventory.json", info)
    if mismatches:
        raise RuntimeError(f"Installed payload mismatches: {mismatches}")
    print(f"Inventory: {len(payload['files'])} payload hashes match; {info['gpu']['stdout'].strip()}", flush=True)
    for backend in args.backends:
        worker = Worker(args, backend, f"inventory-{backend}")
        worker.close()


def wav(path, seconds):
    values = array.array("h", (round(8192 * math.sin(2 * math.pi * 440 * n / 16000))
                              for n in range(round(seconds * 16000))))
    if sys.byteorder != "little":
        values.byteswap()
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16000)
        output.writeframes(values.tobytes())


def prepare(args):
    inputs = args.root / "inputs"
    inputs.mkdir(exist_ok=False)
    seeds = args.root / "seeds"
    seeds.mkdir(exist_ok=False)
    if not args.seed_project:
        raise ValueError("prepare requires --seed-project with a verified original U-Net checkpoint")
    shutil.copytree(args.seed_project, seeds / "training")
    fixture = args.repo / "crates/feathertalk-frame-adapters/tests/fixtures/demo_frame_v1/frame.jpg"
    wav(inputs / "features.wav", args.audio_seconds)
    env = child_env(args, "cpu")
    commands = [
        [args.install / "ffmpeg.exe", "-hide_banner", "-loglevel", "error", "-loop", "1",
         "-framerate", "25", "-i", fixture, "-frames:v", str(args.frame_count), "-an",
         "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", inputs / "faces.mp4"],
        [args.install / "ffmpeg.exe", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
         f"testsrc2=size=1280x720:rate=30:duration={args.normalize_seconds}", "-f", "lavfi", "-i",
         f"sine=frequency=440:sample_rate=48000:duration={args.normalize_seconds}",
         "-c:v", "libx264", "-preset", "fast", "-crf", "23", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-shortest", inputs / "normalize.mp4"],
    ]
    generated = []
    for command in commands:
        result = external(command, env=env, timeout=300)
        generated.append(result)
        if result["returncode"]:
            raise RuntimeError(result["stderr"])
    render = seeds / "render"
    assets = render / "assets"
    (assets / "frames").mkdir(parents=True)
    (assets / "landmarks").mkdir()
    project = json.loads((seeds / "training/project.json").read_text(encoding="utf-8-sig"))
    project.update(project_id="backend-benchmark-render", display_name="Backend benchmark synthetic repeated face")
    write_json(render / "project.json", project)
    quality = json.loads((seeds / "training/assets/quality.json").read_text(encoding="utf-8-sig"))
    exemplar = quality["frames"][0]
    quality.update(frame_count=args.render_frames, accepted_count=args.render_frames, frames=[])
    for index in range(args.render_frames):
        item = copy.deepcopy(exemplar)
        item.update(index=index, frame_file=f"frames/{index:06}.jpg", landmark_file=f"landmarks/{index:06}.lms")
        quality["frames"].append(item)
        shutil.copy2(seeds / "training/assets/frames/000000.jpg", assets / item["frame_file"])
        shutil.copy2(seeds / "training/assets/landmarks/000000.lms", assets / item["landmark_file"])
    write_json(assets / "quality.json", quality)
    video_command = commands[0].copy()
    video_command[video_command.index("-frames:v") + 1] = str(args.render_frames)
    video_command[-1] = assets / "video_25fps.mp4"
    result = external(video_command, env=env)
    generated.append(result)
    if result["returncode"]:
        raise RuntimeError(result["stderr"])
    wav(assets / "audio_16k_mono.wav", args.render_frames / 25)
    worker = Worker(args, "cpu", "prepare-render")
    try:
        for command, params in [
            ("extract_features", {"project_dir": render, "audio": assets / "audio_16k_mono.wav"}),
            ("lock_asset_package", {"project_dir": render}),
        ]:
            result = worker.task(command, params, command)
            if result["status"] != "completed":
                raise RuntimeError(result)
    finally:
        worker.close()
    write_json(args.root / "fixtures.json", {
        "audio_seconds": args.audio_seconds, "frame_count": args.frame_count,
        "render_frames": args.render_frames, "normalize_seconds": args.normalize_seconds,
        "source_face": file_info(fixture), "source_seed": str(args.seed_project),
        "input_files": [file_info(p) for p in sorted(inputs.iterdir())],
        "checkpoint_files": [file_info(p) for p in sorted((seeds / "training/models/unet/checkpoint-00000001").iterdir())],
        "generation": generated, "render_fixture": "Repeated verified frame and landmarks, actual audio extraction and asset lock",
    })
    print("Fixtures prepared and render assets locked", flush=True)


def request_for(args, workload, task_dir):
    task_dir.mkdir(parents=True, exist_ok=False)
    if workload in ("features", "frames"):
        project = task_dir / "project"
        (project / "assets").mkdir(parents=True)
        write_json(project / "project.json", {})
        source, name, command, key = (("features.wav", "audio_16k_mono.wav", "extract_features", "audio")
                                     if workload == "features" else
                                     ("faces.mp4", "video_25fps.mp4", "extract_frames", "video"))
        media = project / "assets" / name
        shutil.copy2(args.root / "inputs" / source, media)
        return command, {"project_dir": project, key: media}
    if workload == "normalize":
        return "normalize_media", {"input": args.root / "inputs/normalize.mp4", "output_dir": task_dir / "normalized"}
    if workload == "train":
        project = task_dir / "project"
        shutil.copytree(args.root / "seeds/training-resume", project)
        return "train", {"project_dir": project, "mode": "baseline", "variant": "original_unet",
                         "epochs": 2, "batch_size": 1, "resume": True}
    if workload == "render":
        project = args.root / "seeds/render"
        return "render", {"project_dir": project, "checkpoint": args.root / "seeds/training/models/unet/checkpoint-00000001",
                          "audio": project / "assets/audio_16k_mono.wav", "output": task_dir / "render.mp4",
                          "max_output_frames": args.render_frames}
    raise ValueError(workload)


def prepare_training(args):
    """Create a genuine checkpoint with two planned epochs, cancelled after one."""
    original = args.root / "seeds/training"
    project = args.root / "seeds/training-resume"
    project.mkdir(exist_ok=False)
    shutil.copy2(original / "project.json", project / "project.json")
    shutil.copytree(original / "assets", project / "assets")
    worker = Worker(args, "cpu", "prepare-training-resume")
    try:
        record = worker.task("train", {"project_dir": project, "mode": "baseline", "variant": "original_unet",
                                      "epochs": 2, "batch_size": 1, "resume": False},
                             "seed", cancel_after_step=1)
        if record["status"] != "cancelled":
            raise RuntimeError(f"Expected a cancelled seed run after step 1: {record}")
    finally:
        worker.close()
    checkpoint = project / "models/unet/checkpoint-00000001"
    manifest = json.loads((checkpoint / "manifest.json").read_text(encoding="utf-8"))
    state = json.loads((checkpoint / "training-state.json").read_text(encoding="utf-8"))
    for name in ("model", "optimizer", "training_state"):
        item = manifest[name]
        assert sha256(checkpoint / item["file_name"]) == item["sha256"]
    write_json(args.root / "training-seed.json", {"generation": "Installed CPU worker, epochs=2, cancel after training event step=1",
               "project_dir": str(project), "state": state,
               "files": [file_info(p) for p in sorted(checkpoint.iterdir())], "record": record})
    print("Real resumable checkpoint created and all component hashes verified", flush=True)


def validate_result(args, record):
    if record["status"] != "completed":
        return {"ok": False, "reason": "task did not complete"}
    result, params = record["result"], record["request"]["params"]
    command = record["command"]
    validation = {"ok": True}
    if command in ("train", "render", "extract_features", "extract_frames"):
        expected = {"cpu": "ndarray-cpu", "wgpu": "wgpu", "cuda": "cuda"}[record["backend_requested"]]
        assert result["backend"] == expected, result
        if expected == "wgpu":
            assert result["graphics_api"].lower() == "vulkan", result
    if command == "extract_features":
        path = Path(params["project_dir"]) / "assets/features/feather_hubert.f32"
        data = path.read_bytes()
        magic, version, tokens, pair, dims, size = struct.unpack("<8sIQQQQ", data[:44])
        assert magic == b"FTF32\0\0\0" and version == 1 and pair == 2 and dims == 1024
        assert size == len(data) - 44 == tokens * dims * 4
        values = array.array("f")
        values.frombytes(data[44:])
        if sys.byteorder != "little":
            values.byteswap()
        assert all(math.isfinite(value) for value in values)
        validation.update(feature=file_info(path), tokens=tokens, dims=dims, finite=True)
    elif command == "extract_frames":
        quality = json.loads((Path(params["project_dir"]) / "assets/quality.json").read_text(encoding="utf-8"))
        assert quality["frame_count"] == args.frame_count
        assert quality["accepted_count"] == args.frame_count and not quality["anomalies"]
        validation.update(frame_count=quality["frame_count"], accepted_count=quality["accepted_count"])
    elif command == "train":
        assert result["global_step"] == 2 and result["checkpoints_written"] == 1
        assert result["samples_seen"] == 1 and math.isfinite(result["total_loss"])
        metrics = Path(params["project_dir"]) / "outputs/metrics/step-00000002.json"
        validation["training_metrics"] = json.loads(metrics.read_text(encoding="utf-8"))
        manifest = Path(params["project_dir"]) / "models/unet/checkpoint-00000002/manifest.json"
        validation["checkpoint_manifest"] = json.loads(manifest.read_text(encoding="utf-8"))
    elif command == "render":
        assert result["frame_count"] == args.render_frames
        validation["output"] = file_info(params["output"])
    elif command == "normalize_media":
        validation["outputs"] = [file_info(p) for p in sorted(Path(params["output_dir"]).iterdir()) if p.is_file()]
        assert len(validation["outputs"]) == 2
    return validation


def run(args):
    config = json.loads((args.root / "fixtures.json").read_text(encoding="utf-8"))
    args.frame_count, args.render_frames = config["frame_count"], config["render_frames"]
    with open(args.root / "results.ndjson", "a", encoding="utf-8") as output:
        for wi, workload in enumerate(args.workloads):
            # Rotate order between workloads; never run competing backends together.
            order = args.backends[wi % len(args.backends):] + args.backends[:wi % len(args.backends)]
            for backend in order:
                name = f"{args.label}-{workload}-{backend}"
                worker = Worker(args, backend, name)
                try:
                    for repeat in range(args.repeats + 1):
                        label = f"r{repeat}"
                        task_dir = args.root / "tasks" / name / label
                        command, params = request_for(args, workload, task_dir)
                        record = worker.task(command, params, label)
                        record.update(workload=workload, repeat=repeat, phase="first_in_worker" if repeat == 0 else "repeat_in_worker",
                                      task_dir=str(task_dir), run_label=args.label)
                        try:
                            record["validation"] = validate_result(args, record)
                        except Exception as error:
                            record["validation"] = {"ok": False, "reason": f"{type(error).__name__}: {error}"}
                        write_json(task_dir / "measurement.json", record)
                        output.write(json.dumps(record, ensure_ascii=False) + "\n")
                        output.flush()
                        if not record["validation"]["ok"]:
                            print(f"FAILED {name}: {record['error']} {record['validation']}", flush=True)
                            break
                finally:
                    worker.info["loaded_gpu_modules_after_tasks"] = worker.modules()
                    write_json(worker.directory / "session.json", worker.info)
                    worker.close()


def monitored_run(args):
    gpu_path = args.root / f"gpu-{args.label}-{int(time.time())}.csv"
    with open(gpu_path, "x", encoding="utf-8") as output:
        monitor = subprocess.Popen(["nvidia-smi", "--query-gpu=timestamp,uuid,pstate,temperature.gpu,utilization.gpu,utilization.memory,memory.used,power.draw,clocks.sm,clocks.mem",
                                    "--format=csv", "--loop-ms=1000"], stdout=output, stderr=subprocess.DEVNULL,
                                   creationflags=subprocess.CREATE_NO_WINDOW)
        try:
            (run_cli if args.action == "cli" else run)(args)
        finally:
            monitor.terminate()
            monitor.wait(timeout=10)


def cli_argv(args, backend, command, params):
    argv = [args.install / "feathertalk.exe", "--worker", args.install / "feathertalk-worker.exe",
            "--backend", backend, "--json", command.replace("_", "-")]
    if command == "train":
        argv += [params["project_dir"], "--mode", "baseline", "--variant", "original-unet",
                 "--epochs", "2", "--batch-size", "1", "--resume"]
    elif command == "render":
        argv += [params[k] for k in ("project_dir", "checkpoint", "audio", "output")]
        argv += ["--max-output-frames", str(args.render_frames)]
    elif command == "normalize_media":
        argv += [params["input"], params["output_dir"]]
    else:
        argv += [params["project_dir"], params["audio" if command == "extract_features" else "video"]]
    return [str(x) for x in argv]


def run_cli(args):
    """Real CLI invocations: a new worker per task, as in the current GUI."""
    config = json.loads((args.root / "fixtures.json").read_text(encoding="utf-8"))
    args.frame_count, args.render_frames = config["frame_count"], config["render_frames"]
    with open(args.root / "cli-results.ndjson", "a", encoding="utf-8") as output:
        for wi, workload in enumerate(args.workloads):
            for repeat in range(args.repeats):
                offset = (wi + repeat) % len(args.backends)
                order = args.backends[offset:] + args.backends[:offset]
                for backend in order:
                    name = f"{args.label}-{workload}-{backend}-r{repeat}"
                    task_dir = args.root / "tasks" / name
                    command, params = request_for(args, workload, task_dir)
                    argv = cli_argv(args, backend, command, params)
                    record = {"run_label": args.label, "workload": workload, "repeat": repeat,
                              "worker_cwd": str(args.worker_cwd),
                              "cudnn_directory": str(args.cudnn) if args.cudnn else None,
                              "phase": "fresh_cli_process", "backend_requested": backend,
                              "command": command, "task_dir": str(task_dir), "argv": argv,
                              "request": {"command": command, "params": {k: str(v) if isinstance(v, Path) else v for k, v in params.items()}},
                              "started_utc": datetime.now(timezone.utc).isoformat()}
                    print(f"CLI START {name}", flush=True)
                    with open(task_dir / "stdout.ndjson", "w", encoding="utf-8") as stdout, open(task_dir / "stderr.log", "w", encoding="utf-8") as stderr:
                        started = time.perf_counter()
                        proc = subprocess.Popen(argv, stdout=stdout, stderr=stderr, env=child_env(args, backend),
                                                cwd=args.worker_cwd, creationflags=subprocess.CREATE_NO_WINDOW)
                        while True:
                            try:
                                proc.wait(timeout=20)
                                break
                            except subprocess.TimeoutExpired:
                                elapsed = time.perf_counter() - started
                                print(f"CLI WAIT {name} {elapsed:.1f}s", flush=True)
                                if elapsed > args.timeout:
                                    if psutil:
                                        for child in psutil.Process(proc.pid).children(recursive=True):
                                            child.kill()
                                    proc.kill()
                                    proc.wait(timeout=10)
                                    break
                        record.update(end_to_end_seconds=time.perf_counter() - started, returncode=proc.returncode)
                    frames = [json.loads(line) for line in (task_dir / "stdout.ndjson").read_text(encoding="utf-8").splitlines() if line]
                    ready = next((f["data"] for f in frames if f.get("frame") == "ready"), None)
                    terminal = next((f["data"] for f in reversed(frames) if f.get("frame") == "event" and f["data"]["stage"]["stage"] in ("completed", "failed", "cancelled")), None)
                    record.update(ready=ready, status=terminal["stage"]["stage"] if terminal else "process_failed",
                                  result=terminal.get("result") if terminal else None,
                                  error=terminal.get("error") if terminal else "No terminal event")
                    try:
                        record["validation"] = validate_result(args, record)
                        if record["returncode"] != 0:
                            record["validation"] = {"ok": False, "reason": f"CLI exit {proc.returncode}"}
                    except Exception as error:
                        record["validation"] = {"ok": False, "reason": f"{type(error).__name__}: {error}"}
                    write_json(task_dir / "measurement.json", record)
                    output.write(json.dumps(record, ensure_ascii=False) + "\n")
                    output.flush()
                    print(f"CLI END {name}: {record['status']} {record['end_to_end_seconds']:.3f}s validation={record['validation']['ok']}", flush=True)


def summarize(args):
    records = [json.loads(line) for line in (args.root / "results.ndjson").read_text(encoding="utf-8").splitlines() if line]
    groups = {}
    for record in records:
        if record["status"] == "completed" and record["validation"]["ok"]:
            groups.setdefault((record["run_label"], record["workload"], record["backend_requested"]), []).append(record)
    rows = []
    for (label, workload, backend), records in groups.items():
        first = [r["task_seconds"] for r in records if r["repeat"] == 0]
        warm = [r["task_seconds"] for r in records if r["repeat"] > 0]
        rows.append({"run_label": label, "workload": workload, "backend": backend,
                     "startup_seconds": records[0]["startup_seconds"], "first_seconds": first[0] if first else None,
                     "warm_n": len(warm), "warm_median_seconds": statistics.median(warm) if warm else None,
                     "warm_min_seconds": min(warm) if warm else None, "warm_max_seconds": max(warm) if warm else None})
    write_json(args.root / "summary.json", rows)
    if rows:
        with open(args.root / "summary.csv", "w", newline="", encoding="utf-8-sig") as output:
            writer = csv.DictWriter(output, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps(rows, ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("inventory", "prepare", "prepare-training", "run", "cli", "summarize"))
    parser.add_argument("--install", type=Path, default=Path(r"D:\tools\ft"))
    parser.add_argument("--cuda", type=Path, default=Path(r"D:\environment\cuda-v12.6"))
    parser.add_argument("--cudnn", type=Path, help="Expose installed cuDNN 9 / CUDA 12.6 DLLs; does not change the application's backend implementation")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--worker-cwd", type=Path, help="Override only for explicitly labelled cache diagnostics")
    parser.add_argument("--seed-project", type=Path)
    parser.add_argument("--backends", nargs="+", choices=("cpu", "wgpu", "cuda"), default=["cpu", "wgpu", "cuda"])
    parser.add_argument("--workloads", nargs="+", choices=("features", "frames", "normalize", "render", "train"),
                        default=["features", "frames", "normalize", "render", "train"])
    parser.add_argument("--repeats", type=int, default=3, help="Warm tasks after the first task for run; independent invocations for cli")
    parser.add_argument("--label", default="main")
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--audio-seconds", type=float, default=5)
    parser.add_argument("--frame-count", type=int, default=25)
    parser.add_argument("--render-frames", type=int, default=4)
    parser.add_argument("--normalize-seconds", type=float, default=10)
    args = parser.parse_args()
    args.repo = Path(__file__).resolve().parents[2]
    args.root, args.install, args.cuda = args.root.resolve(), args.install.resolve(), args.cuda.resolve()
    if args.seed_project:
        args.seed_project = args.seed_project.resolve()
    if args.cudnn:
        args.cudnn = args.cudnn.resolve()
    args.worker_cwd = args.worker_cwd.resolve() if args.worker_cwd else args.root / "cwd"
    args.worker_cwd.mkdir(parents=True, exist_ok=True)
    {"inventory": inventory, "prepare": prepare, "prepare-training": prepare_training,
     "run": monitored_run, "cli": monitored_run, "summarize": summarize}[args.action](args)


if __name__ == "__main__":
    main()
