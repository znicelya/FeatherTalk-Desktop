# NVIDIA CUDA Backend Implementation Plan

> For agentic workers: use subagent-driven-development for independent client/media tasks and review; integrate the worker runtime in the main session.

**Goal:** Accelerate NVIDIA model workloads with burn-cuda and video decoding with FFmpeg CUDA, with automatic wgpu/CPU fallback.
**Architecture:** Discover and validate CUDA independently before advertising it. Select one concrete backend/device per worker and dispatch existing generic Burn functions on it. Keep media retries inside staging.
**Tech Stack:** Rust 2024 / 1.94, Burn 0.21.0, CubeCL 0.10.0, cudarc dynamic loading, FFmpeg.

## Global constraints

- Preserve the existing user modification to .gitignore.
- Build and start without CUDA; keep CUDA dependencies specific to Windows/Linux.
- Honor explicit CPU/wgpu devices. Automatic order: CUDA, wgpu, CPU.
- Do not replay partly committed model jobs.
- Preserve the desktop lockfile's yanked gpui dependency.

## Task 1: Selection and protocol

Files: domain frame/version/fixtures; client compute; CLI arguments; desktop compute; associated tests.
Interfaces: Backend::{Auto,Cpu,Wgpu,Cuda}; actual adapter metadata never uses Auto; ComputeOptions defaults to Auto; CUDA training requires training capability and an eligible CUDA adapter.

- [x] Write and run failing CUDA serialization, auto ordering/fallback, exact device and CLI tests.
- [x] Extend the protocol and increment its version; update fixtures.
- [x] Implement shared auto resolution, CUDA CLI/environment and desktop refresh without pinning automatic choices.
- [x] Run domain/client/CLI and desktop compute checks (including all desktop tests).

## Task 2: CUDA runtime and model dispatch

Files: root/model/worker manifests, models/backend.rs, worker compute/cuda.rs, compute/config/handshake/dispatch/models/telemetry.
Interfaces: registry resolves concrete metadata, retains UUID-to-ordinal mappings and opens a checked GPU context; context exposes synchronization, API and CUDA ordinal.

- [x] Add failing unavailable-runtime/selection tests and CUDA smoke coverage.
- [x] Add burn-cuda 0.21.0 and dynamically loaded cudarc; resolve lockfile.
- [x] Probe driver, NVRTC, headers and a Burn kernel, logging rejection reasons.
- [x] Wire all generic model operations, autodiff, synchronization and allocator metrics.
- [x] Run CPU/wgpu regression checks and compatible CUDA hardware checks.

## Task 3: CUDA decoding

Files: media/frame-pipeline command builders/runners and worker extraction/normalization.
Interface: optional selected CUDA ordinal in media configuration; hardware options precede FFmpeg input.

- [x] Write and run command/retry/cancellation tests.
- [x] Add hardware decoding and staged software retry, clearing partial output.
- [x] Wire selected CUDA device in worker callers and run media/pipeline suites.

## Task 4: Final verification

- [x] Update README prerequisites, automatic behavior, fallback and explicit overrides.
- [x] Run formatting, affected suites, workspace checks and supported desktop checks.
- [x] Review missing call sites, device identity and fallback boundaries.
- [x] Report completed checks and concrete hardware/environment limits.

## Verification results

- `cargo test --locked -p feathertalk-domain -p feathertalk-client -p feathertalk-cli --tests`: passed.
- `cargo test --locked --manifest-path crates/feathertalk-app/Cargo.toml --tests`: passed.
- `cargo test --offline -p feathertalk-media -p feathertalk-frame-pipeline --tests`: passed, including decoder retry/cleanup/timeout/cancellation.
- Worker unit tests and the commands/config/cuda_runtime/handshake/extract_frames/runtime/process_boundary/models/features/adapter_locks suites: 164 passed, including missing-toolkit startup, rejecting unavailable explicit GPUs before touching inputs, and duplicate CUDA identity fallback. The final unit run passed 48 tests; the separate native GPU smoke test remains ignored because the full GPU execution suites were run instead.
- `cargo test --locked -p feathertalk-worker --test cuda_execution -- --ignored --test-threads=1`: all 5 passed on RTX 2060, driver 560.94, isolated CUDA 12.6. Includes both U-Nets/all modes, CPU↔CUDA checkpoints, rendering, feature CPU parity, and SCRFD/PFLD fixture parity.
- `cargo test --locked -j 2 -p feathertalk-worker --test wgpu_execution --test cuda_media -- --ignored --test-threads=1`: all 5 wgpu model tests and the real FFmpeg CUDA decoding/retry test passed after the UUID fix.
- `cargo check --locked --workspace --all-targets -j 2`: passed after the final changes.
- All 44 changed Rust files passed targeted rustfmt checks; `git diff --check` passed. The desktop workspace lockfile is unchanged.
- Real FFmpeg H.264 decoding selected `cuda`/`h264_nvdec`; CLI CUDA normalization produced the existing MPEG-4 video and 16 kHz mono PCM audio contract.
- Fresh worker startup after the UUID fix passed in three environments: CUDA 12.6 advertised the RTX 2060 CUDA device; missing Toolkit and system CUDA 11.8 both retained CPU/wgpu and omitted CUDA.
- System CUDA 11.8 was left unchanged. Compatible NVIDIA wheel components were verified and extracted only into ignored `target/cuda-validation/toolkit` for the hardware tests.
- Independent review completed. CUDA discovery uses the MIG-aware UUID API; duplicate identities are excluded from selection while preserving a valid handshake and automatic fallback.
- Physical hardware coverage is limited to one RTX 2060. Multi-GPU/MIG identity handling has regression coverage but has not been tested on physical multi-GPU/MIG hardware.
