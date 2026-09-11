# NVIDIA CUDA backend

The worker prefers NVIDIA CUDA for model computation when a usable CUDA installation is present. Existing wgpu and CPU implementations remain available. Explicit CPU/wgpu selections retain their meaning.

## Discovery and selection

Add auto and cuda choices. Automatic priority is certified CUDA, certified wgpu, then CPU; stable IDs determine ordering within a backend. Advertise actual backends only. CUDA UUIDs identify devices and retained CUDA ordinals select the exact runtime device. Probe CUDA independently of Vulkan.

Require the CUDA driver, NVRTC, headers and a successful Burn kernel before advertising CUDA. Missing/incompatible components preserve the existing wgpu/CPU paths. Dynamically load CUDA on Windows/Linux so building and starting without CUDA remains supported. Keep macOS on existing backends.

Automatic choices re-resolve at each handshake; explicit device choices remain exact. Extend the backend vocabulary and increment the protocol version so incompatible worker/client pairs fail clearly.

## Model execution

Use Burn 0.21.0 CUDA with fusion and autotuning for U-Net training, perceptual loss, rendering, FeatherHuBERT features, SCRFD and PFLD. Share device checks, result metadata and memory reporting. Synchronize loader threads and check GPU work before publishing. Fall back during startup discovery, never replay a partly committed stateful job.

## Media

Burn handles tensors; FFmpeg handles video decoding. Pass the selected CUDA ordinal to frame extraction and normalization hardware decoding. Failed hardware decoding retries the existing software command inside staging, preserving cancellation/timeouts and removing partial output before retry. Keep the current encoding quality/format contract.

## Verification

Test automatic ordering and fallback, exact choices, protocol and CLI/desktop behavior without GPU hardware. Exercise FFmpeg argument and retry behavior with existing process runner boundaries. Run actual CUDA smoke checks where compatible, report concrete limitations otherwise, and verify fallback plus affected regression suites.
