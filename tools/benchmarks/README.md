**Installed backend comparison**

`backend_comparison.py` tests the installed Release worker and CLI. It requires
Windows, Python 3.11+, a complete installed FeatherTalk payload, CUDA 12.x for
the CUDA cases, and a verified one-frame project with an Original U-Net
checkpoint. `psutil` is optional and adds process/module telemetry.
`analyze_backend_comparison.py` additionally requires NumPy.

All generated projects, media, checkpoints, raw protocol and timing records go
under the requested scratch root. The tools do not rebuild the application,
modify the installed payload, change system environment/power settings, or
clear GPU/driver caches. Run timed workloads sequentially.

Example in PowerShell, using the preserved fixture from the 2026-09-10 run:

```powershell
$benchRoot = 'target/benchmarks/retest'
$seedProject = 'target/benchmarks/2026-09-10-cpu-vulkan-cuda/seeds/training'

python -B tools/benchmarks/backend_comparison.py inventory --root $benchRoot
python -B tools/benchmarks/backend_comparison.py prepare --root $benchRoot --seed-project $seedProject
python -B tools/benchmarks/backend_comparison.py prepare-training --root $benchRoot
python -B tools/benchmarks/backend_comparison.py run --root $benchRoot --label main --repeats 3 --timeout 1200
python -B tools/benchmarks/backend_comparison.py cli --root $benchRoot --label cli --repeats 3 --timeout 1200
python -B tools/benchmarks/analyze_backend_comparison.py --root $benchRoot --out docs/reports/retest-data --verify-media
```

The original one-frame fixture was produced by `installer/windows/verify.ps1`.
It has locked assets and `models/unet/checkpoint-00000001`. Supply its directory
with `--seed-project` if the preserved benchmark directory is unavailable.
`prepare-training` uses the installed CPU worker to start a two-epoch plan and
cancel after the first step. This creates a genuine checkpoint whose plan
allows the one-step resume benchmark; no checkpoint metadata is hand-edited.

Defaults are `--install D:\tools\ft` and
`--cuda D:\environment\cuda-v12.6`. Both may be overridden. The child environment
pins the installed FFmpeg/models and puts the chosen CUDA `bin` first in PATH.
`inventory.json` and session metadata retain payload hashes, Ready frames and
actual loaded CUDA DLLs. A run fails if its installed payload differs from the
repository's `dist/FeatherTalk-0.1.0-x64.payload.json`.

`run --repeats 3` means one first task plus three more tasks in the **same worker**
for each backend/workload. `cli --repeats 3` means three independent CLI processes,
each spawning and closing its own worker, matching the current GUI lifecycle.
The former measures Start to the terminal protocol event; the latter measures
CLI process start to exit. Ready latency is recorded separately for `run`.
Successful GPU model tasks must report the requested backend, and wgpu must
report Vulkan. Completed output is checked before admitting a timing sample.

The normal workload set is 5 s of mono 16 kHz audio, 25 repeated 720p face frames,
10 s of 720p H.264/AAC normalization, a 4-frame render from the same checkpoint,
and a one-step Original U-Net baseline resume with batch size 1. Full training
timing includes restore, VGG19 loss, backward/Adam, checkpoint and preview.
The render fixture uses a verified repeated frame and landmarks plus real
feature extraction and asset locking. It is a performance fixture, not a
lip-sync quality evaluation.

Use `--workloads` / `--backends` to select a subset and a unique `--label` for
additional measurements. Existing task/session directories are not overwritten.
Failures are retained and excluded from summaries. `--worker-cwd` is only for
explicitly labelled configuration diagnostics; changing cwd can change CubeCL
configuration discovery and the autotune cache location. The main measurements
use a common cwd and existing default caches throughout.

To reproduce the additional cuDNN environment comparison after the baseline,
run the following sequentially, using the same scratch root:

```powershell
$cudnnRoot = 'C:\Program Files\NVIDIA\CUDNN\v9.3'
python -B tools/benchmarks/verify_cudnn.py --cudnn $cudnnRoot --output docs/reports/retest-data/cudnn-availability.json
python -B tools/benchmarks/backend_comparison.py run --root $benchRoot --label cudnn-module-probe --backends cuda --workloads features --repeats 0 --cudnn $cudnnRoot
python -B tools/benchmarks/backend_comparison.py cli --root $benchRoot --label cudnn-available --backends cuda --repeats 3 --timeout 1200 --cudnn $cudnnRoot
python -B tools/benchmarks/analyze_backend_comparison.py --root $benchRoot --out docs/reports/retest-data --verify-media
```

`--cudnn` exposes cuDNN 9's CUDA 12.6 DLL directory and sets CUDNN_PATH/HOME in
child processes. It does not add or enable cuDNN operators in FeatherTalk.
`verify_cudnn.py` independently performs a real tiny FP32 convolution and records
library versions and hashes; that check is not an application performance result.
The installed Burn-CUDA backend uses CubeCL/Cubek, with no cuDNN integration.
The analysis exports `cudnn-comparison.json` for the `cudnn-available` label,
comparing timings and outputs against the original `cli` CUDA records with the
same repeat index. Use the same repeat count for both runs. Keep `--verify-media`
when exporting the final report to include media validation and decoded RGB
comparisons. Worker module snapshots are retained in each `session.json`.
