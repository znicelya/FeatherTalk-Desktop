# SCRFD Burn importer

This standalone tool validates `data_utils/scrfd_2.5g_kps.onnx`, generates the
pinned Burn 0.21 graph, converts its temporary burnpack state to safetensors,
and publishes a deterministic four-file artifact tree. The model source hash
is `32d20c77b9e2dc1d07e94c2ab9d25bdd5cd05eddbe0b46e7b38e7a1eca22e99a` and the
tracked ONNX file is 3,291,017 bytes.

Run from `rust/`:

```powershell
cargo run --manifest-path tools/scrfd-import/Cargo.toml --bin generate -- --repo-root .. --destination crates/feathertalk-scrfd/.generated-candidate
cargo run --manifest-path tools/scrfd-import/Cargo.toml --bin generate -- --repo-root .. --verify-against crates/feathertalk-scrfd
```

The `.bpk` file exists only in a temporary staging directory and is never
published. The final tree contains generated Rust source, a source/weight hash
contract, safetensors weights, and the validated manifest. The manifest keeps
license status `NOASSERTION` with `redistribution_approved: false`.

The Python/OpenCV fixture generator is separate and is only used for Task 5:

```powershell
python -m venv tools\scrfd-import\.venv
tools\scrfd-import\.venv\Scripts\python.exe -m pip install --disable-pip-version-check -r tools\scrfd-import\python\requirements-fixture.txt
tools\scrfd-import\.venv\Scripts\python.exe tools\scrfd-import\python\generate_fixture.py --repo-root .. --destination crates\feathertalk-scrfd\tests\fixtures\opencv_cpu_v1
tools\scrfd-import\.venv\Scripts\python.exe tools\scrfd-import\python\generate_fixture.py --repo-root .. --verify-against crates\feathertalk-scrfd\tests\fixtures\opencv_cpu_v1
```

The fixture command requires Python 3.11 exactly at the major/minor level and
pins NumPy 2.2.6 plus opencv-python-headless 4.12.0.88. Runtime and ordinary
Rust tests never invoke Python, OpenCV, or the ONNX parser.

SCRFD acceptance compares every element of all nine raw OpenCV outputs on the
NdArray CPU backend (`max_abs <= 1e-3`, `mean_abs <= 1e-4`):

```powershell
cargo test -p feathertalk-scrfd --test parity -- --nocapture
```

An ignored WGPU smoke test checks artifact loading, one forward pass, and all
public tensor shapes when a compatible adapter is available:

```powershell
cargo test -p feathertalk-scrfd --test wgpu_smoke -- --ignored --nocapture
```

WGPU is not required for ordinary CPU acceptance.
