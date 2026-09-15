# FeatherTalk worker command benchmark

This crate links `feathertalk-worker` directly and benchmarks command execution
in the benchmark process. It does not build or launch a worker executable.

Create a JSON array of worker `Request` values. Any string may contain
`{{repeat}}`, which is replaced by `0` for the first run and `1`, `2`, ... for
warm runs. This keeps commands that write output in separate destinations.

```json
[
  {
    "command": "normalize_media",
    "params": {
      "input": "fixtures/input.mp4",
      "output_dir": "target/benchmark/normalize-{{repeat}}"
    }
  },
  {
    "command": "extract_features",
    "params": {
      "project_dir": "fixtures/project-{{repeat}}",
      "audio": "fixtures/audio_16k_mono.wav"
    }
  }
]
```

Run:

```powershell
cargo run -p feathertalk-benchmark -- --request-file target/benchmark/requests.json --repeats 3 --label main
```

The final report contains first-run and warm median/minimum/maximum times.
Use `--json` for machine-readable output. Worker toolchain and model paths are
read from the same environment variables as the worker; `--backend` and
`--adapter` override compute selection.
