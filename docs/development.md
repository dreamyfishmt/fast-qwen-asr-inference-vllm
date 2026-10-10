# Development

## Layout

```
server.py                  entry point: `uvicorn server:app`
asr_server/
├── config.py              environment settings (env() registers each one for docs/configuration.md)
├── language.py            language codes -> Qwen language names, OpenCC conversion
├── audio.py               upload decoding (libsndfile, ffmpeg), resampling, PCM stream decoding
├── auth.py                API token check, token redaction in logs
├── service.py             shared state: engine, loading status, concurrency limits, batch transcription
├── app.py                 FastAPI app: /health, /ready, /transcribe, /transcribe-streaming, upload size limit
├── openai_api.py          /v1/audio/transcriptions, /v1/models
└── engines/               inference backends (same interface, selected by ASR_BACKEND)
    ├── vllm_engine.py     qwen-asr + vLLM
    └── onnx_engine.py     ONNX Runtime (CPU or CUDA)
tests/                     pytest suite (runs against a fake engine, no model needed)
scripts/                   model download, docs generation
client-streaming.py        streaming test client
benchmark.py               load test
```

## Dependencies

All Python dependencies are declared in `pyproject.toml` and locked in `uv.lock`; the Docker images install them
with `uv sync --frozen`, so every build of a commit gets the same versions. Each backend is an extra:

| Extra | Image | Installs |
|---|---|---|
| `cpu` | `Dockerfile.cpu` | onnxruntime |
| `gpu` | `Dockerfile.gpu` | onnxruntime-gpu (pinned) + CUDA 13 wheels |
| `vllm` | `Dockerfile` | qwen-asr[vllm] (vLLM, PyTorch) |

The extras conflict with each other (`[tool.uv] conflicts`), so only one can be installed at a time.

```bash
uv sync --extra cpu                 # local environment: CPU backend + dev tools (pytest, ruff, httpx)
uv lock --upgrade-package fastapi   # update one dependency
uv lock --upgrade                   # update everything (then test the images, especially the GPU decoder)
```

Commit `uv.lock` together with `pyproject.toml`; `uv lock --check` fails when they disagree (CI runs it).

## Running locally

With the CPU backend and a downloaded model ([deployment-cpu.md](deployment-cpu.md)):

```bash
uv sync --extra cpu
ASR_BACKEND=onnx ASR_MODEL_NAME=/srv/models/qwen3-asr-0.6b-onnx uv run uvicorn server:app --port 8907 --reload
```

## Tests and lint

```bash
uv run pytest
uv run ruff check .
uv run scripts/gen-config-docs.py   # after adding or changing a setting
```

The tests use a fake engine, so they need no model or GPU. They also check that `docs/configuration.md` matches
the settings declared in the code. CI (`.github/workflows/ci.yml`) runs all of this plus a CPU image build on
every push and pull request.

## Trying the server

> **Windows PowerShell:** use `curl.exe` instead of `curl` (in Windows PowerShell 5.1, `curl` is an alias
> for `Invoke-WebRequest`).

If `API_TOKEN` is set, add `-H "Authorization: Bearer $API_TOKEN"` to the curl commands; the streaming client
reads `$API_TOKEN` (or `-t <token>`).

Sample files are in `files/` (`reference.*` — German: "Das ist ein Referenztext.").

```bash
# health / readiness
curl http://127.0.0.1:8907/health
curl -i http://127.0.0.1:8907/ready

# batch: one or more files in one request
curl -X POST "http://127.0.0.1:8907/transcribe?language=de" \
  -F "files=@files/reference.mp3" -F "files=@files/reference.wav"
# expected: [{"text":"Das ist ein Referenztext.","language":"German"}, ...]

# OpenAI-compatible
curl http://127.0.0.1:8907/v1/audio/transcriptions -F file=@files/reference.wav -F response_format=verbose_json

# forced alignment (vLLM image with ENABLE_ALIGNER_MODEL=true)
curl -X POST "http://127.0.0.1:8907/transcribe?language=de&forced_alignment=true" -F "files=@files/reference.wav"
```

`files/reference.m4a` only works with the vLLM image (it needs ffmpeg).

### Streaming

```bash
uv run client-streaming.py -e ws://127.0.0.1:8907/transcribe-streaming -f files/reference.pcm -l de
uv run client-streaming.py -e ws://127.0.0.1:8907/transcribe-streaming -f files/reference.wav -l de   # 24 kHz WAV
```

```
Connecting to ws://127.0.0.1:8907/transcribe-streaming?language=de...
Audio Duration: 2.06s
Streaming files/reference.pcm (16000 Hz)...
[19:00:53.518] [Server Ready]
...
[19:00:53.598] [Final] (German): Das ist ein Referenztext.
```

The client sends 16-bit mono WAV files at their own sample rate (the server resamples), or raw PCM
(16-bit mono, 16 kHz unless `-r` says otherwise). Convert other files with ffmpeg:

```bash
ffmpeg -i input.mp3 -ac 1 -c:a pcm_s16le input.wav
```

## Benchmarking

Defaults below: 4 concurrent clients, 20 requests each.

```bash
# streaming (WebSocket)
uv run benchmark.py --mode streaming --url ws://127.0.0.1:8907/transcribe-streaming --file files/reference.pcm --clients 4 --requests 20

# batch (HTTP, uses curl)
uv run benchmark.py --mode batch --url http://127.0.0.1:8907/transcribe --file files/reference.wav --clients 4 --requests 20
```

Reference numbers from the upstream **vLLM** backend (1x NVIDIA H200 NVL, `Qwen3-ASR-1.7B` + `Qwen3-ForcedAligner-0.6B`);
they don't describe the ONNX Runtime images (for the CPU image see
[How streaming works on CPU](deployment-cpu.md#how-streaming-works-on-cpu)):

- VRAM: < 2 GB at peak, even with 80 concurrent streams.
- Batch: 80 requests, avg QPS 28.5, latency P50 0.139 s.
- Qwen3-ASR runs at roughly 20x real-time with Flash Attention 2.

Upstream streaming numbers were measured before the streaming buffer fix
(audio was re-fed cumulatively), so re-run the streaming benchmark on your hardware.

## Releasing

Pushing a version tag builds the images with GitHub Actions (`.github/workflows/docker-publish.yml`)
and pushes them to GHCR:

```bash
git tag v1.2.3
git push origin v1.2.3
```

| Tag pushed | Image tags |
|---|---|
| `v1.2.3` | ONNX CPU: `1.2.3-cpu`, `1.2-cpu`, `1-cpu`, `latest-cpu` · ONNX GPU: `1.2.3-gpu`, `1.2-gpu`, `1-gpu`, `latest-gpu` |
| `v1.2.3-rc.1` | `1.2.3-rc.1-cpu`, `1.2.3-rc.1-gpu` (don't move `latest-*`) |

The workflow can also be started manually (Actions → Publish Docker image → Run workflow); a manual run on a
branch publishes `latest-cpu` / `latest-gpu` only. The vLLM image is not published; build it with `compose.yaml`.
The CPU image is built for linux/amd64 and linux/arm64.

GHCR packages are private when first published. To pull without `docker login ghcr.io`, open the package
(GitHub profile → Packages → fast-qwen-asr-inference-vllm → Package settings) and change its visibility to public.
