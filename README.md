# Fast Qwen3-ASR Inference Server (vLLM / ONNX CPU backends, FastAPI async processing)

A containerized Qwen3-ASR inference server using FastAPI.
Provides HTTP `/transcribe` and WebSocket `/transcribe-streaming` endpoints.

Two images, same API:

| Image | Backend | Model | Hardware |
|---|---|---|---|
| `…:latest` | vLLM + `qwen-asr` | Qwen3-ASR-1.7B (FP8) | NVIDIA GPU (RTX 30 series or newer) |
| `…:latest-cpu` | ONNX Runtime | Qwen3-ASR-0.6B (int4) | Any x86-64 / ARM64 CPU, ~2 GB RAM. See [CPU deployment](#cpu-deployment) |

The server runs with Docker Compose and loads the models from a **local model directory**
mounted into the container (read-only, offline — the HuggingFace Hub is never contacted).

## Requirements

- NVIDIA GPU + driver
- Docker with Compose v2 and GPU support:
  - Linux: [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
  - Windows: Docker Desktop with the WSL 2 backend (GPU support is built in)
- [uv](https://docs.astral.sh/uv/) — only for the local test/benchmark clients

## Setup

### 1. Download the models

Put each model in its own folder inside one models directory, e.g.:

```
D:/models/                      (or /srv/models on Linux)
├── Qwen3-ASR-1.7B-fp8/         default ASR model (FP8)
├── Qwen3-ASR-1.7B/             optional, original BF16 model
└── Qwen3-ForcedAligner-0.6B/   optional, only for timestamps
```

For example with the HuggingFace CLI:

```bash
uvx --from huggingface_hub hf download vrfai/Qwen3-ASR-1.7B-fp8 --local-dir D:/models/Qwen3-ASR-1.7B-fp8
# optional
uvx --from huggingface_hub hf download Qwen/Qwen3-ASR-1.7B --local-dir D:/models/Qwen3-ASR-1.7B
uvx --from huggingface_hub hf download Qwen/Qwen3-ForcedAligner-0.6B --local-dir D:/models/Qwen3-ForcedAligner-0.6B
```

#### Which model

The server uses the `qwen-asr` package, which needs checkpoints in the **original Qwen3-ASR layout**
(`config.json` with `thinker_config`, weights named `thinker.*`). The Transformers-native conversions
(`Qwen/Qwen3-ASR-1.7B-hf`) and GGUF / MLX / ONNX / OpenVINO builds do **not** work.

| Model | Size | Notes |
|---|---|---|
| [`vrfai/Qwen3-ASR-1.7B-fp8`](https://huggingface.co/vrfai/Qwen3-ASR-1.7B-fp8) (default) | ~2.5 GB weights | NVIDIA ModelOpt FP8, text decoder only (audio encoder and lm_head stay BF16). Reported WER 7.34% → 7.60% vs BF16 and ~25% higher single-request throughput (RTX 5090). Needs `VLLM_QUANTIZATION=modelopt` and an SM80+ GPU (RTX 30 series or newer); native FP8 compute on RTX 40/50 (SM89+), weight-only FP8 via Marlin on RTX 30 |
| [`Qwen/Qwen3-ASR-1.7B`](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) | ~3.9 GB weights | Original BF16. Set `ASR_MODEL_DIR=Qwen3-ASR-1.7B` and an empty `VLLM_QUANTIZATION=` |

### 2. Configure

```bash
cp .env.example .env
```

Set at least `MODEL_DIR` in `.env` (use forward slashes on Windows, e.g. `D:/models`).
Folder names inside it are set with `ASR_MODEL_DIR` / `ALIGNER_MODEL_DIR`.

> **Windows tip:** loading models from a Windows drive goes through the WSL 2 file share and is slow.
> For faster startup, keep the models inside the WSL filesystem (e.g. `\\wsl$\Ubuntu\home\<you>\models`,
> referenced in `.env` as the Linux path when running `docker compose` from WSL).

Only `compose.yaml` and `.env` are needed on the server machine — the image is pulled from GHCR
(`ghcr.io/dreamyfishmt/fast-qwen-asr-inference-vllm`). Set `IMAGE_TAG` in `.env` to pin a release
(e.g. `1.2.3`) instead of `latest`.

### 3. Start

```bash
docker compose up -d      # pulls the image on first run
docker compose logs -f
```

Wait for `Server is ready to accept requests.` — `docker compose ps` shows the container as `healthy` once
models are loaded and warmed up.

**!!FIRST START NOTICE!!** The image is large (CUDA + vLLM + PyTorch), and the first start (CUDA graph compile,
loading the model into VRAM and warmup) can take a while. The vLLM compile cache is kept in the `vllm_cache` volume,
so subsequent starts are faster.

The server is available at `http://127.0.0.1:8907` (bound to localhost only, as it has no authentication).

### Common commands

```bash
docker compose ps                 # status / health
docker compose logs -f            # follow logs
docker compose restart            # restart (e.g. after changing .env: use `up -d` instead)
docker compose up -d              # apply .env changes
docker compose pull && docker compose up -d   # update to the newest image for IMAGE_TAG
docker compose down               # stop and remove the container
```

### Build the image locally

`compose.local.yaml` builds the image from this checkout (tagged `qwen3-asr-server:local`) instead of pulling it.
Pass both files to every command:

```bash
docker compose -f compose.yaml -f compose.local.yaml up -d --build
docker compose -f compose.yaml -f compose.local.yaml build --no-cache   # full rebuild
```

Building compiles/installs flash-attn and takes a while; `BUNDLE_FLASH_ATTENTION` and `MAX_JOBS` in `.env`
control it.

### Development mode

Run `server.py` from the working tree with auto-reload (no rebuild needed after edits). Works with the pulled
image; add `-f compose.local.yaml` as well if you changed the Dockerfile:

```bash
docker compose -f compose.yaml -f compose.dev.yaml up
```

## CPU deployment

For servers without a GPU, e.g. a small VPS that clients reach over the internet. Uses ONNX Runtime with
[`rhasspy/qwen3-asr-0.6b-onnx-int4-merged`](https://huggingface.co/rhasspy/qwen3-asr-0.6b-onnx-int4-merged)
(Qwen3-ASR-0.6B, int4, ~785 MB). The image has no PyTorch or CUDA.

1. Download the model:

   ```bash
   uvx --from huggingface_hub hf download rhasspy/qwen3-asr-0.6b-onnx-int4-merged \
     --local-dir /srv/models/qwen3-asr-0.6b-onnx-int4-merged
   ```

2. Get `compose.cpu.yaml`, `Caddyfile` and `.env.cpu.example` from this repo, then:

   ```bash
   cp .env.cpu.example .env
   # set MODEL_DIR, and API_TOKEN (e.g. `openssl rand -hex 32`)
   ```

3. Start, either:
   - **HTTPS/WSS (recommended on the internet):** point a domain's DNS A record at the server, set `DOMAIN` and
     `BIND_ADDR=127.0.0.1` in `.env`, open ports 80 and 443, and run
     `docker compose -f compose.cpu.yaml --profile tls up -d`. Caddy gets a Let's Encrypt certificate
     automatically. Clients use `wss://DOMAIN/transcribe-streaming`.
   - **Plain WS:** `docker compose -f compose.cpu.yaml up -d` and clients use
     `ws://SERVER_IP:8907/transcribe-streaming`. The token and audio travel unencrypted, so use this only on
     a trusted network or behind your own TLS proxy.

4. Check: `curl -H "Authorization: Bearer $API_TOKEN" https://DOMAIN/health`

**Behaviour on CPU.** This export has no incremental decoder, so a live `partial` means re-transcribing the
whole utterance so far. Partials are therefore sent every `STREAM_PARTIAL_INTERVAL_SEC` (2 s) only while the
utterance is shorter than `STREAM_PARTIAL_MAX_SEC` (10 s); the `final` result is always a full transcription
after `stop`. Utterances are capped at `STREAM_MAX_SEC` (60 s). Requests are processed one at a time.

| Variable | Default | Description |
|---|---|---|
| `API_TOKEN` | — (required) | Shared secret; clients send `Authorization: Bearer <token>` |
| `ASR_MODEL_DIR` | `qwen3-asr-0.6b-onnx-int4-merged` | Model folder inside `MODEL_DIR` (merged or split int4 layout) |
| `ONNX_THREADS` | `0` (all cores) | ONNX Runtime threads |
| `STREAM_PARTIALS` | `true` | Send live partial results |
| `STREAM_PARTIAL_INTERVAL_SEC` | `2.0` | Seconds of new audio between partials |
| `STREAM_PARTIAL_MAX_SEC` | `10` | No partials once the utterance is longer than this |
| `STREAM_MAX_SEC` | `60` | Audio beyond this per utterance is dropped (an `info` message is sent) |
| `DOMAIN` | — | Domain for the `tls` profile (Caddy, automatic HTTPS) |

Build the CPU image locally instead of pulling it:
`docker compose -f compose.cpu.yaml -f compose.cpu.local.yaml up -d --build`.

The CPU image has no ffmpeg: `POST /transcribe` accepts WAV/FLAC/OGG/MP3 (what libsndfile reads), and forced
alignment is not available.

## Authentication

Set `API_TOKEN` to require a shared secret on every endpoint except `GET /ready` (used by the container
healthcheck). Clients send `Authorization: Bearer <token>`, or `?token=<token>` where headers can't be set
(e.g. browser WebSockets). Missing or wrong tokens get HTTP 401, or a rejected WebSocket handshake (HTTP 403).
The CPU compose file requires it; for the GPU compose file it's optional (`API_TOKEN` in `.env`).

## Releasing

Pushing a version tag builds the image with GitHub Actions (`.github/workflows/docker-publish.yml`)
and pushes it to GHCR:

```bash
git tag v1.2.3
git push origin v1.2.3
```

| Tag pushed | Image tags |
|---|---|
| `v1.2.3` | GPU: `1.2.3`, `1.2`, `1`, `latest` · CPU: `1.2.3-cpu`, `1.2-cpu`, `1-cpu`, `latest-cpu` |
| `v1.2.3-rc.1` | `1.2.3-rc.1`, `1.2.3-rc.1-cpu` (don't move `latest`) |

The workflow can also be started manually (Actions → Publish Docker image → Run workflow); a manual run on a
branch publishes `latest` / `latest-cpu` only. The CPU image is built for linux/amd64 and linux/arm64.

GHCR packages are private when first published. To pull without `docker login ghcr.io`, open the package
(GitHub profile → Packages → fast-qwen-asr-inference-vllm → Package settings) and change its visibility to public.

## Configuration

Set in `.env` (see `.env.example`):

| Variable | Default | Description |
|---|---|---|
| `MODEL_DIR` | — (required) | Host directory with model folders, mounted read-only at `/models` |
| `ASR_MODEL_DIR` | `Qwen3-ASR-1.7B-fp8` | ASR model folder name inside `MODEL_DIR` |
| `VLLM_QUANTIZATION` | `modelopt` | vLLM quantization method passed to the model loader; empty for unquantized (BF16) checkpoints |
| `ENABLE_ALIGNER_MODEL` | `false` | Load the forced aligner (timestamps for `POST /transcribe`) |
| `ALIGNER_MODEL_DIR` | `Qwen3-ForcedAligner-0.6B` | Aligner folder name inside `MODEL_DIR` |
| `IMAGE` | `ghcr.io/dreamyfishmt/fast-qwen-asr-inference-vllm` | Image to pull |
| `IMAGE_TAG` | `latest` | Image tag, e.g. `1.2.3` to pin a release |
| `PORT` | `8907` | Host port |
| `BIND_ADDR` | `127.0.0.1` | Host interface to bind; `0.0.0.0` exposes the server to the network |
| `GPU_MEMORY_UTILIZATION` | `0.15` | Fraction of GPU memory vLLM may reserve |
| `STREAM_CHUNK_SIZE_SEC` | `1.0` | Audio seconds per streaming decode step. Smaller = faster partial updates, more GPU work |
| `STREAM_UNFIXED_CHUNK_NUM` | `2` | First N chunks are decoded without a text prefix |
| `STREAM_UNFIXED_TOKEN_NUM` | `5` | Trailing tokens rolled back (re-decodable) on each step |
| `PARTIAL_INTERVAL_MS` | `120` | Minimum interval between `partial` messages |
| `BUNDLE_FLASH_ATTENTION` | `true` | Local build only: install flash-attn |
| `MAX_JOBS` | `8` | Local build only: parallel jobs if flash-attn has to be compiled from source |

Further server settings (`MAX_CONCURRENT_INFER`, `MAX_CONCURRENT_DECODE`, `THREADPOOL_WORKERS`,
`MAX_NEW_TOKENS`, `OPENCC_TW_CONFIG`, `OPENCC_HK_CONFIG`) are read from the environment by `server.py`
and can be added to `compose.yaml`.

## Languages

The `language` parameter is optional (auto-detect when omitted). It accepts ISO 639-1 / BCP-47 codes
(`en`, `en-US`, `de`, `ja`, `ko`, `zh`, `zh-CN`, `yue`, ...) or Qwen language names (`Chinese`, `English`, ...).
Unsupported languages are rejected (HTTP 400 / WebSocket `error` message).

Qwen3-ASR has a single `Chinese` language that outputs Simplified Chinese. Traditional variants are produced
by converting the transcript with OpenCC:

| Code | Output |
|---|---|
| `zh`, `zh-CN`, `zh-Hans`, `zh-SG` | Simplified Chinese |
| `zh-TW`, `zh-Hant` | Traditional Chinese, Taiwan phrasing (`s2twp`) |
| `zh-HK`, `zh-MO` | Traditional Chinese, Hong Kong (`s2hk`) |

## Endpoints

### `GET /health`
Model loading status (`starting` → `loading_models` → `warming_up` → `ready`, or `error`), limits,
streaming settings and memory usage. Always returns 200.

### `GET /ready`
200 `{"status":"ready"}` once models are loaded and warmed up, 503 otherwise. Used by the container healthcheck.

### `POST /transcribe`
Upload one or more audio files (any format ffmpeg can decode).

- **URL**: `http://127.0.0.1:8907/transcribe?language=zh-CN`
- **Body**: multipart/form-data, one or more `files` fields
- **Query**: `language` (optional), `context` (optional, see below), `forced_alignment=true|false` (GPU image with `ENABLE_ALIGNER_MODEL=true`)
- **Response**: `[{"text": "...", "language": "Chinese"}, ...]` (plus `timestamps` with forced alignment)

### `WS /transcribe-streaming`
Stream raw PCM audio for real-time transcription. One connection = one utterance.

- **URL**: `ws://127.0.0.1:8907/transcribe-streaming?language=zh-CN` (`language` is a **query parameter**, optional)

Protocol:

1. Client connects. If the models are still loading, the server holds the connection until they are ready
   (clients should apply a timeout while waiting for `ready`).
2. Server → `{"type": "ready"}`
3. Client → `{"type": "start", "format": "pcm_s16le", "sample_rate_hz": 16000, "context": "..."}`
   (anything other than 16 kHz `pcm_s16le` is rejected with an `error` and close code 1003; `context` is optional)
4. Server → `{"type": "info", "message": "language=Chinese"}` (only when `language` was given)
5. Client → binary frames: raw PCM, 16 kHz, 16-bit little-endian, mono (any size, e.g. 100 ms = 3200 bytes)
6. Server → `{"type": "partial", "text": "...", "language": "Chinese"}` — the **full** transcript so far;
   earlier words may be revised, so replace (don't append) the displayed text
7. Client → `{"type": "stop"}`
8. Server → `{"type": "final", "text": "...", "language": "Chinese"}`, then closes the connection (code 1000)

Errors are sent as `{"type": "error", "message": "..."}` followed by a close
(1002: audio before `start`, 1003: unsupported format/language, 1011: server not ready / internal error).
With `API_TOKEN` set, a missing or wrong token rejects the handshake (HTTP 403).

If the utterance exceeds `STREAM_MAX_SEC`, the server sends `{"type": "info", "message": "max_duration_reached=60s"}`
and ignores further audio; `stop` still returns the final result.

**Context (`context`).** Free text placed in the prompt's system turn to bias recognition toward specific
spellings, e.g. `"Vocabulary: Kubernetes, QwenType, 张三"`. Keep it short; it is part of every decode.

## Testing

> **Windows PowerShell:** use `curl.exe` instead of `curl` (in Windows PowerShell 5.1, `curl` is an alias
> for `Invoke-WebRequest`).

If `API_TOKEN` is set, add `-H "Authorization: Bearer $API_TOKEN"` to the curl commands; the streaming client
reads `$API_TOKEN` (or `-t <token>`).

Sample files are in `files/` (`reference.*` — German: "Das ist ein Referenztext.").

### Health / readiness

```bash
curl http://127.0.0.1:8907/health
curl -i http://127.0.0.1:8907/ready
```

### Batch transcription

```bash
# single file
curl -X POST "http://127.0.0.1:8907/transcribe?language=de" -F "files=@files/reference.wav"

# batch: multiple files in one request
curl -X POST "http://127.0.0.1:8907/transcribe?language=de" \
  -F "files=@files/reference.m4a" -F "files=@files/reference.mp3" -F "files=@files/reference.wav"
```

Expected: `[{"text":"Das ist ein Referenztext.","language":"German"}, ...]`

### Forced alignment

Requires `ENABLE_ALIGNER_MODEL=true` in `.env` (then `docker compose up -d`).

```bash
curl -X POST "http://127.0.0.1:8907/transcribe?language=de&forced_alignment=true" -F "files=@files/reference.wav"
```

### Streaming

The test clients are managed with uv (`pyproject.toml`):

```bash
uv sync
uv run client-streaming.py -e ws://127.0.0.1:8907/transcribe-streaming -f files/reference.pcm -l de
```

```
Connecting to ws://127.0.0.1:8907/transcribe-streaming?language=de...
Audio Duration: 2.06s
Streaming files/reference.pcm...
[19:00:53.518] [Server Ready]
...
[19:00:53.598] [Final] (German): Das ist ein Referenztext.
```

The client expects raw PCM (16 kHz, 16-bit, mono). Convert other files with ffmpeg:

```bash
ffmpeg -i input.mp3 -f s16le -ac 1 -ar 16000 input.pcm
```

## Benchmarking

Defaults below: 4 concurrent clients, 20 requests each.

```bash
# streaming (WebSocket)
uv run benchmark.py --mode streaming --url ws://127.0.0.1:8907/transcribe-streaming --file files/reference.pcm --clients 4 --requests 20

# batch (HTTP, uses curl)
uv run benchmark.py --mode batch --url http://127.0.0.1:8907/transcribe --file files/reference.wav --clients 4 --requests 20
```

Reference numbers from upstream (1x NVIDIA H200 NVL, `Qwen3-ASR-1.7B` + `Qwen3-ForcedAligner-0.6B`):

- VRAM: < 2 GB at peak, even with 80 concurrent streams.
- Batch: 80 requests, avg QPS 28.5, latency P50 0.139 s.
- Qwen3-ASR runs at roughly 20x real-time with Flash Attention 2.

Upstream streaming numbers were measured before the streaming buffer fix
(audio was re-fed cumulatively), so re-run the streaming benchmark on your hardware.
