# Fast Qwen3-ASR Inference Server (vLLM Backend, FastAPI async processing)

A containerized Qwen3-ASR inference server using FastAPI and vLLM.
Provides HTTP `/transcribe` and WebSocket `/transcribe-streaming` endpoints.

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
├── Qwen3-ASR-1.7B/
└── Qwen3-ForcedAligner-0.6B/   (optional, only for timestamps)
```

For example with the HuggingFace CLI (or `modelscope download` from ModelScope):

```bash
uvx --from huggingface_hub hf download Qwen/Qwen3-ASR-1.7B --local-dir D:/models/Qwen3-ASR-1.7B
# optional
uvx --from huggingface_hub hf download Qwen/Qwen3-ForcedAligner-0.6B --local-dir D:/models/Qwen3-ForcedAligner-0.6B
```

### 2. Configure

```bash
cp .env.example .env
```

Set at least `MODEL_DIR` in `.env` (use forward slashes on Windows, e.g. `D:/models`).
Folder names inside it are set with `ASR_MODEL_DIR` / `ALIGNER_MODEL_DIR`.

> **Windows tip:** loading models from a Windows drive goes through the WSL 2 file share and is slow.
> For faster startup, keep the models inside the WSL filesystem (e.g. `\\wsl$\Ubuntu\home\<you>\models`,
> referenced in `.env` as the Linux path when running `docker compose` from WSL).

### 3. Build and start

```bash
docker compose up -d --build
docker compose logs -f
```

Wait for `Server is ready to accept requests.` — `docker compose ps` shows the container as `healthy` once
models are loaded and warmed up.

**!!FIRST START NOTICE!!** Building the image (flash-attn) and the first start (CUDA graph compile, loading
the model into VRAM and warmup) can take a while. The vLLM compile cache is kept in the `vllm_cache` volume,
so subsequent starts are faster.

The server is available at `http://127.0.0.1:8907` (bound to localhost only, as it has no authentication).

### Common commands

```bash
docker compose ps                 # status / health
docker compose logs -f            # follow logs
docker compose restart            # restart (e.g. after changing .env: use `up -d` instead)
docker compose up -d              # apply .env changes
docker compose down               # stop and remove the container
docker compose build --no-cache   # rebuild the image
```

### Development mode

Run `server.py` from the working tree with auto-reload (no rebuild needed after edits):

```bash
docker compose -f compose.yaml -f compose.dev.yaml up
```

## Configuration

Set in `.env` (see `.env.example`):

| Variable | Default | Description |
|---|---|---|
| `MODEL_DIR` | — (required) | Host directory with model folders, mounted read-only at `/models` |
| `ASR_MODEL_DIR` | `Qwen3-ASR-1.7B` | ASR model folder name inside `MODEL_DIR` |
| `ENABLE_ALIGNER_MODEL` | `false` | Load the forced aligner (timestamps for `POST /transcribe`) |
| `ALIGNER_MODEL_DIR` | `Qwen3-ForcedAligner-0.6B` | Aligner folder name inside `MODEL_DIR` |
| `PORT` | `8907` | Host port |
| `BIND_ADDR` | `127.0.0.1` | Host interface to bind; `0.0.0.0` exposes the server to the network |
| `GPU_MEMORY_UTILIZATION` | `0.15` | Fraction of GPU memory vLLM may reserve |
| `STREAM_CHUNK_SIZE_SEC` | `1.0` | Audio seconds per streaming decode step. Smaller = faster partial updates, more GPU work |
| `STREAM_UNFIXED_CHUNK_NUM` | `2` | First N chunks are decoded without a text prefix |
| `STREAM_UNFIXED_TOKEN_NUM` | `5` | Trailing tokens rolled back (re-decodable) on each step |
| `PARTIAL_INTERVAL_MS` | `120` | Minimum interval between `partial` messages |
| `BUNDLE_FLASH_ATTENTION` | `true` | Install flash-attn at build time |
| `MAX_JOBS` | `8` | Parallel jobs if flash-attn has to be compiled from source |

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
- **Query**: `language` (optional), `forced_alignment=true|false` (requires `ENABLE_ALIGNER_MODEL=true`)
- **Response**: `[{"text": "...", "language": "Chinese"}, ...]` (plus `timestamps` with forced alignment)

### `WS /transcribe-streaming`
Stream raw PCM audio for real-time transcription. One connection = one utterance.

- **URL**: `ws://127.0.0.1:8907/transcribe-streaming?language=zh-CN` (`language` is a **query parameter**, optional)

Protocol:

1. Client connects. If the models are still loading, the server holds the connection until they are ready
   (clients should apply a timeout while waiting for `ready`).
2. Server → `{"type": "ready"}`
3. Client → `{"type": "start", "format": "pcm_s16le", "sample_rate_hz": 16000}`
   (anything other than 16 kHz `pcm_s16le` is rejected with an `error` and close code 1003)
4. Server → `{"type": "info", "message": "language=Chinese"}` (only when `language` was given)
5. Client → binary frames: raw PCM, 16 kHz, 16-bit little-endian, mono (any size, e.g. 100 ms = 3200 bytes)
6. Server → `{"type": "partial", "text": "...", "language": "Chinese"}` — the **full** transcript so far;
   earlier words may be revised, so replace (don't append) the displayed text
7. Client → `{"type": "stop"}`
8. Server → `{"type": "final", "text": "...", "language": "Chinese"}`, then closes the connection (code 1000)

Errors are sent as `{"type": "error", "message": "..."}` followed by a close
(1002: audio before `start`, 1003: unsupported format/language, 1011: server not ready / internal error).

## Testing

> **Windows PowerShell:** use `curl.exe` instead of `curl` (in Windows PowerShell 5.1, `curl` is an alias
> for `Invoke-WebRequest`).

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
