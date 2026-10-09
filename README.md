# Fast Qwen3-ASR Inference Server

A containerized [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR) speech recognition server (FastAPI) with HTTP
`/transcribe` and real-time WebSocket `/transcribe-streaming` endpoints. Models are read from a **local model
directory** mounted into the container (read-only, offline).

| Image | Backend | Model | Hardware | Image size |
|---|---|---|---|---|
| **`…:latest-gpu` (recommended)** | ONNX Runtime + CUDA 13 | Qwen3-ASR-1.7B (int4) | NVIDIA GPU, driver R580+. See [Quick start: GPU](#quick-start-gpu-onnx-runtime) | ~3 GB |
| `…:latest-cpu` | ONNX Runtime | Qwen3-ASR-0.6B (int4) | Any x86-64 / ARM64 CPU, 2+ GB RAM. See [CPU deployment](#cpu-deployment) | ~0.5 GB |
| built locally (`compose.yaml`) | vLLM + `qwen-asr` | Qwen3-ASR-1.7B (FP8) | NVIDIA GPU (RTX 30 series or newer). See [vLLM image (advanced)](#vllm-image-advanced) | ~14 GB |

Images: `ghcr.io/dreamyfishmt/fast-qwen-asr-inference-vllm`. All three serve the same API.

The ONNX Runtime images need no PyTorch or vLLM and are the right fit for one user or a few (e.g. voice input).
The vLLM image is mostly PyTorch, vLLM and the full CUDA library set built for every GPU generation; that buys
continuous batching for many concurrent users. It is not published (too large to build in CI), so `compose.yaml`
builds it locally.

## Client: QwenType

[**QwenType**](https://github.com/dreamyfishmt/QwenType) is the Windows voice-typing client for this server:
hold Right Ctrl, speak, release, and the text is typed into the focused app, with a live transcript while you
speak. Download `QwenType.exe` from its [Releases](https://github.com/dreamyfishmt/QwenType/releases), start the
server below, then set the connection in the tray menu → **ASR Server…**:

- **WebSocket URL**: default `ws://127.0.0.1:8907/transcribe-streaming` (server on the same PC); for a remote
  server behind HTTPS, `wss://DOMAIN/transcribe-streaming`
- **API Token**: the server's `API_TOKEN` (required by the CPU image; leave empty if the server has none)
- **Hotwords** (optional): names and terms sent as the stream's [`context`](#ws-transcribe-streaming)

## Quick start: GPU (ONNX Runtime)

Qwen3-ASR-1.7B (int4) on an NVIDIA GPU with ONNX Runtime's CUDA execution provider. The image (`…:latest-gpu`,
~3 GB installed) contains no PyTorch or vLLM.

**Requirements**

- NVIDIA GPU with a driver that supports CUDA 13 (R580 or newer)
- Docker with Compose v2 and GPU support:
  - Linux: [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
  - Windows: Docker Desktop with the WSL 2 backend (GPU support is built in)
- [uv](https://docs.astral.sh/uv/) (or the `hf` CLI), to download the model

Get this repository first (the compose files, the download script and the test samples live here):

```bash
git clone https://github.com/dreamyfishmt/fast-qwen-asr-inference-vllm
cd fast-qwen-asr-inference-vllm
```

Or, to only run the server, download just the compose file and the settings template into an empty folder
(`Caddyfile` is only needed with `--profile tls`):

```bash
curl -LO https://raw.githubusercontent.com/dreamyfishmt/fast-qwen-asr-inference-vllm/main/compose.gpu.yaml
curl -L -o .env https://raw.githubusercontent.com/dreamyfishmt/fast-qwen-asr-inference-vllm/main/.env.gpu.example
```

On Windows PowerShell, use `curl.exe` instead of `curl`. This already creates `.env`, so skip the `cp` in step 2.

1. Download the model (~2.7 GB) from [`dreamyfishmt/qwen3-asr-1.7b-onnx`](https://huggingface.co/dreamyfishmt/qwen3-asr-1.7b-onnx),
   pinned to the tested revision:

   ```bash
   uvx --from huggingface_hub hf download dreamyfishmt/qwen3-asr-1.7b-onnx \
     --revision f4a19c9705b87ea06685b1ffddd95772ffbde1b0 --local-dir /srv/models/qwen3-asr-1.7b-onnx
   # or: scripts/download-models.sh 1.7b /srv/models   # -> /srv/models/qwen3-asr-1.7b-onnx
   ```

   On Windows (PowerShell; `uvx` from [uv](https://docs.astral.sh/uv/)):

   ```powershell
   uvx --from huggingface_hub hf download dreamyfishmt/qwen3-asr-1.7b-onnx --revision f4a19c9705b87ea06685b1ffddd95772ffbde1b0 --local-dir D:/models/qwen3-asr-1.7b-onnx
   ```

   The repo repackages, unmodified, the encoder, embeddings and tokenizer of
   [`andrewleech/qwen3-asr-1.7b-onnx`](https://huggingface.co/andrewleech/qwen3-asr-1.7b-onnx) and the int4
   GroupQueryAttention decoder with fp16 I/O from [`sorryhyun/qwen3-asr-onnx-gqa`](https://huggingface.co/sorryhyun/qwen3-asr-onnx-gqa);
   see its [model card](https://huggingface.co/dreamyfishmt/qwen3-asr-1.7b-onnx) for the exact source revisions.

   If the download fails with a 401 from `cas-server.xethub.hf.co` (some proxies block Hugging Face's Xet storage),
   set `HF_HUB_DISABLE_XET=1` and run it again (the script does this automatically).

   The 0.6B model folder from [CPU deployment](#cpu-deployment) works on the GPU too (set `ASR_MODEL_DIR`).

2. `cp .env.gpu.example .env` and set `MODEL_DIR` (use forward slashes on Windows, e.g. `D:/models`).

   > **Windows tip:** loading models from a Windows drive goes through the WSL 2 file share and is slow.
   > For faster startup, keep the models inside the WSL filesystem (e.g. `\\wsl$\Ubuntu\home\<you>\models`),
   > referenced in `.env` as the Linux path when running `docker compose` from WSL.

3. Start:

   ```bash
   docker compose -f compose.gpu.yaml up -d
   docker compose -f compose.gpu.yaml logs -f   # look for "provider cuda", then "Server is ready"
   ```

   The server listens on `http://127.0.0.1:8907` (this machine only). If CUDA can't be initialized (driver too
   old, no GPU visible to the container), it logs a warning and falls back to the CPU.

   For other machines, set `API_TOKEN` and either `BIND_ADDR=0.0.0.0` (LAN, plain WS) or a `DOMAIN` with
   `--profile tls` (Caddy with automatic HTTPS, as in [CPU deployment](#cpu-deployment)).

On the GPU a full transcription takes a fraction of a second, so the defaults differ from the CPU image:
partials every second for the whole utterance, and the final result is always a fresh full decode
(`STREAM_FINAL_REUSE_PARTIAL=false`). The KV cache stays on the GPU between decode steps.

Build locally instead of pulling: `docker compose -f compose.gpu.yaml -f compose.gpu.local.yaml up -d --build`.

| Variable | Default | Description |
|---|---|---|
| `ASR_MODEL_DIR` | `qwen3-asr-1.7b-onnx` | Model folder inside `MODEL_DIR` |
| `ONNX_PROVIDER` | `cuda` (image default) | `cuda` or `cpu` |
| `ONNX_DEVICE_ID` | `0` | GPU index |
| `ONNX_DECODER` / `ONNX_ENCODER` | auto | Decoder / encoder file in the model folder; auto prefers `decoder-*fp16*` on the GPU and `decoder-*fp32*` on the CPU, and `encoder.int4.onnx` over `encoder.onnx` |
| `STREAM_PARTIAL_INTERVAL_SEC` / `STREAM_PARTIAL_MAX_SEC` | `1.0` / `60` | Partial cadence and cutoff |
| `STREAM_FINAL_REUSE_PARTIAL` | `false` | Continue the final result from the last partial (see [CPU deployment](#cpu-deployment)) |
| `API_TOKEN` | — | Optional shared secret; see [Authentication](#authentication) |
| `STREAM_MAX_SEC` | `120` | Audio beyond this per utterance is dropped |

## CPU deployment

For servers without a GPU, e.g. a small VPS that clients reach over the internet. The same ONNX Runtime backend as
the GPU image, with Qwen3-ASR-0.6B quantized to int4 (~840 MB of model files); the image has no PyTorch or CUDA.

The model folder combines two Hugging Face repos: the audio encoder, embeddings and tokenizer from
[`rhasspy/qwen3-asr-0.6b-onnx-int4-merged`](https://huggingface.co/rhasspy/qwen3-asr-0.6b-onnx-int4-merged), and the
text decoder from [`sorryhyun/qwen3-asr-onnx-gqa`](https://huggingface.co/sorryhyun/qwen3-asr-onnx-gqa)
(built with GroupQueryAttention, so the cost per generated token stays almost flat as the utterance grows).

On the server, get this repository first (`git clone https://github.com/dreamyfishmt/fast-qwen-asr-inference-vllm`
and `cd` into it); it has the compose files and the download script.

1. Download the model with [`scripts/download-models.sh`](scripts/download-models.sh) (needs the `hf` CLI or
   [uv](https://docs.astral.sh/uv/)). It fetches exactly the files the server uses from the two repos, pinned to the
   tested revisions, into `/srv/models/qwen3-asr-0.6b-onnx`:

   ```bash
   scripts/download-models.sh 0.6b /srv/models
   ```

   <details><summary>Manual commands (same files and revisions)</summary>

   ```bash
   D=/srv/models/qwen3-asr-0.6b-onnx
   uvx --from huggingface_hub hf download rhasspy/qwen3-asr-0.6b-onnx-int4-merged \
     config.json tokenizer.json embed_tokens.bin encoder.int4.onnx encoder.int4.onnx.data \
     --revision 9ea8c26bbf497ef74a84ce19202ce62246af8ab4 --local-dir $D
   uvx --from huggingface_hub hf download sorryhyun/qwen3-asr-onnx-gqa \
     decoder-0.6b-fp32.onnx decoder-0.6b-fp32.onnx.data \
     --revision 075249f70b56cdded1cf4b189cbdde0fb77aeec1 --local-dir $D
   ```

   </details>

   If a download fails with a 401 from `cas-server.xethub.hf.co` (some proxies block Hugging Face's Xet storage),
   set `HF_HUB_DISABLE_XET=1`; the script retries that way automatically.

   (The rhasspy repo's own `decoder_merged.int4.onnx` also works if present and no `decoder-*.onnx` is, but it is
   ~2× slower and more prone to repetition loops; `ONNX_DECODER` picks a decoder file explicitly.)

2. In the repository clone:

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

### How streaming works on CPU

This export has no incremental decoder, so a `partial` result re-transcribes the utterance so far. To keep that
affordable:

- Partials run in the background every `STREAM_PARTIAL_INTERVAL_SEC` (2 s) for the first `STREAM_PARTIAL_MAX_SEC`
  (20 s) of an utterance. Each one continues from the previous partial minus its last `STREAM_UNFIXED_TOKEN_NUM`
  tokens (the rollback strategy of qwen-asr's own streaming), so it only generates the new part.
- On `stop`, a partial still running is aborted and the final result is computed right away, again continuing
  from the last partial.
- Requests are transcribed one at a time; a partial is skipped when another request is being transcribed.

Measured with 2 cores of an Intel Xeon (2.1 GHz, AVX-512) and audio sent at real-time speed; expect slower
results on CPUs without AVX-512 / AMX:

| Utterance | Delay from `stop` to `final` |
|---|---|
| 4 s | 1.4 s |
| 6 s, fast speech (~60 characters) | 2.3–2.5 s |
| 11 s | 2.9 s |
| 23 s | 3.9 s |

Memory: ~0.9 GB resident after loading, ~1.1 GB while serving short utterances, ~1.45 GB peak for a 51 s
utterance. 2 GB of RAM works for dictation-length audio; 4 GB leaves headroom for `STREAM_MAX_SEC=60`.

| Variable | Default | Description |
|---|---|---|
| `API_TOKEN` | — (required) | Shared secret; clients send `Authorization: Bearer <token>` |
| `ASR_MODEL_DIR` | `qwen3-asr-0.6b-onnx` | Model folder inside `MODEL_DIR` |
| `ONNX_DECODER` | auto | Decoder file in the model folder (`decoder-*.onnx`, then `decoder_merged.int4.onnx`, then split) |
| `ONNX_THREADS` | `0` (all cores) | ONNX Runtime threads |
| `STREAM_PARTIALS` | `true` | Send live partial results |
| `STREAM_PARTIAL_INTERVAL_SEC` | `2.0` | Seconds of new audio between partials |
| `STREAM_PARTIAL_MAX_SEC` | `20` | No partials once the utterance is longer than this |
| `STREAM_REUSE_PARTIAL` | `true` | Continue from the previous partial instead of decoding from scratch |
| `STREAM_FINAL_REUSE_PARTIAL` | `true` | The final result also continues from the last partial (faster on CPU; an error in an early partial can survive into the final text) |
| `STREAM_UNFIXED_CHUNK_NUM` / `STREAM_UNFIXED_TOKEN_NUM` | `2` / `5` | First N partials start from scratch; last K tokens are re-decoded |
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
The CPU compose file requires it; for the GPU and vLLM compose files it's optional (`API_TOKEN` in `.env`).

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
Model loading status (`starting` → `loading_models` → `warming_up` → `ready`, or `error`), backend, limits,
streaming settings and memory usage. Returns 200 regardless of the loading status; with `API_TOKEN` set it needs the
token (401 otherwise), so it also works as a token check.

### `GET /ready`
200 `{"status":"ready"}` once models are loaded and warmed up, 503 otherwise. Never needs a token; used by the
container healthcheck.

### `POST /transcribe`
Upload one or more audio files. The ONNX Runtime images read WAV, FLAC, OGG and MP3 (libsndfile; no ffmpeg);
the vLLM image also accepts anything ffmpeg can decode (M4A, …).

- **URL**: `http://127.0.0.1:8907/transcribe?language=zh-CN`
- **Body**: multipart/form-data, one or more `files` fields
- **Query**: `language` (optional), `context` (optional, see below), `forced_alignment=true|false` (vLLM image with `ENABLE_ALIGNER_MODEL=true` only)
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
  -F "files=@files/reference.mp3" -F "files=@files/reference.wav"
```

`files/reference.m4a` only works with the vLLM image (it needs ffmpeg).

Expected: `[{"text":"Das ist ein Referenztext.","language":"German"}, ...]`

### Forced alignment

Only in the [vLLM image](#vllm-image-advanced), with `ENABLE_ALIGNER_MODEL=true` in `.env` (then `docker compose up -d`).

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

Reference numbers from the upstream **vLLM** backend (1x NVIDIA H200 NVL, `Qwen3-ASR-1.7B` + `Qwen3-ForcedAligner-0.6B`);
they don't describe the ONNX Runtime images (for the CPU image see [How streaming works on CPU](#how-streaming-works-on-cpu)):

- VRAM: < 2 GB at peak, even with 80 concurrent streams.
- Batch: 80 requests, avg QPS 28.5, latency P50 0.139 s.
- Qwen3-ASR runs at roughly 20x real-time with Flash Attention 2.

Upstream streaming numbers were measured before the streaming buffer fix
(audio was re-fed cumulatively), so re-run the streaming benchmark on your hardware.

## vLLM image (advanced)

The original backend: [qwen-asr](https://github.com/QwenLM/Qwen3-ASR) on vLLM, with Qwen3-ASR-1.7B in FP8.
It supports continuous batching for many concurrent streams and the forced aligner (timestamps), at the cost of a
~14 GB image that `compose.yaml` builds locally. Requirements are the same as for the
[GPU quick start](#quick-start-gpu-onnx-runtime), plus an SM80+ GPU (RTX 30 series or newer).

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
(`Qwen/Qwen3-ASR-1.7B-hf`) and GGUF / MLX / ONNX / OpenVINO builds do **not** work with the vLLM image (the ONNX
Runtime images use the ONNX models described in their own sections).

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

### 3. Build and start

```bash
docker compose up -d --build   # builds the image on first run
docker compose logs -f
```

Wait for `Server is ready to accept requests.` — `docker compose ps` shows the container as `healthy` once
models are loaded and warmed up.

**!!FIRST START NOTICE!!** The image is large (CUDA + vLLM + PyTorch), and the first start (CUDA graph compile,
loading the model into VRAM and warmup) can take a while. The vLLM compile cache is kept in the `vllm_cache` volume,
so subsequent starts are faster.

The server is available at `http://127.0.0.1:8907` (bound to localhost only; set `API_TOKEN` before exposing it).

### Common commands

```bash
docker compose ps                 # status / health
docker compose logs -f            # follow logs
docker compose restart            # restart (e.g. after changing .env: use `up -d` instead)
docker compose up -d              # apply .env changes
git pull && docker compose up -d --build      # update: rebuild from the latest code
docker compose build --no-cache               # full rebuild (also picks up new dependency versions)
docker compose down               # stop and remove the container
```

### Image build options

The image is based on `nvidia/cuda:12.8.0-runtime` (torch and vLLM bring their own CUDA kernels and libraries).
flash-attn is not installed by default: vLLM has its own attention kernels, and flash-attn only speeds up the forced
aligner. To include it, set `CUDA_FLAVOR=devel` (adds the CUDA toolkit, needed to compile it) and
`BUNDLE_FLASH_ATTENTION=true` in `.env`; compiling it takes a long time and a lot of RAM (`MAX_JOBS` limits the
parallel jobs).

### Development mode

Run `server.py` from the working tree with auto-reload (no rebuild needed after edits):

```bash
docker compose -f compose.yaml -f compose.dev.yaml up
```

### Configuration

Set in `.env` (see `.env.example`):

| Variable | Default | Description |
|---|---|---|
| `MODEL_DIR` | — (required) | Host directory with model folders, mounted read-only at `/models` |
| `ASR_MODEL_DIR` | `Qwen3-ASR-1.7B-fp8` | ASR model folder name inside `MODEL_DIR` |
| `VLLM_QUANTIZATION` | `modelopt` | vLLM quantization method passed to the model loader; empty for unquantized (BF16) checkpoints |
| `ENABLE_ALIGNER_MODEL` | `false` | Load the forced aligner (timestamps for `POST /transcribe`) |
| `ALIGNER_MODEL_DIR` | `Qwen3-ForcedAligner-0.6B` | Aligner folder name inside `MODEL_DIR` |
| `PORT` | `8907` | Host port |
| `BIND_ADDR` | `127.0.0.1` | Host interface to bind; `0.0.0.0` exposes the server to the network |
| `GPU_MEMORY_UTILIZATION` | `0.15` | Fraction of GPU memory vLLM may reserve |
| `STREAM_CHUNK_SIZE_SEC` | `1.0` | Audio seconds per streaming decode step. Smaller = faster partial updates, more GPU work |
| `STREAM_UNFIXED_CHUNK_NUM` | `2` | First N chunks are decoded without a text prefix |
| `STREAM_UNFIXED_TOKEN_NUM` | `5` | Trailing tokens rolled back (re-decodable) on each step |
| `PARTIAL_INTERVAL_MS` | `120` | Minimum interval between `partial` messages |
| `CUDA_FLAVOR` | `runtime` | Local build only: CUDA base image, `runtime` or `devel` (CUDA toolkit) |
| `BUNDLE_FLASH_ATTENTION` | `false` | Local build only: install flash-attn (speeds up the forced aligner; needs `CUDA_FLAVOR=devel`) |
| `MAX_JOBS` | `8` | Local build only: parallel jobs if flash-attn has to be compiled from source |

Further server settings (`MAX_CONCURRENT_INFER`, `MAX_CONCURRENT_DECODE`, `THREADPOOL_WORKERS`,
`MAX_NEW_TOKENS`, `OPENCC_TW_CONFIG`, `OPENCC_HK_CONFIG`) are read from the environment by `server.py`
and can be added to `compose.yaml`.

## Releasing

Pushing a version tag builds the image with GitHub Actions (`.github/workflows/docker-publish.yml`)
and pushes it to GHCR:

```bash
git tag v1.2.3
git push origin v1.2.3
```

| Tag pushed | Image tags |
|---|---|
| `v1.2.3` | ONNX CPU: `1.2.3-cpu`, `1.2-cpu`, `1-cpu`, `latest-cpu` · ONNX GPU: `1.2.3-gpu`, `1.2-gpu`, `1-gpu`, `latest-gpu` |
| `v1.2.3-rc.1` | `1.2.3-rc.1-cpu`, `1.2.3-rc.1-gpu` (don't move `latest-*`) |

The workflow can also be started manually (Actions → Publish Docker image → Run workflow); a manual run on a
branch publishes `latest-cpu` / `latest-gpu` only. The vLLM image is not published; build it with `compose.yaml`. The CPU image is built for linux/amd64 and linux/arm64.

GHCR packages are private when first published. To pull without `docker login ghcr.io`, open the package
(GitHub profile → Packages → fast-qwen-asr-inference-vllm → Package settings) and change its visibility to public.
