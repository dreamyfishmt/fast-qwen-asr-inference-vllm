# Fast Qwen3-ASR Inference Server

A containerized [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR) speech recognition server (FastAPI) with HTTP
`/transcribe`, an OpenAI-compatible `/v1/audio/transcriptions` and real-time WebSocket `/transcribe-streaming`
endpoints. Models are read from a **local model directory** mounted into the container (read-only, offline).

| Image | Backend | Model | Hardware | Image size |
|---|---|---|---|---|
| **`…:latest-gpu` (recommended)** | ONNX Runtime + CUDA 13 | Qwen3-ASR-1.7B (int4 decoder, FP16 encoder) | NVIDIA GPU, driver R580+. See [GPU deployment](docs/deployment-gpu.md) | ~3 GB |
| `…:latest-cpu` | ONNX Runtime | Qwen3-ASR-0.6B (int4) | Any x86-64 / ARM64 CPU, 2+ GB RAM. See [CPU deployment](docs/deployment-cpu.md) | ~0.5 GB |
| built locally (`compose.yaml`) | vLLM + `qwen-asr` | Qwen3-ASR-1.7B (FP8) | NVIDIA GPU (RTX 30 series or newer). See [vLLM image](docs/deployment-vllm.md) | ~14 GB |

Images: `ghcr.io/dreamyfishmt/fast-qwen-asr-inference-vllm`. All three serve the same API.

The ONNX Runtime images need no PyTorch or vLLM and are the right fit for one user or a few (e.g. voice input).
The vLLM image is mostly PyTorch, vLLM and the full CUDA library set built for every GPU generation; that buys
continuous batching for many concurrent users. It is not published (too large to build in CI), so `compose.yaml`
builds it locally.

## Documentation

| Page | Contents |
|---|---|
| [GPU deployment](docs/deployment-gpu.md) | ONNX Runtime on an NVIDIA GPU (recommended) |
| [CPU deployment](docs/deployment-cpu.md) | ONNX Runtime on a CPU-only server, HTTPS with Caddy, CPU streaming performance |
| [vLLM image](docs/deployment-vllm.md) | qwen-asr + vLLM, FP8 model, forced aligner |
| [API](docs/api.md) | Endpoints, OpenAI compatibility, streaming protocol, authentication, languages, limits |
| [Configuration](docs/configuration.md) | Every environment variable |
| [Development](docs/development.md) | Code layout, dependencies (uv), tests, trying the server, benchmarks, releasing |

## Quick start (GPU)

Requires an NVIDIA driver R580+, Docker with GPU support, and [uv](https://docs.astral.sh/uv/) for the download.

```bash
git clone https://github.com/dreamyfishmt/fast-qwen-asr-inference-vllm
cd fast-qwen-asr-inference-vllm
scripts/download-models.sh 1.7b /srv/models     # ~2.25 GB -> /srv/models/qwen3-asr-1.7b-onnx
cp .env.gpu.example .env                        # set MODEL_DIR=/srv/models
docker compose -f compose.gpu.yaml up -d
curl -X POST "http://127.0.0.1:8907/transcribe?language=de" -F "files=@files/reference.wav"
# [{"text":"Das ist ein Referenztext.","language":"German"}]
```

Details, Windows notes and settings: [docs/deployment-gpu.md](docs/deployment-gpu.md). No GPU? Use the
[CPU deployment](docs/deployment-cpu.md).

## Client: QwenType

[**QwenType**](https://github.com/dreamyfishmt/QwenType) is the Windows voice-typing client for this server:
hold Right Ctrl, speak, release, and the text is typed into the focused app, with a live transcript while you
speak. Download `QwenType.exe` from its [Releases](https://github.com/dreamyfishmt/QwenType/releases), start the
server, then set the connection in the tray menu → **ASR Server…**:

- **WebSocket URL**: default `ws://127.0.0.1:8907/transcribe-streaming` (server on the same PC); for a remote
  server behind HTTPS, `wss://DOMAIN/transcribe-streaming`
- **API Token**: the server's `API_TOKEN` (required by the CPU image; leave empty if the server has none)
- **Hotwords** (optional): names and terms sent as the stream's [`context`](docs/api.md#context)

Other clients: anything built for OpenAI's transcription API works with `base_url=http://HOST:8907/v1`
(see [API](docs/api.md#post-v1audiotranscriptions-openai-compatible)).
