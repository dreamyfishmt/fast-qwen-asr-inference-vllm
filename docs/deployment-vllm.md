# vLLM image (advanced)

The original backend: [qwen-asr](https://github.com/QwenLM/Qwen3-ASR) on vLLM, with Qwen3-ASR-1.7B in FP8.
It supports continuous batching for many concurrent streams and the forced aligner (timestamps), at the cost of a
~14 GB image that `compose.yaml` builds locally. Requirements are the same as for the
[GPU deployment](deployment-gpu.md#requirements), plus an SM80+ GPU (RTX 30 series or newer).

## 1. Download the models

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

### Which model

The server uses the `qwen-asr` package, which needs checkpoints in the **original Qwen3-ASR layout**
(`config.json` with `thinker_config`, weights named `thinker.*`). The Transformers-native conversions
(`Qwen/Qwen3-ASR-1.7B-hf`) and GGUF / MLX / ONNX / OpenVINO builds do **not** work with the vLLM image (the ONNX
Runtime images use the ONNX models described in their own guides).

| Model | Size | Notes |
|---|---|---|
| [`vrfai/Qwen3-ASR-1.7B-fp8`](https://huggingface.co/vrfai/Qwen3-ASR-1.7B-fp8) (default) | ~2.5 GB weights | NVIDIA ModelOpt FP8, text decoder only (audio encoder and lm_head stay BF16). Reported WER 7.34% → 7.60% vs BF16 and ~25% higher single-request throughput (RTX 5090). Needs `VLLM_QUANTIZATION=modelopt` and an SM80+ GPU (RTX 30 series or newer); native FP8 compute on RTX 40/50 (SM89+), weight-only FP8 via Marlin on RTX 30 |
| [`Qwen/Qwen3-ASR-1.7B`](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) | ~3.9 GB weights | Original BF16. Set `ASR_MODEL_DIR=Qwen3-ASR-1.7B` and an empty `VLLM_QUANTIZATION=` |

## 2. Configure

```bash
cp .env.example .env
```

Set at least `MODEL_DIR` in `.env` (use forward slashes on Windows, e.g. `D:/models`).
Folder names inside it are set with `ASR_MODEL_DIR` / `ALIGNER_MODEL_DIR`.

> **Windows tip:** loading models from a Windows drive goes through the WSL 2 file share and is slow.
> For faster startup, keep the models inside the WSL filesystem (e.g. `\\wsl$\Ubuntu\home\<you>\models`,
> referenced in `.env` as the Linux path when running `docker compose` from WSL).

## 3. Build and start

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

## Common commands

```bash
docker compose ps                 # status / health
docker compose logs -f            # follow logs
docker compose restart            # restart (e.g. after changing .env: use `up -d` instead)
docker compose up -d              # apply .env changes
git pull && docker compose up -d --build      # update: rebuild from the latest code and uv.lock
docker compose down               # stop and remove the container
```

## Image build options

The image is based on `nvidia/cuda:12.8.0-runtime` (torch and vLLM bring their own CUDA kernels and libraries).
Python and all dependencies are installed by `uv sync --frozen --extra vllm` from `uv.lock`, so a rebuild of the
same commit installs the same versions; see [development.md](development.md#dependencies) for updating them.

flash-attn is not installed by default: vLLM has its own attention kernels, and flash-attn only speeds up the forced
aligner. To include it, set `CUDA_FLAVOR=devel` (adds the CUDA toolkit, needed to compile it) and
`BUNDLE_FLASH_ATTENTION=true` in `.env`; compiling it takes a long time and a lot of RAM (`MAX_JOBS` limits the
parallel jobs). It is compiled against the locked torch and is not itself part of `uv.lock`.

## Development mode

Run the server from the working tree with auto-reload (no rebuild needed after edits):

```bash
docker compose -f compose.yaml -f compose.dev.yaml up
```

## Settings

Set in `.env` (see `.env.example`); all server settings are listed in [configuration.md](configuration.md).

| Variable | Default | Description |
|---|---|---|
| `MODEL_DIR` | — (required) | Host directory with model folders, mounted read-only at `/models` |
| `ASR_MODEL_DIR` | `Qwen3-ASR-1.7B-fp8` | ASR model folder name inside `MODEL_DIR` |
| `VLLM_QUANTIZATION` | `modelopt` | vLLM quantization method passed to the model loader; empty for unquantized (BF16) checkpoints |
| `ENABLE_ALIGNER_MODEL` | `false` | Load the forced aligner (timestamps for `POST /transcribe` and word timestamps in the OpenAI API) |
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
