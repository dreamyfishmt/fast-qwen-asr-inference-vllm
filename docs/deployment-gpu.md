# GPU deployment (ONNX Runtime)

Qwen3-ASR-1.7B (int4 decoder, FP16 encoder) on an NVIDIA GPU with ONNX Runtime's CUDA execution provider. The image
(`…:latest-gpu`, ~3 GB installed) contains no PyTorch or vLLM.

## Requirements

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

## Steps

1. Download the model (~2.25 GB) from [`dreamyfishmt/qwen3-asr-1.7b-onnx`](https://huggingface.co/dreamyfishmt/qwen3-asr-1.7b-onnx):

   ```bash
   uvx --from huggingface_hub hf download dreamyfishmt/qwen3-asr-1.7b-onnx --local-dir /srv/models/qwen3-asr-1.7b-onnx
   # or: scripts/download-models.sh 1.7b /srv/models   # -> /srv/models/qwen3-asr-1.7b-onnx
   ```

   On Windows (PowerShell; `uvx` from [uv](https://docs.astral.sh/uv/)):

   ```powershell
   uvx --from huggingface_hub hf download dreamyfishmt/qwen3-asr-1.7b-onnx --local-dir D:/models/qwen3-asr-1.7b-onnx
   ```

   The repo repackages, unmodified, the embeddings and tokenizer of
   [`andrewleech/qwen3-asr-1.7b-onnx`](https://huggingface.co/andrewleech/qwen3-asr-1.7b-onnx) and the int4
   GroupQueryAttention decoder with fp16 I/O from [`sorryhyun/qwen3-asr-onnx-gqa`](https://huggingface.co/sorryhyun/qwen3-asr-onnx-gqa).
   The encoder `encoder.fp16.onnx` is an FP16 conversion of andrewleech's FP32 `encoder.onnx`; see the
   [model card](https://huggingface.co/dreamyfishmt/qwen3-asr-1.7b-onnx) for source revisions and validation.

   If the download fails with a 401 from `cas-server.xethub.hf.co` (some proxies block Hugging Face's Xet storage),
   set `HF_HUB_DISABLE_XET=1` and run it again (the script does this automatically).

   The 0.6B model folder from the [CPU deployment](deployment-cpu.md) works on the GPU too (set `ASR_MODEL_DIR`).

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
   `--profile tls` (Caddy with automatic HTTPS, as in the [CPU deployment](deployment-cpu.md)).

On the GPU a full transcription takes a fraction of a second, so the defaults differ from the CPU image:
partials every second for the whole utterance, and the final result is always a fresh full decode
(`STREAM_FINAL_REUSE_PARTIAL=false`). The KV cache stays on the GPU between decode steps.

Build locally instead of pulling: `docker compose -f compose.gpu.yaml -f compose.gpu.local.yaml up -d --build`.

## Settings

Set in `.env` (see `.env.gpu.example`); all server settings are listed in [configuration.md](configuration.md).

| Variable | Default | Description |
|---|---|---|
| `ASR_MODEL_DIR` | `qwen3-asr-1.7b-onnx` | Model folder inside `MODEL_DIR` |
| `ONNX_PROVIDER` | `cuda` (image default) | `cuda` or `cpu` |
| `ONNX_DEVICE_ID` | `0` | GPU index |
| `ONNX_DECODER` / `ONNX_ENCODER` | auto | Decoder / encoder file in the model folder; auto prefers `decoder-*fp16*` on the GPU and `decoder-*fp32*` on the CPU, and `encoder.int4.onnx`, then `encoder.fp16.onnx` on the GPU and `encoder.onnx` on the CPU |
| `STREAM_PARTIAL_INTERVAL_SEC` / `STREAM_PARTIAL_MAX_SEC` | `1.0` / `60` | Partial cadence and cutoff |
| `STREAM_FINAL_REUSE_PARTIAL` | `false` | Continue the final result from the last partial (see [How streaming works on CPU](deployment-cpu.md#how-streaming-works-on-cpu)) |
| `API_TOKEN` | — | Optional shared secret; see [Authentication](api.md#authentication) |
| `STREAM_MAX_SEC` | `120` | Audio beyond this per utterance is dropped |
