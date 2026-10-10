# CPU deployment

For servers without a GPU, e.g. a small VPS that clients reach over the internet. The same ONNX Runtime backend as
the [GPU image](deployment-gpu.md), with Qwen3-ASR-0.6B quantized to int4 (~840 MB of model files); the image has no
PyTorch or CUDA.

The model folder combines two Hugging Face repos: the audio encoder, embeddings and tokenizer from
[`rhasspy/qwen3-asr-0.6b-onnx-int4-merged`](https://huggingface.co/rhasspy/qwen3-asr-0.6b-onnx-int4-merged), and the
text decoder from [`sorryhyun/qwen3-asr-onnx-gqa`](https://huggingface.co/sorryhyun/qwen3-asr-onnx-gqa)
(built with GroupQueryAttention, so the cost per generated token stays almost flat as the utterance grows).

On the server, get this repository first (`git clone https://github.com/dreamyfishmt/fast-qwen-asr-inference-vllm`
and `cd` into it); it has the compose files and the download script.

## Steps

1. Download the model with [`scripts/download-models.sh`](../scripts/download-models.sh) (needs the `hf` CLI or
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

Build the CPU image locally instead of pulling it:
`docker compose -f compose.cpu.yaml -f compose.cpu.local.yaml up -d --build`.

The CPU image has no ffmpeg: uploads can be WAV/FLAC/OGG/MP3 (what libsndfile reads), and forced
alignment is not available.

## How streaming works on CPU

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

## Settings

Set in `.env` (see `.env.cpu.example`); all server settings are listed in [configuration.md](configuration.md).

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
