# Configuration

The server reads its settings from environment variables. With Docker Compose, set them in `.env`; the compose
files pass the common ones through (and set image-specific defaults that can differ from the server defaults
below). Others can be added to the `environment:` section of the compose file.

## Compose variables

Used by the compose files themselves (not read by the server):

| Variable | Used by | Description |
|---|---|---|
| `MODEL_DIR` | all | Host directory with the model folders, mounted read-only at `/models` |
| `ASR_MODEL_DIR` | all | Model folder inside `MODEL_DIR` (becomes `ASR_MODEL_NAME=/models/<dir>`) |
| `ALIGNER_MODEL_DIR` | all | Forced aligner folder inside `MODEL_DIR` (becomes `ALIGNER_MODEL_NAME`) |
| `DIARIZATION_MODEL_DIR` | `compose.cpu.yaml`, `compose.gpu.yaml` | Speaker diarization folder inside `MODEL_DIR` (becomes `DIARIZATION_MODEL_NAME`) |
| `PORT` | all | Host port (default `8907`) |
| `BIND_ADDR` | all | Host interface (`127.0.0.1` for GPU/vLLM, `0.0.0.0` for CPU) |
| `CONTAINER` | all | Container name |
| `IMAGE` / `IMAGE_TAG` | `compose.cpu.yaml`, `compose.gpu.yaml` | Image to pull |
| `DOMAIN` | `--profile tls` | Domain for Caddy (automatic HTTPS) |
| `CUDA_FLAVOR`, `BUNDLE_FLASH_ATTENTION`, `MAX_JOBS` | `compose.yaml` | vLLM image build options, see [deployment-vllm.md](deployment-vllm.md#image-build-options) |

## Server settings

Generated from the code (`asr_server/config.py` and the engines) by `scripts/gen-config-docs.py`; defaults are the
server's own, before any compose file overrides them.

<!-- BEGIN GENERATED: scripts/gen-config-docs.py -->

### Server

| Variable | Default | Description |
|---|---|---|
| `API_TOKEN` | — | Shared secret; clients send `Authorization: Bearer <token>` (or `?token=`). Empty = no auth |
| `ASR_BACKEND` | `vllm` | Inference backend: `vllm` or `onnx` (set by each image) |
| `CORS_ORIGINS` | `*` | Comma-separated origins allowed by CORS (`*` = any) |

### Limits

| Variable | Default | Description |
|---|---|---|
| `EXIT_ON_LOAD_FAILURE` | `true` | Stop the server when the model fails to load, so the container restart policy retries |
| `FFMPEG_TIMEOUT_SEC` | `60.0` | Time limit for decoding one upload with ffmpeg |
| `MAX_CONCURRENT_DECODE` | `4` | Audio files decoded in parallel |
| `MAX_CONCURRENT_INFER` | `1` | Inference calls run in parallel (GPU: usually 1); streams go before queued uploads |
| `MAX_FILES` | `16` | Most files accepted in one `POST /transcribe` request |
| `MAX_UPLOAD_MB` | `100.0` | Largest accepted request body for the upload endpoints (0 = unlimited) |
| `THREADPOOL_WORKERS` | `0` | Worker threads for blocking calls (0 = 5 × CPU cores) |

### Streaming

| Variable | Default | Description |
|---|---|---|
| `MAX_STREAMS` | `64` | Most concurrent WebSocket streams; more are refused (0 = unlimited) |
| `PARTIAL_INTERVAL_MS` | `120` | Minimum interval between `partial` messages |
| `STREAM_EXPECT_SR` | `16000` | Sample rate assumed when `start` has no `sample_rate_hz` |
| `STREAM_IDLE_TIMEOUT_SEC` | `30.0` | Close a stream that sends nothing for this long (0 = never) |
| `STREAM_MAX_SEC` | `300.0` | Audio beyond this many seconds per utterance is dropped (0 = unlimited) |
| `STREAM_MIN_SAMPLES` | `1600` | Samples (16 kHz) buffered before they are fed to the stream (1600 = 100 ms; vLLM: at least one chunk) |

### Languages

| Variable | Default | Description |
|---|---|---|
| `OPENCC_HK_CONFIG` | `s2hk` | OpenCC config for `zh-HK` / `zh-MO` |
| `OPENCC_TW_CONFIG` | `s2twp` | OpenCC config for `zh-TW` / `zh-Hant` |

### OpenAI API

| Variable | Default | Description |
|---|---|---|
| `OPENAI_MODEL_NAME` | `qwen3-asr` | Model id listed by `GET /v1/models` (the `model` form field is accepted but ignored) |

### ONNX backend

| Variable | Default | Description |
|---|---|---|
| `ALIGNER_MODEL_NAME` | `/models/qwen3-forcedaligner-0.6b-onnx` | Forced aligner directory (ONNX export of Qwen3-ForcedAligner-0.6B, see `scripts/download-models.sh aligner`) |
| `ASR_MODEL_NAME` | `/models/qwen3-asr-0.6b-onnx` | Model directory |
| `ENABLE_ALIGNER_MODEL` | `false` | Load the forced aligner (word timestamps; ~1.3 GB more memory) |
| `MAX_NEW_TOKENS` | `1024` | Most tokens generated per transcription (per segment, see `ONNX_SEGMENT_SEC`) |
| `ONNX_DECODER` | — | Decoder graph file in the model directory (empty = auto-detect: gqa, then merged, then split) |
| `ONNX_DEVICE_ID` | `0` | GPU index for `ONNX_PROVIDER=cuda` |
| `ONNX_ENCODER` | — | Encoder graph file (empty = `encoder.int4.onnx`, else `encoder.fp16.onnx` on the GPU / `encoder.onnx` on the CPU) |
| `ONNX_ENCODER_SEGMENT_WINDOWS` | `1` | Encoder segment length in 8 s windows (0 = whole utterance in one pass) |
| `ONNX_MEM_ARENA` | `false` | Use ONNX Runtime's CPU memory arena (faster, higher RSS) |
| `ONNX_PROVIDER` | `cpu` | Execution provider: `cpu`, or `cuda` (onnxruntime-gpu; falls back to CPU if CUDA can't be loaded) |
| `ONNX_SEGMENT_SEC` | `30.0` | Longer audio is transcribed in segments of at most this many seconds, cut at pauses (0 = in one piece) |
| `ONNX_THREADS` | `0` | ONNX Runtime intra-op threads (0 = all cores) |
| `STREAM_FINAL_REUSE_PARTIAL` | `true` | The final result also continues from the last partial (faster on CPU; an early partial's error can survive) |
| `STREAM_PARTIALS` | `true` | Send live partial results |
| `STREAM_PARTIAL_INTERVAL_SEC` | `2.0` | Seconds of new audio between partials |
| `STREAM_PARTIAL_MAX_SEC` | `20.0` | No partials once the utterance is longer than this |
| `STREAM_REUSE_PARTIAL` | `true` | Continue from the previous partial instead of decoding from scratch |
| `STREAM_UNFIXED_CHUNK_NUM` | `2` | First N partials start from scratch |
| `STREAM_UNFIXED_TOKEN_NUM` | `5` | Last K tokens of the previous partial are re-decoded |

### vLLM backend

| Variable | Default | Description |
|---|---|---|
| `ALIGNER_MODEL_NAME` | `Qwen/Qwen3-ForcedAligner-0.6B` | Forced aligner path (or Hugging Face id) |
| `ASR_MODEL_NAME` | `Qwen/Qwen3-ASR-1.7B` | ASR model path (or Hugging Face id) |
| `ENABLE_ALIGNER_MODEL` | `true` | Load the forced aligner (timestamps) |
| `ENABLE_ASR_MODEL` | `true` | Load the ASR model |
| `GPU_MEMORY_UTILIZATION` | `0.75` | Fraction of GPU memory vLLM may reserve |
| `MAX_NEW_TOKENS` | `4096` | Most tokens generated per transcription |
| `STREAM_CHUNK_SIZE_SEC` | `2.0` | Audio seconds per streaming decode step (smaller = faster partials, more GPU work) |
| `STREAM_UNFIXED_CHUNK_NUM` | `2` | First N chunks are decoded without a text prefix |
| `STREAM_UNFIXED_TOKEN_NUM` | `5` | Trailing tokens rolled back (re-decoded) on each step |
| `VLLM_QUANTIZATION` | — | vLLM quantization method, e.g. `modelopt` for ModelOpt FP8 checkpoints (empty = from the checkpoint config) |

### Speaker diarization

| Variable | Default | Description |
|---|---|---|
| `DIARIZATION_MIN_DURATION_OFF` | `0.5` | Pauses of one speaker shorter than this are bridged, in seconds |
| `DIARIZATION_MIN_DURATION_ON` | `0.3` | Shortest speech turn kept, in seconds |
| `DIARIZATION_MODEL_NAME` | `/models/speaker-diarization` | Directory with `segmentation.onnx` and `embedding.onnx` (see `scripts/download-models.sh diarization`) |
| `DIARIZATION_THREADS` | `0` | Threads per diarization model (0 = all cores) |
| `DIARIZATION_THRESHOLD` | `0.5` | Clustering threshold when the number of speakers is not given: higher = fewer speakers |
| `ENABLE_DIARIZATION` | `false` | Load the speaker diarization models (`diarize=true`) |

<!-- END GENERATED -->
