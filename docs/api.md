# API

All images serve the same API on port 8000 inside the container (8907 on the host with the provided compose files).

| Endpoint | Purpose |
|---|---|
| [`GET /health`](#get-health) | Loading status, limits, memory |
| [`GET /ready`](#get-ready) | Readiness probe (no token needed) |
| [`POST /transcribe`](#post-transcribe) | Transcribe one or more uploaded files |
| [`POST /v1/audio/transcriptions`](#post-v1audiotranscriptions-openai-compatible) | OpenAI-compatible transcription |
| [`GET /v1/models`](#get-v1models) | OpenAI-compatible model list |
| [`WS /transcribe-streaming`](#ws-transcribe-streaming) | Real-time transcription of streamed PCM |

## Authentication

Set `API_TOKEN` to require a shared secret on every endpoint except `GET /ready` (used by the container
healthcheck). Clients send `Authorization: Bearer <token>`, or `?token=<token>` where headers can't be set
(e.g. browser WebSockets). Missing or wrong tokens get HTTP 401, or a rejected WebSocket handshake (HTTP 403).
The CPU compose file requires it; for the GPU and vLLM compose files it's optional (`API_TOKEN` in `.env`).

`?token=` values are masked (`token=***`) in the server's access log.

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

## Audio formats and limits

Uploads are decoded with libsndfile (WAV, FLAC, OGG, MP3). The vLLM image also has ffmpeg, so it accepts anything
ffmpeg can decode (M4A, …); decoding is aborted after `FFMPEG_TIMEOUT_SEC`. Stereo is mixed down to mono.

| Limit | Default | Response when exceeded |
|---|---|---|
| `MAX_UPLOAD_MB` (request body of the upload endpoints) | 100 | HTTP 413 |
| `MAX_FILES` (files per `POST /transcribe`) | 16 | HTTP 413 |
| `MAX_STREAMS` (concurrent WebSocket streams) | 64 | `error`, close code 1013 |
| `STREAM_IDLE_TIMEOUT_SEC` (no message on a stream) | 30 | `error`, close code 1008 |
| `STREAM_MAX_SEC` (audio per stream) | 300 (`compose.cpu.yaml`: 60, `compose.gpu.yaml`: 120) | `info`, further audio ignored |

Errors return a short message (`{"detail": "..."}`, or OpenAI's error shape on `/v1`); internal details such as
exception text stay in the server log.

## `GET /health`

Model loading status (`starting` → `loading_models` → `warming_up` → `ready`, or `error`), backend, limits,
streaming settings and memory usage. Returns 200 regardless of the loading status; with `API_TOKEN` set it needs the
token (401 otherwise), so it also works as a token check.

If the model fails to load, the server logs the error and exits with code 1 (unless `EXIT_ON_LOAD_FAILURE=false`),
so the container restart policy retries.

## `GET /ready`

200 `{"status":"ready"}` once models are loaded and warmed up, 503 otherwise. Never needs a token; used by the
container healthcheck.

## `POST /transcribe`

Upload one or more audio files.

- **URL**: `http://127.0.0.1:8907/transcribe?language=zh-CN`
- **Body**: multipart/form-data, one or more `files` fields
- **Query**: `language` (optional), `context` (optional, see [Context](#context)), `forced_alignment=true|false`
  (vLLM image with `ENABLE_ALIGNER_MODEL=true` only)
- **Response**: `[{"text": "...", "language": "Chinese"}, ...]` (plus `timestamps` with forced alignment:
  `{"items": [{"text", "start_time", "end_time"}, ...]}`, times in seconds)

```bash
curl -X POST "http://127.0.0.1:8907/transcribe?language=de" -F "files=@files/reference.wav"
```

## `POST /v1/audio/transcriptions` (OpenAI-compatible)

Implements [OpenAI's transcription API](https://platform.openai.com/docs/api-reference/audio/createTranscription),
so OpenAI SDKs and tools built for Whisper work with this server by pointing their base URL at `http://HOST:8907/v1`
and using `API_TOKEN` as the API key (any value if the server has no token).

| Form field | Support |
|---|---|
| `file` | Required. Same formats as `POST /transcribe` |
| `model` | Accepted and ignored (the server runs one model) |
| `language` | Optional; any code from [Languages](#languages) |
| `prompt` | Used as the [context](#context) |
| `response_format` | `json` (default), `text`, `srt`, `vtt`, `verbose_json` |
| `timestamp_granularities[]` | `segment` (default) and/or `word` with `verbose_json`; `word` needs the forced aligner |
| `temperature` | Accepted and ignored (decoding is greedy) |
| `stream` | `true` is rejected; use [`WS /transcribe-streaming`](#ws-transcribe-streaming) |

Timestamps: with the forced aligner (vLLM image, `ENABLE_ALIGNER_MODEL=true`), `srt`, `vtt` and `verbose_json`
segments are built from word timings (a new segment after a 0.8 s pause or 10 s). Without it, the whole file is one
segment from 0 to its duration. `verbose_json` reports `language` in lowercase (`"german"`), like OpenAI.

```bash
curl http://127.0.0.1:8907/v1/audio/transcriptions -H "Authorization: Bearer $API_TOKEN" \
  -F file=@files/reference.wav -F model=qwen3-asr -F response_format=srt
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8907/v1", api_key="<API_TOKEN>")
with open("files/reference.wav", "rb") as f:
    print(client.audio.transcriptions.create(model="qwen3-asr", file=f, language="de").text)
```

Errors use OpenAI's shape: `{"error": {"message": "...", "type": "invalid_request_error", "param": "...", "code": null}}`.

## `GET /v1/models`

`{"object": "list", "data": [{"id": "qwen3-asr", ...}]}` — the id is `OPENAI_MODEL_NAME`.

## `WS /transcribe-streaming`

Stream raw PCM audio for real-time transcription. One connection = one utterance.

- **URL**: `ws://127.0.0.1:8907/transcribe-streaming?language=zh-CN` (`language` is a **query parameter**, optional)

Protocol:

1. Client connects. If the models are still loading, the server holds the connection until they are ready
   (clients should apply a timeout while waiting for `ready`).
2. Server → `{"type": "ready"}`
3. Client → `{"type": "start", "format": "pcm_s16le", "sample_rate_hz": 16000, "context": "..."}`
   - `format`: `pcm_s16le` (optional; the only format)
   - `sample_rate_hz`: any rate from 8000 to 192000 (default 16000). Audio at other rates than 16 kHz is
     resampled on the server, so e.g. a browser can send its native 48 kHz directly
   - `channels`: `1` (optional; only mono)
   - `context`: optional, see [Context](#context)

   Anything else is rejected with an `error` and close code 1003.
4. Server → `{"type": "info", "message": "language=Chinese"}` (only when `language` was given)
5. Client → binary frames: raw PCM, 16-bit little-endian, mono, at `sample_rate_hz`. Frames can have any size
   (e.g. 100 ms at 16 kHz = 3200 bytes), and a frame may end in the middle of a sample
6. Server → `{"type": "partial", "text": "...", "language": "Chinese"}` — the **full** transcript so far;
   earlier words may be revised, so replace (don't append) the displayed text
7. Client → `{"type": "stop"}`
8. Server → `{"type": "final", "text": "...", "language": "Chinese"}`, then closes the connection (code 1000)

Errors are sent as `{"type": "error", "message": "..."}` followed by a close
(1002: audio before `start`, `start` sent twice, or `stop` before `start`; 1003: unsupported format/language;
1008: idle timeout; 1011: server not ready / internal error; 1013: too many streams, try again later).
With `API_TOKEN` set, a missing or wrong token rejects the handshake (HTTP 403).

If the utterance exceeds `STREAM_MAX_SEC`, the server sends `{"type": "info", "message": "max_duration_reached=60s"}`
and ignores further audio; `stop` still returns the final result.

## Context

Free text placed in the prompt's system turn to bias recognition toward specific spellings, e.g.
`"Vocabulary: Kubernetes, QwenType, 张三"`. Keep it short; it is part of every decode. Sent as the `context` query
parameter of `POST /transcribe`, the `prompt` field of `POST /v1/audio/transcriptions`, or the `context` field of
the streaming `start` message.
