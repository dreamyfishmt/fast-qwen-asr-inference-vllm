import os
import json
import io
import hmac
import asyncio
import logging
import subprocess
from typing import Optional, List, Tuple
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor

import uvicorn
import numpy as np
import soundfile as sf
import psutil
from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect, Query, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from engines import create_engine

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------
# Config
# -----------------------------
ASR_BACKEND = os.getenv("ASR_BACKEND", "vllm")

MAX_CONCURRENT_DECODE = int(os.getenv("MAX_CONCURRENT_DECODE", "4"))
MAX_CONCURRENT_INFER = int(os.getenv("MAX_CONCURRENT_INFER", "1"))  # GPU: usually 1
THREADPOOL_WORKERS = int(os.getenv("THREADPOOL_WORKERS", str((os.cpu_count() or 4) * 5)))

# Streaming buffering/throttling
STREAM_MIN_SAMPLES = int(os.getenv("STREAM_MIN_SAMPLES", "1600"))  # 100ms @ 16kHz
PARTIAL_INTERVAL_MS = int(os.getenv("PARTIAL_INTERVAL_MS", "120"))  # throttle partials
STREAM_EXPECT_SR = int(os.getenv("STREAM_EXPECT_SR", "16000"))
# Audio beyond this many seconds per utterance is dropped (0 = unlimited)
STREAM_MAX_SEC = float(os.getenv("STREAM_MAX_SEC", "0"))

# Shared secret for clients: "Authorization: Bearer <token>" (or ?token=<token>). Empty = no auth.
API_TOKEN = os.getenv("API_TOKEN", "").strip()

# OpenCC config used when a Traditional Chinese variant (zh-TW, zh-HK, zh-Hant) is requested.
OPENCC_TW_CONFIG = os.getenv("OPENCC_TW_CONFIG", "s2twp")
OPENCC_HK_CONFIG = os.getenv("OPENCC_HK_CONFIG", "s2hk")

SUPPORTED_LANGUAGES = [
    "Chinese", "English", "Cantonese", "Arabic", "German", "French", "Spanish", "Portuguese",
    "Indonesian", "Italian", "Korean", "Russian", "Thai", "Vietnamese", "Japanese", "Turkish",
    "Hindi", "Malay", "Dutch", "Swedish", "Danish", "Finnish", "Polish", "Czech", "Filipino",
    "Persian", "Greek", "Romanian", "Hungarian", "Macedonian",
]

# -----------------------------
# App state
# -----------------------------
engine = create_engine(ASR_BACKEND)
model_status = "starting"
model_ready_event = asyncio.Event()

decode_sem = asyncio.Semaphore(MAX_CONCURRENT_DECODE)
infer_sem = asyncio.Semaphore(MAX_CONCURRENT_INFER)

# -----------------------------
# Helpers
# -----------------------------
async def to_thread_limited(sem: asyncio.Semaphore, fn, *args, **kwargs):
    async with sem:
        return await asyncio.to_thread(fn, *args, **kwargs)

def token_ok(headers, query_params) -> bool:
    if not API_TOKEN:
        return True
    supplied = query_params.get("token") or ""
    auth = headers.get("authorization") or ""
    if auth.lower().startswith("bearer "):
        supplied = auth[7:].strip()
    return hmac.compare_digest(supplied.encode(), API_TOKEN.encode())

LANGUAGE_MAP = {
    "en": "English", "de": "German", "fr": "French", "es": "Spanish",
    "it": "Italian", "ja": "Japanese", "ko": "Korean", "zh": "Chinese",
    "ru": "Russian", "pt": "Portuguese", "nl": "Dutch", "tr": "Turkish",
    "sv": "Swedish", "id": "Indonesian", "vi": "Vietnamese",
    "hi": "Hindi", "ar": "Arabic", "yue": "Cantonese", "th": "Thai",
    "ms": "Malay", "da": "Danish", "fi": "Finnish", "pl": "Polish",
    "cs": "Czech", "fil": "Filipino", "tl": "Filipino", "fa": "Persian",
    "el": "Greek", "ro": "Romanian", "hu": "Hungarian", "mk": "Macedonian",
}

# Chinese script variants -> OpenCC config (None = keep the model's Simplified output)
ZH_SCRIPT_MAP = {
    "zh-cn": None, "zh-sg": None, "zh-hans": None,
    "zh-tw": OPENCC_TW_CONFIG, "zh-hant": OPENCC_TW_CONFIG,
    "zh-hk": OPENCC_HK_CONFIG, "zh-mo": OPENCC_HK_CONFIG,
}

def map_language(lang_code: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """
    Map a language code (ISO 639-1 or BCP-47, e.g. "en", "zh-CN", "zh-TW")
    to (Qwen full language name, OpenCC config or None).
    Qwen3-ASR has a single "Chinese" language, so Traditional output is
    produced by converting the Simplified transcript with OpenCC.
    Full language names (e.g. "Chinese") are accepted as well.
    Raises ValueError for languages the model does not support.
    """
    if not lang_code or not lang_code.strip():
        return None, None
    code = lang_code.strip().lower().replace("_", "-")
    if code in ZH_SCRIPT_MAP:
        return "Chinese", ZH_SCRIPT_MAP[code]
    base = code.split("-", 1)[0]
    if base in LANGUAGE_MAP:
        return LANGUAGE_MAP[base], None
    name = lang_code.strip().capitalize()
    if name in SUPPORTED_LANGUAGES:
        return name, None
    raise ValueError(f"Unsupported language: {lang_code}")

_opencc_converters = {}

def get_converter(config: Optional[str]):
    """Return a cached text converter for the OpenCC config (identity if None)."""
    if config is None:
        return lambda text: text
    if config not in _opencc_converters:
        import opencc
        _opencc_converters[config] = opencc.OpenCC(config).convert
    return _opencc_converters[config]

def read_audio_file(file_bytes: bytes) -> Tuple[np.ndarray, int]:
    """
    Sync decode. Must be called via asyncio.to_thread (or threadpool).
    soundfile first; fallback to ffmpeg for mp3/m4a/etc.
    """
    try:
        with io.BytesIO(file_bytes) as f:
            wav, sr = sf.read(f, dtype="float32", always_2d=False)
            return wav, sr
    except Exception:
        process = subprocess.Popen(
            ["ffmpeg", "-i", "pipe:0", "-f", "wav", "-"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        out, err = process.communicate(input=file_bytes)
        if process.returncode != 0:
            raise ValueError(f"FFmpeg decoding failed: {err.decode(errors='ignore')}")
        with io.BytesIO(out) as f:
            wav, sr = sf.read(f, dtype="float32", always_2d=False)
            return wav, sr

# -----------------------------
# Model loading
# -----------------------------
async def load_models_background():
    global model_status
    logger.info(f"Background task: Loading models (backend: {engine.name})...")
    model_status = "loading_models"
    try:
        await asyncio.to_thread(engine.load)
    except Exception as e:
        logger.exception(f"Failed to load models: {e}")
        model_status = "error"
        model_ready_event.set()  # don't hang endpoints
        return

    # Warmup (best-effort)
    logger.info("Warming up ASR model (best-effort)...")
    model_status = "warming_up"
    try:
        async with infer_sem:
            await asyncio.to_thread(engine.warmup)
        logger.info("Warmup complete.")
    except Exception as e:
        logger.warning(f"Warmup failed (non-critical): {e}")

    model_status = "ready"
    model_ready_event.set()
    logger.info("Server is ready to accept requests.")

# -----------------------------
# Lifespan
# -----------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up Qwen3-ASR Server...")
    if not API_TOKEN:
        logger.warning("API_TOKEN is not set: the server accepts unauthenticated requests.")

    # Bigger threadpool helps when decoding + websocket buffering + other to_thread calls happen together.
    executor = ThreadPoolExecutor(max_workers=THREADPOOL_WORKERS)
    app.state.executor = executor
    asyncio.get_running_loop().set_default_executor(executor)

    task = asyncio.create_task(load_models_background())
    try:
        yield
    finally:
        # Shutdown
        task.cancel()
        executor.shutdown(wait=False, cancel_futures=True)
        logger.info("Shutdown complete.")

# -----------------------------
# App
# -----------------------------
app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def require_token(request: Request, call_next):
    # /ready stays open for container healthchecks; it only reports the loading status.
    if request.url.path != "/ready" and request.method != "OPTIONS" and not token_ok(request.headers, request.query_params):
        return JSONResponse(status_code=401, content={"detail": "Invalid or missing API token"})
    return await call_next(request)

# -----------------------------
# Endpoints
# -----------------------------
@app.get("/health")
async def health():
    mem = psutil.virtual_memory()
    proc = psutil.Process()
    info = {
        "status": model_status,
        "backend": engine.name,
        "limits": {
            "max_concurrent_decode": MAX_CONCURRENT_DECODE,
            "max_concurrent_infer": MAX_CONCURRENT_INFER,
            "threadpool_workers": THREADPOOL_WORKERS,
            "stream_max_sec": STREAM_MAX_SEC,
        },
        "streaming": {**engine.streaming_config(), "partial_interval_ms": PARTIAL_INTERVAL_MS},
        "memory": {
            "ram_total_mb": mem.total // (1024 * 1024),
            "ram_available_mb": mem.available // (1024 * 1024),
            "ram_percent": mem.percent,
            "process_rss_mb": proc.memory_info().rss // (1024 * 1024),
        },
    }
    info["memory"].update(engine.memory_info())
    return info

@app.get("/ready")
async def ready():
    """200 once models are loaded and warmed up, 503 otherwise (for container healthchecks)."""
    if model_status != "ready":
        return JSONResponse(status_code=503, content={"status": model_status})
    return {"status": model_status}

@app.post("/transcribe")
async def transcribe(
    files: List[UploadFile] = File(...),
    language: Optional[str] = Query(None, description="Language code (e.g. en, de, zh-CN, zh-TW). None for auto-detect."),
    context: str = Query("", description="Context / vocabulary hints, e.g. 'Vocabulary: Kubernetes, QwenType.'"),
    forced_alignment: bool = Query(False, description="Enable forced alignment (timestamps)"),
):
    await model_ready_event.wait()

    if model_status != "ready":
        raise HTTPException(status_code=503, detail=f"Server not ready: {model_status}")
    if forced_alignment and not engine.supports_alignment:
        raise HTTPException(status_code=503, detail="Aligner model is not enabled or not supported by this backend.")

    try:
        full_lang, opencc_config = map_language(language)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    convert = get_converter(opencc_config)

    async def decode_one(f: UploadFile):
        content = await f.read()
        return await to_thread_limited(decode_sem, read_audio_file, content)

    # Decode concurrently (limited)
    try:
        audio_batch = await asyncio.gather(*(decode_one(f) for f in files))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid audio file: {e}")

    # Inference (explicitly limited, because GPU/CPU concurrency is not free)
    try:
        async with infer_sem:
            results = await asyncio.to_thread(engine.transcribe, audio_batch, full_lang, context)

        if forced_alignment:
            texts = [text for text, _ in results]  # align against the model's original script
            async with infer_sem:
                alignment_results = await asyncio.to_thread(engine.align, audio_batch, texts, full_lang)
            return [
                {"text": convert(text), "language": lang, "timestamps": alignment_results[i]}
                for i, (text, lang) in enumerate(results)
            ]
        return [{"text": convert(text), "language": lang} for text, lang in results]

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Inference failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/transcribe-streaming")
async def websocket_endpoint(
    ws: WebSocket,
    language: Optional[str] = Query(None),
    forced_alignment: bool = Query(False),  # kept for API symmetry; not used in streaming
):
    if not token_ok(ws.headers, ws.query_params):
        # closing before accept rejects the handshake (HTTP 403)
        await ws.close(code=1008, reason="Invalid or missing API token")
        return

    await ws.accept()

    # do wait until we know the outcome
    await model_ready_event.wait()

    if model_status != "ready":
        await ws.close(code=1011, reason=f"Server not ready: {model_status}")
        return

    try:
        full_lang, opencc_config = map_language(language)
    except ValueError as e:
        await ws.send_json({"type": "error", "message": str(e)})
        await ws.close(code=1003)
        return
    convert = get_converter(opencc_config)

    # Send ready
    try:
        await ws.send_json({"type": "ready"})
    except Exception:
        return

    stream = None
    buf_parts: List[np.ndarray] = []
    buf_n = 0
    total_samples = 0
    max_samples = int(STREAM_MAX_SEC * STREAM_EXPECT_SR) if STREAM_MAX_SEC > 0 else 0
    last_partial_ts = 0.0
    last_partial_text = ""

    async def flush_and_infer(send_partial: bool):
        nonlocal buf_parts, buf_n, last_partial_ts, last_partial_text
        if buf_n <= 0:
            return
        chunk = np.concatenate(buf_parts, axis=0) if len(buf_parts) > 1 else buf_parts[0]
        # streams buffer/accumulate internally: feed only new samples
        buf_parts = []
        buf_n = 0

        async with infer_sem:
            updated = await asyncio.to_thread(stream.feed, chunk, send_partial)

        if send_partial and updated:
            now = asyncio.get_running_loop().time()
            text = convert(stream.text)
            if text != last_partial_text and (now - last_partial_ts) * 1000.0 >= PARTIAL_INTERVAL_MS:
                last_partial_ts = now
                last_partial_text = text
                await ws.send_json({"type": "partial", "text": text, "language": stream.language})

    try:
        while True:
            msg = await ws.receive()

            if msg["type"] == "websocket.disconnect":
                break

            if msg["type"] != "websocket.receive":
                continue

            # Control messages
            if msg.get("text"):
                try:
                    data = json.loads(msg["text"])
                except json.JSONDecodeError:
                    data = None

                if isinstance(data, dict):
                    t = data.get("type")

                    if t == "start":
                        client_sr = int(data.get("sample_rate_hz", 0)) if data.get("sample_rate_hz") else None
                        fmt = data.get("format")

                        if client_sr != STREAM_EXPECT_SR or fmt not in (None, "pcm_s16le"):
                            await ws.send_json(
                                {"type": "error", "message": f"Only pcm_s16le @ {STREAM_EXPECT_SR}Hz supported"}
                            )
                            await ws.close(code=1003)
                            return

                        context = data.get("context") or ""
                        if not isinstance(context, str):
                            context = ""
                        try:
                            async with infer_sem:
                                stream = await asyncio.to_thread(engine.new_stream, full_lang, context)
                        except Exception as e:
                            logger.exception(f"Failed to init stream: {e}")
                            await ws.close(code=1011, reason="stream init failed")
                            return

                        # Optional: acknowledge language selection
                        if full_lang is not None:
                            await ws.send_json({"type": "info", "message": f"language={full_lang}"})
                        continue

                    if t == "stop":
                        if stream is None:
                            await ws.send_json({"type": "error", "message": "stop before start"})
                            await ws.close(code=1002)
                            return
                        # Flush remainder, finish, send final
                        await flush_and_infer(send_partial=False)
                        async with infer_sem:
                            await asyncio.to_thread(stream.finish)

                        await ws.send_json({"type": "final", "text": convert(stream.text), "language": stream.language})
                        await ws.close(code=1000)
                        return

            # Audio frames
            if msg.get("bytes"):
                if stream is None:
                    # Require explicit start so we can validate format.
                    await ws.send_json({"type": "error", "message": "Send {type:'start', format:'pcm_s16le', sample_rate_hz:16000} first"})
                    await ws.close(code=1002)
                    return

                chunk_bytes = msg["bytes"]
                # int16 mono little-endian -> float32 [-1, 1]
                audio_int16 = np.frombuffer(chunk_bytes, dtype=np.int16)
                if audio_int16.size == 0:
                    continue

                if max_samples:
                    room = max_samples - total_samples
                    if room <= 0:
                        continue
                    if audio_int16.size >= room:
                        audio_int16 = audio_int16[:room]
                        await ws.send_json({"type": "info", "message": f"max_duration_reached={STREAM_MAX_SEC:g}s"})
                total_samples += audio_int16.size

                audio_f32 = audio_int16.astype(np.float32) / 32768.0
                buf_parts.append(audio_f32)
                buf_n += audio_f32.size

                if buf_n >= STREAM_MIN_SAMPLES:
                    await flush_and_infer(send_partial=True)

    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.exception(f"WS Error: {e}")
        try:
            await ws.close(code=1011, reason="internal error")
        except Exception:
            pass

if __name__ == "__main__":
    # NOTE: for GPU models, keep workers=1 unless you deliberately replicate the model per worker.
    uvicorn.run(app, host="0.0.0.0", port=8000)
