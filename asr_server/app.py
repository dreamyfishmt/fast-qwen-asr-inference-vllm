"""FastAPI app: /health, /ready, /transcribe and /transcribe-streaming (plus the OpenAI-compatible routes)."""

import asyncio
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import List, Optional

import numpy as np
import psutil
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from . import config
from .audio import MODEL_RATE, PcmS16Decoder, StreamResampler
from .auth import install_log_redaction, token_ok
from .engines import Engine, create_engine
from .language import get_converter, map_language
from .openai_api import router as openai_router
from .service import Service, ServiceError

logger = logging.getLogger(__name__)

# Paths that accept uploads (request body size limit)
UPLOAD_PATHS = ("/transcribe", "/v1/audio/transcriptions")
STREAM_MIN_RATE, STREAM_MAX_RATE = 8000, 192000


class BodySizeLimit:
    """Rejects upload requests whose body exceeds `max_bytes` with 413, before it is buffered in full."""

    def __init__(self, app, max_bytes: int, paths=UPLOAD_PATHS):
        self.app = app
        self.max_bytes = max_bytes
        self.paths = paths

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or not self.max_bytes or scope["path"] not in self.paths:
            return await self.app(scope, receive, send)

        too_large = HTTPException(413, f"Request body exceeds MAX_UPLOAD_MB={config.MAX_UPLOAD_MB:g}")
        for key, value in scope.get("headers", []):
            if key == b"content-length" and value.isdigit() and int(value) > self.max_bytes:
                return await _send_json(send, 413, too_large.detail)

        received = 0

        async def limited_receive():
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    raise too_large
            return message

        await self.app(scope, limited_receive, send)


async def _send_json(send, status: int, detail: str) -> None:
    body = json.dumps({"detail": detail}).encode()
    await send({"type": "http.response.start", "status": status,
                "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())]})
    await send({"type": "http.response.body", "body": body})


def create_app(engine: Optional[Engine] = None) -> FastAPI:
    service = Service(engine or create_engine(config.ASR_BACKEND))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger.info("Starting up Qwen3-ASR Server...")
        install_log_redaction()
        if not config.API_TOKEN:
            logger.warning("API_TOKEN is not set: the server accepts unauthenticated requests.")

        # Bigger threadpool helps when decoding + websocket buffering + other to_thread calls happen together.
        executor = ThreadPoolExecutor(max_workers=config.THREADPOOL_WORKERS)
        asyncio.get_running_loop().set_default_executor(executor)

        task = asyncio.create_task(service.load())
        try:
            yield
        finally:
            task.cancel()
            executor.shutdown(wait=False, cancel_futures=True)
            logger.info("Shutdown complete.")
            if service.exit_code:
                # uvicorn exits with 0 after SIGTERM; report the failed load to the container runtime
                logging.shutdown()
                os._exit(service.exit_code)

    app = FastAPI(lifespan=lifespan)
    app.state.service = service

    origins = [o.strip() for o in config.CORS_ORIGINS.split(",") if o.strip()] or ["*"]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        # Auth is a bearer token, not a cookie; credentials are only allowed for explicitly listed origins.
        allow_credentials="*" not in origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(BodySizeLimit, max_bytes=int(config.MAX_UPLOAD_MB * 1024 * 1024))

    @app.middleware("http")
    async def require_token(request: Request, call_next):
        # /ready stays open for container healthchecks; it only reports the loading status.
        if request.url.path != "/ready" and request.method != "OPTIONS" and not token_ok(request.headers, request.query_params):
            return JSONResponse(status_code=401, content={"detail": "Invalid or missing API token"})
        return await call_next(request)

    @app.exception_handler(ServiceError)
    async def service_error(request: Request, exc: ServiceError):
        return JSONResponse(status_code=exc.status, content={"detail": exc.message})

    @app.get("/health")
    async def health():
        mem = psutil.virtual_memory()
        proc = psutil.Process()
        info = {
            "status": service.status,
            "backend": service.engine.name,
            "limits": {
                "max_concurrent_decode": config.MAX_CONCURRENT_DECODE,
                "max_concurrent_infer": config.MAX_CONCURRENT_INFER,
                "threadpool_workers": config.THREADPOOL_WORKERS,
                "max_upload_mb": config.MAX_UPLOAD_MB,
                "max_files": config.MAX_FILES,
                "max_streams": config.MAX_STREAMS,
                "active_streams": service.active_streams,
                "stream_max_sec": config.STREAM_MAX_SEC,
                "stream_idle_timeout_sec": config.STREAM_IDLE_TIMEOUT_SEC,
            },
            "streaming": {**service.engine.streaming_config(), "partial_interval_ms": config.PARTIAL_INTERVAL_MS},
            "memory": {
                "ram_total_mb": mem.total // (1024 * 1024),
                "ram_available_mb": mem.available // (1024 * 1024),
                "ram_percent": mem.percent,
                "process_rss_mb": proc.memory_info().rss // (1024 * 1024),
            },
        }
        info["memory"].update(service.engine.memory_info())
        return info

    @app.get("/ready")
    async def ready():
        """200 once models are loaded and warmed up, 503 otherwise (for container healthchecks)."""
        if service.status != "ready":
            return JSONResponse(status_code=503, content={"status": service.status})
        return {"status": service.status}

    @app.post("/transcribe")
    async def transcribe(
        files: List[UploadFile] = File(...),
        language: Optional[str] = Query(None, description="Language code (e.g. en, de, zh-CN, zh-TW). None for auto-detect."),
        context: str = Query("", description="Context / vocabulary hints, e.g. 'Vocabulary: Kubernetes, QwenType.'"),
        forced_alignment: bool = Query(False, description="Enable forced alignment (timestamps)"),
    ):
        if config.MAX_FILES and len(files) > config.MAX_FILES:
            raise ServiceError(413, f"Too many files: {len(files)} (MAX_FILES={config.MAX_FILES})")
        try:
            full_lang, opencc_config = map_language(language)
        except ValueError as e:
            raise ServiceError(400, str(e))
        await service.wait_ready()

        contents = [await f.read() for f in files]
        audios = await service.decode(contents, [f.filename for f in files])
        return await service.transcribe(audios, full_lang, opencc_config, context, align=forced_alignment)

    @app.websocket("/transcribe-streaming")
    async def transcribe_streaming(
        ws: WebSocket,
        language: Optional[str] = Query(None),
        forced_alignment: bool = Query(False),  # kept for API symmetry; not used in streaming
    ):
        if not token_ok(ws.headers, ws.query_params):
            # closing before accept rejects the handshake (HTTP 403)
            await ws.close(code=1008, reason="Invalid or missing API token")
            return

        await ws.accept()

        if config.MAX_STREAMS and service.active_streams >= config.MAX_STREAMS:
            await _ws_error(ws, f"Too many concurrent streams (MAX_STREAMS={config.MAX_STREAMS}); try again later", 1013)
            return

        service.active_streams += 1
        try:
            await _run_stream(ws, service, language)
        finally:
            service.active_streams -= 1

    app.include_router(openai_router)
    return app


async def _ws_error(ws: WebSocket, message: str, code: int) -> None:
    try:
        await ws.send_json({"type": "error", "message": message})
        await ws.close(code=code)
    except Exception:
        pass


async def _run_stream(ws: WebSocket, service: Service, language: Optional[str]) -> None:
    # do wait until we know the outcome
    await service.ready_event.wait()
    if service.status != "ready":
        await ws.close(code=1011, reason=f"Server not ready: {service.status}")
        return

    try:
        full_lang, opencc_config = map_language(language)
    except ValueError as e:
        await _ws_error(ws, str(e), 1003)
        return
    convert = get_converter(opencc_config)

    try:
        await ws.send_json({"type": "ready"})
    except Exception:
        return

    stream = None
    pcm = PcmS16Decoder()
    resampler: Optional[StreamResampler] = None
    buf_parts: List[np.ndarray] = []
    buf_n = 0
    total_samples = 0  # at the client's sample rate
    max_samples = 0
    last_partial_ts = 0.0
    last_partial_text = ""
    idle_timeout = config.STREAM_IDLE_TIMEOUT_SEC if config.STREAM_IDLE_TIMEOUT_SEC > 0 else None

    def buffer(audio: np.ndarray) -> None:
        nonlocal buf_n
        if audio.size:
            buf_parts.append(audio)
            buf_n += audio.size

    async def flush_and_infer(send_partial: bool):
        nonlocal buf_parts, buf_n, last_partial_ts, last_partial_text
        if buf_n <= 0:
            return
        chunk = np.concatenate(buf_parts, axis=0) if len(buf_parts) > 1 else buf_parts[0]
        # streams buffer/accumulate internally: feed only new samples
        buf_parts = []
        buf_n = 0

        updated = await service.run_infer(stream.feed, chunk, send_partial)

        if send_partial and updated:
            now = asyncio.get_running_loop().time()
            text = convert(stream.text)
            if text != last_partial_text and (now - last_partial_ts) * 1000.0 >= config.PARTIAL_INTERVAL_MS:
                last_partial_ts = now
                last_partial_text = text
                await ws.send_json({"type": "partial", "text": text, "language": stream.language})

    try:
        while True:
            try:
                msg = await asyncio.wait_for(ws.receive(), timeout=idle_timeout)
            except asyncio.TimeoutError:
                await _ws_error(ws, f"No data for {config.STREAM_IDLE_TIMEOUT_SEC:g}s (STREAM_IDLE_TIMEOUT_SEC)", 1008)
                return

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
                if not isinstance(data, dict):
                    continue
                t = data.get("type")

                if t == "start":
                    if stream is not None:
                        await _ws_error(ws, "start was already sent; one connection = one utterance", 1002)
                        return
                    client_sr = _parse_rate(data.get("sample_rate_hz"))
                    fmt = data.get("format")
                    channels = data.get("channels", 1)
                    if client_sr is None or fmt not in (None, "pcm_s16le") or channels not in (1, "1"):
                        await _ws_error(
                            ws,
                            f"Only mono pcm_s16le at {STREAM_MIN_RATE}-{STREAM_MAX_RATE} Hz is supported "
                            f"(sample_rate_hz defaults to {config.STREAM_EXPECT_SR})",
                            1003,
                        )
                        return

                    context = data.get("context") or ""
                    if not isinstance(context, str):
                        context = ""
                    try:
                        stream = await service.run_infer(service.engine.new_stream, full_lang, context)
                    except Exception as e:
                        logger.exception(f"Failed to init stream: {e}")
                        await ws.close(code=1011, reason="stream init failed")
                        return
                    resampler = StreamResampler(client_sr, MODEL_RATE)
                    max_samples = int(config.STREAM_MAX_SEC * client_sr) if config.STREAM_MAX_SEC > 0 else 0

                    # Optional: acknowledge language selection
                    if full_lang is not None:
                        await ws.send_json({"type": "info", "message": f"language={full_lang}"})
                    continue

                if t == "stop":
                    if stream is None:
                        await _ws_error(ws, "stop before start", 1002)
                        return
                    # Flush remainder, finish, send final
                    buffer(resampler.flush())
                    await flush_and_infer(send_partial=False)
                    await service.run_infer(stream.finish)
                    await ws.send_json({"type": "final", "text": convert(stream.text), "language": stream.language})
                    await ws.close(code=1000)
                    return
                continue

            # Audio frames
            if msg.get("bytes"):
                if stream is None:
                    # Require explicit start so we can validate format.
                    await _ws_error(ws, "Send {type:'start', format:'pcm_s16le', sample_rate_hz:16000} first", 1002)
                    return

                audio = pcm.decode(msg["bytes"])
                if audio.size == 0:
                    continue
                if max_samples:
                    room = max_samples - total_samples
                    if room <= 0:
                        continue
                    if audio.size >= room:
                        audio = audio[:room]
                        await ws.send_json({"type": "info", "message": f"max_duration_reached={config.STREAM_MAX_SEC:g}s"})
                total_samples += audio.size

                buffer(resampler.process(audio))
                if buf_n >= config.STREAM_MIN_SAMPLES:
                    await flush_and_infer(send_partial=True)

    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.exception(f"WS Error: {e}")
        try:
            await ws.close(code=1011, reason="internal error")
        except Exception:
            pass


def _parse_rate(value) -> Optional[int]:
    """sample_rate_hz from `start` (default STREAM_EXPECT_SR); None if invalid or out of range."""
    if value is None or value == "":
        return config.STREAM_EXPECT_SR
    try:
        rate = int(value)
    except (TypeError, ValueError):
        return None
    if rate != value and str(rate) != str(value):
        return None  # e.g. 16000.5
    return rate if STREAM_MIN_RATE <= rate <= STREAM_MAX_RATE else None
