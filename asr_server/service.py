"""Shared server state: the engine, its loading status and the concurrency limits."""

import asyncio
import logging
import os
import signal
from typing import List, Optional, Sequence

from . import config
from .audio import DecodeError, read_audio_file
from .engines import Audio, Engine
from .language import get_converter

logger = logging.getLogger(__name__)


class ServiceError(Exception):
    """An error with a message and HTTP status that are safe to return to the client."""

    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


class Service:
    def __init__(self, engine: Engine):
        self.engine = engine
        self.status = "starting"
        self.ready_event = asyncio.Event()
        self.decode_sem = asyncio.Semaphore(config.MAX_CONCURRENT_DECODE)
        self.infer_sem = asyncio.Semaphore(config.MAX_CONCURRENT_INFER)
        self.active_streams = 0
        self.exit_code = 0  # set when the server stops itself after a failed load

    async def run_infer(self, fn, *args):
        """Run a blocking engine call on a worker thread, within the inference limit."""
        async with self.infer_sem:
            return await asyncio.to_thread(fn, *args)

    # ---- loading ----

    async def load(self) -> None:
        logger.info(f"Background task: Loading models (backend: {self.engine.name})...")
        self.status = "loading_models"
        try:
            await asyncio.to_thread(self.engine.load)
        except Exception as e:
            logger.exception(f"Failed to load models: {e}")
            self.status = "error"
            self.ready_event.set()  # don't hang endpoints
            if config.EXIT_ON_LOAD_FAILURE:
                # Let the container restart policy retry instead of staying up, unhealthy, forever.
                logger.error("Stopping the server (EXIT_ON_LOAD_FAILURE=true)")
                self.exit_code = 1
                os.kill(os.getpid(), signal.SIGTERM)
            return

        logger.info("Warming up ASR model (best-effort)...")
        self.status = "warming_up"
        try:
            await self.run_infer(self.engine.warmup)
            logger.info("Warmup complete.")
        except Exception as e:
            logger.warning(f"Warmup failed (non-critical): {e}")

        self.status = "ready"
        self.ready_event.set()
        logger.info("Server is ready to accept requests.")

    async def wait_ready(self) -> None:
        await self.ready_event.wait()
        if self.status != "ready":
            raise ServiceError(503, f"Server not ready: {self.status}")

    # ---- batch transcription ----

    async def decode(self, files: Sequence[bytes], names: Sequence[Optional[str]]) -> List[Audio]:
        async def one(data: bytes, name: Optional[str]) -> Audio:
            async with self.decode_sem:
                return await asyncio.to_thread(read_audio_file, data, name)

        try:
            return list(await asyncio.gather(*(one(d, n) for d, n in zip(files, names))))
        except DecodeError as e:
            raise ServiceError(400, str(e))

    async def transcribe(
        self,
        audios: List[Audio],
        language: Optional[str],
        opencc_config: Optional[str],
        context: str = "",
        align: bool = False,
    ) -> List[dict]:
        """Returns [{"text", "language"[, "timestamps"]}] per input; timestamps are the aligner's result."""
        if align and not self.engine.supports_alignment:
            raise ServiceError(503, "Aligner model is not enabled or not supported by this backend.")
        convert = get_converter(opencc_config)
        try:
            results = await self.run_infer(self.engine.transcribe, audios, language, context)
            out = [{"text": convert(text), "language": lang} for text, lang in results]
            if align:
                texts = [text for text, _ in results]  # align against the model's original script
                alignments = await self.run_infer(self.engine.align, audios, texts, language)
                for item, alignment in zip(out, alignments):
                    item["timestamps"] = alignment
            return out
        except Exception as e:
            logger.exception(f"Inference failed: {e}")
            raise ServiceError(500, "Transcription failed (see server logs)")
