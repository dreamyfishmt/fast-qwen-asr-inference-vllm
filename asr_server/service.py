"""Shared server state: the engine, its loading status and the concurrency limits."""

import asyncio
import heapq
import itertools
import logging
import os
import signal
from contextlib import asynccontextmanager
from typing import List, Optional, Sequence

from . import config
from .audio import DecodeError, read_audio_file
from .engines import Audio, Engine
from .language import get_converter
from .timeline import assign_speakers, build_segments, words_from_alignment

logger = logging.getLogger(__name__)


class ServiceError(Exception):
    """An error with a message and HTTP status that are safe to return to the client."""

    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


class PrioritySemaphore:
    """A semaphore whose waiters are served by priority (lower first), then in arrival order."""

    def __init__(self, value: int):
        self._free = max(1, value)
        self._waiters: list = []  # heap of (priority, seq, future)
        self._seq = itertools.count()

    @asynccontextmanager
    async def slot(self, priority: int):
        if self._free > 0 and not self._waiters:
            self._free -= 1
        else:
            fut = asyncio.get_running_loop().create_future()
            heapq.heappush(self._waiters, (priority, next(self._seq), fut))
            try:
                await fut
            except asyncio.CancelledError:
                if fut.done() and not fut.cancelled():
                    self._release()  # the slot was handed over just as we were cancelled
                raise
        try:
            yield
        finally:
            self._release()

    def _release(self) -> None:
        while self._waiters:
            _, _, fut = heapq.heappop(self._waiters)
            if not fut.done():
                fut.set_result(None)  # hand the slot over directly
                return
        self._free += 1


async def _in_thread(fn, *args):
    """asyncio.to_thread(), except that when cancelled it still waits for the call to return.

    A thread can't be interrupted: without this, a cancelled caller would leave its inference slot
    while the engine is still running, and the next call could overlap it.
    """
    task = asyncio.ensure_future(asyncio.to_thread(fn, *args))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        await asyncio.wait({task})
        raise


# Priorities for PrioritySemaphore: streams are latency-sensitive, uploads wait behind them
STREAM, BATCH = 0, 1


class Service:
    def __init__(self, engine: Engine, diarizer=None):
        self.engine = engine
        self.diarizer = diarizer  # asr_server.diarization.Diarizer, or None
        self.status = "starting"
        self.ready_event = asyncio.Event()
        self.decode_sem = asyncio.Semaphore(config.MAX_CONCURRENT_DECODE)
        self.infer_sem = PrioritySemaphore(config.MAX_CONCURRENT_INFER)
        self.active_streams = 0
        self.exit_code = 0  # set when the server stops itself after a failed load

    async def run_infer(self, fn, *args):
        """Run a blocking engine call (batch work) on a worker thread, within the inference limit."""
        async with self.infer_sem.slot(BATCH):
            return await _in_thread(fn, *args)

    async def run_stream(self, fn, *args):
        """Run a blocking streaming call on a worker thread.

        Engines whose calls must not overlap (`exclusive`, e.g. vLLM) share the inference limit with
        uploads but go first; other engines (ONNX) synchronize internally and are called directly, so a
        long upload doesn't hold up the streams.
        """
        if not self.engine.exclusive:
            return await _in_thread(fn, *args)
        async with self.infer_sem.slot(STREAM):
            return await _in_thread(fn, *args)

    # ---- loading ----

    async def load(self) -> None:
        logger.info(f"Background task: Loading models (backend: {self.engine.name})...")
        self.status = "loading_models"
        try:
            await asyncio.to_thread(self.engine.load)
            if self.diarizer is not None:
                await asyncio.to_thread(self.diarizer.load)
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
            if self.diarizer is not None:
                await self.run_infer(self.diarizer.warmup)
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
        diarize: bool = False,
        num_speakers: Optional[int] = None,
    ) -> List[dict]:
        """Returns [{"text", "language"[, "timestamps"][, "speakers"][, "segments"]}] per input.

        timestamps: the aligner's result (align); speakers: speaker turns (diarize); segments (with timestamps):
        the transcript in timed pieces, each with its speaker when diarized. `diarize` also aligns when the
        engine can, so that the words can be given speakers.
        """
        if align and not self.engine.supports_alignment:
            raise ServiceError(503, "Aligner model is not enabled or not supported by this backend.")
        if diarize and self.diarizer is None:
            raise ServiceError(503, "Speaker diarization is not enabled (ENABLE_DIARIZATION=true).")
        align = align or (diarize and self.engine.supports_alignment)
        convert = get_converter(opencc_config)
        try:
            if align:
                results = await self.run_infer(self.engine.transcribe_aligned, audios, language, context)
            else:
                results = [(text, lang, None) for text, lang in await self.run_infer(self.engine.transcribe, audios, language, context)]
            out = []
            for (wav, sr), (text, lang, alignment) in zip(audios, results):
                item = {"text": convert(text), "language": lang}
                if align:
                    item["timestamps"] = alignment
                if diarize:
                    item["speakers"] = await self.run_infer(self.diarizer.diarize, wav, sr, num_speakers)
                if align:
                    words = words_from_alignment(alignment)  # timed words in the model's original script
                    assign_speakers(words, item.get("speakers", []))
                    item["segments"] = build_segments(words, text, round(wav.shape[0] / sr, 3), convert)
                out.append(item)
            return out
        except Exception as e:
            logger.exception(f"Inference failed: {e}")
            raise ServiceError(500, "Transcription failed (see server logs)")
