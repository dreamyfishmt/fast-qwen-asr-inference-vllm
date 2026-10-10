"""qwen-asr + vLLM backend (NVIDIA GPU)."""

import logging
from typing import List, Optional, Sequence, Tuple

import numpy as np

from ..config import env
from . import Audio, Engine, Stream

logger = logging.getLogger(__name__)


GROUP = "vLLM backend"

ENABLE_ASR_MODEL = env("ENABLE_ASR_MODEL", True, "Load the ASR model", group=GROUP)
ASR_MODEL_NAME = env("ASR_MODEL_NAME", "Qwen/Qwen3-ASR-1.7B", "ASR model path (or Hugging Face id)", group=GROUP)
GPU_MEMORY_UTILIZATION = env("GPU_MEMORY_UTILIZATION", 0.75, "Fraction of GPU memory vLLM may reserve", group=GROUP)
MAX_NEW_TOKENS = env("MAX_NEW_TOKENS", 4096, "Most tokens generated per transcription", group=GROUP)
VLLM_QUANTIZATION = env(
    "VLLM_QUANTIZATION", "", "vLLM quantization method, e.g. `modelopt` for ModelOpt FP8 checkpoints (empty = from the checkpoint config)",
    group=GROUP,
)
ENABLE_ALIGNER_MODEL = env("ENABLE_ALIGNER_MODEL", True, "Load the forced aligner (timestamps)", group=GROUP)
ALIGNER_MODEL_NAME = env("ALIGNER_MODEL_NAME", "Qwen/Qwen3-ForcedAligner-0.6B", "Forced aligner path (or Hugging Face id)", group=GROUP)

# Streaming decoder params (qwen_asr init_streaming_state).
# Smaller chunk size -> partial text updates more often, at the cost of more GPU calls.
STREAM_CHUNK_SIZE_SEC = env(
    "STREAM_CHUNK_SIZE_SEC", 2.0, "Audio seconds per streaming decode step (smaller = faster partials, more GPU work)", group=GROUP,
)
STREAM_UNFIXED_CHUNK_NUM = env("STREAM_UNFIXED_CHUNK_NUM", 2, "First N chunks are decoded without a text prefix", group=GROUP)
STREAM_UNFIXED_TOKEN_NUM = env("STREAM_UNFIXED_TOKEN_NUM", 5, "Trailing tokens rolled back (re-decoded) on each step", group=GROUP)


class VllmStream(Stream):
    def __init__(self, model, state):
        self._model = model
        self._state = state

    @property
    def text(self) -> str:
        return self._state.text

    @property
    def language(self) -> str:
        return self._state.language

    def feed(self, pcm: np.ndarray, partial: bool = True) -> bool:
        # streaming_transcribe() buffers internally and decodes once per full chunk
        self._model.streaming_transcribe(pcm, self._state)
        return True

    def finish(self) -> None:
        self._model.finish_streaming_transcribe(self._state)


class VllmEngine(Engine):
    name = "vllm"

    def __init__(self):
        self.asr = None
        self.aligner = None

    @property
    def supports_alignment(self) -> bool:
        return self.aligner is not None

    def load(self) -> None:
        from qwen_asr import Qwen3ASRModel, Qwen3ForcedAligner

        if ENABLE_ASR_MODEL:
            llm_kwargs = {"quantization": VLLM_QUANTIZATION} if VLLM_QUANTIZATION else {}
            if VLLM_QUANTIZATION:
                logger.info(f"Using vLLM quantization: {VLLM_QUANTIZATION}")
            logger.info(f"Loading ASR Model: {ASR_MODEL_NAME}...")
            self.asr = Qwen3ASRModel.LLM(
                model=ASR_MODEL_NAME,
                gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
                max_new_tokens=MAX_NEW_TOKENS,
                **llm_kwargs,
            )
            logger.info("ASR Model loaded successfully.")
        else:
            logger.info("ASR Model disabled via ENABLE_ASR_MODEL.")

        if ENABLE_ALIGNER_MODEL:
            import torch

            logger.info(f"Loading Aligner Model: {ALIGNER_MODEL_NAME}...")
            self.aligner = Qwen3ForcedAligner.from_pretrained(
                ALIGNER_MODEL_NAME, dtype=torch.bfloat16, device_map="cuda:0"
            )
            logger.info("Aligner Model loaded successfully.")
        else:
            logger.info("Aligner Model disabled via ENABLE_ALIGNER_MODEL.")

        if self.asr is None:
            raise RuntimeError("ASR model is not enabled (ENABLE_ASR_MODEL=false).")

    def warmup(self) -> None:
        dummy_wav = np.zeros(16000, dtype=np.float32)
        self.asr.transcribe(audio=[(dummy_wav, 16000)], language=["English"], return_time_stamps=False)
        stream = self.new_stream(None)
        for n in [320, 640, 1024, 3200] + [3200] * 25:
            stream.feed(dummy_wav[:n])
        stream.finish()

    def transcribe(
        self, audios: Sequence[Audio], language: Optional[str], context: str = ""
    ) -> List[Tuple[str, str]]:
        results = self.asr.transcribe(
            audio=list(audios),
            context=[context] * len(audios),
            language=[language] * len(audios),
            return_time_stamps=False,
        )
        return [(r.text, r.language) for r in results]

    def align(self, audios: Sequence[Audio], texts: Sequence[str], language: Optional[str]) -> list:
        if self.aligner is None:
            raise RuntimeError("Aligner model is not enabled or failed to load.")
        return self.aligner.align(audio=list(audios), text=list(texts), language=[language] * len(audios))

    def new_stream(self, language: Optional[str], context: str = "") -> Stream:
        state = self.asr.init_streaming_state(
            context=context,
            language=language,
            unfixed_chunk_num=STREAM_UNFIXED_CHUNK_NUM,
            unfixed_token_num=STREAM_UNFIXED_TOKEN_NUM,
            chunk_size_sec=STREAM_CHUNK_SIZE_SEC,
        )
        return VllmStream(self.asr, state)

    def memory_info(self) -> dict:
        try:
            import torch
        except ImportError:
            return {}
        if not torch.cuda.is_available():
            return {}
        return {
            "gpu_allocated_mb": torch.cuda.memory_allocated() // (1024 * 1024),
            "gpu_reserved_mb": torch.cuda.memory_reserved() // (1024 * 1024),
        }

    def streaming_config(self) -> dict:
        return {
            "chunk_size_sec": STREAM_CHUNK_SIZE_SEC,
            "unfixed_chunk_num": STREAM_UNFIXED_CHUNK_NUM,
            "unfixed_token_num": STREAM_UNFIXED_TOKEN_NUM,
        }
