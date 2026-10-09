"""qwen-asr + vLLM backend (NVIDIA GPU)."""

import logging
import os
from typing import List, Optional, Sequence, Tuple

import numpy as np

from . import Audio, Engine, Stream

logger = logging.getLogger(__name__)


def _env_bool(key: str, default: str) -> bool:
    return os.getenv(key, default).lower() in ("true", "1", "yes", "on")


# Streaming decoder params (qwen_asr init_streaming_state).
# Smaller chunk size -> partial text updates more often, at the cost of more GPU calls.
STREAM_CHUNK_SIZE_SEC = float(os.getenv("STREAM_CHUNK_SIZE_SEC", "2.0"))
STREAM_UNFIXED_CHUNK_NUM = int(os.getenv("STREAM_UNFIXED_CHUNK_NUM", "2"))
STREAM_UNFIXED_TOKEN_NUM = int(os.getenv("STREAM_UNFIXED_TOKEN_NUM", "5"))


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

    def feed(self, pcm: np.ndarray) -> bool:
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

        if _env_bool("ENABLE_ASR_MODEL", "true"):
            model_name = os.getenv("ASR_MODEL_NAME", "Qwen/Qwen3-ASR-1.7B")
            gpu_mem = float(os.getenv("GPU_MEMORY_UTILIZATION", "0.75"))
            max_new_tokens = int(os.getenv("MAX_NEW_TOKENS", "4096"))
            # vLLM quantization method, e.g. "modelopt" for ModelOpt FP8 checkpoints. Empty = from checkpoint config.
            quantization = os.getenv("VLLM_QUANTIZATION", "").strip() or None
            llm_kwargs = {"quantization": quantization} if quantization else {}
            if quantization:
                logger.info(f"Using vLLM quantization: {quantization}")
            logger.info(f"Loading ASR Model: {model_name}...")
            self.asr = Qwen3ASRModel.LLM(
                model=model_name,
                gpu_memory_utilization=gpu_mem,
                max_new_tokens=max_new_tokens,
                **llm_kwargs,
            )
            logger.info("ASR Model loaded successfully.")
        else:
            logger.info("ASR Model disabled via ENABLE_ASR_MODEL.")

        if _env_bool("ENABLE_ALIGNER_MODEL", "true"):
            import torch

            aligner_name = os.getenv("ALIGNER_MODEL_NAME", "Qwen/Qwen3-ForcedAligner-0.6B")
            logger.info(f"Loading Aligner Model: {aligner_name}...")
            self.aligner = Qwen3ForcedAligner.from_pretrained(
                aligner_name, dtype=torch.bfloat16, device_map="cuda:0"
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
