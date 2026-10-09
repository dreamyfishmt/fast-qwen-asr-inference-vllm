"""ASR inference backends.

server.py owns the HTTP/WebSocket protocol; an engine owns model loading and
inference. All engine methods are blocking and are called from worker threads.

Select one with ASR_BACKEND:
  vllm  - qwen-asr + vLLM on an NVIDIA GPU (engines/vllm_engine.py)
  onnx  - ONNX Runtime on CPU (engines/onnx_engine.py)
"""

from typing import List, Optional, Sequence, Tuple

import numpy as np

# (waveform float32 mono, sample rate)
Audio = Tuple[np.ndarray, int]


class Stream:
    """One streaming utterance. `text` / `language` hold the latest result."""

    text: str = ""
    language: str = ""

    def feed(self, pcm: np.ndarray, partial: bool = True) -> bool:
        """Add 16 kHz float32 samples. Returns True if `text` may have changed.
        partial=False: more audio is not expected (stop), don't start partial work."""
        raise NotImplementedError

    def finish(self) -> None:
        """Decode any remaining audio and set the final `text`."""
        raise NotImplementedError


class Engine:
    name = "base"
    supports_alignment = False

    def load(self) -> None:
        raise NotImplementedError

    def warmup(self) -> None:
        pass

    def transcribe(
        self, audios: Sequence[Audio], language: Optional[str], context: str = ""
    ) -> List[Tuple[str, str]]:
        """Batch transcription. Returns [(text, language)] per input."""
        raise NotImplementedError

    def align(self, audios: Sequence[Audio], texts: Sequence[str], language: Optional[str]) -> list:
        raise NotImplementedError(f"{self.name} backend does not support forced alignment")

    def new_stream(self, language: Optional[str], context: str = "") -> Stream:
        raise NotImplementedError

    def streaming_config(self) -> dict:
        return {}

    def memory_info(self) -> dict:
        """Backend-specific memory stats for /health (e.g. GPU usage)."""
        return {}


def create_engine(backend: str) -> Engine:
    backend = backend.strip().lower()
    if backend == "vllm":
        from .vllm_engine import VllmEngine
        return VllmEngine()
    if backend == "onnx":
        from .onnx_engine import OnnxEngine
        return OnnxEngine()
    raise ValueError(f"Unknown ASR_BACKEND: {backend!r} (expected 'vllm' or 'onnx')")
