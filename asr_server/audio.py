"""Audio decoding (uploads) and resampling (uploads and streams)."""

import io
import logging
import os
import subprocess
import tempfile
from typing import Optional, Tuple

import numpy as np
import soundfile as sf

from .config import FFMPEG_TIMEOUT_SEC

logger = logging.getLogger(__name__)

MODEL_RATE = 16000


class DecodeError(ValueError):
    """An upload that can't be decoded. The message is safe to return to the client."""


def read_audio_file(file_bytes: bytes, filename: Optional[str] = None) -> Tuple[np.ndarray, int]:
    """
    Decode an uploaded file to (mono float32 waveform, sample rate). Blocking: call it from a worker thread.
    libsndfile first (WAV/FLAC/OGG/MP3); ffmpeg for everything else (M4A, ...) when it is installed.
    """
    if not file_bytes:
        raise DecodeError(f"{_label(filename)} is empty")
    try:
        with io.BytesIO(file_bytes) as f:
            wav, sr = sf.read(f, dtype="float32", always_2d=False)
    except Exception:
        wav, sr = _ffmpeg_decode(file_bytes, filename), MODEL_RATE
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if wav.size == 0:
        raise DecodeError(f"{_label(filename)} contains no audio")
    return np.ascontiguousarray(wav, dtype=np.float32), int(sr)


def _ffmpeg_decode(file_bytes: bytes, filename: Optional[str]) -> np.ndarray:
    # A seekable temp file instead of a pipe: MP4/M4A files with the index at the end can't be read from a pipe.
    suffix = os.path.splitext(filename or "")[1][:16]
    with tempfile.NamedTemporaryFile(suffix=suffix) as tmp:
        tmp.write(file_bytes)
        tmp.flush()
        try:
            proc = subprocess.run(
                ["ffmpeg", "-nostdin", "-v", "error", "-i", tmp.name,
                 "-vn", "-ac", "1", "-ar", str(MODEL_RATE), "-f", "f32le", "-acodec", "pcm_f32le", "-"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=FFMPEG_TIMEOUT_SEC if FFMPEG_TIMEOUT_SEC > 0 else None,
            )
        except FileNotFoundError:
            raise DecodeError(
                f"Unsupported audio format: {_label(filename)} (this server reads WAV, FLAC, OGG and MP3)"
            ) from None
        except subprocess.TimeoutExpired:
            logger.warning(f"ffmpeg timed out after {FFMPEG_TIMEOUT_SEC:g}s decoding {_label(filename)}")
            raise DecodeError(f"Decoding {_label(filename)} timed out") from None
    if proc.returncode != 0:
        logger.info(f"ffmpeg could not decode {_label(filename)}: {proc.stderr.decode(errors='ignore').strip()[-500:]}")
        raise DecodeError(f"Could not decode {_label(filename)}: unsupported or corrupt audio")
    return np.frombuffer(proc.stdout, dtype=np.float32)


def _label(filename: Optional[str]) -> str:
    return f"file '{filename}'" if filename else "audio file"


def resample(audio: np.ndarray, sr: int, target: int = MODEL_RATE) -> np.ndarray:
    """Resample a whole mono float32 waveform."""
    if sr == target:
        return audio
    import soxr

    return soxr.resample(audio, sr, target).astype(np.float32, copy=False)


class PcmS16Decoder:
    """Converts a stream of little-endian int16 bytes to float32 samples in [-1, 1].

    Frames don't have to end on a sample boundary: a trailing odd byte is kept for the next frame.
    """

    def __init__(self):
        self._carry = b""

    def decode(self, data: bytes) -> np.ndarray:
        if self._carry:
            data = self._carry + data
        n = len(data) & ~1
        self._carry = data[n:]
        return np.frombuffer(data, dtype="<i2", count=n // 2).astype(np.float32) / 32768.0


class StreamResampler:
    """Resamples a mono float32 stream chunk by chunk without seams at chunk boundaries."""

    def __init__(self, in_rate: int, out_rate: int = MODEL_RATE):
        self.in_rate = in_rate
        self.out_rate = out_rate
        self._stream = None
        if in_rate != out_rate:
            import soxr

            self._stream = soxr.ResampleStream(in_rate, out_rate, 1, dtype="float32")

    def process(self, audio: np.ndarray, last: bool = False) -> np.ndarray:
        if self._stream is None:
            return audio
        return self._stream.resample_chunk(audio, last=last)

    def flush(self) -> np.ndarray:
        """Remaining samples held back by the filter (call once, at the end of the stream)."""
        return self.process(np.zeros(0, dtype=np.float32), last=True)
