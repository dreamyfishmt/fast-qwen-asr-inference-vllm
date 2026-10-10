import io
from typing import List, Optional

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from asr_server import config
from asr_server.app import create_app
from asr_server.engines import Engine, Stream


class FakeStream(Stream):
    def __init__(self):
        self.samples: List[np.ndarray] = []
        self.language = "English"
        self.text = ""

    def feed(self, pcm, partial=True):
        self.samples.append(pcm)
        self.text = f"{self.n} samples"
        return True

    def finish(self):
        self.text = f"final {self.n} samples"

    @property
    def n(self) -> int:
        return int(sum(s.size for s in self.samples))


class FakeEngine(Engine):
    """Reports what it was given instead of transcribing."""

    name = "fake"

    def __init__(self, supports_alignment=False, fail_load=False, fail_transcribe=False):
        self.supports_alignment = supports_alignment
        self.fail_load = fail_load
        self.fail_transcribe = fail_transcribe
        self.calls = []
        self.streams: List[FakeStream] = []

    def load(self):
        if self.fail_load:
            raise RuntimeError("model files missing at /secret/path")

    def transcribe(self, audios, language: Optional[str], context: str = ""):
        if self.fail_transcribe:
            raise RuntimeError("CUDA error at /secret/path")
        self.calls.append((audios, language, context))
        return [(f"{wav.shape[0]} samples at {sr} Hz", language or "German") for wav, sr in audios]

    def align(self, audios, texts, language):
        return [{"items": [{"text": "Das", "start_time": 0.1, "end_time": 0.3},
                           {"text": "ist", "start_time": 0.35, "end_time": 0.5},
                           {"text": "gut", "start_time": 2.0, "end_time": 2.4}]} for _ in audios]

    def new_stream(self, language, context=""):
        stream = FakeStream()
        self.streams.append(stream)
        return stream


@pytest.fixture
def make_client(monkeypatch):
    clients = []

    def make(engine: Optional[FakeEngine] = None, **settings):
        monkeypatch.setattr(config, "EXIT_ON_LOAD_FAILURE", False)
        for key, value in settings.items():
            monkeypatch.setattr(config, key, value)
        engine = engine or FakeEngine()
        client = TestClient(create_app(engine))
        client.__enter__()
        clients.append(client)
        client.engine = engine
        return client

    yield make
    for c in clients:
        c.__exit__(None, None, None)


def wav_bytes(seconds: float = 1.0, sr: int = 16000, channels: int = 1) -> bytes:
    t = np.arange(int(seconds * sr)) / sr
    wav = 0.1 * np.sin(2 * np.pi * 440 * t)
    if channels > 1:
        wav = np.stack([wav] * channels, axis=1)
    buf = io.BytesIO()
    sf.write(buf, wav, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()
