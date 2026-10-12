import time

import numpy as np
import pytest
from starlette.websockets import WebSocketDisconnect


def pcm(seconds: float, sr: int = 16000) -> bytes:
    return (np.zeros(int(seconds * sr), dtype="<i2") + 100).tobytes()


def run(ws, start: dict, chunks, chunk_bytes=None):
    assert ws.receive_json() == {"type": "ready"}
    ws.send_json(start)
    for c in chunks:
        ws.send_bytes(c)
    ws.send_json({"type": "stop"})
    msgs = []
    while True:
        m = ws.receive_json()
        msgs.append(m)
        if m["type"] in ("final", "error"):
            return msgs


def test_stream_16k(make_client):
    c = make_client()
    with c.websocket_connect("/transcribe-streaming") as ws:
        msgs = run(ws, {"type": "start", "format": "pcm_s16le", "sample_rate_hz": 16000}, [pcm(1.0)])
    assert msgs[-1] == {"type": "final", "text": "final 16000 samples", "language": "English"}


def test_odd_sized_frames(make_client):
    c = make_client()
    data = pcm(0.5)
    chunks = [data[i:i + 333] for i in range(0, len(data), 333)]  # odd sizes split samples
    with c.websocket_connect("/transcribe-streaming") as ws:
        msgs = run(ws, {"type": "start", "sample_rate_hz": 16000}, chunks)
    assert msgs[-1]["text"] == "final 8000 samples"
    fed = np.concatenate(c.engine.streams[0].samples)
    np.testing.assert_allclose(fed, 100 / 32768.0)


@pytest.mark.parametrize("sr", [8000, 44100, 48000])
def test_other_sample_rates_are_resampled(make_client, sr):
    c = make_client()
    with c.websocket_connect("/transcribe-streaming") as ws:
        msgs = run(ws, {"type": "start", "format": "pcm_s16le", "sample_rate_hz": sr}, [pcm(0.1, sr)] * 10)
    n = c.engine.streams[0].n
    assert msgs[-1]["type"] == "final" and abs(n - 16000) <= 2


def test_default_sample_rate(make_client):
    c = make_client()
    with c.websocket_connect("/transcribe-streaming") as ws:
        msgs = run(ws, {"type": "start"}, [pcm(0.2)])
    assert msgs[-1]["text"] == "final 3200 samples"


@pytest.mark.parametrize("start", [
    {"type": "start", "sample_rate_hz": 4000},
    {"type": "start", "sample_rate_hz": "abc"},
    {"type": "start", "format": "opus"},
    {"type": "start", "channels": 2},
])
def test_bad_start(make_client, start):
    c = make_client()
    with c.websocket_connect("/transcribe-streaming") as ws:
        assert ws.receive_json()["type"] == "ready"
        ws.send_json(start)
        assert ws.receive_json()["type"] == "error"
        with pytest.raises(WebSocketDisconnect) as e:
            ws.receive_json()
        assert e.value.code == 1003


def test_max_duration(make_client):
    c = make_client(STREAM_MAX_SEC=0.5)
    with c.websocket_connect("/transcribe-streaming") as ws:
        msgs = run(ws, {"type": "start", "sample_rate_hz": 48000}, [pcm(0.2, 48000)] * 5)
    assert {"type": "info", "message": "max_duration_reached=0.5s"} in msgs
    assert abs(c.engine.streams[0].n - 8000) <= 2


def test_idle_timeout(make_client):
    c = make_client(STREAM_IDLE_TIMEOUT_SEC=0.2)
    with c.websocket_connect("/transcribe-streaming") as ws:
        assert ws.receive_json()["type"] == "ready"
        msg = ws.receive_json()
        assert msg["type"] == "error" and "STREAM_IDLE_TIMEOUT_SEC" in msg["message"]
        with pytest.raises(WebSocketDisconnect) as e:
            ws.receive_json()
        assert e.value.code == 1008


def test_max_streams(make_client):
    c = make_client(MAX_STREAMS=1)
    with c.websocket_connect("/transcribe-streaming") as ws1:
        assert ws1.receive_json()["type"] == "ready"
        with c.websocket_connect("/transcribe-streaming") as ws2:
            assert ws2.receive_json()["type"] == "error"
            with pytest.raises(WebSocketDisconnect) as e:
                ws2.receive_json()
            assert e.value.code == 1013
    # the slot is released after a stream ends
    with c.websocket_connect("/transcribe-streaming") as ws:
        assert run(ws, {"type": "start"}, [pcm(0.1)])[-1]["type"] == "final"


def test_ws_auth(make_client):
    c = make_client(API_TOKEN="s3cret")
    with pytest.raises(WebSocketDisconnect):
        with c.websocket_connect("/transcribe-streaming") as ws:
            ws.receive_json()
    with c.websocket_connect("/transcribe-streaming?token=s3cret") as ws:
        assert ws.receive_json()["type"] == "ready"


def test_audio_is_read_while_inference_runs(make_client):
    from .conftest import FakeEngine, FakeStream

    class Slow(FakeStream):
        def feed(self, pcm, partial=True):
            time.sleep(0.05)
            return super().feed(pcm, partial)

    class Engine(FakeEngine):
        def new_stream(self, language, context=""):
            self.streams.append(Slow())
            return self.streams[-1]

    c = make_client(Engine())
    with c.websocket_connect("/transcribe-streaming") as ws:
        msgs = run(ws, {"type": "start"}, [pcm(0.1)] * 30)
    stream = c.engine.streams[0]
    assert msgs[-1]["text"] == "final 48000 samples"
    # audio that arrived during a feed() call was fed in one piece afterwards
    assert len(stream.samples) < 30


def test_feed_size_follows_the_engine(make_client):
    from .conftest import FakeEngine

    class Engine(FakeEngine):
        def stream_feed_samples(self):
            return 8000

    c = make_client(Engine())
    with c.websocket_connect("/transcribe-streaming") as ws:
        msgs = run(ws, {"type": "start"}, [pcm(0.1)] * 13)
    sizes = [s.size for s in c.engine.streams[0].samples]
    assert msgs[-1]["text"] == "final 20800 samples"
    assert all(n >= 8000 for n in sizes[:-1]) and sum(sizes) == 20800


def test_throttled_partial_is_sent_later(make_client):
    c = make_client(PARTIAL_INTERVAL_MS=300)
    with c.websocket_connect("/transcribe-streaming") as ws:
        assert ws.receive_json()["type"] == "ready"
        ws.send_json({"type": "start"})
        ws.send_bytes(pcm(0.1))
        assert ws.receive_json() == {"type": "partial", "text": "1600 samples", "language": "English"}
        sent = time.monotonic()
        ws.send_bytes(pcm(0.1))  # within PARTIAL_INTERVAL_MS: held back, not dropped
        assert ws.receive_json() == {"type": "partial", "text": "3200 samples", "language": "English"}
        assert time.monotonic() - sent >= 0.2
        ws.send_json({"type": "stop"})
        assert ws.receive_json()["type"] == "final"


def test_disconnect_mid_stream_releases_the_slot(make_client):
    c = make_client(MAX_STREAMS=1)
    with c.websocket_connect("/transcribe-streaming") as ws:
        assert ws.receive_json()["type"] == "ready"
        ws.send_json({"type": "start"})
        ws.send_bytes(pcm(0.5))
    with c.websocket_connect("/transcribe-streaming") as ws:
        assert run(ws, {"type": "start"}, [pcm(0.1)])[-1]["type"] == "final"
