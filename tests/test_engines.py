"""Engine logic that runs without a model."""

from types import SimpleNamespace

import numpy as np


def test_vllm_feed_decodes_a_backlog_in_one_step():
    from asr_server.engines.vllm_engine import VllmStream

    steps = []

    class Model:
        def streaming_transcribe(self, pcm, state):  # same chunking as qwen_asr
            state.buffer = np.concatenate([state.buffer, pcm])
            while state.buffer.shape[0] >= state.chunk_size_samples:
                state.buffer = state.buffer[state.chunk_size_samples:]
                steps.append(state.chunk_size_samples)

    state = SimpleNamespace(chunk_size_samples=16000, buffer=np.zeros(0, np.float32), text="", language="")
    stream = VllmStream(Model(), state)
    assert stream.feed(np.zeros(8000, np.float32)) is False
    assert stream.feed(np.zeros(8000, np.float32)) is True
    assert stream.feed(np.zeros(16000 * 3 + 500, np.float32)) is True
    assert steps == [16000, 48000]
    assert state.chunk_size_samples == 16000 and state.buffer.shape[0] == 500


def test_onnx_long_audio_is_transcribed_in_segments(monkeypatch):
    from asr_server.engines import onnx_engine

    engine = onnx_engine.OnnxEngine()
    calls = []

    def transcribe_tokens(audio, language, context, prefix=()):
        calls.append((audio.shape[0], tuple(prefix)))
        return [len(calls)]

    monkeypatch.setattr(engine, "transcribe_tokens", transcribe_tokens)
    monkeypatch.setattr(engine, "parse", lambda tokens, language: (f"part{tokens[0]}", "English" if tokens[0] != 2 else "German"))
    audio = np.random.default_rng(0).standard_normal(16000 * 70).astype(np.float32)

    text, language = engine.transcribe_long(audio, None, "", prefix=[7, 8], prefix_samples=16000 * 15)
    assert text == "part1 part2 part3" and language == "English"
    assert len(calls) == 3 and sum(n for n, _ in calls) == audio.shape[0]
    assert all(n <= 16000 * onnx_engine.SEGMENT_SEC for n, _ in calls)
    assert [p for _, p in calls] == [(7, 8), (), ()]  # the partial's tokens continue the first segment only

    # a prefix covering more audio than the first segment is not used
    calls.clear()
    engine.transcribe_long(audio, None, "", prefix=[7], prefix_samples=16000 * 40)
    assert calls[0][1] == ()
