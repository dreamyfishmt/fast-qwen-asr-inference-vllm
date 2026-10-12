import shutil

import numpy as np
import pytest

from asr_server.audio import DecodeError, PcmS16Decoder, StreamResampler, read_audio_file, resample

from .conftest import wav_bytes


def test_read_wav_downmixes_to_mono():
    wav, sr = read_audio_file(wav_bytes(0.5, 22050, channels=2), "a.wav")
    assert sr == 22050 and wav.ndim == 1 and wav.dtype == np.float32 and wav.shape[0] == 11025


def test_empty_upload_is_rejected():
    with pytest.raises(DecodeError, match="empty"):
        read_audio_file(b"", "a.wav")


def test_garbage_upload_gives_client_safe_error():
    with pytest.raises(DecodeError) as e:
        read_audio_file(b"not audio at all" * 10, "x.bin")
    assert "x.bin" in str(e.value) and "ffmpeg" not in str(e.value).lower()


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_ffmpeg_decodes_m4a():
    with open("files/reference.m4a", "rb") as f:
        wav, sr = read_audio_file(f.read(), "reference.m4a")
    assert sr == 16000 and 1.5 < wav.shape[0] / sr < 3


def test_ffmpeg_missing(monkeypatch):
    import subprocess

    def missing(*a, **k):
        raise FileNotFoundError("ffmpeg")

    monkeypatch.setattr(subprocess, "run", missing)
    with pytest.raises(DecodeError, match="Unsupported audio format"):
        read_audio_file(b"\x00\x01" * 100, "a.m4a")


def test_ffmpeg_timeout(monkeypatch):
    import subprocess

    def slow(*a, **k):
        raise subprocess.TimeoutExpired("ffmpeg", k.get("timeout"))

    monkeypatch.setattr(subprocess, "run", slow)
    with pytest.raises(DecodeError, match="timed out"):
        read_audio_file(b"\x00\x01" * 100, "a.m4a")


def test_pcm_decoder_carries_odd_byte():
    samples = np.array([1000, -2000, 3000, -32768], dtype="<i2")
    data = samples.tobytes()
    dec = PcmS16Decoder()
    parts = [dec.decode(data[:3]), dec.decode(data[3:4]), dec.decode(data[4:7]), dec.decode(data[7:])]
    out = np.concatenate(parts)
    np.testing.assert_allclose(out, samples / 32768.0)


def test_stream_resampler_matches_length_and_is_seamless():
    sr = 48000
    t = np.arange(sr) / sr
    tone = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    rs = StreamResampler(sr, 16000)
    out = np.concatenate([rs.process(tone[i:i + 480]) for i in range(0, sr, 480)] + [rs.flush()])
    assert abs(out.size - 16000) <= 2
    whole = resample(tone, sr, 16000)
    # chunked and one-shot resampling agree away from the edges
    np.testing.assert_allclose(out[200:15800], whole[200:15800], atol=2e-3)


def test_stream_resampler_passthrough():
    x = np.ones(10, dtype=np.float32)
    rs = StreamResampler(16000)
    assert rs.process(x) is x and rs.flush().size == 0


def test_split_on_silence_cuts_at_pauses():
    from asr_server.audio import split_on_silence

    rng = np.random.default_rng(0)
    audio = (0.3 * rng.standard_normal(16000 * 70)).astype(np.float32)
    audio[16000 * 25 : 16000 * 25 + 4000] = 0  # pause at 25 s
    audio[16000 * 52 : 16000 * 52 + 4000] = 0  # pause at 52 s
    pieces = split_on_silence(audio, 16000 * 30, 16000 * 8)
    assert len(pieces) == 3
    assert all(p.shape[0] <= 16000 * 30 for p in pieces)
    np.testing.assert_array_equal(np.concatenate(pieces), audio)
    cuts = np.cumsum([p.shape[0] for p in pieces])[:-1] / 16000
    assert 25 <= cuts[0] <= 25.25 and 52 <= cuts[1] <= 52.25


def test_split_on_silence_short_or_disabled():
    from asr_server.audio import split_on_silence

    audio = np.ones(16000 * 10, dtype=np.float32)
    assert len(split_on_silence(audio, 16000 * 30, 16000 * 8)) == 1
    assert len(split_on_silence(np.ones(16000 * 100, dtype=np.float32), 0, 16000 * 8)) == 1
