from .conftest import FakeEngine, wav_bytes


def post(c, data=None, name="a.wav", audio=None):
    return c.post("/v1/audio/transcriptions", data=data or {}, files={"file": (name, audio or wav_bytes(2.5))})


def test_json(make_client):
    r = post(make_client(), {"model": "whisper-1", "language": "de"})
    assert r.status_code == 200 and r.json() == {"text": "40000 samples at 16000 Hz"}


def test_text(make_client):
    r = post(make_client(), {"response_format": "text"})
    assert r.text == "40000 samples at 16000 Hz" and r.headers["content-type"].startswith("text/plain")


def test_verbose_json_without_aligner(make_client):
    body = post(make_client(), {"response_format": "verbose_json", "language": "de"}).json()
    assert body["language"] == "german" and body["duration"] == 2.5
    assert [(s["start"], s["end"]) for s in body["segments"]] == [(0.0, 2.5)]
    assert "words" not in body


def test_srt_and_vtt_without_aligner(make_client):
    c = make_client()
    assert post(c, {"response_format": "srt"}).text == "1\n00:00:00,000 --> 00:00:02,500\n40000 samples at 16000 Hz\n\n"
    assert post(c, {"response_format": "vtt"}).text.startswith("WEBVTT\n\n00:00:00.000 --> 00:00:02.500\n")


def test_word_timestamps(make_client):
    c = make_client(FakeEngine(supports_alignment=True))
    body = post(c, {"response_format": "verbose_json", "timestamp_granularities[]": ["word", "segment"]}).json()
    assert body["words"][0] == {"word": "Das", "start": 0.1, "end": 0.3}
    # the 1.5 s pause starts a new segment
    assert [(s["start"], s["end"], s["text"]) for s in body["segments"]] == [(0.1, 0.5, "Das ist"), (2.0, 2.4, "gut")]
    words_only = post(c, {"response_format": "verbose_json", "timestamp_granularities[]": "word"}).json()
    assert "segments" not in words_only and len(words_only["words"]) == 3
    srt = post(c, {"response_format": "srt"}).text
    assert "00:00:02,000 --> 00:00:02,400\ngut" in srt


def test_word_timestamps_need_aligner(make_client):
    r = post(make_client(), {"response_format": "verbose_json", "timestamp_granularities[]": "word"})
    assert r.status_code == 400 and r.json()["error"]["param"] == "timestamp_granularities[]"


def test_errors_are_openai_shaped(make_client):
    c = make_client()
    assert post(c, {"response_format": "xml"}).json()["error"]["type"] == "invalid_request_error"
    assert post(c, {"language": "xx"}).status_code == 400
    assert post(c, {"stream": "true"}).status_code == 400
    r = post(c, audio=b"garbage" * 30)
    assert r.status_code == 400 and r.json()["error"]["param"] == "file"


def test_internal_error_hidden(make_client):
    r = post(make_client(FakeEngine(fail_transcribe=True)))
    assert r.status_code == 500 and "secret" not in r.text


def test_auth_and_models(make_client):
    c = make_client(API_TOKEN="s3cret")
    assert post(c).status_code == 401
    r = c.get("/v1/models", headers={"Authorization": "Bearer s3cret"})
    assert r.json()["data"][0]["id"] == "qwen3-asr"


def test_upload_limit_applies(make_client):
    c = make_client(MAX_UPLOAD_MB=0.05)
    assert post(c, audio=wav_bytes(3.0)).status_code == 413
