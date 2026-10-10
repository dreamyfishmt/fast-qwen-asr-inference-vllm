import logging

from .conftest import FakeEngine, wav_bytes


def test_ready_and_health(make_client):
    c = make_client()
    assert c.get("/ready").json() == {"status": "ready"}
    assert c.get("/health").json()["backend"] == "fake"


def test_auth(make_client):
    c = make_client(API_TOKEN="s3cret")
    assert c.get("/health").status_code == 401
    assert c.get("/health", headers={"Authorization": "Bearer s3cret"}).status_code == 200
    assert c.get("/health?token=s3cret").status_code == 200
    assert c.get("/ready").status_code == 200


def test_token_not_in_access_log(make_client, caplog):
    from asr_server.auth import RedactTokenFilter

    make_client(API_TOKEN="s3cret")
    access = logging.getLogger("uvicorn.access")
    assert any(isinstance(f, RedactTokenFilter) for f in access.filters)
    with caplog.at_level(logging.INFO, logger="uvicorn.access"):
        access.info('%s - "%s %s HTTP/%s" %d', "127.0.0.1:1", "GET", "/health?token=s3cret", "1.1", 200)
    assert "s3cret" not in caplog.text and "token=***" in caplog.text


def test_transcribe(make_client):
    c = make_client()
    r = c.post("/transcribe?language=de", files=[("files", ("a.wav", wav_bytes(1.0, 8000))), ("files", ("b.wav", wav_bytes(0.5)))])
    assert r.status_code == 200
    assert r.json() == [{"text": "8000 samples at 8000 Hz", "language": "German"},
                        {"text": "8000 samples at 16000 Hz", "language": "German"}]


def test_transcribe_bad_language(make_client):
    r = make_client().post("/transcribe?language=xx", files=[("files", ("a.wav", wav_bytes()))])
    assert r.status_code == 400


def test_transcribe_bad_audio(make_client):
    r = make_client().post("/transcribe", files=[("files", ("a.wav", b"garbage" * 50))])
    assert r.status_code == 400 and "a.wav" in r.json()["detail"]


def test_too_many_files(make_client):
    c = make_client(MAX_FILES=2)
    r = c.post("/transcribe", files=[("files", (f"{i}.wav", wav_bytes(0.1))) for i in range(3)])
    assert r.status_code == 413


def test_upload_too_large(make_client):
    c = make_client(MAX_UPLOAD_MB=0.05)  # ~52 kB
    r = c.post("/transcribe", files=[("files", ("a.wav", wav_bytes(3.0)))])  # ~96 kB
    assert r.status_code == 413
    assert c.post("/transcribe", files=[("files", ("a.wav", wav_bytes(0.5)))]).status_code == 200


def test_upload_too_large_chunked(make_client):
    c = make_client(MAX_UPLOAD_MB=0.05)
    body = b"x" * 100_000

    def chunks():
        for i in range(0, len(body), 10_000):
            yield body[i:i + 10_000]

    r = c.post("/transcribe", content=chunks(), headers={"content-type": "multipart/form-data; boundary=abc"})
    assert r.status_code == 413


def test_internal_errors_are_not_leaked(make_client):
    c = make_client(FakeEngine(fail_transcribe=True))
    r = c.post("/transcribe", files=[("files", ("a.wav", wav_bytes()))])
    assert r.status_code == 500 and "secret" not in r.text


def test_alignment_unsupported(make_client):
    r = make_client().post("/transcribe?forced_alignment=true", files=[("files", ("a.wav", wav_bytes()))])
    assert r.status_code == 503


def test_alignment(make_client):
    c = make_client(FakeEngine(supports_alignment=True))
    r = c.post("/transcribe?forced_alignment=true", files=[("files", ("a.wav", wav_bytes()))])
    assert r.json()[0]["timestamps"]["items"][0]["text"] == "Das"


def test_load_failure_stops_server(make_client, monkeypatch):
    import os
    import time

    killed = []
    monkeypatch.setattr(os, "kill", lambda pid, sig: killed.append(sig))
    monkeypatch.setattr(os, "_exit", lambda code: killed.append(("exit", code)))
    c = make_client(FakeEngine(fail_load=True), EXIT_ON_LOAD_FAILURE=True)
    for _ in range(100):
        if c.get("/ready").json()["status"] == "error":
            break
        time.sleep(0.02)
    assert c.get("/ready").status_code == 503
    assert len(killed) == 1
    r = c.post("/transcribe", files=[("files", ("a.wav", wav_bytes()))])
    assert r.status_code == 503
    c.__exit__(None, None, None)
    assert killed[-1] == ("exit", 1)
